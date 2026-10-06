"""Read-only access to stable-worldmodel style HDF5 recordings.

The layout is flat: every per-frame column has one row per frame, with all
episodes concatenated, and ``ep_len`` / ``ep_offset`` index the episodes.
Recordings without those two columns fall back to an ``episode_id``-like
column, or to a single episode spanning the whole file.
"""

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

INDEX_KEYS = ("ep_len", "ep_offset")
EPISODE_ID_KEYS = ("episode_id", "episode_index", "ep_idx")
STATE_COLUMNS = ("observation", "state", "proprio")
H5_SUFFIXES = (".h5", ".hdf5")


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "image" or "series"
    shape: tuple
    dtype: str
    names: list | None = None


def discover(target):
    """Map display name -> path for one HDF5 file or every HDF5 file under a directory."""
    target = Path(target)
    if target.is_file():
        return {target.name: target}
    if not target.is_dir():
        raise FileNotFoundError(target)
    paths = sorted(p for p in target.rglob("*") if p.suffix in H5_SUFFIXES and p.is_file())
    if not paths:
        raise FileNotFoundError(f"No HDF5 files under {target}")
    return {str(p.relative_to(target)): p for p in paths}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _image_layout(shape, dtype):
    """Return "hwc", "chw", "hw" for per-frame image columns, else None."""
    if dtype != np.uint8:
        return None
    frame = shape[1:]
    if len(frame) == 3 and frame[-1] in (1, 3, 4) and min(frame[:2]) >= 8:
        return "hwc"
    if len(frame) == 3 and frame[0] in (1, 3, 4) and min(frame[1:]) >= 8:
        return "chw"
    if len(frame) == 2 and min(frame) >= 8:
        return "hw"
    return None


class Recording:
    """One open HDF5 recording. Reads are serialized because h5py handles are not reentrant."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.file = h5py.File(self.path, "r")
        self.metadata = self._read_metadata()
        self.lengths, self.offsets = self._read_episodes()
        self.rows = int(self.lengths.sum())
        self.columns = self._read_columns()

    def close(self):
        self.file.close()

    def _read_metadata(self):
        metadata = {}
        for key, value in self.file.attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str) and key.endswith("json"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = value
                if isinstance(parsed, dict) and key == "metadata_json":
                    metadata.update(parsed)
                    continue
                value = parsed
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[key] = value
        return metadata

    def _datasets(self):
        found = {}

        def visit(name, obj):
            # visititems stops at the first non-None return, so never return a value.
            if isinstance(obj, h5py.Dataset):
                found[name] = obj

        self.file.visititems(visit)
        return found

    def _read_episodes(self):
        if all(key in self.file for key in INDEX_KEYS):
            lengths = self.file["ep_len"][:].astype(np.int64)
            offsets = self.file["ep_offset"][:].astype(np.int64)
            return lengths, offsets
        for key in EPISODE_ID_KEYS:
            if key in self.file and self.file[key].ndim == 1:
                ids = self.file[key][:]
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(ids)])
                return lengths.astype(np.int64), starts.astype(np.int64)
        rows = max((len(d) for d in self._datasets().values() if d.ndim), default=0)
        return np.array([rows], dtype=np.int64), np.array([0], dtype=np.int64)

    def _dim_names(self, name, width):
        meta = self.metadata
        candidates = [
            (meta.get("names") or {}).get(name) if isinstance(meta.get("names"), dict) else None,
            meta.get(f"{name}_names"),
            meta.get("state_order") if name.split("/")[-1] in STATE_COLUMNS else None,
        ]
        for names in candidates:
            if isinstance(names, list) and len(names) == width:
                return [str(n) for n in names]
        return None

    def _read_columns(self):
        columns = {}
        for name, data in self._datasets().items():
            if name in INDEX_KEYS or data.ndim == 0 or len(data) != self.rows:
                continue
            if _image_layout(data.shape, data.dtype):
                columns[name] = Column(name, "image", data.shape, str(data.dtype))
            elif data.ndim <= 2 and (data.dtype.kind in "biuf"):
                width = 1 if data.ndim == 1 else data.shape[1]
                names = self._dim_names(name, width)
                columns[name] = Column(name, "series", data.shape, str(data.dtype), names)
        return columns

    def episode_attributes(self):
        """Per-episode scalar metadata, when the recording stores an aligned ``episodes`` list."""
        episodes = self.metadata.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != len(self.lengths):
            return None
        scalar = (str, int, float, bool)
        return [
            _json_safe({k: v for k, v in ep.items() if isinstance(v, scalar)})
            if isinstance(ep, dict)
            else {}
            for ep in episodes
        ]

    def info(self):
        metadata = {k: v for k, v in self.metadata.items() if k != "episodes"}
        return _json_safe(
            {
                "path": str(self.path),
                "rows": self.rows,
                "episodes": len(self.lengths),
                "columns": [asdict(c) for c in self.columns.values()],
                "fps": self.default_fps(),
                "metadata": metadata,
            }
        )

    def default_fps(self):
        meta, config = self.metadata, self.metadata.get("config") or {}
        for source in (meta, config):
            if isinstance(source.get("fps"), (int, float)) and source["fps"] > 0:
                return float(source["fps"])
            if isinstance(source.get("control_dt"), (int, float)) and source["control_dt"] > 0:
                return 1.0 / source["control_dt"]
        return None

    def _region(self, episode):
        if not 0 <= episode < len(self.lengths):
            raise IndexError(f"Episode {episode} out of range [0, {len(self.lengths)})")
        start = int(self.offsets[episode])
        return start, start + int(self.lengths[episode])

    def episode(self, episode):
        start, stop = self._region(episode)
        series = {}
        with self.lock:
            for name, column in self.columns.items():
                if column.kind != "series":
                    continue
                values = np.asarray(self.file[name][start:stop], dtype=np.float64)
                values = values.reshape(len(values), -1)
                values = np.where(np.isfinite(values), values, np.nan)
                series[name] = [[None if math.isnan(v) else v for v in row] for row in values]
        attributes = self.episode_attributes()
        return {
            "index": episode,
            "offset": start,
            "length": stop - start,
            "attributes": attributes[episode] if attributes else None,
            "series": series,
        }

    def frame(self, name, episode, step):
        """Return one frame of an image column as an HxWxC (C in 1, 3, 4) uint8 array."""
        column = self.columns.get(name)
        if column is None or column.kind != "image":
            raise KeyError(f"Not an image column: {name}")
        start, stop = self._region(episode)
        if not 0 <= step < stop - start:
            raise IndexError(f"Step {step} out of range [0, {stop - start})")
        with self.lock:
            image = self.file[name][start + step]
        layout = _image_layout(column.shape, np.dtype(column.dtype))
        if layout == "chw":
            image = np.transpose(image, (1, 2, 0))
        elif layout == "hw":
            image = image[..., None]
        return image

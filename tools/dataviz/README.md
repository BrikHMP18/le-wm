# dataviz

Local browser viewer for stable-worldmodel style HDF5 recordings (the format
`swm.data.load_dataset` reads for LeWM training): flat per-frame columns with
episodes concatenated and indexed by `ep_len` / `ep_offset`.

```bash
cd tools/dataviz
uv run dataviz ../../experiments/teaching-cartpole/data/cartpole-10k-v1 --open
# or a single file:  uv run dataviz path/to/train.h5 --port 8765
```

The target can be one `.h5`/`.hdf5` file or a directory (searched recursively;
pick the file in the header dropdown).

## What it shows

- **Image columns** (`uint8`, HWC / CHW / HW) as a synced player: play/pause,
  scrubber, 0.25–4× speed, loop, auto-advance to the next episode.
- **Series columns** (numeric/bool, 1-D or 2-D) as one plot per column, with a
  time cursor; click or drag a plot to seek, click a legend entry to hide a dim.
- **Episode list** with facet filters built from string/bool fields of an
  aligned `episodes` list in `metadata_json` (e.g. `policy`, `end_reason`).
- **Recording metadata** from the file attributes (`metadata_json` is parsed).

Keys: `space` play/pause, `←/→` frame, `↑/↓` episode. The URL hash keeps the
file/episode/frame, so a view can be reloaded or bookmarked.

## Conventions it understands

| Field | Use |
|---|---|
| `ep_len`, `ep_offset` | Episode index. Fallback: change points of `episode_id` / `episode_index` / `ep_idx`, else one episode. |
| `timestamp` | Time axis and real-time playback; otherwise `fps` / `control_dt` from metadata, else 10 fps. |
| `state_order`, `<col>_names`, `names: {col: [...]}` in metadata | Dimension labels (`state_order` applies to `observation`/`state`/`proprio`). |
| `episodes: [{...}]` in metadata, one per episode | Per-episode tags and filters. |

NaN values (e.g. the last action of each episode) are drawn as gaps.

## Notes

- Read-only; files are opened with `h5py` in `"r"` mode and frames are encoded
  to JPEG on demand (`--quality`), so multi-GB files open instantly.
- Lance (`.lance`) recordings are not supported yet.

```bash
uv run pytest -q && uv run ruff check .
```

"""Collect episode-contiguous HDF5 datasets without GUI or training dependencies."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np

from .environment import PARAMETERS, STATE_ORDER, CartPole, Policy, render, sample_initial_state


@dataclass(frozen=True)
class Config:
    episodes: int = 100
    steps: int = 200
    seed: int = 42
    physics_dt: float = 0.005
    substeps: int = 10
    image_size: int = 224
    force_limit: float = 10.0
    track_limit: float = 2.5
    policy: str = "random"
    initial: str = "mixed"
    action_hold: int = 2
    noise_std: float = 0.5
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    def validate(self):
        if self.episodes < 3 or self.steps < 3:
            raise ValueError("Need >=3 episodes and >=3 transitions per episode")
        if self.seed < 0 or self.substeps < 1 or self.action_hold < 1 or self.image_size < 64:
            raise ValueError("Invalid seed, substeps, action hold, or image size")
        if not all(
            np.isfinite(v) and v > 0 for v in (self.physics_dt, self.force_limit, self.track_limit)
        ):
            raise ValueError("dt and force/track limits must be positive and finite")
        if self.track_limit > 2.5:
            raise ValueError("Track limit >2.5 can clip the pole in the fixed camera")
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("Noise standard deviation must be finite and nonnegative")
        if not (
            0 < self.val_fraction < 1
            and 0 < self.test_fraction < 1
            and self.val_fraction + self.test_fraction < 1
        ):
            raise ValueError("Split fractions must be positive with sum <1")
        if self.policy not in {"random", "zero", "lqr", "mixed"}:
            raise ValueError("Unknown policy")
        if self.initial not in {"mixed", "upright", "downward", "uniform"}:
            raise ValueError("Unknown initial state distribution")


def collect_episode(cfg, episode_id):
    # Episode RNG is independent of split assignment and other episodes' length.
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, episode_id]))
    policy_name = cfg.policy
    initial = cfg.initial
    if policy_name == "mixed":
        policy_name = "lqr" if episode_id % 2 else "random"
        if policy_name == "lqr":
            initial = "upright"
    state, initial_mode = sample_initial_state(rng, initial)
    state[0] *= min(1.0, 0.8 * cfg.track_limit / 0.5)
    env = CartPole(state)
    policy = Policy(rng, policy_name, cfg.force_limit, cfg.action_hold, cfg.noise_std)
    pixels, states, times, actions = [render(state, cfg.image_size)], [state.copy()], [0.0], []
    reason = "time_limit"
    for step in range(cfg.steps):
        # Store the exact float32 force we apply; action[t] advances state[t].
        force = np.float32(policy.action(env.state, step))
        for _ in range(cfg.substeps):
            state = env.step(force, cfg.physics_dt)
        actions.append([force])
        states.append(state.copy())
        times.append(env.time)
        pixels.append(render(state, cfg.image_size))
        if abs(state[0]) >= cfg.track_limit:
            reason = "track_limit"
            break
    # Retain the final observation, but it has no outgoing action.
    actions.append([np.nan])
    length = len(states)
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    arrays = {
        "pixels": np.asarray(pixels, dtype=np.uint8),
        "action": np.asarray(actions, dtype=np.float32),
        "observation": np.asarray(states, dtype=np.float64),
        "timestamp": np.asarray(times, dtype=np.float64),
        "episode_id": np.full(length, episode_id, dtype=np.int64),
        "terminated": done & (reason == "track_limit"),
        "truncated": done & (reason == "time_limit"),
    }
    info = {
        "episode_id": episode_id,
        "rng_seed_components": [cfg.seed, episode_id],
        "policy": policy_name,
        "initial_mode": initial_mode,
        "initial_state": states[0].tolist(),
        "rows": length,
        "transitions": length - 1,
        "end_reason": reason,
    }
    return arrays, info


class Writer:
    """Standard HDF5 columns + ep_len/ep_offset, one episode committed at a time."""

    def __init__(self, path, metadata):
        self.file = h5py.File(path, "x", libver="latest")
        self.file.attrs["metadata_json"] = json.dumps(metadata)
        self.file.create_dataset("ep_len", (0,), maxshape=(None,), dtype=np.int32)
        self.file.create_dataset("ep_offset", (0,), maxshape=(None,), dtype=np.int64)
        self.offset = 0

    def append(self, episode):
        length = len(episode["action"])
        for name, data in episode.items():
            if name not in self.file:
                kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                self.file.create_dataset(
                    name,
                    (0, *data.shape[1:]),
                    maxshape=(None, *data.shape[1:]),
                    dtype=data.dtype,
                    chunks=(1, *data.shape[1:]),
                    **kwargs,
                )
            column = self.file[name]
            column.resize(self.offset + length, axis=0)
            column[self.offset : self.offset + length] = data
        for name, value in (("ep_len", length), ("ep_offset", self.offset)):
            column = self.file[name]
            column.resize(len(column) + 1, axis=0)
            column[-1] = value
        self.offset += length
        self.file.flush()

    def close(self):
        self.file.close()


def save_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def episodes(cfg, workers):
    if workers == 1:
        for episode_id in range(cfg.episodes):
            yield collect_episode(cfg, episode_id)
        return
    # Spawn avoids inheriting open HDF5 files. Small batches bound frame memory;
    # ordered results and per-episode RNG keep datasets independent of worker count.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for start in range(0, cfg.episodes, 2 * workers):
            ids = range(start, min(start + 2 * workers, cfg.episodes))
            yield from pool.map(partial(collect_episode, cfg), ids)


def run(cfg, output, workers=1):
    cfg.validate()
    if workers < 1:
        raise ValueError("workers must be >=1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    permutation = np.random.default_rng(cfg.seed).permutation(cfg.episodes)
    n_val = max(1, int(cfg.episodes * cfg.val_fraction))
    n_test = max(1, int(cfg.episodes * cfg.test_fraction))
    # Leave at least one training episode even for small pilots.
    if n_val + n_test >= cfg.episodes:
        n_val = n_test = 1
    splits = {int(i): "train" for i in permutation}
    splits.update({int(i): "validation" for i in permutation[:n_val]})
    splits.update({int(i): "test" for i in permutation[n_val : n_val + n_test]})
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    manifest = {
        "schema_version": 1,
        "status": "collecting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "collection_workers": workers,
        "physics": PARAMETERS,
        "state_order": STATE_ORDER,
        "control_dt": cfg.physics_dt * cfg.substeps,
        "git_revision": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": source_hashes,
        "dependency_versions": {
            name: version(name) for name in ("numpy", "scipy", "pygame", "h5py", "pillow")
        },
        "episodes": [],
        "action_alignment": "action[t] advances observation[t] to observation[t+1]; last is NaN",
    }
    writers = {}
    save_manifest(output, manifest)
    try:
        for split in ("train", "validation", "test"):
            writers[split] = Writer(output / f"{split}.h5", {**manifest, "split": split})
        for episode_id, (arrays, info) in enumerate(episodes(cfg, workers)):
            split = splits[episode_id]
            writers[split].append(arrays)
            manifest["episodes"].append({**info, "split": split})
            if episode_id % 100 == 0:
                save_manifest(output, manifest)
            print(
                f"episode {episode_id + 1}/{cfg.episodes}: {split}, "
                f"{info['transitions']} transitions, {info['end_reason']}",
                flush=True,
            )
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        for split, writer in writers.items():
            writer.file.attrs["metadata_json"] = json.dumps(
                {
                    **manifest,
                    "split": split,
                    "episodes": [ep for ep in manifest["episodes"] if ep["split"] == split],
                }
            )
            writer.close()
        save_manifest(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New run directory; never overwritten"
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel episode collectors")
    for name in ("episodes", "steps", "seed", "substeps", "image_size", "action_hold"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(Config(), name)
        )
    for name in (
        "physics_dt",
        "force_limit",
        "track_limit",
        "noise_std",
        "val_fraction",
        "test_fraction",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(Config(), name)
        )
    parser.add_argument("--policy", choices=["random", "zero", "lqr", "mixed"], default="random")
    parser.add_argument(
        "--initial", choices=["mixed", "upright", "downward", "uniform"], default="mixed"
    )
    args = vars(parser.parse_args())
    output = args.pop("output")
    workers = args.pop("workers")
    try:
        run(Config(**args), output, workers)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

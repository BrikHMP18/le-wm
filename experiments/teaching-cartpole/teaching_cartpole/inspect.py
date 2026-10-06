"""Validate collection schema/alignment and optionally export a visual preview."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .environment import CartPole


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(output, check_physics=False, physics_sample=0):
    require(physics_sample >= 0, "Physics sample must be nonnegative")
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Collection is incomplete")
    cfg = manifest["config"]
    expected = {ep["episode_id"]: ep for ep in manifest["episodes"]}
    require(len(expected) == cfg["episodes"], "Missing or duplicate manifest episodes")
    seen, counts = set(), {}
    for split in ("train", "validation", "test"):
        rows, transitions, windows = 0, 0, 0
        with h5py.File(output / f"{split}.h5", "r") as file:
            lengths, offsets = file["ep_len"][:], file["ep_offset"][:]
            require(len(lengths) > 0 and len(lengths) == len(offsets), f"Empty/invalid {split}")
            replay_indices = set(
                np.linspace(0, len(lengths) - 1, min(physics_sample, len(lengths)), dtype=int)
            )
            require(np.array_equal(offsets, np.cumsum(np.r_[0, lengths[:-1]])), "Bad offsets")
            total = int(lengths.sum())
            for key in (
                "pixels",
                "action",
                "observation",
                "timestamp",
                "episode_id",
                "terminated",
                "truncated",
            ):
                require(len(file[key]) == total, f"Wrong row count: {key}")
            require(
                file["pixels"].shape == (total, cfg["image_size"], cfg["image_size"], 3)
                and file["pixels"].dtype == np.uint8,
                "Invalid RGB frame schema",
            )
            require(file["observation"].shape == (total, 4), "Invalid physical state schema")
            require(file["action"].shape == (total, 1), "Invalid scalar action schema")
            for ep_index, (length, offset) in enumerate(zip(lengths, offsets, strict=True)):
                length, offset = int(length), int(offset)
                require(length >= 2, "Episode must have at least one transition")
                region = slice(offset, offset + length)
                ids = file["episode_id"][region]
                episode_id = int(ids[0])
                require(
                    np.all(ids == episode_id) and episode_id not in seen,
                    "Mixed, duplicated, or leaked episode",
                )
                seen.add(episode_id)
                require(episode_id in expected, "Unknown episode")
                info = expected[episode_id]
                require(info["split"] == split and info["rows"] == length, "Manifest mismatch")
                state = file["observation"][region]
                action = file["action"][region, 0]
                times = file["timestamp"][region]
                require(np.isfinite(state).all(), "Nonfinite physical state")
                require(np.array_equal(state[0], info["initial_state"]), "Wrong initial state")
                require(
                    np.isfinite(action[:-1]).all() and np.isnan(action[-1]),
                    "Only the last outgoing action may be NaN",
                )
                require(np.all(np.abs(action[:-1]) <= cfg["force_limit"]), "Force bound violated")
                require(
                    np.allclose(
                        times, np.arange(length) * manifest["control_dt"], rtol=0, atol=1e-9
                    ),
                    "Invalid timestamps or reset crossing",
                )
                terminated = file["terminated"][region]
                truncated = file["truncated"][region]
                require(
                    not (terminated[:-1] | truncated[:-1]).any()
                    and bool(terminated[-1]) != bool(truncated[-1]),
                    "Bad terminal flags",
                )
                require(
                    bool(terminated[-1]) == (info["end_reason"] == "track_limit"),
                    "End reason mismatch",
                )
                require(np.all(np.abs(state[:-1, 0]) < cfg["track_limit"]), "Late track reset")
                if terminated[-1]:
                    require(abs(state[-1, 0]) >= cfg["track_limit"], "False track termination")
                else:
                    require(length == cfg["steps"] + 1, "Early time limit")
                if check_physics or ep_index in replay_indices:
                    env = CartPole(state[0])
                    for t, force in enumerate(action[:-1]):
                        for _ in range(cfg["substeps"]):
                            predicted = env.step(float(force), cfg["physics_dt"])
                        require(
                            np.allclose(predicted, state[t + 1], rtol=0, atol=1e-9),
                            f"Misaligned transition in episode {episode_id}, row {t}",
                        )
                rows += length
                transitions += length - 1
                windows += max(0, length - 4 + 1)
            counts[split] = {
                "episodes": len(lengths),
                "rows": rows,
                "transitions": transitions,
                "four_frame_windows": windows,
            }
    require(seen == set(expected), "Missing episodes across splits")
    return counts


def preview(output):
    output = Path(output)
    destination = output / "preview"
    destination.mkdir(exist_ok=True)
    with h5py.File(output / "train.h5", "r") as file:
        index = int(np.argmax(file["ep_len"][:]))
        length, offset = int(file["ep_len"][index]), int(file["ep_offset"][index])
        frames = file["pixels"][offset : offset + length]
        times = file["timestamp"][offset : offset + length]
        episode_id = int(file["episode_id"][offset])
    size = frames.shape[1]
    sheet = Image.new("RGB", (size * 3, (size + 25) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for panel, t in enumerate(np.linspace(0, length - 1, 6, dtype=int)):
        x, y = panel % 3 * size, panel // 3 * (size + 25)
        sheet.paste(Image.fromarray(frames[t]), (x, y))
        draw.text((x + 5, y + size + 5), f"ep {episode_id}, t={times[t]:.2f}s", fill="black")
    sheet.save(destination / "frames.png")
    images = [Image.fromarray(frame) for frame in frames]
    dt = float(times[1] - times[0])
    images[0].save(
        destination / "episode.gif",
        save_all=True,
        append_images=images[1:],
        duration=max(10, round(dt * 1000)),
        loop=0,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--check-physics", action="store_true", help="Replay every stored transition"
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--physics-sample",
        type=int,
        default=0,
        help="Replay this many evenly spaced episodes per split",
    )
    args = parser.parse_args()
    try:
        counts = validate(args.output, args.check_physics, args.physics_sample)
        print(json.dumps(counts, indent=2))
        if args.preview:
            print(f"Preview: {preview(args.output)}")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

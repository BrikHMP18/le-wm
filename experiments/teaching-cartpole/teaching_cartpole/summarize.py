"""Export collection counts and state-coverage diagnostics for the assignment report."""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Collection must be complete before summarizing")
    episodes = manifest["episodes"]
    angle_edges = np.linspace(-np.pi, np.pi, 37)
    position_edges = np.linspace(
        -manifest["config"]["track_limit"], manifest["config"]["track_limit"], 41
    )
    policies = sorted({ep["policy"] for ep in episodes})
    histograms = {
        p: {"angle": np.zeros(36, dtype=np.int64), "position": np.zeros(40, dtype=np.int64)}
        for p in policies
    }
    ids = {ep["episode_id"]: ep for ep in episodes}
    splits = {}
    state_min, state_max = np.full(4, np.inf), np.full(4, -np.inf)
    for split in ("train", "validation", "test"):
        with h5py.File(output / f"{split}.h5", "r") as file:
            lengths = file["ep_len"][:]
            splits[split] = {
                "episodes": len(lengths),
                "frames": int(lengths.sum()),
                "transitions": int(lengths.sum() - len(lengths)),
                "four_frame_windows": int(np.maximum(0, lengths - 3).sum()),
            }
            for offset, length in zip(file["ep_offset"][:], lengths, strict=True):
                offset, length = int(offset), int(length)
                episode_id = int(file["episode_id"][offset])
                policy = ids[episode_id]["policy"]
                # Exclude terminal-only frames from coverage: there is no outgoing action.
                state = file["observation"][offset : offset + length - 1]
                state_min = np.minimum(state_min, state.min(axis=0))
                state_max = np.maximum(state_max, state.max(axis=0))
                error = np.arctan2(np.sin(state[:, 3] - np.pi), np.cos(state[:, 3] - np.pi))
                histograms[policy]["angle"] += np.histogram(error, angle_edges)[0]
                histograms[policy]["position"] += np.histogram(state[:, 0], position_edges)[0]
    lengths = np.array([ep["transitions"] for ep in episodes])
    result = {
        "dataset_path": str(output.resolve()),
        "status": "complete",
        "config": manifest["config"],
        "git_revision": manifest["git_revision"],
        "collector_source_sha256": manifest["source_sha256"],
        "control_dt": manifest["control_dt"],
        "splits": splits,
        "total_episodes": len(episodes),
        "total_transitions": int(lengths.sum()),
        "total_frames": int(lengths.sum() + len(episodes)),
        "total_simulated_seconds": float(lengths.sum() * manifest["control_dt"]),
        "episode_transitions": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        },
        "episodes_by_policy": dict(Counter(ep["policy"] for ep in episodes)),
        "transitions_by_policy": {
            p: sum(ep["transitions"] for ep in episodes if ep["policy"] == p) for p in policies
        },
        "episodes_by_initial_mode": dict(Counter(ep["initial_mode"] for ep in episodes)),
        "episodes_by_end_reason": dict(Counter(ep["end_reason"] for ep in episodes)),
        "state_order": manifest["state_order"],
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
        "file_bytes": {p.name: p.stat().st_size for p in output.glob("*.h5")},
        "angle_error_bin_edges": angle_edges.tolist(),
        "position_bin_edges": position_edges.tolist(),
        "coverage_counts": {
            p: {key: values.tolist() for key, values in counts.items()}
            for p, counts in histograms.items()
        },
        "scope": "Collection diagnostics only; LeWM has not been trained or evaluated.",
        "paper_reference": {
            "url": "https://arxiv.org/html/2603.19312v3",
            "section": "Appendix E",
            "reacher_and_cube": {"episodes": 10000, "steps_per_episode": 200},
            "interpretation": "Experimental dataset sizes, not a demonstrated CartPole minimum.",
            "temporal_difference": (
                "Paper frameskip=5; frames are stored at 20 Hz and training loads them "
                "with frameskip=5 (0.25 s per latent step)."
            ),
        },
    }
    return result


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for policy, counts in summary["coverage_counts"].items():
        for ax, field, edges_name in zip(
            axes,
            ("position", "angle"),
            ("position_bin_edges", "angle_error_bin_edges"),
            strict=True,
        ):
            values = np.array(counts[field], dtype=float)
            edges = np.array(summary[edges_name])
            density = values / (values.sum() * np.diff(edges))
            ax.stairs(density, edges, label=policy, linewidth=1.5)
    axes[0].set(xlabel="Cart position x (m)", ylabel="Density")
    axes[1].set(xlabel="Wrapped angular error from upright (rad)", ylabel="Density")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"CartPole collection: {summary['total_episodes']:,} episodes, "
        f"{summary['total_transitions']:,} transitions"
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    summary = summarize(args.dataset)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plot(summary, args.figure)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("total_episodes", "total_transitions", "total_frames", "splits")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

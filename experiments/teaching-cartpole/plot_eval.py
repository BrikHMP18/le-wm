"""Figures from eval_lewm.py reports (one or more runs, overlaid).

    .venv/bin/python experiments/teaching-cartpole/plot_eval.py \
        experiments/teaching-cartpole/outputs/eval-cartpole-fs5-epoch9.json [more.json ...]

Writes, next to the first report (or under --out-dir):
  eval-rollout.png  open-loop latent error vs horizon (s), per run, with the
                    "nothing changes" baseline (copy the last context embedding)
  eval-probes.png   linear-probe test R^2 per state variable, trained vs untrained encoder
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (dataviz reference palette, light): runs get hues;
# baselines/controls are recessive gray reference marks, labelled directly.
RUN_COLORS = ["#2a78d6", "#eb6834"]
REF = "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABELS = {
    "x": "cart position $x$",
    "x_dot": "cart velocity $\\dot{x}$",
    "theta_dot": "angular velocity $\\dot\\theta$",
    "sin_theta": "$\\sin\\theta$ (left / right)",
    "cos_theta": "$\\cos\\theta$ (up / down)",
}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def run_name(r):
    epoch = r["weights"].removesuffix(".pt").replace("weights_", "")
    return f"LeWM frameskip {r['frameskip']} ({epoch})"


def rollout(reports, split, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    style(ax)
    for i, r in enumerate(reports):
        m = r["trained"]["latent"][split]
        h = [0.0] + m["rollout_horizon_s"]
        c = RUN_COLORS[i]
        ax.plot(
            h,
            [0.0] + m["rollout_rel_mse"],
            color=c,
            lw=2,
            marker="o",
            ms=4,
            markevery=max(1, len(h) // 8),
            label=run_name(r),
        )
        if i == 0:
            ax.plot(
                h,
                [0.0] + m["rollout_copy_last_rel_mse"],
                color=REF,
                lw=2,
                ls="--",
                label='baseline: "nothing changes"',
            )
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.text(
        0.01,
        1.02,
        "1.0 = predicting the mean embedding",
        color=MUTED,
        fontsize=8,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlabel("open-loop prediction horizon (s)", color=INK, fontsize=10)
    ax.set_ylabel("latent error / embedding variance", color=INK, fontsize=10)
    ax.set_title(
        f"Latent prediction on held-out {split} episodes", color=INK, fontsize=11, loc="left"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def probes(reports, out):
    # One bar group per state variable: each trained run, then the untrained control (first run).
    keys = ["x", "cos_theta", "sin_theta", "x_dot", "theta_dot"]

    def r2(rep, model):
        single, hist = rep[model]["probes"]["single_frame"], rep[model]["probes"]["history"]
        vals = dict(zip(hist["targets"], hist["test"]))
        vals.update(dict(zip(single["targets"], single["test"])))  # position/angle: one frame
        return [vals[k] for k in keys]

    series = [(run_name(r), r2(r, "trained"), RUN_COLORS[i]) for i, r in enumerate(reports)]
    series.append(("untrained encoder (control)", r2(reports[0], "random_init"), REF))

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    style(ax)
    ax.grid(axis="y", visible=False)
    n, bar = len(series), 0.8 / len(series)
    for j, (name, vals, c) in enumerate(series):
        ys = [k + (j - (n - 1) / 2) * bar for k in range(len(keys))]
        ax.barh(ys, vals, height=bar * 0.85, color=c, label=name)
        for y, v in zip(ys, vals):
            ax.text(max(v, 0) + 0.01, y, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(keys)), [LABELS[k] for k in keys], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(min(v) for _, v, _ in series)) - 0.02, 1.08)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("test $R^2$ of a linear probe from the embedding", color=INK, fontsize=10)
    ax.set_title(
        "What the embedding encodes about the physical state", color=INK, fontsize=11, loc="left"
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.text(
        0.01,
        0.01,
        "Position and angle from one frame; velocities from 3 history frames.",
        color=MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--prefix", default="eval")
    args = p.parse_args()
    reports = [json.loads(r.read_text()) for r in args.reports]
    out = args.out_dir or args.reports[0].parent
    out.mkdir(parents=True, exist_ok=True)
    rollout(reports, args.split, out / f"{args.prefix}-rollout.png")
    probes(reports, out / f"{args.prefix}-probes.png")
    print("wrote", out / f"{args.prefix}-rollout.png", out / f"{args.prefix}-probes.png")


if __name__ == "__main__":
    main()

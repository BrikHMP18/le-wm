"""Q2(e) report material: figures (vector PDF) and q2e_results.json.

Reads the epoch-9 evaluation reports (eval_lewm.py) and decodes imagined frames
with the visualization decoders, writing into the assignment report:

    .venv/bin/python experiments/teaching-cartpole/report_q2e.py

  q2e_rollout.pdf      open-loop latent error vs horizon (s), fs5 and fs1, each
                       with its own "nothing changes" baseline
  q2e_physical.pdf     open-loop error in physical units (cart cm, pole degrees) read
                       by a linear probe, per behavior policy (physical_rollout.py)
  q2e_imagination.pdf  real vs decoded open-loop imagination of both models,
                       two held-out episodes at three instants
  q2e_results.json     every number quoted in the report text and tables
"""

import json
import subprocess
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from decode_video import DATA, HERE, episode_panels, read_col  # noqa: E402
from decoder import load_decoder, load_world_model  # noqa: E402

REPORT = Path(
    "/home/autobrik/Projects/personal/brik-meza-knowledge-base/education/mcgill-msc/"
    "fall-2026/comp-765/assignments/assignment-1/report/figures"
)
RUNS = {"fs5": "cartpole-fs5", "fs1": "cartpole-fs1"}
COLORS = {"fs5": "C0", "fs1": "C1"}  # matplotlib defaults, as in the other Q2 figures
EPISODES = [(11, "noisy LQR"), (478, "random forces")]  # held-out test episodes
# s; t = 0 is the first frame both models see (decoded, not predicted); the others are
# after both models' context (up to 0.5 s for fs5), so they are open-loop predictions
TIMES = [0.0, 1.0, 3.0, 5.75]


def results():
    out = {"source": {}, "runs": {}}
    out["source"]["fork_commit"] = subprocess.check_output(
        ["git", "-C", str(HERE.parents[1]), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    out["source"]["huggingface"] = {
        "dataset": "https://huggingface.co/datasets/Autobrik/teaching-cartpole-10k-v1",
        "fs5": "https://huggingface.co/Autobrik/lewm-teaching-cartpole-fs5",
        "fs1": "https://huggingface.co/Autobrik/lewm-teaching-cartpole-fs1",
    }
    for key, run in RUNS.items():
        ev = json.loads((HERE / "outputs" / f"eval-{run}-epoch9.json").read_text())
        lt = ev["trained"]["latent"]["test"]
        h = lt["rollout_horizon_s"]
        at = {
            f"{s:.2f}": i
            for i, s in enumerate(h)
            if any(abs(s - x) < 1e-6 for x in (0.25, 0.5, 1.0))
        }

        def probes(model):
            p = ev[model]["probes"]
            v = dict(zip(p["history"]["targets"], p["history"]["test"]))
            v.update(dict(zip(p["single_frame"]["targets"], p["single_frame"]["test"])))
            return {
                k: round(v[k], 3) for k in ("x", "cos_theta", "sin_theta", "x_dot", "theta_dot")
            }

        out["runs"][key] = {
            "weights": ev["weights"],
            "frameskip": ev["frameskip"],
            "latent_step_s": ev["latent_step_s"],
            "test_windows": ev["windows_per_split"]["test"],
            "onestep_rel_mse": round(lt["onestep_rel_mse"], 3),
            "shuffled_actions_rel_mse": round(lt["shuffled_actions_rel_mse"], 3),
            "copy_current_rel_mse": round(lt["copy_current_rel_mse"], 3),
            "rollout_rel_mse": {s: round(lt["rollout_rel_mse"][i], 3) for s, i in at.items()},
            "rollout_copy_last_rel_mse": {
                s: round(lt["rollout_copy_last_rel_mse"][i], 3) for s, i in at.items()
            },
            "rollout_steps": {s: i + 1 for s, i in at.items()},
            "probe_r2_test": probes("trained"),
            "probe_r2_test_untrained": probes("random_init"),
            "_curve": {
                "h": h,
                "model": lt["rollout_rel_mse"],
                "copy": lt["rollout_copy_last_rel_mse"],
            },
        }
    return out


def rollout_figure(res, path):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for key, r in res["runs"].items():
        c = r.pop("_curve")
        h = [0.0] + c["h"]
        label = f"frameskip {r['frameskip']} ({r['latent_step_s']:.2f} s/step)"
        ax.plot(h, [0.0] + c["model"], color=COLORS[key], marker="o", ms=3,
                markevery=max(1, len(h) // 8), label=f"LeWM, {label}")  # fmt: skip
        ax.plot(h, [0.0] + c["copy"], color=COLORS[key], ls="--", lw=1,
                label=f'"nothing changes", {label}')  # fmt: skip
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("open-loop horizon (s)")
    ax.set_ylabel("relative latent MSE")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def physical_results(res):
    """Add physical-unit errors at 0.25 / 0.5 / 1 s (physical_rollout.py outputs)."""
    curves = {}
    for key, run in RUNS.items():
        d = json.loads((HERE / "outputs" / f"physical-{run}-epoch9.json").read_text())
        at = {
            f"{s:.2f}": i for i, s in enumerate(d["horizon_s"]) if round(s, 2) in (0.25, 0.5, 1.0)
        }
        res["runs"][key]["physical_test_windows"] = d["test_windows"]
        for q in ("cart_position_error_cm", "pole_angle_error_deg"):
            res["runs"][key][q] = {
                ref: {
                    g: {s: v[g]["mean"][i] for s, i in at.items()} for g in ("all", "lqr", "random")
                }
                for ref, v in d[q].items()
            }
        curves[key] = d
    return curves


def physical_figure(curves, path):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    quantities = [("cart_position_error_cm", "cart position error (cm)"),
                  ("pole_angle_error_deg", "pole angle error (deg)")]  # fmt: skip
    groups = [("lqr", "noisy LQR episodes"), ("random", "random-force episodes")]
    for i, (q, ylabel) in enumerate(quantities):
        for j, (g, title) in enumerate(groups):
            ax = axes[i, j]
            for key, d in curves.items():
                h = d["horizon_s"]  # no point at t = 0: the probe already has error there
                fs = d["frameskip"]
                ax.plot(h, d[q]["model"][g]["mean"], color=COLORS[key], marker="o", ms=2.5,
                        markevery=max(1, len(h) // 8), label=f"LeWM, frameskip {fs}")  # fmt: skip
                ax.plot(d["horizon_s"], d[q]["probe_floor"][g]["mean"], color=COLORS[key], ls=":",
                        lw=1, label=f"probe floor, frameskip {fs}")  # fmt: skip
            base = curves["fs1"]  # finest horizon grid; the fs5 windows give nearly the same curve
            ax.plot([0.0] + base["horizon_s"], [0.0] + base[q]["nothing_changes"][g]["mean"],
                    color="0.45", ls="--", lw=1, label='"nothing changes"')  # fmt: skip
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if i == 1:
                ax.set_xlabel("open-loop horizon (s)", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def imagination_figure(path):
    dt = json.loads((DATA / "manifest.json").read_text())["control_dt"]
    with h5py.File(DATA / "train.h5", "r") as f:
        a = read_col(f, "action")
    a = a[~np.isnan(a).any(1)]
    act_mean, act_std = float(a.mean()), float(a.std(ddof=1))
    models = {}
    for key, run in RUNS.items():
        ck = HERE / "checkpoints" / run
        cfg = OmegaConf.load(ck / "config.yaml")
        models[key] = dict(
            fs=cfg.data.dataset.frameskip, hist=cfg.history_size,
            world=load_world_model(ck, "weights_epoch_9.pt").cuda(),
            dec=load_decoder(HERE / "checkpoints/decoders" / f"{run}-weights_epoch_9"),
        )  # fmt: skip
    cols = []  # (episode label, column title, [images at TIMES])
    with h5py.File(DATA / "test.h5", "r") as f:
        for ep, label in EPISODES:
            for key, m in models.items():
                idx = [int(round(t / (m["fs"] * dt))) for t in TIMES]
                real, _, pred, _ = episode_panels(
                    m["world"], m["dec"], f, ep, m["fs"], m["hist"], act_mean, act_std,
                    max(idx) + 1,
                )  # fmt: skip
                if key == "fs5":
                    cols.append((label, "real", [real[i] for i in idx]))
                cols.append((label, f"frameskip {m['fs']}", [pred[i] for i in idx]))
    # crop the letterbox: keep rows where any real frame has content
    stack = np.stack([img for c in cols if c[1] == "real" for img in c[2]])
    rows = np.flatnonzero((stack < 245).any(axis=(0, 2, 3)))
    r0, r1 = max(rows.min() - 4, 0), min(rows.max() + 5, stack.shape[1])

    cell = 6.6 / len(cols)  # inches per panel, full text width
    height = len(TIMES) * cell * (r1 - r0) / stack.shape[2] + 0.85  # titles + legend
    fig, axes = plt.subplots(len(TIMES), len(cols), figsize=(6.6, height))
    for j, (label, title, imgs) in enumerate(cols):
        for i, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img[r0:r1])
            ax.set_xticks([])
            ax.set_yticks([])
            seen = (
                title == "real" or TIMES[i] == 0
            )  # real frame, or the decoded frame the model saw
            for s in ax.spines.values():
                imagined = COLORS["fs5"] if "5" in title else COLORS["fs1"]
                s.set_color("0.75" if seen else imagined)
                s.set_linewidth(0.8 if seen else 1.6)
            if i == 0:
                color = (
                    "0.1" if title == "real" else (COLORS["fs5"] if "5" in title else COLORS["fs1"])
                )
                ax.set_title(title, fontsize=7.5, color=color)
            if j == 0:
                ax.set_ylabel("t = 0 s\n(seen)" if TIMES[i] == 0 else f"t = {TIMES[i]:.2f} s",
                              fontsize=7.5)  # fmt: skip
    for k, (_, label) in enumerate(EPISODES):
        x = (k + 0.5) / len(EPISODES)
        fig.text(x, 0.995, f"test episode {EPISODES[k][0]} ({label})", ha="center", va="top",
                 fontsize=8.5)  # fmt: skip
    from matplotlib.patches import Patch

    legend = [
        Patch(
            facecolor="white", edgecolor="0.75", lw=0.8, label="real frame, or frame the model sees"
        ),
        Patch(facecolor="white", edgecolor=COLORS["fs5"], lw=1.6, label="frameskip-5 prediction"),
        Patch(facecolor="white", edgecolor=COLORS["fs1"], lw=1.6, label="frameskip-1 prediction"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=0.3, w_pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    res = results()
    rollout_figure(res, REPORT / "q2e_rollout.pdf")
    physical_figure(physical_results(res), REPORT / "q2e_physical.pdf")
    imagination_figure(REPORT / "q2e_imagination.pdf")
    res["imagination_figure"] = {"episodes": [e for e, _ in EPISODES], "times_s": TIMES}
    (REPORT / "q2e_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(
        "wrote",
        *(
            REPORT / n
            for n in (
                "q2e_rollout.pdf",
                "q2e_physical.pdf",
                "q2e_imagination.pdf",
                "q2e_results.json",
            )
        ),
    )


if __name__ == "__main__":
    main()

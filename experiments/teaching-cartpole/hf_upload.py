# ruff: noqa: E501  (model-card Markdown tables cannot be wrapped)
"""Back up epoch-9 LeWM TeachingCartpole checkpoints (world model + decoder) to Hugging Face.

One private model repo per run, with a generated model card whose numbers are
read from the evaluation reports (not typed by hand):

    .venv/bin/python experiments/teaching-cartpole/hf_upload.py            # stage + upload
    .venv/bin/python experiments/teaching-cartpole/hf_upload.py --dry-run  # stage only
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the HDF5 compression filters
import numpy as np

HERE = Path(__file__).resolve().parent
FORK = HERE.parents[1]
CK, OUT = HERE / "checkpoints", HERE / "outputs"
OWNER, WEIGHTS = "Autobrik", "weights_epoch_9.pt"
SWM_REV = "21446f1ede6d5284e981bd7b47f432b994e6d812"

RUNS = {
    "cartpole-fs5": dict(
        repo="lewm-teaching-cartpole-fs5",
        status="Completed all 10 epochs (7,476 steps/epoch, ~4.4 it/s, ~4.7 h). "
        "Epoch 9 is published to match the frameskip-1 run; epoch 10 exists but is not the "
        "comparison point.",
        videos=[
            "cartpole-fs5-weights_epoch_9-ep11-lqr",
            "cartpole-fs5-weights_epoch_9-ep478-random",
        ],
    ),
    "cartpole-fs1": dict(
        repo="lewm-teaching-cartpole-fs1",
        status="Epochs 1-9 completed (8,376 steps/epoch, ~4.4 it/s). The run was cancelled at "
        "epoch 10, step 7,300/8,376, when the launching client disconnected; epoch 9 is the "
        "last complete checkpoint and was not retrained.",
        videos=[
            "cartpole-fs1-weights_epoch_9-ep11-lqr",
            "cartpole-fs1-weights_epoch_9-ep478-random",
        ],
    ),
}
COMPARE = [
    "compare-cartpole-fs5-vs-cartpole-fs1-ep11-lqr",
    "compare-cartpole-fs5-vs-cartpole-fs1-ep478-random",
]


def action_stats():
    """z-score stats train.py fits on train.h5 (non-NaN rows, unbiased std)."""
    with h5py.File(HERE / "data/cartpole-10k-v1/train.h5", "r") as f:
        col = f["action"]
        a = np.concatenate([col[i : i + 50_000] for i in range(0, len(col), 50_000)])
    a = a[~np.isnan(a).any(1)]
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "rows": int(len(a))}


def card(run, meta, ev, dec, act, commit):
    fs, step_s = ev["frameskip"], ev["latent_step_s"]
    lt, pr = ev["trained"]["latent"]["test"], ev["trained"]["probes"]
    rd = ev["random_init"]["probes"]
    h = lt["rollout_horizon_s"]
    keep = [i for i, x in enumerate(h) if any(abs(x - s) < 1e-6 for s in (0.25, 0.5, 1.0))]
    roll = "\n".join(
        f"| {h[i]:.2f} s ({i + 1} step{'s' if i else ''}) | {lt['rollout_rel_mse'][i]:.3f} | "
        f"{lt['rollout_copy_last_rel_mse'][i]:.3f} |"
        for i in keep
    )

    def r2(block, key):
        return dict(zip(block["targets"], block["test"]))[key]

    probes = "\n".join(
        f"| {label} | {r2(pr[b], k):.3f} | {r2(rd[b], k):.3f} |"
        for label, b, k in [
            ("cart position x", "single_frame", "x"),
            ("cos θ (up/down)", "single_frame", "cos_theta"),
            ("sin θ (left/right)", "single_frame", "sin_theta"),
            ("cart velocity ẋ", "history", "x_dot"),
            ("angular velocity θ̇", "history", "theta_dot"),
        ]
    )
    other = "fs1" if fs == 5 else "fs5"
    return f"""---
license: mit
library_name: pytorch
tags: [world-model, jepa, lewm, le-world-model, cartpole, model-based-rl, research]
---

# LeWM on TeachingCartpole — frameskip {fs}, epoch 9

A [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) JEPA world model trained from
raw pixels and force actions on a single-pole CartPole dataset, plus a post-hoc image
decoder used only for visualization. Research backup for COMP 765 (McGill) Assignment 1,
Q2(e). Companion run: [`{OWNER}/lewm-teaching-cartpole-{other}`](https://huggingface.co/{OWNER}/lewm-teaching-cartpole-{other}).

| | |
|---|---|
| Latent step | frameskip {fs} × 0.05 s = **{step_s:.2f} s** |
| History | 3 frames (+1 predicted) |
| Inputs | 224×224 RGB (ImageNet-normalized) + actions; physical state is **not** a model input |
| Action input | {fs} force value(s) in N per latent step, z-scored with train.h5 stats (below), NaN → 0 |
| Checkpoint | `weights_epoch_9.pt` of a 10-epoch cosine schedule |
| Code | fork `BrikHMP18/le-wm`, branch `feat/teaching-cartpole-data`, commit `{commit}`; stable-worldmodel `{SWM_REV[:7]}` |

**Training status:** {meta["status"]}

## Files

| Path | Content |
|---|---|
| `world_model/weights_epoch_9.pt` | `JEPA` state dict (ViT-tiny/14 encoder, 6-layer AR predictor, MLP projectors; ~18M params) |
| `world_model/config.json` | Hydra model config (`_target_: jepa.JEPA`), loadable by `swm.wm.utils.load_pretrained` |
| `world_model/train_config.yaml` | Full training config as run |
| `world_model/action_normalizer.json` | Action z-score stats from train.h5 (mean {act["mean"]:.6f}, std {act["std"]:.6f}) |
| `decoder/decoder.pt`, `decoder/decoder.json` | Visualization decoder (~8M params) and its recipe |
| `eval/eval-epoch9.json` | Full held-out evaluation report |
| `eval/*.png` | Rollout and probe figures (both runs overlaid) |
| `media/*.mp4` | Decoded videos: real / encode→decode / open-loop prediction, and a fs5-vs-fs1 comparison |

## Data and training

- **Dataset** `cartpole-10k-v1` (not uploaded): 10,000 episodes, 1,507,974 transitions at
  20 Hz, 224×224 RGB, split by episode 8,000 / 1,000 / 1,000; half noisy-LQR balancing, half
  random forces (±10 N). Training uses `train.h5` only.
- **Training**: upstream LeWM `train.py`, unmodified (prediction MSE + SIGReg λ=0.09,
  AdamW lr 5e-5, batch 128, bf16), 1× NVIDIA L40S on Modal.
- The `validate/*` losses logged during training use a random split of windows from
  `train.h5` (shared episodes), so they are optimistic; the numbers below are on held-out
  episodes.

## Held-out evaluation (test split, {ev["windows_per_split"]["test"]} windows)

Latent errors are MSE relative to the mean per-dimension embedding variance
(1.0 = predicting the mean embedding). They are comparable to this model's own baselines,
not across models (each model has its own embedding space).

| | relative MSE |
|---|---|
| One step, teacher-forced (training objective) | {lt["onestep_rel_mse"]:.3f} |
| Same, actions shuffled across the batch | {lt["shuffled_actions_rel_mse"]:.3f} |
| Baseline: copy current embedding | {lt["copy_current_rel_mse"]:.3f} |

Open-loop rollout from 3 context frames with recorded actions:

| Horizon | LeWM | Baseline "nothing changes" |
|---|---|---|
{roll}

Linear probes (ridge; α chosen on validation; test R²). Position/angle from one frame,
velocities from the 3 history frames; control = same architecture, untrained:

| Variable | LeWM | Untrained control |
|---|---|---|
{probes}

## Decoder

Paper-style diagnostic decoder: the projected embedding (the space the predictor works in)
is lifted to 8 context tokens; one learnable query per 14×14 patch cross-attends to them
through 4 residual blocks (dim 384, 6 heads); a linear head emits patch pixels. Trained
with the world model frozen: MSE on ImageNet-normalized pixels, AdamW lr 3e-4 (cosine),
batch 256, {dec["steps"]:,} steps (5 epochs over the 1,215,292 training frames). The paper
does not report its decoder's training recipe; this one is our choice. Decoded images show
what the embedding retains; they are not model outputs used for training or planning.

## Usage

Requires the fork's `jepa.py` / `module.py` (and `decoder.py` for the decoder) on `sys.path`.

```python
from huggingface_hub import snapshot_download
import stable_worldmodel as swm

d = snapshot_download("{OWNER}/{meta["repo"]}")
model = swm.wm.utils.load_pretrained(f"{{d}}/world_model/weights_epoch_9.pt").eval()

from decoder import load_decoder  # experiments/teaching-cartpole/decoder.py
dec = load_decoder(f"{{d}}/decoder")
```

## Limitations

- Trained on one simulator, camera, and dataset; the data contain little swing-up behavior.
- Linear probes bound what is *linearly* readable; low R² does not prove absence.
- No control (MPC) results yet; this checkpoint is not a validated controller.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--public", action="store_true", help="default: private repos")
    args = p.parse_args()

    commit = subprocess.check_output(
        ["git", "-C", str(FORK), "rev-parse", "--short", "HEAD"], text=True
    )
    act = action_stats()
    stage_root = OUT / "hf_stage"
    for run, meta in RUNS.items():
        stage = stage_root / meta["repo"]
        shutil.rmtree(stage, ignore_errors=True)
        for sub in ("world_model", "decoder", "eval", "media"):
            (stage / sub).mkdir(parents=True)
        shutil.copy(CK / run / WEIGHTS, stage / "world_model" / WEIGHTS)
        shutil.copy(CK / run / "config.json", stage / "world_model/config.json")
        shutil.copy(CK / run / "config.yaml", stage / "world_model/train_config.yaml")
        (stage / "world_model/action_normalizer.json").write_text(
            json.dumps({**act, "source": "train.h5 action column, as fitted by train.py"}, indent=2)
        )
        dec_dir = CK / "decoders" / f"{run}-weights_epoch_9"
        for f in ("decoder.pt", "decoder.json"):
            shutil.copy(dec_dir / f, stage / "decoder" / f)
        shutil.copy(OUT / f"eval-{run}-epoch9.json", stage / "eval/eval-epoch9.json")
        for fig in ("rollout", "probes"):
            shutil.copy(
                OUT / f"eval-fs5-vs-fs1-epoch9-{fig}.png", stage / f"eval/fs5-vs-fs1-{fig}.png"
            )
        for v in meta["videos"] + COMPARE:
            shutil.copy(OUT / "videos" / f"{v}.mp4", stage / "media" / f"{v}.mp4")
        ev = json.loads((stage / "eval/eval-epoch9.json").read_text())
        dec = json.loads((stage / "decoder/decoder.json").read_text())
        (stage / "README.md").write_text(card(run, meta, ev, dec, act, commit.strip()))
        print("staged", stage)

        if not args.dry_run:
            from huggingface_hub import HfApi

            api, repo_id = HfApi(), f"{OWNER}/{meta['repo']}"
            api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=stage,
                commit_message=f"Epoch-9 world model, decoder, eval, and media ({commit.strip()})",
            )
            print("uploaded", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

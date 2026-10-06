"""Held-out evaluation of a trained LeWM checkpoint on TeachingCartpole.

Run from the fork root with the fork's environment (torch, stable-worldmodel):

    .venv/bin/python experiments/teaching-cartpole/eval_lewm.py \
        --run experiments/teaching-cartpole/checkpoints/cartpole-fs5

All metrics use episode-held-out files (validation.h5 / test.h5); train.h5 is
used only for the action normalizer (as in training) and for fitting probes.

Metrics (latent errors are reported relative to the per-dimension variance of
the target embeddings, so 1.0 = predicting the mean embedding):

1. One-step prediction, teacher-forced exactly as in training (`pred_loss`),
   against baselines: copy the current embedding, and the model with actions
   shuffled across the batch (does the predictor use the action?).
2. Open-loop rollout: autoregressive prediction for `--rollout` latent steps
   from a 3-frame context, against copying the last context embedding.
3. Linear probes (ridge; alpha selected on validation, reported on test) from
   embeddings to physical state. Single frame -> [x, sin(theta), cos(theta)];
   history of `history_size` frames -> also velocities. A randomly initialised
   model with the same architecture is the control.
4. `--epochs`: one-step metrics for every `weights_epoch_N.pt` on validation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the HDF5 compression filters
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK))  # jepa.py, module.py, utils.py

from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "cartpole-10k-v1"
STATE = ["x", "x_dot", "theta_dot", "theta"]  # observation order in the HDF5 files


def load_split(split, frameskip, num_steps):
    """HDF5 reader with action/observation cached in memory, read in slices.

    The collector wrote these columns with one-row gzip chunks; reading a whole
    column at once (what `keys_to_cache` does) costs ~8 GB of HDF5 bookkeeping for
    train.h5 and ran a laptop out of memory. Slices give identical arrays at
    <0.5 GB. `_cache` is the reader's own column cache (stable-worldmodel 21446f1).
    """
    path = DATA / f"{split}.h5"
    ds = swm.data.load_dataset(
        str(path),
        frameskip=frameskip,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "observation"],
    )
    with h5py.File(path, "r") as f:
        for key in ("action", "observation"):
            col = f[key]
            arr = np.empty(col.shape, col.dtype)
            for i in range(0, len(col), 50_000):
                arr[i : i + 50_000] = col[i : i + 50_000]
            ds._cache[key] = arr
    return ds


def batches(ds, n, batch_size, workers, seed):
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, np.sort(idx)),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )
    yield from loader


def load_model(cfg_json, weights=None):
    model = hydra.utils.instantiate(OmegaConf.create(json.loads(cfg_json.read_text())))
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.cuda().eval()


@torch.no_grad()
def run_model(model, ds, args, hist, rollout):
    """Embeddings, one-step and rollout errors for a sample of windows."""
    acc = {k: [] for k in ["onestep", "copy1", "shuffled", "roll", "copyk", "var"]}
    feats, states = [], []
    gen = torch.Generator().manual_seed(args.seed)
    for batch in batches(ds, args.max_windows, args.batch_size, args.workers, args.seed):
        pix = batch["pixels"].cuda()
        act = torch.nan_to_num(batch["action"].cuda(), 0.0)  # as in training
        # fp32 throughout: an untrained model's embeddings vary on a ~1e-3 scale,
        # which bf16 rounding would erase.
        emb = model.encode({"pixels": pix})["emb"]  # (B, T, D)
        act_emb = model.action_encoder(act)

        # 1. teacher-forced one step: the training objective (all `hist` positions,
        # so early positions see less than the full history)
        pred = model.predict(emb[:, :hist], act_emb[:, :hist])
        perm = torch.randperm(emb.size(0), generator=gen).cuda()
        pred_shuf = model.predict(emb[:, :hist], act_emb[perm, :hist])

        # 2. open-loop rollout from the first `hist` frames
        cur = emb[:, :hist]
        roll = []
        for t in range(rollout):
            step = model.predict(cur[:, -hist:], act_emb[:, t : t + hist])[:, -1:]
            cur = torch.cat([cur, step], dim=1)
            roll.append(step[:, 0])
        tgt = emb[:, 1 : hist + 1]
        acc["onestep"].append(((pred - tgt) ** 2).mean((1, 2)))
        acc["copy1"].append(((emb[:, :hist] - tgt) ** 2).mean((1, 2)))
        acc["shuffled"].append(((pred_shuf - tgt) ** 2).mean((1, 2)))
        roll = torch.stack(roll, 1)  # (B, rollout, D)
        future = emb[:, hist : hist + rollout]
        acc["roll"].append(((roll - future) ** 2).mean(2))
        acc["copyk"].append(((emb[:, hist - 1 : hist] - future) ** 2).mean(2))
        acc["var"].append(emb.reshape(-1, emb.size(-1)))
        feats.append(emb[:, :hist].cpu())
        states.append(batch["observation"][:, :hist].float())
    var = torch.cat(acc.pop("var")).var(0).mean().item()  # mean per-dim variance
    out = {k: torch.cat(v).mean(0).cpu().numpy() / var for k, v in acc.items()}
    out["emb_var"] = var
    return out, torch.cat(feats).numpy(), torch.cat(states).numpy()


def probe_targets(states, hist):
    """Single-frame and history targets; theta is unwrapped, so use sin/cos."""
    x, x_dot, th_dot, th = (states[..., i] for i in range(4))
    single = np.stack([x, np.sin(th), np.cos(th)], -1)  # per frame
    last = hist - 1
    history = np.stack(
        [x[:, last], x_dot[:, last], th_dot[:, last], np.sin(th[:, last]), np.cos(th[:, last])], -1
    )
    return single, history


def ridge_r2(train, val, test, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """Fit ridge on train, choose alpha by mean validation R^2, report R^2 per target."""
    (xtr, ytr), (xva, yva), (xte, yte) = train, val, test
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    ym = ytr.mean(0)
    z = lambda x: np.c_[(x - mu) / sd]  # noqa: E731
    a = z(xtr)

    def r2(w, x, y):
        res = ((z(x) @ w + ym - y) ** 2).sum(0)
        return 1 - res / ((y - y.mean(0)) ** 2).sum(0)

    best = None
    for alpha in alphas:
        w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (ytr - ym))
        score = r2(w, xva, yva)
        if best is None or score.mean() > best[0].mean():
            best = (score, alpha, w)
    score_va, alpha, w = best
    return {"alpha": alpha, "val": score_va.tolist(), "test": r2(w, xte, yte).tolist()}


def probes(feats, states, hist):
    out = {}
    single = {s: probe_targets(states[s], hist)[0] for s in feats}
    history = {s: probe_targets(states[s], hist)[1] for s in feats}
    flat = lambda s: (  # noqa: E731
        feats[s].reshape(-1, feats[s].shape[-1]),
        single[s].reshape(-1, single[s].shape[-1]),
    )
    out["single_frame"] = dict(
        targets=["x", "sin_theta", "cos_theta"],
        **ridge_r2(flat("train"), flat("validation"), flat("test")),
    )
    cat = lambda s: (feats[s].reshape(len(feats[s]), -1), history[s])  # noqa: E731
    out["history"] = dict(
        targets=["x", "x_dot", "theta_dot", "sin_theta", "cos_theta"],
        **ridge_r2(cat("train"), cat("validation"), cat("test")),
    )
    return out


def summarize(m, frameskip, dt):
    step_s = frameskip * dt
    return {
        "onestep_rel_mse": float(m["onestep"]),
        "copy_current_rel_mse": float(m["copy1"]),
        "shuffled_actions_rel_mse": float(m["shuffled"]),
        "rollout_rel_mse": m["roll"].tolist(),  # [0] = one step with full history
        "rollout_copy_last_rel_mse": m["copyk"].tolist(),
        "rollout_horizon_s": [round((k + 1) * step_s, 4) for k in range(len(m["roll"]))],
        "embedding_mean_dim_variance": m["emb_var"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True, help="downloaded checkpoints/<run-name>")
    p.add_argument("--weights", default=None, help="default: highest weights_epoch_N.pt")
    p.add_argument("--rollout", type=int, default=4, help="open-loop latent steps")
    p.add_argument("--max-windows", type=int, default=8192, help="per split")
    p.add_argument("--epoch-windows", type=int, default=2048, help="per epoch, validation")
    p.add_argument("--epochs", action="store_true", help="also evaluate every epoch")
    # Small defaults: a 64-window batch is ~270 MB of float32 frames, and 6 prefetching
    # workers exhausted laptop RAM (OOM, 2026-10-06).
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.run / "config.yaml")
    hist, fs = cfg.history_size, cfg.data.dataset.frameskip
    manifest = json.loads((DATA / "manifest.json").read_text())
    dt = manifest.get("control_dt") or manifest.get("args", {}).get("control_dt") or 0.05
    ckpts = sorted(args.run.glob("weights_epoch_*.pt"), key=lambda q: int(q.stem.split("_")[-1]))
    weights = Path(args.weights) if args.weights else ckpts[-1]
    torch.manual_seed(cfg.seed)

    splits = {s: load_split(s, fs, hist + args.rollout) for s in ("train", "validation", "test")}
    # Action normalizer fitted on train.h5, identical to train.py.
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size),
        get_column_normalizer(splits["train"], "action", "action"),
    )
    for ds in splits.values():
        ds.transform = transform

    report = {
        "run": str(args.run),
        "weights": weights.name,
        "frameskip": fs,
        "history_size": hist,
        "latent_step_s": fs * dt,
        "windows_per_split": {s: min(args.max_windows, len(d)) for s, d in splits.items()},
        "note": "Latent errors are relative to the mean per-dim variance of target embeddings.",
    }
    for name, w in [("trained", weights), ("random_init", None)]:
        t0 = time.time()
        model = load_model(args.run / "config.json", w)
        feats, states, latent = {}, {}, {}
        for s, ds in splits.items():
            m, feats[s], states[s] = run_model(model, ds, args, hist, args.rollout)
            if s != "train" and w is not None:  # an untrained predictor is meaningless
                latent[s] = summarize(m, fs, dt)
        report[name] = {"latent": latent, "probes": probes(feats, states, hist)}
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
        print(json.dumps(report[name], indent=1), flush=True)

    if args.epochs:
        args.max_windows = args.epoch_windows
        report["per_epoch_validation"] = []
        for ck in ckpts:
            model = load_model(args.run / "config.json", ck)
            m, _, _ = run_model(model, splits["validation"], args, hist, args.rollout)
            row = {"epoch": int(ck.stem.split("_")[-1]), **summarize(m, fs, dt)}
            report["per_epoch_validation"].append(row)
            print(f"epoch {row['epoch']}: onestep {row['onestep_rel_mse']:.4f}", flush=True)

    out = args.out or Path(__file__).resolve().parent / "outputs" / f"eval-{args.run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

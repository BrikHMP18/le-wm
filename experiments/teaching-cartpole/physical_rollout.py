"""Open-loop prediction error in physical units (cart position in cm, pole angle in degrees).

A linear probe (ridge, fitted on training embeddings, alpha chosen on validation)
reads x, sin(theta) and cos(theta) from one embedding. Applied to the model's
open-loop predicted embeddings, it turns each prediction into a physical state,
compared with the true state of the test episode at that instant:

    model     probe(predicted embedding at t+k)  vs. true state at t+k
    floor     probe(real embedding at t+k)       vs. true state at t+k   (probe error alone)
    baseline  true state at t (last context frame) vs. true state at t+k  ("nothing changes")

    .venv/bin/python experiments/teaching-cartpole/physical_rollout.py \
        --run experiments/teaching-cartpole/checkpoints/cartpole-fs5 --rollout 4
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from eval_lewm import (
    DATA,
    batches,
    get_column_normalizer,
    get_img_preprocessor,
    load_model,
    load_split,
    spt,
)


@torch.no_grad()
def collect(model, ds, args, hist, rollout):
    """Real embeddings, open-loop predicted embeddings, and true states per window."""
    real, pred, states = [], [], []
    for batch in batches(ds, args.max_windows, args.batch_size, args.workers, args.seed):
        pix = batch["pixels"].cuda()
        act = torch.nan_to_num(batch["action"].cuda(), 0.0)  # as in training
        emb = model.encode({"pixels": pix})["emb"]  # (B, hist + rollout, D)
        act_emb = model.action_encoder(act)
        cur = emb[:, :hist]
        for t in range(rollout):
            step = model.predict(cur[:, -hist:], act_emb[:, t : t + hist])[:, -1:]
            cur = torch.cat([cur, step], dim=1)
        real.append(emb.cpu())
        pred.append(cur[:, hist:].cpu())
        states.append(batch["observation"].float())
    return torch.cat(real).numpy(), torch.cat(pred).numpy(), torch.cat(states).numpy()


def window_policies(ds, split, n, seed):
    """Behavior policy of each sampled window, in the order batches() yields them."""
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False))
    with h5py.File(DATA / f"{split}.h5", "r") as f:
        ep_ids = f["episode_id"][f["ep_offset"][:]]
    manifest = json.loads((DATA / "manifest.json").read_text())
    policy = {e["episode_id"]: e["policy"] for e in manifest["episodes"]}
    return np.array([policy[int(ep_ids[ds.clip_indices[i][0]])] for i in idx])


def targets(states):  # (..., 4) [x, x_dot, theta_dot, theta] -> [x, sin, cos]
    return np.stack([states[..., 0], np.sin(states[..., 3]), np.cos(states[..., 3])], -1)


def fit_probe(x_tr, y_tr, x_va, y_va, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    mu, sd, ym = x_tr.mean(0), x_tr.std(0) + 1e-6, y_tr.mean(0)
    a = (x_tr - mu) / sd
    best = None
    for alpha in alphas:
        w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y_tr - ym))
        err = (((x_va - mu) / sd @ w + ym - y_va) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, alpha, w)
    _, alpha, w = best
    return (lambda x: (x - mu) / sd @ w + ym), alpha


def errors(read, true):  # read: (..., 3) [x, sin, cos]; true: (..., 4) states
    dx = np.abs(read[..., 0] - true[..., 0]) * 100  # cm
    ang = np.arctan2(read[..., 1], read[..., 2])
    dth = np.abs(np.angle(np.exp(1j * (ang - true[..., 3])))) * 180 / np.pi  # wrapped, deg
    return dx, dth


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--weights", default="weights_epoch_9.pt")
    p.add_argument(
        "--rollout", type=int, required=True, help="latent steps (1 s: 4 for fs5, 20 for fs1)"
    )
    p.add_argument("--max-windows", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.run / "config.yaml")
    hist, fs = cfg.history_size, cfg.data.dataset.frameskip
    dt = json.loads((DATA / "manifest.json").read_text())["control_dt"]
    model = load_model(args.run / "config.json", args.run / args.weights)
    data = {}
    for split in ("train", "validation", "test"):
        ds = load_split(split, fs, hist + args.rollout)
        if split == "train":
            norm = get_column_normalizer(ds, "action", "action")  # train.h5, as in train.py
        ds.transform = spt.data.transforms.Compose(
            get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size), norm
        )
        data[split] = collect(model, ds, args, hist, args.rollout)
        if split == "test":
            policies = window_policies(ds, split, args.max_windows, args.seed)
        print(split, "windows", len(data[split][0]), flush=True)

    # probe: single frame -> [x, sin, cos], fitted on real training embeddings
    flat = lambda split: (  # noqa: E731
        data[split][0].reshape(-1, data[split][0].shape[-1]),
        targets(data[split][2]).reshape(-1, 3),
    )
    probe, alpha = fit_probe(*flat("train"), *flat("validation"))

    real, pred, states = data["test"]
    future = states[:, hist:]  # true states at the predicted instants
    model_dx, model_dth = errors(probe(pred), future)
    floor_dx, floor_dth = errors(probe(real[:, hist:]), future)
    last = np.repeat(states[:, hist - 1 : hist], args.rollout, axis=1)
    base_dx = np.abs(last[..., 0] - future[..., 0]) * 100
    base_dth = np.abs(np.angle(np.exp(1j * (last[..., 3] - future[..., 3])))) * 180 / np.pi

    groups = {
        "all": np.ones(len(real), bool),
        "lqr": policies == "lqr",
        "random": policies == "random",
    }

    def stats(e):  # mean and median per horizon, for all windows and per behavior policy
        return {
            g: {
                "mean": e[m].mean(0).round(3).tolist(),
                "median": np.median(e[m], 0).round(3).tolist(),
            }
            for g, m in groups.items()
        }

    out = {
        "run": str(args.run), "weights": args.weights, "frameskip": fs, "probe_alpha": alpha,
        "test_windows": {g: int(m.sum()) for g, m in groups.items()},
        "horizon_s": [round((k + 1) * fs * dt, 4) for k in range(args.rollout)],
        "cart_position_error_cm": {"model": stats(model_dx), "probe_floor": stats(floor_dx),
                                   "nothing_changes": stats(base_dx)},
        "pole_angle_error_deg": {"model": stats(model_dth), "probe_floor": stats(floor_dth),
                                 "nothing_changes": stats(base_dth)},
    }  # fmt: skip
    path = (
        args.out
        or Path(__file__).resolve().parent / "outputs" / f"physical-{args.run.name}-epoch9.json"
    )
    path.write_text(json.dumps(out, indent=2) + "\n")
    for key in ("cart_position_error_cm", "pole_angle_error_deg"):
        for g in groups:
            d = {k: v[g]["mean"][-1] for k, v in out[key].items()}
            print(f"{key} {g} at {out['horizon_s'][-1]} s:", d)
    print("wrote", path)


if __name__ == "__main__":
    main()

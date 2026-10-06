"""LeWM-paper-style decoded videos for held-out TeachingCartpole episodes.

Three panels per frame, as on the LeWM website:
  (i)   real frames,
  (ii)  each frame encoded and decoded independently (what the embedding keeps),
  (iii) open-loop prediction: the world model sees only the first `history_size`
        frames plus the recorded actions, rolls out in latent space, and each
        predicted embedding is decoded.

Frames are shown at the model's latent step (frameskip x control_dt). Run from the
fork root with the fork's environment:

    .venv/bin/python experiments/teaching-cartpole/decode_video.py \
        --run experiments/teaching-cartpole/checkpoints/cartpole-fs5 --weights weights_epoch_9.pt \
        --decoder experiments/teaching-cartpole/checkpoints/decoders/cartpole-fs5-weights_epoch_9
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the HDF5 compression filters
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decoder import load_decoder, load_world_model  # noqa: E402

DATA = HERE / "data" / "cartpole-10k-v1"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet, as in training
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_col(f, key, rows=None):
    """Slice-read: whole-column reads of these one-row-chunk columns cost GBs (see eval_lewm)."""
    col = f[key]
    if rows is not None:
        return col[rows.start : rows.stop]
    out = np.empty(col.shape, col.dtype)
    for i in range(0, len(col), 50_000):
        out[i : i + 50_000] = col[i : i + 50_000]
    return out


def to_input(pixels):  # (T, H, W, 3) uint8 -> (T, 3, H, W) ImageNet-normalized
    x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD


def to_rgb(x):  # (3, H, W) normalized -> (H, W, 3) uint8
    x = (x.cpu().float() * STD + MEAN).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def pick_episodes(n_per_policy):
    """Longest held-out (test split) episodes of each behavior policy."""
    m = json.loads((DATA / "manifest.json").read_text())
    test = [e for e in m["episodes"] if e["split"] == "test"]
    picks = []
    for policy in ("lqr", "random"):
        eps = sorted((e for e in test if e["policy"] == policy), key=lambda e: -e["rows"])
        picks += eps[:n_per_policy]
    return picks


@torch.no_grad()
def episode_panels(world, dec, f, episode_id, fs, hist, act_mean, act_std, max_steps):
    ids = f["episode_id"][f["ep_offset"][:]]
    k = int(np.flatnonzero(ids == episode_id)[0])
    start, length = int(f["ep_offset"][k]), int(f["ep_len"][k])
    steps = min((length - 1) // fs + 1, max_steps)  # frames at 0, fs, 2fs, ...
    rows = slice(start, start + (steps - 1) * fs + 1)
    pixels = read_col(f, "pixels", rows)[::fs]
    raw_act = read_col(f, "action", rows).astype(np.float32)  # (rows, 1)
    act = np.zeros((steps, fs), np.float32)  # action block driving step t -> t+1
    for t in range(steps):
        blk = raw_act[t * fs : (t + 1) * fs, 0]
        act[t, : len(blk)] = blk
    act = np.nan_to_num((act - act_mean) / act_std, nan=0.0)  # normalize, then NaN -> 0

    x = to_input(pixels).cuda()
    emb = world.encode({"pixels": x[None]})["emb"][0]  # (T, D)
    act_emb = world.action_encoder(torch.from_numpy(act).cuda()[None])[0]  # (T, A)
    cur = emb[:hist]
    for t in range(steps - hist):
        nxt = world.predict(cur[None, -hist:], act_emb[None, t : t + hist])[0, -1:]
        cur = torch.cat([cur, nxt])
    recon, pred = dec(emb), dec(cur)
    # raw latent MSE: SIGReg keeps embeddings at ~unit variance, so this is on the scale of
    # eval_lewm's relative errors (per-episode normalization would inflate calm episodes)
    err = ((cur - emb) ** 2).mean(1).cpu().numpy()
    return pixels, [to_rgb(r) for r in recon], [to_rgb(p) for p in pred], err


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose(real, recon, pred, t, hist, step_s, err, title, scale=2):
    w, pad, top = real.shape[1] * scale, 12, 64
    canvas = Image.new("RGB", (3 * w + 4 * pad, w + top + pad), "white")
    d = ImageDraw.Draw(canvas)
    heads = ["(i) real", "(ii) encode \u2192 decode", "(iii) predicted (open loop)"]
    for i, (img, head) in enumerate(zip([real, recon, pred], heads)):
        x0 = pad + i * (w + pad)
        canvas.paste(Image.fromarray(img).resize((w, w), Image.NEAREST), (x0, top))
        d.text((x0, top - 24), head, fill=(11, 11, 11), font=font(18))
    status = "context (observed)" if t < hist else f"imagined, latent MSE {err:.2f}"
    d.text(
        (pad, 8), f"{title}   t = {t * step_s:.2f} s   [{status}]", fill=(82, 81, 78), font=font(16)
    )
    if t >= hist:  # mark the imagined panel
        x0 = pad + 2 * (w + pad)
        d.rectangle([x0 - 2, top - 2, x0 + w + 1, top + w + 1], outline=(42, 120, 214), width=3)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--weights", default="weights_epoch_9.pt")
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2, help="per policy (lqr, random)")
    p.add_argument("--max-steps", type=int, default=24, help="latent steps per video")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "videos")
    args = p.parse_args()

    cfg = OmegaConf.load(args.run / "config.yaml")
    fs, hist = cfg.data.dataset.frameskip, cfg.history_size
    step_s = fs * json.loads((DATA / "manifest.json").read_text())["control_dt"]
    world = load_world_model(args.run, args.weights).cuda()
    dec = load_decoder(args.decoder)
    with h5py.File(DATA / "train.h5", "r") as f:  # action normalizer: train.h5, as in train.py
        a = read_col(f, "action")
    a = a[~np.isnan(a).any(1)]
    act_mean, act_std = float(a.mean()), float(a.std(ddof=1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.run.name}-{Path(args.weights).stem}"
    with h5py.File(DATA / "test.h5", "r") as f:
        for e in pick_episodes(args.episodes):
            real, recon, pred, err = episode_panels(
                world, dec, f, e["episode_id"], fs, hist, act_mean, act_std, args.max_steps
            )
            title = f"{tag}  test ep {e['episode_id']} ({e['policy']})"
            frames = [
                compose(real[t], recon[t], pred[t], t, hist, step_s, err[t], title)
                for t in range(len(real))
            ]
            stem = args.out_dir / f"{tag}-ep{e['episode_id']}-{e['policy']}"
            frames[0].save(
                f"{stem}.gif", save_all=True, append_images=frames[1:],
                duration=int(1000 / args.fps), loop=0,
            )  # fmt: skip
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{frames[0].width}x{frames[0].height}", "-r", str(args.fps), "-i", "-",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{stem}.mp4"],
                stdin=subprocess.PIPE,
            )  # fmt: skip
            for fr in frames:
                enc.stdin.write(fr.tobytes())
            enc.stdin.close()
            enc.wait()
            print(f"{stem}.gif/.mp4  steps={len(real)}  final latent err={err[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()

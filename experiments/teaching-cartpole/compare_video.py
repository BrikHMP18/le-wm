"""Side-by-side open-loop imagination of two LeWM runs on the same held-out episodes.

Panels: (i) real, then each run's decoded open-loop prediction, sampled at the
coarser run's latent step so all panels show the same instant. Each run sees only
its own `history_size` context frames (so a finer frameskip starts imagining
earlier) plus the recorded actions.

    .venv/bin/python experiments/teaching-cartpole/compare_video.py \
        --runs cartpole-fs5 cartpole-fs1 --weights weights_epoch_9.pt
"""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from decode_video import DATA, HERE, episode_panels, font, pick_episodes, read_col
from decoder import load_decoder, load_world_model

COLORS = [(42, 120, 214), (235, 104, 52)]  # dataviz slots 1-2, as in plot_eval.py


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs=2, default=["cartpole-fs5", "cartpole-fs1"])
    p.add_argument("--weights", default="weights_epoch_9.pt")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--episodes", type=int, default=2, help="per policy (lqr, random)")
    p.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "videos")
    args = p.parse_args()

    dt = json.loads((DATA / "manifest.json").read_text())["control_dt"]
    with h5py.File(DATA / "train.h5", "r") as f:  # action normalizer, as in train.py
        a = read_col(f, "action")
    a = a[~np.isnan(a).any(1)]
    act_mean, act_std = float(a.mean()), float(a.std(ddof=1))

    runs = []
    for name in args.runs:
        run = HERE / "checkpoints" / name
        cfg = OmegaConf.load(run / "config.yaml")
        stem = Path(args.weights).stem
        runs.append(
            dict(
                name=name,
                fs=cfg.data.dataset.frameskip,
                hist=cfg.history_size,
                world=load_world_model(run, args.weights).cuda(),
                dec=load_decoder(HERE / "checkpoints" / "decoders" / f"{name}-{stem}"),
            )
        )
    coarse = max(r["fs"] for r in runs)
    step_s = coarse * dt
    n_out = int(round(args.seconds / step_s))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA / "test.h5", "r") as f:
        for e in pick_episodes(args.episodes):
            real, preds = None, []
            for r in runs:
                stride = coarse // r["fs"]  # model steps per output frame
                steps = (n_out - 1) * stride + 1
                rl, _, pr, _ = episode_panels(
                    r["world"], r["dec"], f, e["episode_id"], r["fs"], r["hist"],
                    act_mean, act_std, steps,
                )  # fmt: skip
                preds.append((pr[::stride], r["hist"] * r["fs"] * dt))
                if r["fs"] == coarse:
                    real = rl
            n = min(len(real), *(len(p) for p, _ in preds))
            w, pad, top = 224 * 2, 12, 64
            frames = []
            for t in range(n):
                canvas = Image.new("RGB", (3 * w + 4 * pad, w + top + pad), "white")
                d = ImageDraw.Draw(canvas)
                panels = [(real[t], "(i) real", None, False)]
                for i, (r, (pr, ctx_s)) in enumerate(zip(runs, preds)):
                    imagined = t * step_s >= ctx_s
                    tag = "imagined" if imagined else "observed context"
                    panels.append((pr[t], f"{r['name']} ({tag})", COLORS[i], imagined))
                for i, (img, head, color, boxed) in enumerate(panels):
                    x0 = pad + i * (w + pad)
                    canvas.paste(Image.fromarray(img).resize((w, w), Image.NEAREST), (x0, top))
                    d.text((x0, top - 24), head, fill=color or (11, 11, 11), font=font(18))
                    if boxed:
                        d.rectangle(
                            [x0 - 2, top - 2, x0 + w + 1, top + w + 1], outline=color, width=3
                        )
                d.text(
                    (pad, 8),
                    f"test ep {e['episode_id']} ({e['policy']})   t = {t * step_s:.2f} s   "
                    f"open loop: each model sees 3 frames + recorded actions",
                    fill=(82, 81, 78),
                    font=font(16),
                )
                frames.append(canvas)
            stem = (
                args.out_dir / f"compare-{'-vs-'.join(args.runs)}-ep{e['episode_id']}-{e['policy']}"
            )
            frames[0].save(
                f"{stem}.gif", save_all=True, append_images=frames[1:], duration=250, loop=0
            )
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{frames[0].width}x{frames[0].height}", "-r", "4", "-i", "-",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{stem}.mp4"],
                stdin=subprocess.PIPE,
            )  # fmt: skip
            for fr in frames:
                enc.stdin.write(fr.tobytes())
            enc.stdin.close()
            enc.wait()
            print(f"{stem}.gif/.mp4  frames={n}", flush=True)


if __name__ == "__main__":
    main()

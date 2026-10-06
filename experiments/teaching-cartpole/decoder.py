"""Post-hoc image decoder for a frozen LeWM, used only for visualization.

Follows the LeWM paper's diagnostic decoder: the 192-d embedding is projected
to a few context tokens that serve as keys/values for cross-attention; one
learnable query per image patch attends to them through residual
cross-attention + MLP blocks, and a linear head emits each patch's pixels.

It decodes the *projected* embedding (`JEPA.encode(...)["emb"]`), the space the
predictor works in, so predicted embeddings can be decoded the same way as
encoded ones. The paper does not report the decoder's training recipe; here it
is MSE on ImageNet-normalized pixels with AdamW, the world model frozen.

Train (from the fork root; on Modal see modal_train.py --decoder):
    .venv/bin/python experiments/teaching-cartpole/decoder.py \
        --run experiments/teaching-cartpole/checkpoints/cartpole-fs5 \
        --weights weights_epoch_9.pt --data <dir with train.h5> --out <decoder dir>
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the HDF5 compression filters
import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from torch import nn

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK))  # jepa.py, module.py, utils.py

from utils import get_img_preprocessor  # noqa: E402


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )

    def forward(self, q, kv):
        kv = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class LatentDecoder(nn.Module):
    def __init__(
        self, embed_dim=192, dim=384, depth=4, heads=6, ctx_tokens=8, img_size=224, patch=14
    ):
        super().__init__()
        self.config = dict(
            embed_dim=embed_dim, dim=dim, depth=depth, heads=heads,
            ctx_tokens=ctx_tokens, img_size=img_size, patch=patch,
        )  # fmt: skip
        self.grid, self.patch = img_size // patch, patch
        self.ctx_tokens, self.dim = ctx_tokens, dim
        self.lift = nn.Linear(embed_dim, ctx_tokens * dim)
        self.queries = nn.Parameter(torch.randn(1, self.grid**2, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3 * patch * patch)

    def forward(self, z):  # (B, D) -> (B, 3, H, W), ImageNet-normalized pixels
        b = z.size(0)
        kv = self.lift(z).view(b, self.ctx_tokens, self.dim)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        x = self.head(self.norm(q))  # (B, G*G, 3*p*p)
        g, p = self.grid, self.patch
        return x.view(b, g, g, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, g * p, g * p)


def load_world_model(run, weights):
    model = hydra.utils.instantiate(OmegaConf.create(json.loads((run / "config.json").read_text())))
    model.load_state_dict(torch.load(run / weights, map_location="cpu"))
    return model.eval().requires_grad_(False)


def load_decoder(path, device="cuda"):
    path = Path(path)
    dec = LatentDecoder(**json.loads((path / "decoder.json").read_text())["config"])
    dec.load_state_dict(torch.load(path / "decoder.pt", map_location="cpu"))
    return dec.to(device).eval()


def frames(h5, img_size):
    """Single frames (num_steps=1) with the training image preprocessing."""
    # absolute path: a relative one is resolved as a HuggingFace repo id
    ds = swm.data.load_dataset(
        str(Path(h5).resolve()), frameskip=1, num_steps=1, keys_to_load=["pixels"]
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    )
    return ds


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True, help="checkpoint dir with config.json")
    p.add_argument("--weights", default="weights_epoch_9.pt")
    p.add_argument("--data", type=Path, required=True, help="dir containing train.h5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    torch.manual_seed(0)
    world = load_world_model(args.run, args.weights).cuda()
    dec = LatentDecoder().cuda()
    ds = frames(args.data / "train.h5", img_size=224)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )  # fmt: skip
    total = int(args.epochs * len(loader))
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.05)
    warm = min(500, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(
        config=dec.config, world_model=str(args.run / args.weights), steps=total,
        batch_size=args.batch_size, lr=args.lr, loss="MSE on ImageNet-normalized pixels",
        input="projected embedding JEPA.encode()['emb']", world_model_frozen=True,
    )  # fmt: skip
    print(f"{len(ds)} frames, {len(loader)} batches/epoch, {total} steps", flush=True)

    step, t0, run_loss = 0, time.time(), 0.0
    while step < total:
        for batch in loader:
            pix = batch["pixels"].cuda(non_blocking=True)  # (B, 1, 3, H, W)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    z = world.encode({"pixels": pix})["emb"][:, 0]
                loss = nn.functional.mse_loss(dec(z.float()).float(), pix[:, 0].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % 100 == 0 or step == total:
                rate = step / (time.time() - t0)
                print(
                    f"step {step}/{total} loss {run_loss / 100:.4f} ({rate:.1f} it/s)", flush=True
                )
                run_loss = 0.0
            if step % 2000 == 0 or step == total:
                torch.save(dec.state_dict(), args.out / "decoder.pt")
                (args.out / "decoder.json").write_text(json.dumps({**meta, "step": step}, indent=2))
            if step >= total:
                break
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()

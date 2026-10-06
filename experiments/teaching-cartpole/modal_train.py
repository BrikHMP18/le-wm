"""Run upstream LeWM `train.py` on Modal for the TeachingCartpole dataset.

One-time upload (from the fork root):
    modal volume create lewm-cartpole
    modal volume put lewm-cartpole \
        experiments/teaching-cartpole/data/cartpole-10k-v1/train.h5 \
        stablewm/datasets/cartpole-10k-v1/train.h5

Preflight (~1 min: image, GPU, dataset on the volume, W&B secret):
    modal run experiments/teaching-cartpole/modal_train.py --check-only

Smoke test (attached; Ctrl+C or closing the terminal cancels it):
    modal run experiments/teaching-cartpole/modal_train.py \
        --smoke --run-name smoke --wandb-entity mu-robotics

Long runs must use --background. `modal run --detach` does NOT survive the laptop
shutting down: the server cancels the in-flight call when the client disconnects
(this killed cartpole-fs1 at epoch 10 on 2026-10-06). --background spawns the call
on the deployed app, which runs to completion with no client attached. Redeploy
after any code or config change, since the deployed app is a snapshot:
    modal deploy experiments/teaching-cartpole/modal_train.py
    modal run experiments/teaching-cartpole/modal_train.py --background \
        --run-name cartpole-fs5 --wandb-entity mu-robotics
Follow it with `modal app logs lewm-cartpole` or the Modal dashboard (W&B gets
metrics only; weights stay on the volume).

Visualization decoder for a finished run (frozen world model; ~30 min on an L40S;
saved to stablewm/decoders/<run-name>-<weights stem>/ on the volume):
    modal run experiments/teaching-cartpole/modal_train.py --background --decoder \
        --run-name cartpole-fs5 --decoder-weights weights_epoch_9.pt

Download weights afterwards:
    modal volume get lewm-cartpole stablewm/checkpoints/cartpole-fs5 \
        experiments/teaching-cartpole/checkpoints/
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

# Fork root, only meaningful locally; inside the container this file sits at /root.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
REMOTE = "/root/le-wm"
VOL = "/vol"
DATASET = "cartpole-10k-v1/train.h5"
# Exact pins frozen from le-wm/.venv (stable-worldmodel[train] @ 21446f1 + hdf5plugin),
# so Modal runs the same stack that was smoke-tested locally.
REQS = Path(__file__).resolve().parent / "requirements-modal.txt"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_pip_install(requirements=[str(REQS)])
    .add_local_dir(ROOT / "config", f"{REMOTE}/config")
    .add_local_file(ROOT / "train.py", f"{REMOTE}/train.py")
    .add_local_file(ROOT / "jepa.py", f"{REMOTE}/jepa.py")
    .add_local_file(ROOT / "module.py", f"{REMOTE}/module.py")
    .add_local_file(ROOT / "utils.py", f"{REMOTE}/utils.py")
    .add_local_file(
        ROOT / "experiments/teaching-cartpole/decoder.py",
        f"{REMOTE}/experiments/teaching-cartpole/decoder.py",
    )
)
volume = modal.Volume.from_name("lewm-cartpole", create_if_missing=True)
app = modal.App("lewm-cartpole", image=image)


@app.function(gpu="L40S", cpu=16, memory=65536, timeout=24 * 3600, volumes={VOL: volume})
def train(overrides: list[str]):
    subprocess.run(["nvidia-smi"], check=True)
    copy_dataset()

    env = {
        **os.environ,  # keep secrets such as WANDB_API_KEY
        "STABLEWM_HOME": f"{VOL}/stablewm",  # weights_epoch_N.pt land here
        "SPT_CACHE_DIR": f"{VOL}/spt",  # Lightning checkpoints and run metadata
        "LOCAL_DATASET_DIR": "/tmp/data",
    }
    cmd = ["python", "train.py", "data=cartpole", *overrides]
    print("+", shlex.join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=REMOTE, env=env, check=True)
    finally:
        volume.commit()


def copy_dataset():
    """Copy train.h5 to container disk; random HDF5 reads over the volume are slow."""
    src = Path(VOL, "stablewm/datasets", DATASET)
    dst = Path("/tmp/data/datasets", DATASET)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst.parent


@app.function(gpu="L40S", cpu=16, memory=65536, timeout=24 * 3600, volumes={VOL: volume})
def train_decoder(run_name: str, weights: str, args: list[str]):
    """Visualization decoder for a frozen checkpoint (decoder.py); saved on the volume."""
    data = copy_dataset()
    out = Path(VOL, "stablewm/decoders", f"{run_name}-{Path(weights).stem}")
    cmd = [
        "python", "experiments/teaching-cartpole/decoder.py",
        "--run", f"{VOL}/stablewm/checkpoints/{run_name}", "--weights", weights,
        "--data", str(data), "--out", str(out), *args,
    ]  # fmt: skip
    print("+", shlex.join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=REMOTE, check=True)
    finally:
        volume.commit()


@app.function(
    gpu="L40S", timeout=600, volumes={VOL: volume}, secrets=[modal.Secret.from_name("wandb-secret")]
)
def check():
    """Preflight (~1 min): same image, GPU, dataset on the volume, and the W&B secret."""
    import h5py
    import hdf5plugin  # noqa: F401  registers the HDF5 compression filters
    import stable_pretraining  # noqa: F401
    import stable_worldmodel  # noqa: F401
    import torch
    import wandb

    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).float().abs().mean() > 0
    print("gpu", torch.cuda.get_device_name(), "- bf16 matmul ok")

    src = Path(VOL, "stablewm/datasets", DATASET)
    with h5py.File(src, "r") as f:
        print("dataset", src, f"{src.stat().st_size / 1e9:.2f} GB", "pixels", f["pixels"].shape)
        assert f["pixels"][0].shape == (224, 224, 3)

    print("code", sorted(p.name for p in Path(REMOTE).iterdir()))
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing from wandb-secret"
    viewer = wandb.Api().viewer
    print("wandb user", viewer.username, "- entities you can log to:", viewer.teams)
    print("CHECK OK")


@app.local_entrypoint()
def main(
    run_name: str = "cartpole-fs5",
    max_epochs: int = 10,
    frameskip: int = 5,
    gpu: str = "L40S",
    wandb_entity: str = "",
    smoke: bool = False,
    extra: str = "",
    check_only: bool = False,
    background: bool = False,
    decoder: bool = False,
    decoder_weights: str = "weights_epoch_9.pt",
):
    # --background: look the function up on the deployed app (`modal deploy`) and spawn,
    # so the call is not tied to this client process.
    def fn(local, name):
        return modal.Function.from_name("lewm-cartpole", name) if background else local

    def launch(f, *a):
        if background:
            call = f.spawn(*a)
            print(f"Spawned {call.object_id} on the deployed app; safe to close this terminal.")
        else:
            f.remote(*a)

    if check_only:
        launch(fn(check, "check"))
        return
    if decoder:  # --run-name selects the trained checkpoint; --extra goes to decoder.py
        launch(fn(train_decoder, "train_decoder"), run_name, decoder_weights, shlex.split(extra))
        return
    overrides = [
        f"output_model_name={run_name}",
        f"subdir={run_name}",
        f"trainer.max_epochs={1 if smoke else max_epochs}",
        f"data.dataset.frameskip={frameskip}",
        "num_workers=14",
    ]
    if smoke:
        overrides += ["+trainer.limit_train_batches=50", "+trainer.limit_val_batches=10"]
    secrets = []
    if wandb_entity:
        secrets = [modal.Secret.from_name("wandb-secret")]
        overrides += [
            "wandb.enabled=true",
            f"wandb.config.entity={wandb_entity}",
            "wandb.config.project=lewm-cartpole",
        ]
    overrides += shlex.split(extra)
    launch(fn(train, "train").with_options(gpu=gpu, secrets=secrets), overrides)

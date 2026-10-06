# TeachingCartpole — Q2(e) data collection

Prepared October 5, 2026. Branch: `feat/teaching-cartpole-data`.
Covers single-pole data collection and LeWM training on Modal; evaluation and
MPC integration follow.
Simulator and schema provenance: [SOURCES.md](SOURCES.md).

## What LeWM learns

LeWM's `jepa.py` takes a short history of RGB images and actions. An image encoder
maps each image to an embedding; an action-conditioned predictor predicts the
next embedding. Training combines embedding prediction loss with SIGReg to
discourage collapsed representations. The upstream history default is three
frames. There is no image decoder: a predicted embedding is not a predicted
video frame or a direct physical-state estimate.

For this experiment, the action is one scalar horizontal force in newtons.
Images and actions are learning inputs; physical states are collected as ground
truth for diagnostics/probes. They must not silently become privileged inputs
to the visual model. Later, MPC will choose forces by evaluating predicted
latent trajectories against a goal; it is not implemented in this stage.

## Setup and a pilot

Run from the fork root `/home/autobrik/Projects/master/le-wm`:

```bash
uv sync --project experiments/teaching-cartpole --python 3.11

uv run --directory experiments/teaching-cartpole python -m teaching_cartpole.collect \
  --output data/pilot-v1 --episodes 30 --steps 200 --seed 42 --policy mixed

uv run --directory experiments/teaching-cartpole python -m teaching_cartpole.inspect \
  data/pilot-v1 --check-physics --preview
```

`--directory` changes the working directory, so these outputs are under
`experiments/teaching-cartpole/data/pilot-v1/`. Collection opens no window and
needs no GPU. The pinned `uv.lock` belongs to this experiment; it does not
install or alter the environment used by the assignment or upstream training.

Every output directory must be new: existing runs are never overwritten or
appended. If interrupted, completed episodes are retained and the manifest is
marked interrupted. Start a fresh output directory rather than treating that
run as complete. Concurrent collectors must use distinct output directories.

## Format and alignment

Each run contains `train.h5`, `validation.h5`, `test.h5`, and `manifest.json`.
Splits are selected deterministically **by whole episode**, approximately
80/10/10, with at least one episode in each split. Small runs use rounded counts.
Rows are flat and episode-contiguous; images use lossless gzip compression.

| HDF5 field | Shape / content |
| --- | --- |
| `pixels` | `(N, 224, 224, 3)`, RGB uint8; fixed-camera image, resized with letterboxing |
| `action` | `(N, 1)`, float32 applied force, in N |
| `observation` | `(N, 4)`, float64 `[x, x_dot, theta_dot, theta]` |
| `timestamp` | `(N,)`, simulated seconds from episode reset |
| `episode_id` | `(N,)`, unique ID across the run's three splits |
| `terminated` | Track-boundary termination, true only on the final row |
| `truncated` | Time-limit termination, true only on the final row |
| `ep_len` | Number of observation rows per episode |
| `ep_offset` | Start row of each episode in this split |

The table describes default image resolution; `--image-size` can change it.
Position is in metres, angle in radians, velocities in m/s and rad/s.
Theta = 0 is downward, theta = pi is upright; angles remain unwrapped.

**Row t contains image/state at t and the force that produces row t+1.** An
episode of T transitions has T+1 observations. Its final action is NaN because
there is no outgoing transition. This matches LeWM's existing treatment of
boundary-action NaNs. It must never be treated as an observed zero-force action
by a downstream physical-dynamics evaluator. Temporal windows stay within one
episode; the final image remains available as a prediction target.

The manifest records arguments, physics constants, state order, simulated
control interval, per-episode RNG seed components, initial states, policy,
split, end reason, Git revision, and hashes of collector sources. Source hashes
identify the executed code even before the working tree is committed.

## Collection choices

Defaults: physics dt = 0.005 s, ten physics substeps per observation/action,
so the control interval is **0.05 s (20 Hz)**. Frames are stored at every
control step; temporal subsampling happens at load time via the loader's
`frameskip` (training uses **frameskip = 5**, see below), so the same files serve
any frameskip without recollection.

- `--policy random`: uniform forces within +/-10 N, held for two control steps
  by default (0.1 s); provides exploration rather than balance demonstrations.
- `--policy lqr`: local balancing LQR, clipped to the same force bound, with
  Gaussian force noise (default standard deviation 0.5 N).
- `--policy mixed`: alternate exploratory random episodes and noisy LQR
  episodes. The LQR episodes start near upright; random episodes use the
  requested initial-state distribution.
- `--policy zero`: zero force, useful for sanity checks.

`--initial mixed` samples upright (pi +/-0.15 rad), downward (+/-0.3 rad), or
uniform angles in [-pi, pi]. Other options select one distribution. Initial cart
position is uniform within +/-min(0.5, 0.8 * track_limit) m, velocity within
+/-0.5 m/s, angular velocity within +/-1 rad/s.

Episodes end at the requested number of transitions or when abs(x) >= 2.5 m
at a control endpoint. This is a camera/data-collection bound, not a collision
model in the original simulator. We retain the final boundary observation.
Mixed/random data are not guaranteed to achieve swing-up; a dedicated swing-up
collection policy remains future work.

After inspecting the pilot and split coverage, a larger run can use:

```bash
uv run --directory experiments/teaching-cartpole python -m teaching_cartpole.collect \
  --output data/cartpole-v1 --episodes 300 --steps 200 --seed 43 --policy mixed
```

The larger run is a suggested starting point, not an established data requirement.
It has not been launched. Review the number of valid temporal windows and
coverage of positions/angles/velocities before deciding how much data to collect.

## Inspection and training boundary

### Paper-scale collection

Appendix E of [LeWM v3](https://arxiv.org/html/2603.19312v3) uses 10,000
200-step episodes for Reacher and OGBench-Cube (about two million steps),
10,000 episodes averaging 92 steps for TwoRoom, and 20,000 episodes averaging
196 steps for PushT. These are benchmark dataset sizes, not a demonstrated
minimum for TeachingCartpole. The paper trains each model for ten epochs;
the inspected fork's default `max_epochs` is 100; `modal_train.py` overrides it
to 10 by default.

The October 5 collection uses the benchmark episode count and a 200-transition
cap, while retaining the camera-bound termination. Thus the actual transition
count is below two million. It mixes equal numbers of random-exploration and
noisy local-LQR episodes. The temporal discretization is a documented CartPole
adaptation: frames are stored at 20 Hz and loaded with the paper's frameskip 5,
so one latent step is 0.25 s (an optional frameskip-1 ablation is planned).

Completed run: `data/cartpole-10k-v1/`, seed 43, **10,000 episodes and
1,507,974 transitions** (1,517,974 observation frames), with 8,000 training,
1,000 validation, and 1,000 test episodes. Collection wall time was 1,095.92 s.
These counts describe a new TeachingCartpole dataset, not a replication of
the paper's environments or evidence of sufficient training data for control.
All 5,000 LQR episodes reach the 200-transition cap (1,000,000 transitions).
The 5,000 exploratory episodes contribute 507,974 transitions; 4,321 terminate
at the track/camera bound. Validation passed for all episode structures and
boundaries, a physical replay sample of 10 episodes per split, and native-reader
first/last windows in every split. Coverage densities are normalized separately
per behavior policy.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --directory experiments/teaching-cartpole python -m teaching_cartpole.collect \
  --output data/cartpole-10k-v1 --episodes 10000 --steps 200 --seed 43 \
  --policy mixed --workers 6
```

`--workers` parallelizes independent episodes; arrays and splits are identical
to single-worker execution. Pending frame buffers are bounded to small batches.
The manifest is checkpointed every 100 episodes and finalized on completion or
a handled interruption; HDF5 episodes are flushed individually. A hard process
kill may leave an incomplete manifest, and such runs must not be used as complete
datasets.

For large runs, validate every episode's structure and replay a deterministic
sample of ten episodes per split:

```bash
uv run --directory experiments/teaching-cartpole python -m teaching_cartpole.inspect \
  data/cartpole-10k-v1 --physics-sample 10 --preview
```

Export counts and physical-state coverage (terminal-only rows excluded):

```bash
uv run --directory experiments/teaching-cartpole --extra analysis \
  python -m teaching_cartpole.summarize data/cartpole-10k-v1 \
  --output-json outputs/cartpole-10k-v1-summary.json \
  --figure outputs/cartpole-10k-v1-coverage.png
```

The optional `analysis` extra adds Matplotlib; it is unnecessary for collection.
Selected final summary/figure artifacts are copied to the assignment report's
`figures/` directory. Collection diagnostics must not be described as trained
model or control results.

### Dataset validation and training

The inspector checks shapes, episode offsets, force bounds, timestamps, final
flags, manifest agreement, unique episode IDs across splits, and optional full
physics replay of the stored forces. `--preview` writes a six-frame contact sheet
and GIF from the longest training episode, under the run's `preview/` directory.

The HDF5 layout follows the stable-worldmodel native reader at the revision in
SOURCES.md. Its loader converts RGB HWC to CHW and groups actions according to
frameskip.

**Decision (supersedes an earlier plan to modify `train.py`):** training runs the
upstream `train.py` **unmodified**, as in the paper, on `train.h5` only, through
`config/train/data/cartpole.yaml` (frameskip 5, history 3 + 1 prediction, so each
window spans 20 stored frames; the action input has dimension 5). Consequences:

- Normalizers are fitted on `train.h5` only, since validation/test files are never
  loaded during training.
- The `validate/*` losses logged during training use a **random split of
  temporal windows from `train.h5`**, so windows from the same episode appear in
  both parts. They are optimistic and serve only as a convergence signal; do not
  report them as generalization results.
- Held-out evaluation uses `validation.h5` and `test.h5` (split by episode) after
  training: latent prediction error and physical-state probes, then MPC.
- The model sees only pixels and actions; `observation` is loaded for diagnostics
  and probes, never as a model input.

An optional native-reader compatibility check can run in a temporary CPU
environment, without changing the collection environment (from the fork root):

```bash
uv run --directory experiments/teaching-cartpole --no-project --python 3.11 \
  --with 'stable-worldmodel @ git+https://github.com/galilai-group/stable-worldmodel.git@21446f1ede6d5284e981bd7b47f432b994e6d812' \
  --with h5py --with hdf5plugin --index https://download.pytorch.org/whl/cpu \
  python -m teaching_cartpole.check_native data/pilot-v1
```

This optional check downloads PyTorch CPU and data-reading dependencies. They
are not required for collection. It verifies first/last windows in every split
against the stored arrays and reports loader tensor shapes.

### Browsing the data

[`tools/dataviz`](../../tools/dataviz/README.md) is a local browser viewer for
these HDF5 files: frame player synced with `observation`/`action`/flag plots,
and an episode list filterable by `policy`, `initial_mode`, and `end_reason`.
It has its own `uv` environment and reads the files without modifying them.
From the fork root:

```bash
uv run --directory tools/dataviz dataviz \
  ../../experiments/teaching-cartpole/data/cartpole-10k-v1 --open
```

Pass a run directory to switch between `train.h5`, `validation.h5`, and
`test.h5` in the header, or a single `.h5` file. Defaults to
`http://127.0.0.1:8765/`; use `--port` to change it. Keys: `space` play/pause,
`←/→` frame, `↑/↓` episode. On a remote machine, run with the default host and
forward the port (`ssh -L 8765:127.0.0.1:8765 <host>`).

## Training on Modal

`modal_train.py` runs the unmodified `train.py` on an L40S (16 CPU, 64 GB RAM,
24 h timeout, Modal's maximum) in workspace `murobotics-ai-9230`. The image
installs `requirements-modal.txt`, an exact freeze of the fork's `.venv`
(stable-worldmodel[train] at the SOURCES.md revision, torch 2.14.1+cu130,
hdf5plugin), so Modal runs the same stack that was smoke-tested locally. Only
`train.h5` is uploaded; the job copies it from the volume to container disk.

One-time setup (from the fork root):

```bash
modal volume create lewm-cartpole
modal volume put lewm-cartpole experiments/teaching-cartpole/data/cartpole-10k-v1/train.h5 \
  stablewm/datasets/cartpole-10k-v1/train.h5
# Modal secret "wandb-secret" with WANDB_API_KEY (created in the Modal dashboard)
```

Run (from the fork root, since the script path is relative):

```bash
modal run experiments/teaching-cartpole/modal_train.py --check-only   # ~1 min preflight
modal run experiments/teaching-cartpole/modal_train.py --smoke --run-name smoke --wandb-entity mu-robotics
modal deploy experiments/teaching-cartpole/modal_train.py              # snapshot; redeploy after edits
modal run experiments/teaching-cartpole/modal_train.py --background --run-name cartpole-fs5 --wandb-entity mu-robotics
```

Flags: `--run-name`, `--max-epochs` (10), `--frameskip` (5), `--gpu` (L40S),
`--wandb-entity`, `--smoke`, `--extra` (raw Hydra overrides, or decoder.py args
with `--decoder`), `--check-only`, `--background`, `--decoder`, `--decoder-weights`.

- **Use `--background` for long jobs.** `modal run --detach` does not survive the
  laptop shutting down: the server cancels the in-flight call when the client
  disconnects. `--background` spawns on the deployed app instead (verified by
  closing the client mid-run).

- **W&B:** metrics only (`log_model: False`); weights are never uploaded. Use
  entity `mu-robotics`: the W&B account `autobrik` has no personal entity, and
  `--wandb-entity autobrik` fails with "entity not found". Project:
  `lewm-cartpole`. Without `--wandb-entity`, W&B is disabled.
- **Outputs on volume `lewm-cartpole`:** per-epoch weights at
  `stablewm/checkpoints/<run-name>/weights_epoch_N.pt` (plus `config.yaml`);
  Lightning checkpoints at `spt/runs/<date>/<time>/<id>/checkpoints/`. Volumes
  commit in the background, so saved files survive timeouts and preemption.
  Download with `modal volume get lewm-cartpole stablewm/checkpoints/<run-name>
  experiments/teaching-cartpole/checkpoints/` (gitignored).
- **Preemption:** Modal may preempt GPU functions and restarts them on the same
  input. `train.py` does not resume, so a preempted run restarts from epoch 0;
  weights already saved are kept.
- **Cost estimate:** 1,063,292 windows → 7,476 training batches/epoch at batch
  128. The October 5 Modal smoke ran at about 4.2 it/s (~30 min/epoch, so ~5 h for
  10 epochs). Treat this as an estimate; check it/s in the first minutes of a run.

### Completed runs (October 5–6, 2026)

| Run | Frameskip | Epochs saved | Notes |
| --- | --- | --- | --- |
| `cartpole-fs5` | 5 (0.25 s/latent step) | 1–10 | Completed; ~4.4 it/s, ~4.7 h. |
| `cartpole-fs1` | 1 (0.05 s/latent step) | 1–9 | Cancelled at epoch 10 (step 7,300/8,376) when the launching laptop shut down (`--detach`). |

**Decision: compare both runs at `weights_epoch_9.pt`; fs1 is not retrained.**
Both used `max_epochs=10`, so epoch 9 sits at the same point of the cosine LR
schedule in each. The fs5 epoch-10 weights exist but are not the comparison point.

Validation on October 5, 2026: a local emulation of the container environment
(online W&B) and a Modal `--check-only` preflight passed. A Modal smoke run (50
train / 10 validation batches) completed, logged to
`mu-robotics/lewm-cartpole/runs/smoke`, and saved `weights_epoch_1.pt` to the
volume. These are engineering checks, not model results.

## Evaluation and report artifacts

All run from the fork root with the fork's `.venv`, after downloading the epoch-9
checkpoints and decoders (`modal volume get lewm-cartpole stablewm/checkpoints/<run>`
and `stablewm/decoders/<run>-weights_epoch_9` into `checkpoints/` and
`checkpoints/decoders/`). On a laptop, wrap each command in
`systemd-run --user --scope -p MemoryMax=6G`: `action`/`observation` use one-row
HDF5 chunks, and whole-column reads cost ~8 GB, so these scripts read them in slices.

| Script | Produces |
| --- | --- |
| `eval_lewm.py --run checkpoints/<run> --weights weights_epoch_9.pt [--rollout 20]` | `outputs/eval-<run>-epoch9.json`: held-out latent one-step and open-loop errors with "nothing changes" and shuffled-action baselines, linear probes with an untrained control |
| `physical_rollout.py --run checkpoints/<run> --rollout <4 or 20>` | `outputs/physical-<run>-epoch9.json`: open-loop error in cm and degrees through the probes, per policy (not used in the report) |
| `modal_train.py --background --decoder --run-name <run>` | visualization decoder on the volume (`decoder.py`: frozen world model, 5 epochs) |
| `decode_video.py`, `compare_video.py` | paper-style decoded videos in `outputs/videos/` |
| `report_q2e.py` | `q2e_imagination.pdf` and `q2e_results.json` in the assignment report (every Q2(e) number) |
| `plot_eval.py` | quick-look comparison figures in `outputs/` |
| `hf_upload.py` | public Hugging Face dataset and model repos with generated cards |

Rollout 4 (frameskip 5) and 20 (frameskip 1) both cover 1 s. Published copies:
[dataset](https://huggingface.co/datasets/Autobrik/teaching-cartpole-10k-v1),
[frameskip 5](https://huggingface.co/Autobrik/lewm-teaching-cartpole-fs5),
[frameskip 1](https://huggingface.co/Autobrik/lewm-teaching-cartpole-fs1).

## Verification and assignment report

```bash
uv run --directory experiments/teaching-cartpole ruff check .
uv run --directory experiments/teaching-cartpole ruff format --check .
uv run --directory experiments/teaching-cartpole pytest -q
```

Preparation verification on October 5, 2026: 15 tests pass, Ruff lint/format
checks pass, and 600 nonlinear transitions agree with the assignment simulator
to a maximum state difference of 7.77e-16. The six-episode mixed-policy smoke
run has 313 transitions and 301 four-frame windows across its three splits;
stored transitions were replayed and preview frames visually inspected. The
native stable-worldmodel reader returned RGB tensors of shape `(4, 3, 224, 224)`
and scalar actions. These are collection checks, not model-performance results.

Code and reproducible run artifacts live in this fork. Final scientific results,
figures, and the Q2(e) manuscript belong in the assignment's
`/home/autobrik/Projects/personal/brik-meza-knowledge-base/education/mcgill-msc/fall-2026/comp-765/assignments/assignment-1/report/`.
Copy selected final figures into `report/figures/` and reference the experiment
configuration and fork commit in `report/main.tex`. The collection smoke test is
engineering validation, not evidence that LeWM was trained or controls CartPole.

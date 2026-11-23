# RoboCasa Human-50 (v0.x, 24 atomic tasks)

StarVLA training and evaluation recipe for the original
[RoboCasa](https://robocasa.ai/) atomic-task benchmark: **24 kitchen tasks, 50 human
demonstrations each (Human-50)**, PandaOmron mobile manipulator, evaluated with the
official RoboCasa protocol (50 rollouts per task, unseen objects and held-out scenes).

> [!IMPORTANT]
> This is **not** RoboCasa365 ([`../Robocasa_365`](../Robocasa_365/README.md), RoboCasa v1.0)
> and not the GR1 tabletop fork ([`../Robocasa_tabletop`](../Robocasa_tabletop/README.md)).
> It uses RoboCasa **v0.2** (robosuite **v1.5.1**) and the `v0.1` human datasets.
> State/action key names follow the same PandaOmron convention as RoboCasa365, but the
> environments, scenes, datasets, and scores are not interchangeable.

## What is included

| Component | Files |
| --- | --- |
| Task table (single source of truth) | `human50_tasks.py`: 24 tasks, dataset paths, horizons, eval scene protocol |
| Download | `train_files/download_human50.sh`: upstream `download_datasets.py --ds_types human_im` |
| Conversion | `train_files/convert_human50_to_lerobot.py`: HDF5 `50_demos` -> LeRobot v2.1 |
| Data registry | `train_files/data_registry/data_config.py`: robot type `panda_omron_robocasa_human50`, mixtures |
| Training | `train_files/run_robocasa_human50.sh` + QwenOFT / SmolVLA YAMLs |
| Evaluation | `eval_files/simulation_env.py` (robosuite client), `model2robocasa_human50_interface.py`, `run_eval.sh`, `summarize_results.py` |

### Tasks

| Category | Tasks |
| --- | --- |
| Pick-and-place (8) | PnPCounterToCab, PnPCabToCounter, PnPCounterToSink, PnPSinkToCounter, PnPCounterToMicrowave, PnPMicrowaveToCounter, PnPCounterToStove, PnPStoveToCounter |
| Doors (4) | OpenSingleDoor, CloseSingleDoor, OpenDoubleDoor, CloseDoubleDoor |
| Drawers (2) | OpenDrawer, CloseDrawer |
| Sink (3) | TurnOnSinkFaucet, TurnOffSinkFaucet, TurnSinkSpout |
| Stove (2) | TurnOnStove, TurnOffStove |
| Coffee (3) | CoffeeSetupMug, CoffeeServeMug, CoffeePressButton |
| Microwave (2) | TurnOnMicrowave, TurnOffMicrowave |

`NavigateKitchen` (the 25th v0.x atomic task) is excluded, as in the RoboCasa benchmark.

## 1. Environment setup

Training/serving use the regular `starVLA` env
([guideline](../../../docs/starVLA_guideline.md)). DeepSpeed needs a CUDA toolkit; if
`nvcc` is missing, install one into the env and export `CUDA_HOME`:

```bash
conda install -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4
export CUDA_HOME=$CONDA_PREFIX
```

Simulation uses a separate `robocasa` env with RoboCasa **v0.2** and robosuite **v1.5.1**:

```bash
conda create -n robocasa -c conda-forge python=3.10 -y
conda activate robocasa
# gcc + kernel headers are only needed to build `evdev` (a teleoperation dependency)
conda install -c conda-forge gcc=12 gxx=12 kernel-headers_linux-64 libegl libgl libopengl -y

mkdir -p playground/Code && cd playground/Code
git clone https://github.com/ARISE-Initiative/robosuite.git && git -C robosuite checkout v1.5.1
git clone https://github.com/robocasa/robocasa.git robocasa_v0 && git -C robocasa_v0 checkout v0.2
pip install -e robosuite -e robocasa_v0
pip install numpy==1.23.3 opencv-python-headless==4.10.0.84 mink==0.0.5 \
    h5py "pandas<2.3" "pyarrow<18" av imageio imageio-ffmpeg tyro websockets msgpack
python robocasa_v0/robocasa/scripts/download_kitchen_assets.py   # ~5 GB
python robocasa_v0/robocasa/scripts/setup_macros.py
cd ../..
```

For headless rendering use `MUJOCO_GL=egl`; `MUJOCO_EGL_DEVICE_ID` selects the GPU.

## 2. Data

```bash
# robocasa env, repo root: ~13 GB of HDF5 (128x128 images, 3 cameras)
bash examples/simBenchmarks/Robocasa_human50/train_files/download_human50.sh

# HDF5 (Human-50 = `50_demos` filter key) -> LeRobot v2.1, ~2 GB, ~2 min with 24 workers
python -m examples.simBenchmarks.Robocasa_human50.train_files.convert_human50_to_lerobot \
    --raw-root playground/Datasets/robocasa_human50_raw \
    --out-root playground/Datasets/robocasa_human50 --num-workers 24
```

Resulting layout (the loader writes caches into `meta/`, so keep it writable):

```text
playground/Datasets/robocasa_human50/v0.1/single_stage/<category>/<Task>/<date>/lerobot/
├── data/chunk-000/episode_*.parquet       # observation.state (16), action (12), task index
├── videos/chunk-000/observation.images.{robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand}/*.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, robocasa_env_args.json}
```

24 tasks × 50 demos = 1,200 episodes, ~332k frames.

## 3. Observation and action contract

| Modality | Shape / order |
| --- | --- |
| Images | left agentview, right agentview, eye-in-hand; 128×128 source, resized to 224×224 |
| State | 16-D: base pos (3), base quat (4), base-relative eef pos (3), eef quat (4), gripper qpos (2); `[sin, cos]` -> 32-D when `include_state: true` |
| Action | 12-D in native `env.step()` order: eef pos (3), eef rot (3), gripper (1), base motion (4), control mode (1); `min_max` per key |
| Language | per-episode RoboCasa instruction (`ep_meta["lang"]`) |

The action layout was verified by replaying recorded demos open-loop through the
evaluation environment (all replays succeed). robosuite renders with the OpenGL
convention, so the eval client flips camera frames upright to match the HDF5 images.

## 4. Training

```bash
conda activate starVLA
# SmolVLA (all 3 cameras + state)
CONFIG_YAML=examples/simBenchmarks/Robocasa_human50/train_files/starvla_smolvla_robocasa_human50.yaml \
MODEL_TAG=smolvla NUM_GPUS=8 BATCH=8 MAX_STEPS=60000 \
  bash examples/simBenchmarks/Robocasa_human50/train_files/run_robocasa_human50.sh

# QwenOFT (Qwen3-VL-4B-Instruct-Action at playground/Pretrained_models/, no state)
NUM_GPUS=8 BATCH=8 MAX_STEPS=60000 \
  bash examples/simBenchmarks/Robocasa_human50/train_files/run_robocasa_human50.sh
```

### SmolVLA full fine-tuning (LeRobot `smolvla_robocasa` recipe)

`starvla_smolvla_full_robocasa_human50.yaml` trains every SmolVLA weight with 50-step
action chunks (mixture `robocasa_human50_all_chunk50`), batch 32, AdamW 1e-4, 1k warmup and
cosine decay to 2.5e-6, as in the official LeRobot RoboCasa checkpoint. On one H200 it runs
at ~2.6 it/s (~5.3 h for 50k steps). Evaluate with `N_ACT=50`.

```bash
CONFIG_YAML=examples/simBenchmarks/Robocasa_human50/train_files/starvla_smolvla_full_robocasa_human50.yaml \
MIXTURE=robocasa_human50_all_chunk50 MODEL_TAG=smolvla_full NUM_GPUS=1 BATCH=32 MAX_STEPS=50000 \
  bash examples/simBenchmarks/Robocasa_human50/train_files/run_robocasa_human50.sh
```

All launcher knobs are environment variables (`MIXTURE`, `DATA_ROOT`, `BATCH`, `MAX_STEPS`,
`SAVE_EVERY`, `RUN_ROOT_DIR`, `RUN_ID`, `STARVLA_ENV`, ...). Mixtures:
`robocasa_human50_all` (default), `robocasa_human50_pnp`, `robocasa_human50_non_pnp`, and
`robocasa_human50_<Task>` for single-task runs. Runs go to
`playground/Checkpoints/robocasa_human50_<model>_<mixture>/`. During training only the latest
`checkpoints/steps_<N>_pytorch_model.pt` is kept (for resuming); when training finishes it is
removed and `final_model/pytorch_model.pt` (model weights only) is the checkpoint to evaluate.
Pass `--trainer.keep_all_checkpoints true` to keep every `SAVE_EVERY` checkpoint.

## 5. Evaluation

Official protocol (robomimic `robocasa` branch): object split `B` (unseen instances),
layout/style pairs `(1,1) (2,2) (4,4) (6,9) (7,10)`, no generative textures, fixed
cameras, per-task horizon (300–1000 steps), 50 rollouts per task, success =
`env._check_success()` at any step. The env seed (`SEED`, default 0) is forwarded, so
the scene sequence of a task is reproducible.

```bash
# all 24 tasks: policy servers on half of the GPUs, 24 parallel simulator clients
# rendering on the other half, then a summary
CKPT=playground/Checkpoints/<run>/final_model/pytorch_model.pt \
NUM_GPUS=8 bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh all
# conda env names default to starVLA / robocasa (STARVLA_ENV, ROBOCASA_ENV, or *_PYTHON paths)

# single task (two terminals)
CUDA_VISIBLE_DEVICES=0 CKPT=... \
  bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh server          # starVLA env
MUJOCO_EGL_DEVICE_ID=1 CKPT=... TASK=OpenDrawer N_EPISODES=5 \
  bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh client          # robocasa env
```

> [!WARNING]
> Keep the simulator's EGL renderer off the GPUs that run policy servers. With a CUDA
> policy server and the EGL renderer on the same GPU, the NVIDIA EGL driver aborted the
> client inside `mjr_readPixels` mid-episode in our tests (driver 580, H200). `run_eval.sh all`
> therefore splits `GPU_IDS` into server GPUs (first half) and render GPUs (second half);
> override with `SERVER_GPU_IDS` / `RENDER_GPU_IDS`. For the single-task `client` mode, set
> `MUJOCO_EGL_DEVICE_ID` to a GPU without a server. On a single GPU, `MUJOCO_GL=osmesa`
> (needs `conda install -c conda-forge "mesalib<25"`) avoids the issue but is ~15x slower.

A policy server answers one request at a time; `SERVERS_PER_GPU=<n>` starts replicas so
24 parallel clients do not queue on the policy. Each result records a per-episode
reset / policy / simulator time split, summarized in `summary.txt`.

Results: `<ckpt>.eval/robocasa_human50/<Task>.json` (per-episode success, steps,
instruction, scene), `summary.json` / `summary.txt` (per-category and 24-task average),
and a few rollout videos in `videos/`. `run_eval.sh all` skips tasks that already have
a result unless `OVERWRITE=1`.

Train/test consistency handled by the client:

- the three camera views are sent in training order, flipped upright and resized to
  224×224 with PIL bicubic (same as the dataloader);
- `state` is sent only if the checkpoint's `config.yaml` has `include_state: true`
  (override with `SEND_STATE=0/1`); QwenOFT changes its prompt whenever state is present;
- actions are un-normalized by the server and executed `N_ACT` (default 8) per query.

## 6. Reproducibility checklist

Record with every result: checkpoint + training step, training mixture and config,
RoboCasa (`v0.2`) / robosuite (`v1.5.1`) / MuJoCo (`3.2.6`) versions, camera count and
image size, state usage, action horizon and `N_ACT`, eval seed, and the 24 per-task
success rates over 50 rollouts each.

## References

- [RoboCasa paper (RSS 2024)](https://robocasa.ai/assets/robocasa_rss24.pdf) and [code (v0.2)](https://github.com/robocasa/robocasa/tree/v0.2)
- [RoboCasa policy learning / evaluation (robomimic `robocasa` branch)](https://github.com/ARISE-Initiative/robomimic/tree/robocasa)
- [RoboCasa dataset guide](https://robocasa.ai/docs/use_cases/downloading_datasets.html)

#!/usr/bin/env bash
# RoboCasa (v0.x) Human-50 — multi-task training on the 24 atomic tasks (50 human demos each).
#
# Run from the repo root AFTER converting the data:
#   bash examples/simBenchmarks/Robocasa_human50/train_files/download_human50.sh
#   python -m examples.simBenchmarks.Robocasa_human50.train_files.convert_human50_to_lerobot ...
#
# Every knob below can be overridden from the environment, e.g.
#   CONFIG_YAML=examples/simBenchmarks/Robocasa_human50/train_files/starvla_smolvla_robocasa_human50.yaml \
#   MODEL_TAG=smolvla NUM_GPUS=8 BATCH=8 MAX_STEPS=60000 \
#     bash examples/simBenchmarks/Robocasa_human50/train_files/run_robocasa_human50.sh
# Extra arguments are forwarded to train_starvla.py as config overrides, e.g.
#   ... run_robocasa_human50.sh --trainer.learning_rate.base 5e-5
set -euo pipefail

# ==============================================================
#  Platform-specific settings — adjust these for YOUR cluster
# ==============================================================
# WandB — run `wandb login` (or export WANDB_API_KEY) before launching,
# or export WANDB_MODE=offline / WANDB_MODE=disabled to skip syncing.

# NCCL networking — uncomment and edit to match your InfiniBand / RoCE setup:
#   export NCCL_SOCKET_IFNAME=<your_network_interface>   # e.g. eth0, bond0
#   export NCCL_IB_HCA=<your_ib_devices>                # e.g. mlx5_0,mlx5_1
# ==============================================================

# Activate the training env if not already active (name or prefix path).
STARVLA_ENV=${STARVLA_ENV:-starVLA}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${STARVLA_ENV}" && "${CONDA_PREFIX:-}" != "${STARVLA_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${STARVLA_ENV}"
fi

NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# ---- training knobs ----
CONFIG_YAML=${CONFIG_YAML:-./examples/simBenchmarks/Robocasa_human50/train_files/starvla_qwenoft_robocasa_human50.yaml}
MODEL_TAG=${MODEL_TAG:-qwenoft}
# robocasa_human50_all (24 tasks) | robocasa_human50_pnp | robocasa_human50_non_pnp | robocasa_human50_<Task>
MIXTURE=${MIXTURE:-robocasa_human50_all}
DATA_ROOT=${DATA_ROOT:-playground/Datasets/robocasa_human50}
BATCH=${BATCH:-8}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_EVERY=${SAVE_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-100}
NUM_WORKERS_PER_GPU=${NUM_WORKERS_PER_GPU:-}
# Use distinct MAIN_PORTs when running several jobs on one node.

run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}
run_id=${RUN_ID:-robocasa_human50_${MODEL_TAG}_${MIXTURE}}
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

extra_args=()
if [[ -n "${NUM_WORKERS_PER_GPU}" ]]; then
  extra_args+=(--datasets.vla_data.num_workers "${NUM_WORKERS_PER_GPU}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PORT:-29500}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${MIXTURE}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robocasa_human50 \
  "${extra_args[@]}" "$@"

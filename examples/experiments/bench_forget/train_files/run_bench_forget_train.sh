#!/usr/bin/env bash
# bench_forget stage-2 training on one GPU.  Usage (from the repo root):
#   CUDA_VISIBLE_DEVICES=0 MAIN_PORT=29500 bash examples/experiments/bench_forget/train_files/run_bench_forget_train.sh B
#   CUDA_VISIBLE_DEVICES=1 MAIN_PORT=29501 bash examples/experiments/bench_forget/train_files/run_bench_forget_train.sh C
set -euo pipefail
VARIANT=${1:?B or C}
# C: only attention projections (q/k/v/o) of the SigLIP encoder, the VLM text layers and the
# action expert are trained; MLPs, norms, embeddings, connector and state/action/time
# projections stay at the run-A values.
ATTN_ONLY='(vision_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj|text_model\.layers\.\d+\.self_attn\.[qkvo]_proj|lm_expert\.layers\.\d+\.self_attn\.[qkvo]_proj)\.'
extra=()
case "${VARIANT}" in
  B) RUN_ID=${RUN_ID:-bench_forget_B_full} ;;
  C) RUN_ID=${RUN_ID:-bench_forget_C_attn_only}; extra+=(--trainer.trainable_param_regex "${ATTN_ONLY}") ;;
  *) echo "variant must be B or C"; exit 1 ;;
esac
CONFIG_YAML=examples/experiments/bench_forget/train_files/starvla_smolvla_libero_object_from_human50.yaml \
MIXTURE=libero_object_robocasa_layout_chunk50 DATA_ROOT=playground/Datasets/libero_robocasa_layout \
MODEL_TAG=smolvla NUM_GPUS=1 BATCH=${BATCH:-32} MAX_STEPS=${MAX_STEPS:-20000} SAVE_EVERY=${SAVE_EVERY:-20000} \
EVAL_EVERY=${EVAL_EVERY:-100000} LOG_EVERY=${LOG_EVERY:-50} RUN_ID=${RUN_ID} \
  bash examples/simBenchmarks/Robocasa_human50/train_files/run_robocasa_human50.sh "${extra[@]}" "${@:2}"

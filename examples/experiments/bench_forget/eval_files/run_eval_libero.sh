#!/usr/bin/env bash
# LIBERO evaluation of a RoboCasa-layout checkpoint: policy servers on SERVER_GPU, one LIBERO
# client per task rendering on RENDER_GPU (keep them apart: EGL + CUDA on one GPU aborted in
# our RoboCasa runs), then a summary.  Run from the repo root:
#   CKPT=playground/Checkpoints/bench_forget_B_full/final_model/pytorch_model.pt \
#   SERVER_GPU=0 RENDER_GPU=1 BASE_PORT=6300 bash examples/experiments/bench_forget/eval_files/run_eval_libero.sh
set -euo pipefail
CKPT=${CKPT:?checkpoint .pt}
SUITE=${SUITE:-libero_object}
N_EPISODES=${N_EPISODES:-50}
N_ACT=${N_ACT:-10}
SEED=${SEED:-7}
SERVER_GPU=${SERVER_GPU:-0}
RENDER_GPU=${RENDER_GPU:-1}
SERVERS=${SERVERS:-2}
BASE_PORT=${BASE_PORT:-6300}
NUM_TASKS=${NUM_TASKS:-10}
STARVLA_PYTHON=${STARVLA_PYTHON:?python of the starVLA env}
LIBERO_PYTHON=${LIBERO_PYTHON:?python of the LIBERO env}
LIBERO_HOME=${LIBERO_HOME:?LIBERO repo}
OUT_DIR=${OUT_DIR:-${CKPT%.*}.eval/${SUITE}}
mkdir -p "${OUT_DIR}/logs"
export PYTHONPATH=$(pwd):${LIBERO_HOME}:${PYTHONPATH:-}

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
for i in $(seq 0 $((SERVERS - 1))); do
  ( export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; CUDA_VISIBLE_DEVICES=${SERVER_GPU} exec "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
      --ckpt_path "${CKPT}" --port $((BASE_PORT + i)) --use_bf16 > "${OUT_DIR}/logs/server_$((BASE_PORT + i)).log" 2>&1 ) &
  PIDS+=($!)
done
CLIENTS=()
for t in $(seq 0 $((NUM_TASKS - 1))); do
  if [[ -f "${OUT_DIR}/task_$(printf %02d $t).json" && "${OVERWRITE:-0}" != "1" ]]; then echo "[libero] task $t exists, skipping"; continue; fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${RENDER_GPU} MUJOCO_EGL_DEVICE_ID=${RENDER_GPU} \
    "${LIBERO_PYTHON}" -m examples.experiments.bench_forget.eval_files.libero_eval \
      --args.port $((BASE_PORT + t % SERVERS)) --args.pretrained-path "${CKPT}" --args.task-suite-name "${SUITE}" \
      --args.task-ids "$t" --args.n-episodes "${N_EPISODES}" --args.n-action-steps "${N_ACT}" --args.seed "${SEED}" \
      --args.out-dir "${OUT_DIR}" ${CLIENT_ARGS:-} > "${OUT_DIR}/logs/task_$(printf %02d $t).log" 2>&1 &
  CLIENTS+=($!)
done
for p in "${CLIENTS[@]}"; do wait "$p" || echo "[libero] a client failed (see ${OUT_DIR}/logs)"; done
"${STARVLA_PYTHON}" - "${OUT_DIR}" "${NUM_TASKS}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import json, sys, pathlib
d, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads((d / f"task_{t:02d}.json").read_text()) for t in range(n) if (d / f"task_{t:02d}.json").is_file()]
for r in rows: print(f"{r['task_id']:2d} {100 * r['success_rate']:5.1f}  {r['num_success']:2d}/{r['num_episodes']}  {r['task']}")
avg = sum(r["success_rate"] for r in rows) / max(len(rows), 1)
print(f"AVERAGE {100 * avg:.1f}  ({len(rows)}/{n} tasks)")
(d / "summary.json").write_text(json.dumps({"average_success_rate": avg, "num_tasks": len(rows), "task_success_rate": {r["task"]: r["success_rate"] for r in rows}}, indent=2))
PY

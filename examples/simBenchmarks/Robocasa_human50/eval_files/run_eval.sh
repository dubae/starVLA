#!/usr/bin/env bash
# RoboCasa (v0.x) Human-50 evaluation (24 atomic tasks x 50 rollouts, official scene protocol).
#
#   # single task, two terminals
#   bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh server   # starVLA env
#   bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh client   # robocasa env
#
#   # all 24 tasks: policy servers + parallel simulator clients on disjoint GPUs, then a summary
#   CKPT=... NUM_GPUS=8 bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh all
#
# Run from the StarVLA repo root. Env vars: CKPT, TASK, PORT, BASE_PORT, NUM_GPUS, GPU_IDS,
# SERVER_GPU_IDS, RENDER_GPU_IDS, SERVERS_PER_GPU,
# N_EPISODES, N_ACT, SEED, MAX_PARALLEL, OVERWRITE, SEND_STATE, STARVLA_ENV / ROBOCASA_ENV (conda env
# names) or STARVLA_PYTHON / ROBOCASA_PYTHON (interpreter paths),
# CLIENT_ARGS (extra flags for simulation_env.py, e.g. "--args.n-video-episodes 0"),
# EPISODE_SHARDS (`all`: split a 500-step task's episodes over this many clients, scaled by
# task horizon; physics is single-threaded, so this is what uses a many-core machine),
# CLIENT_THREADS / SERVER_THREADS (OMP/BLAS threads per process; default 1 / 8).
# Keep the simulator EGL renderer off the policy-server GPU(s) (`all` splits GPU_IDS into
# server / render halves; for `client` set MUJOCO_EGL_DEVICE_ID to a GPU without a server):
# sharing one GPU made the NVIDIA EGL driver abort mid-episode in our tests.
set -euo pipefail

CKPT=${CKPT:-}
TASK=${TASK:-OpenDrawer}
PORT=${PORT:-5678}
BASE_PORT=${BASE_PORT:-6100}
N_EPISODES=${N_EPISODES:-50}
N_ACT=${N_ACT:-8}
SEED=${SEED:-0}
CLIENT_ARGS=${CLIENT_ARGS:-}
env_python() {  # conda env name -> its python (works for envs outside $(conda info --base)/envs)
  local prefix
  prefix=$(conda env list 2>/dev/null | awk -v n="$1" '$1 == n {print $NF}')
  echo "${prefix:-$(conda info --base 2>/dev/null)/envs/$1}/bin/python"
}
STARVLA_PYTHON=${STARVLA_PYTHON:-$(env_python "${STARVLA_ENV:-starVLA}")}
ROBOCASA_PYTHON=${ROBOCASA_PYTHON:-$(env_python "${ROBOCASA_ENV:-robocasa}")}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# SEND_STATE=1/0 forces the state input; unset -> read include_state from the checkpoint config.
state_flag=""
case "${SEND_STATE:-}" in
  1|true|True) state_flag="--args.send-state True" ;;
  0|false|False) state_flag="--args.send-state False" ;;
esac

require_ckpt() {
  if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "CKPT must point to a checkpoint file (got '${CKPT}')" >&2
    exit 1
  fi
}

CLIENT_THREADS=${CLIENT_THREADS:-1}
SERVER_THREADS=${SERVER_THREADS:-8}
threads_env() { echo "OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1"; }

run_server() {  # gpu port log  (exec: the backgrounded PID is the server itself, so cleanup can kill it)
  # many replicas start at once: uncapped torch CPU threads made them thrash for minutes
  export $(threads_env "${SERVER_THREADS}")
  CUDA_VISIBLE_DEVICES=$1 exec "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port "$2" --use_bf16 > "$3" 2>&1
}

run_client() {  # task port egl_device out_dir [shard_id num_shards]
  env $(threads_env "${CLIENT_THREADS}") CUDA_VISIBLE_DEVICES=$3 MUJOCO_EGL_DEVICE_ID=$3 "${ROBOCASA_PYTHON}" -m examples.simBenchmarks.Robocasa_human50.eval_files.simulation_env \
    --args.task "$1" \
    --args.port "$2" \
    --args.pretrained-path "${CKPT}" \
    --args.n-episodes "${N_EPISODES}" \
    --args.n-action-steps "${N_ACT}" \
    --args.seed "${SEED}" \
    --args.out-dir "$4" \
    --args.shard-id "${5:-0}" \
    --args.num-shards "${6:-1}" \
    ${state_flag} ${CLIENT_ARGS}
}

case "${1:-}" in
  server)
    require_ckpt
    exec "${STARVLA_PYTHON}" deployment/model_server/server_policy.py --ckpt_path "${CKPT}" --port "${PORT}" --use_bf16
    ;;
  client)
    require_ckpt
    run_client "${TASK}" "${PORT}" "${MUJOCO_EGL_DEVICE_ID:-0}" "${CKPT%.*}.eval/robocasa_human50"
    ;;
  all)
    require_ckpt
    IFS=',' read -r -a ALL_GPUS <<< "${GPU_IDS:-$(seq -s, 0 $(( ${NUM_GPUS:-$(nvidia-smi -L | wc -l)} - 1 )))}"
    # Policy servers and simulator renderers use disjoint GPUs: with a CUDA policy server and
    # the EGL renderer on the same GPU, the NVIDIA EGL driver aborted in mjr_readPixels
    # mid-episode in our tests. Default split: first half serves, second half renders.
    n_srv=$(( (${#ALL_GPUS[@]} + 1) / 2 ))
    IFS=',' read -r -a GPUS <<< "${SERVER_GPU_IDS:-$(IFS=,; echo "${ALL_GPUS[*]:0:${n_srv}}")}"
    IFS=',' read -r -a RENDER_GPUS <<< "${RENDER_GPU_IDS:-$(IFS=,; echo "${ALL_GPUS[*]:${n_srv}}")}"
    if (( ${#RENDER_GPUS[@]} == 0 )); then
      RENDER_GPUS=("${GPUS[@]}")
      echo "[human50] warning: only one GPU -> server and renderer share it; MUJOCO_GL=osmesa avoids EGL driver aborts (slower)" >&2
    fi
    OUT_DIR=${OUT_DIR:-${CKPT%.*}.eval/robocasa_human50}
    LOG_DIR="${OUT_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    # "task:horizon" pairs
    TASKS=($("${ROBOCASA_PYTHON}" -c "from examples.simBenchmarks.Robocasa_human50.human50_tasks import HUMAN50_TASKS as T; print(' '.join(f'{k}:{v[1]}' for k, v in T.items()))"))
    EPISODE_SHARDS=${EPISODE_SHARDS:-1}
    CLIENT_JOBS=()  # "task shard_id num_shards"
    for th in "${TASKS[@]}"; do
      task=${th%%:*}; horizon=${th##*:}
      if [[ -f "${OUT_DIR}/${task}.json" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "[human50] ${task}: result exists, skipping"; continue
      fi
      ns=1
      if (( EPISODE_SHARDS > 1 )); then  # longer horizons get proportionally more shards
        ns=$(( (EPISODE_SHARDS * horizon + 250) / 500 )); (( ns < 1 )) && ns=1; (( ns > N_EPISODES )) && ns=${N_EPISODES}
      fi
      for (( i = 0; i < ns; i++ )); do
        if (( ns > 1 )) && [[ -f "${OUT_DIR}/${task}.shard${i}of${ns}.json" && "${OVERWRITE:-0}" != "1" ]]; then continue; fi
        CLIENT_JOBS+=("${task} ${i} ${ns}")
      done
    done
    MAX_PARALLEL=${MAX_PARALLEL:-${#CLIENT_JOBS[@]}}

    # A server answers one request at a time; with many clients per GPU, extra replicas
    # (SERVERS_PER_GPU) keep clients from queueing on the policy.
    SERVERS_PER_GPU=${SERVERS_PER_GPU:-1}
    SERVER_PIDS=(); SERVER_PORTS=(); SERVER_GPUS=()
    cleanup() { for pid in "${SERVER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
    trap cleanup EXIT
    for gpu in "${GPUS[@]}"; do
      for r in $(seq 1 "${SERVERS_PER_GPU}"); do
        port=$((BASE_PORT + ${#SERVER_PORTS[@]}))
        run_server "${gpu}" "${port}" "${LOG_DIR}/server_gpu${gpu}_${port}.log" &
        SERVER_PIDS+=($!); SERVER_PORTS+=("${port}"); SERVER_GPUS+=("${gpu}")
      done
    done
    echo "[human50] ${#SERVER_PORTS[@]} policy server(s) on GPUs ${GPUS[*]} (ports ${SERVER_PORTS[*]}), renderers on GPUs ${RENDER_GPUS[*]}; logs -> ${LOG_DIR}"

    echo "[human50] ${#CLIENT_JOBS[@]} client job(s), at most ${MAX_PARALLEL} at once"
    k=0
    for job in "${CLIENT_JOBS[@]}"; do
      read -r task shard_id num_shards <<< "${job}"
      tag=${task}; (( num_shards > 1 )) && tag=${task}.shard${shard_id}of${num_shards}
      while (( $(jobs -rp | wc -l) >= MAX_PARALLEL + ${#SERVER_PORTS[@]} )); do sleep 5; done
      slot=$((k % ${#SERVER_PORTS[@]}))
      render_gpu=${RENDER_GPUS[$(( k % ${#RENDER_GPUS[@]} ))]}
      echo "[human50] ${tag} -> server GPU ${SERVER_GPUS[$slot]} (port ${SERVER_PORTS[$slot]}), render GPU ${render_gpu}"
      run_client "${task}" "${SERVER_PORTS[$slot]}" "${render_gpu}" "${OUT_DIR}" "${shard_id}" "${num_shards}" > "${LOG_DIR}/${tag}.log" 2>&1 &
      k=$((k + 1))
    done
    # wait for the clients only (servers keep running until cleanup)
    for pid in $(jobs -rp); do
      [[ " ${SERVER_PIDS[*]} " == *" ${pid} "* ]] || wait "${pid}" || echo "[human50] a client exited with an error (see ${LOG_DIR})"
    done
    "${ROBOCASA_PYTHON}" -m examples.simBenchmarks.Robocasa_human50.eval_files.summarize_results "${OUT_DIR}" | tee "${OUT_DIR}/summary.txt"
    ;;
  *)
    sed -n '2,18p' "$0"
    ;;
esac

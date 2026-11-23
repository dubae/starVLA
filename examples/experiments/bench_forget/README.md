# bench_forget — does continuing on another benchmark erase RoboCasa skills?

Branch `exp/bench-forget`. Question: after SmolVLA is fine-tuned on RoboCasa Human-50, does
further training on LIBERO-object lower the RoboCasa success rate (catastrophic forgetting)?

| run | start | stage-2 data | trained params | steps |
|---|---|---|---|---|
| A | `lerobot/smolvla_base` | — (Human-50 only, existing run `human50_smolvla_full_50k`) | all | 50k |
| B | A `final_model` | LIBERO-object | all (as A) | 20k |
| C | A `final_model` | LIBERO-object | attention q/k/v/o only (SigLIP, VLM text, action expert); MLPs, norms, embeddings, connector, state/action/time projections frozen | 20k |

Same recipe as A otherwise (batch 32, AdamW 1e-4 (0.9, 0.95), 1k warmup, cosine to 2.5e-6,
1 GPU each, seed 42). Only the final model is kept. Evaluation: RoboCasa Human-50 (24 tasks
x 50, official split, `N_ACT=50`, seed 0 — A: 30.4%) and LIBERO-object (10 tasks x 50
fixed initial states, `N_ACT=10`) for B and C.

## One input/output format for both benchmarks

LIBERO is rewritten in the RoboCasa layout (`robocasa_layout.py`); both drive a Franka arm with
the same robosuite OSC_POSE controller (delta pose, input [-1, 1], output 0.05 m / 0.5 rad,
kp 150, 20 Hz; LIBERO robot base at (-0.6, 0, 0) with identity orientation, so world-frame
deltas are base-frame deltas):

- action, 12-D, Human-50 order: eef delta pos(3) + rot(3) + gripper (+1 close, -1 open;
  LIBERO's 1=open/0=close is mapped) + base_motion(4) = 0 + control_mode = -1 (arm, as in all
  Human-50 frames). Chunk 50 for both (same data config class).
- state, 16-D -> per-key sin/cos 32-D: base pos/quat, base->eef pos, base->eef quat
  (xyzw, w >= 0 like RoboCasa), gripper qpos. The training path (IPEC axis-angle) and the eval
  path (env quaternion) agree to 1e-15 (`check_state_path.py` in the analysis dir).
- views: [left agentview, right agentview, eye-in-hand]; LIBERO agentview -> left, wrist ->
  eye-in-hand, right = all-black and masked out of attention
  (`framework.smolvla.mask_blank_images`, baseline feature).
- normalization: separate statistics keys, `new_embodiment` (Human-50) and `franka`
  (LIBERO, data config `libero_robocasa_layout_chunk50`). The action dims that are constant in
  LIBERO (base_motion, control_mode) take the RoboCasa statistics, so raw 0 / -1 normalize to
  the same value in both; otherwise min == max maps them to 0, which RoboCasa un-normalization
  would read as base motion. RoboCasa evaluation of B/C uses `unnorm_key=new_embodiment`
  (added to the run with `add_unnorm_stats.py`), LIBERO evaluation `unnorm_key=franka`.

## Reproduce

```bash
# data (LeRobot v2.1, IPEC LIBERO-object) -> RoboCasa layout, then the stats patch
python -m examples.experiments.bench_forget.convert_libero_to_robocasa_layout \
  --src <libero_object_no_noops_1.0.0_lerobot> --out playground/Datasets/libero_robocasa_layout/libero_object
# build the dataloader once (computes meta/stats_gr00t.json), then:
python -m examples.experiments.bench_forget.convert_libero_to_robocasa_layout --src x \
  --out playground/Datasets/libero_robocasa_layout/libero_object --patch-stats-only \
  --robocasa-stats playground/Checkpoints/human50_smolvla_full_50k/dataset_statistics.json

# training (one GPU each)
CUDA_VISIBLE_DEVICES=0 MAIN_PORT=29500 bash examples/experiments/bench_forget/train_files/run_bench_forget_train.sh B
CUDA_VISIBLE_DEVICES=1 MAIN_PORT=29501 bash examples/experiments/bench_forget/train_files/run_bench_forget_train.sh C

# RoboCasa eval with the RoboCasa statistics
python -m examples.experiments.bench_forget.add_unnorm_stats playground/Checkpoints/bench_forget_B_full \
  playground/Checkpoints/human50_smolvla_full_50k/dataset_statistics.json new_embodiment
CKPT=playground/Checkpoints/bench_forget_B_full/final_model/pytorch_model.pt N_ACT=50 \
CLIENT_ARGS="--args.unnorm-key new_embodiment" SERVER_GPU_IDS=0 RENDER_GPU_IDS=1 EPISODE_SHARDS=2 \
  bash examples/simBenchmarks/Robocasa_human50/eval_files/run_eval.sh all

# LIBERO eval (LIBERO env: robosuite 1.4.0, LIBERO on PYTHONPATH)
CKPT=... STARVLA_PYTHON=... LIBERO_PYTHON=... LIBERO_HOME=... \
  bash examples/experiments/bench_forget/eval_files/run_eval_libero.sh
```

# Experiment branches

One branch per experiment (`exp/<name>`), branched from `starVLA_dev`. This file on `starVLA_dev`
is the index: add a row when a branch is created and update it when results come in.
Common baseline unless stated: SmolVLA (`lerobot/smolvla_base`) full fine-tuning on RoboCasa
Human-50 (`starvla_smolvla_full_robocasa_human50.yaml`: chunk 50, 32 obs/step, AdamW 1e-4,
cosine to 2.5e-6, 50k steps). Eval: 24 tasks x 50 episodes, official split, `N_ACT=50`, seed 0.
Baseline **A = 30.4%** (flow matching, Beta time sampling, 10 Euler steps, EGL); OSMesa gives 30.3%
(measured 2026-11-25, per-task r = 0.992; W&B `starVLA_dev_human50_smolvla_full_50k`).
W&B: https://forge.coreweave.com/wandb/duhoe/starvla_poc, runs named `{branch}_{run_id}` (see `AGENTS.md`).

| branch | question | change vs. baseline | status (date) | RoboCasa Human-50 |
|---|---|---|---|---|
| `exp/smolvla-kv-stopgrad` | Does cutting upper-layer gradients at the VLM -> expert K/V hand-off help? | `framework.smolvla.vlm_kv_stopgrad: true` (VLM q_proj frozen; forward unchanged) | done (2026-11-23) | 27.0% (vs 30.4%) |
| `exp/bench-forget` | Does continuing on LIBERO-object erase RoboCasa skills? | stage 2 from A on LIBERO-object, 20k steps: B full / C attention q,k,v,o only / D MLP gate,up only | done (2026-11-25) | B 1.5%, C 3.8%, D 7.0% (LIBERO-object 92.4 / 93.2 / 90.4%); most of the drop happens in the first 1k steps |
| `exp/drift` | Can a one-step Drifting generator replace flow matching (1 NFE)? | `objective: drift`, G=16 samples/obs, R=[0.02, 0.05, 0.2] | G16 50k done (2026-11-24); B1 (+ forward-dynamics latent loss) failed twice, in progress | 26.0% (OSMesa); EGL partial 5 tasks 36.4% |
| `exp/timestep` | Train flow matching only at the timesteps used at inference? | `time_sampling: discrete`, t ~ U{0.1, ..., 1.0} (`train_num_steps: 10`) instead of Beta(1.5, 1) on (0, 1]; 2 GPUs x 16 | done (2026-11-25) | 30.7% (OSMesa; A 30.3% OSMesa) |
| `exp/cfg` | Does classifier-free guidance on the observation help the VLA policy? | train with p=0.1 null conditioning (vision + language replaced by null tokens together, state kept); sample v = v_null + w (v_cond - v_null) | training 50k on 1x H200 (2026-11-26); eval OSMesa at w = 1, 2 | pending (vs A 30.4% EGL / 30.3% OSMesa) |

## Details

### exp/smolvla-kv-stopgrad
W&B: `exp/smolvla-kv-stopgrad_human50_smolvla_kvsg_50k` (backfilled).
Stop-gradient at each VLM block's K/V hand-off to the action expert: the VLM's own attention uses
stop-grad q/k/v and the block-input residual is detached, while the K/V given to the expert stay
attached. Final train loss 0.0250 vs 0.0159 (full FT); 27.0% vs 30.4%. Test: `tests/test_smolvla_kv_stopgrad.py`.

### exp/bench-forget
Catastrophic-forgetting benchmark: LIBERO rewritten into the RoboCasa I/O layout
(`examples/experiments/bench_forget/`), continued training from A. All variants keep LIBERO-object
>= 90% but RoboCasa drops to 1.5-7.0%; a 2x2 weights x eval-setup check shows the drop comes from
the weights, not the stage-2 eval setup. See `examples/experiments/bench_forget/README.md` on the branch.

Forgetting curve (re-runs keeping checkpoints at 1k/5k/10k/15k; RoboCasa / LIBERO-object success %, 20k = runs above):

| stage-2 step | B full | C attention q,k,v,o | D MLP gate,up (+ SigLIP fc1) |
|---|---|---|---|
| 0 (A) | 30.4 / - | 30.4 / - | 30.4 / - |
| 1k | 5.0 / 13.8 | 10.5 / 24.2 | 11.1 / 38.4 |
| 5k | 0.7 / 59.8 | 4.2 / 69.6 | 7.2 / 70.8 |
| 10k | 0.8 / 85.6 | 2.7 / 89.6 | 6.5 / 88.2 |
| 15k | 1.0 / 93.0 | 3.6 / 88.8 | 7.6 / 90.4 |
| 20k | 1.5 / 92.4 | 3.8 / 93.2 | 7.0 / 90.4 |

RoboCasa falls during the 1k warmup, before LIBERO is learned, then stays flat; D keeps Drawers at 34-43%
(A 82%), doors and microwave go to ~0% for all three. W&B (backfilled from offline runs):
`exp/bench-forget_bench_forget_{B_full,C_attn_only,D_mlp_in_only}` and `exp/bench-forget_bench_forget_{B,C,D}_curve`.

### exp/drift
SmolVLA expert as a one-step generator a = eps - v(eps, t=1), trained with the Drifting objective
(arXiv 2602.04770): G=16 chunks per observation (VLM prefix computed once), attracted to the
ground-truth chunk and repelled from each other. Per category (OSMesa): PnP 3.0, Doors 37.5,
Drawers 78.0, Sink 48.7, Stove 18.0, Coffee 10.0, Microwave 41.0.
B1 adds a forward-dynamics latent loss: without normalization it was stopped at 6.7k steps, and with
LayerNorm (weight 100) the target latent collapsed at 1.2k steps. All three runs are on W&B (`exp/drift_*`).

### exp/timestep
Flow matching is trained only at the times the Euler sampler visits at inference
(t = 1, 0.9, ..., 0.1 for 10 steps; 4 steps -> 0.25 grid via `train_num_steps: 4`); t = 0 is never
queried, so it is never trained. Config `starvla_smolvla_tsdiscrete_robocasa_human50.yaml`.
W&B run `exp/timestep_human50_smolvla_tsdiscrete10_50k`. 50k steps on 2x H200 in 3h35 (3.87 it/s).
Eval with 10 Euler steps (OSMesa): **30.7%**, i.e. the same as A (OSMesa 30.3%, EGL 30.4%; +-1.3 pp
standard error over 1,200 episodes). Per category, tsdiscrete vs A: PnP 2.5 / 4.0, Doors 46.5 / 47.5, Drawers 82.0 / 82.0,
Sink 47.3 / 44.0, Stove 26.0 / 18.0, Coffee 4.7 / 10.7, Microwave 79.0 / 72.0.

### exp/cfg
Classifier-free guidance for the flow-matching action expert. Training: per sample, with probability
p = 0.1 the image tokens and the language tokens are replaced by null tokens (both at once; the
proprio-state token is kept). Inference: the conditional and null prefixes are run as one batch and each
Euler step uses v = v_null + w (v_cond - v_null) (w = 1: plain conditional sampling). Same recipe as A
(SmolVLA full FT on Human-50, 50k steps, from `lerobot/smolvla_base`); null tokens are learned vectors (one for
vision, one for language) and null text uses a fixed-length mask. Config `starvla_smolvla_cfg_robocasa_human50.yaml`,
test `tests/test_smolvla_cfg.py`. Eval: same protocol as A, OSMesa, guidance w = 1 (no guidance) and w = 2.
W&B run `exp/cfg_human50_smolvla_cfg_p01_50k`.

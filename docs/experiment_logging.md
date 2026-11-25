# Experiment logging (team convention)

All training runs on this fork log to the team W&B on CoreWeave Forge:

**https://forge.coreweave.com/wandb/duhoe/starvla_poc**

| item | value | override (env var) |
|---|---|---|
| server | `https://api.forge.coreweave.com` (the SDK rejects the `forge.coreweave.com` UI address) | `WANDB_BASE_URL` |
| team (entity) | `duhoe` | `WANDB_ENTITY` |
| project | `starvla_poc` | `WANDB_PROJECT` |
| run name | `{git branch}_{run_id}`, e.g. `exp/timestep_human50_smolvla_tsdiscrete10_50k` | — |
| API key | `WANDB_KEY=...` in the repo-root `.env` (git-ignored; a `git worktree` falls back to the main checkout's `.env`) | `WANDB_API_KEY` |

This is implemented in `starVLA/training/trainer_utils/team_wandb.py` and used by
`starVLA/training/train_starvla.py`, so nothing has to be passed on the command line; the
`wandb_entity` / `wandb_project` config keys (and `--wandb_project` in the launchers) are ignored.
The full merged config is attached to the run.

Notes
- Launch with `WANDB_MODE=online` (several experiment scripts default to `offline`). `WANDB_MODE=disabled` still skips W&B.
- Keep one branch per experiment (`exp/<name>`): the branch name in the run name is how runs are grouped.
- Never commit `.env` or print the key.

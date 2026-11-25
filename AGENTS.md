# Agent instructions (starVLA fork)

These rules apply to every agent (Claude Code, Codex, ...) working in this repository, on any branch or machine.

## Experiment branches

`EXPERIMENTS.md` (on `starVLA_dev`) lists every `exp/<name>` branch: the question, the change,
status and results. When you create an experiment branch, add its row on `starVLA_dev` (commit
prefix `[baseline]`); when results come in, update the row. Keep experiment code on its own branch.

## Experiment logging: W&B (mandatory)

Every training run is logged to the team W&B on CoreWeave Forge:
**https://forge.coreweave.com/wandb/duhoe/starvla_poc**

| item | value |
|---|---|
| SDK server (`WANDB_BASE_URL`) | `https://api.forge.coreweave.com` (the SDK rejects the UI address `forge.coreweave.com`) |
| team (`WANDB_ENTITY`) | `duhoe` |
| project (`WANDB_PROJECT`) | `starvla_poc` |
| run name | `{git branch}_{run_id}`, e.g. `exp/timestep_human50_smolvla_tsdiscrete10_50k` |
| API key | `WANDB_KEY=...` in the repo-root `.env` (git-ignored). Never print, log or commit it. |

- `starVLA/training/train_starvla.py` applies all of this automatically
  (`starVLA/training/trainer_utils/team_wandb.py`): it reads `.env` (a `git worktree` falls back
  to the main checkout's `.env`), sets the server, and names the run. The `wandb_entity` /
  `wandb_project` config keys and `--wandb_project` launcher flags are ignored. The full merged
  config is attached to the run.
- Launch with `WANDB_MODE=online`. Some experiment scripts default to `offline`; override it.
- Use the git branch the code actually runs from: one branch per experiment (`exp/<name>`),
  started from `starVLA_dev`. The branch prefix in the run name is how runs are grouped.
- Other trainers / scripts that call `wandb.init` themselves must use the same entity, project,
  server and run-name format (call `team_wandb_init_kwargs(run_id)` from `team_wandb.py`).

### Offline runs and results logged afterwards
- Sync with `wandb sync --entity duhoe --project starvla_poc <.../wandb/offline-run-*>`, then
  rename the run to `{branch}_{run_id}` (the branch the run was trained from), and add the
  config (`<run_dir>/config.full.yaml`), `git_branch` / `git_commit`, tags and a short note.
- Put evaluation results on the training run: summary keys
  `eval_<renderer>/success_rate`, `eval_<renderer>/category/<Category>`, `eval_<renderer>/task/<Task>`,
  plus a per-task table and the raw result JSONs as an `eval_results` artifact. State the protocol
  (tasks x episodes, `N_ACT`, seed, renderer, NFE) in `eval_<renderer>/protocol`; mark partial evals
  as such (e.g. `eval_egl_partial`).
- Failed / stopped runs are uploaded too, tagged `failed` with the reason in the notes.
- Example: `docs/experiment_logging.md`.

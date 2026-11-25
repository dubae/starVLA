"""Team W&B convention for this fork (see docs/experiment_logging.md).

Every training run logs to the CoreWeave-hosted W&B
    https://forge.coreweave.com/wandb/duhoe/starvla_poc
  * server : WANDB_BASE_URL  (default https://api.forge.coreweave.com)
  * team   : WANDB_ENTITY    (default duhoe)
  * project: WANDB_PROJECT   (default starvla_poc)
  * run    : "{git branch}_{run_id}", e.g. "exp/timestep_human50_smolvla_tsdiscrete10_50k"
  * key    : WANDB_KEY in the repo-root `.env` (git-ignored), exported as WANDB_API_KEY.
             For a `git worktree`, the main checkout's `.env` is used as a fallback.
Exported environment variables take precedence over all defaults above; the
`wandb_entity` / `wandb_project` config keys are ignored.
"""

import os
import subprocess
from pathlib import Path

TEAM_WANDB_BASE_URL = "https://api.forge.coreweave.com"
TEAM_WANDB_ENTITY = "duhoe"
TEAM_WANDB_PROJECT = "starvla_poc"

_CODE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=_CODE_DIR, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:
        return ""


def git_branch() -> str:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch
    sha = _git("rev-parse", "--short", "HEAD")
    return f"detached-{sha}" if sha else "nogit"


def _env_files() -> list[Path]:
    files = []
    top = _git("rev-parse", "--show-toplevel")
    if top:
        files.append(Path(top) / ".env")
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if common:  # main checkout of a `git worktree`
        files.append(Path(common).parent / ".env")
    return files


def _read_env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        out[key] = val.strip().strip("'\"")
    return out


def setup_team_wandb_env() -> None:
    """Fill WANDB_* env vars from the team defaults and `.env` (never overrides exported vars)."""
    if not os.environ.get("WANDB_API_KEY"):
        for f in _env_files():
            if f.is_file():
                env = _read_env_file(f)
                key = env.get("WANDB_API_KEY") or env.get("WANDB_KEY")
                if key:
                    os.environ["WANDB_API_KEY"] = key
                    break
    os.environ.setdefault("WANDB_BASE_URL", TEAM_WANDB_BASE_URL)


def team_wandb_init_kwargs(run_id: str) -> dict:
    setup_team_wandb_env()
    return dict(
        name=f"{git_branch()}_{run_id}",
        entity=os.environ.get("WANDB_ENTITY", TEAM_WANDB_ENTITY),
        project=os.environ.get("WANDB_PROJECT", TEAM_WANDB_PROJECT),
    )

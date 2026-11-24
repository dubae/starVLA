"""RoboCasa (v0.x) Human-50 evaluation client for a StarVLA websocket policy server.

Runs ``n_episodes`` rollouts of one atomic task with the official RoboCasa evaluation
protocol (see ``human50_tasks.EVAL_ENV_KWARGS``): unseen object instances (split B),
the five held-out layout/style pairs, no generative textures, fixed cameras, and the
per-task horizon from the RoboCasa dataset registry. An episode is a success as soon
as ``env._check_success()`` holds (same rule as robomimic's rollout loop).

Run inside the ``robocasa`` (v0.2) simulation env, from the StarVLA repo root::

    python -m examples.simBenchmarks.Robocasa_human50.eval_files.simulation_env \
        --args.task OpenDrawer --args.port 5678 --args.pretrained-path <ckpt.pt>

Results: ``<ckpt>.eval/robocasa_human50/<Task>.json`` (override with ``--args.out-dir``).
With ``--args.num-shards K`` the task's episodes are split over K clients (shard i runs
episodes i, i+K, ... in its own env seeded ``seed + 1000 * i``) and each writes
``<Task>.shard<i>of<K>.json``; ``summarize_results.py`` merges complete shard sets.
"""

from __future__ import annotations

import dataclasses
import json
import os
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import robocasa  # noqa: F401  (registers the kitchen environments)
import robosuite
import tyro

from examples.simBenchmarks.Robocasa_human50.eval_files.model2robocasa_human50_interface import (
    DEFAULT_IMAGE_KEYS,
    PolicyWarper,
)
from examples.simBenchmarks.Robocasa_human50.human50_tasks import (
    CAMERA_NAMES,
    CONTROLLER_CONFIG,
    EVAL_ENV_KWARGS,
    EVAL_EPISODES_PER_TASK,
    HUMAN50_TASKS,
    IMAGE_SIZE,
    ROBOT,
)


@dataclasses.dataclass
class Args:
    task: str = "OpenDrawer"  # one of the 24 Human-50 tasks
    host: str = "127.0.0.1"
    port: int = 5678
    pretrained_path: str = ""  # checkpoint served by the policy server (used for result paths / config lookup)
    n_episodes: int = EVAL_EPISODES_PER_TASK
    horizon: Optional[int] = None  # None -> RoboCasa registry horizon for the task
    n_action_steps: int = 8  # actions executed per policy query (<= model action_horizon)
    seed: int = 0  # forwarded to the env: rollout i of a task is reproducible for a given seed
    image_size: Tuple[int, int] = (224, 224)  # must match training (dataset image_size, default 224)
    image_keys: Tuple[str, ...] = DEFAULT_IMAGE_KEYS  # camera views in training order
    # None -> read `datasets.vla_data.include_state` from the checkpoint's config.yaml
    send_state: Optional[bool] = None
    unnorm_key: Optional[str] = None
    num_steps: Optional[int] = None  # flow-matching denoising steps override (SmolVLA / PI-style heads)
    out_dir: Optional[str] = None
    n_video_episodes: int = 3  # save mp4s of the first N episodes (0 disables)
    env_kwargs_json: Optional[str] = None  # JSON dict merged over the official eval env kwargs (ablations only)
    shard_id: int = 0  # run episodes shard_id, shard_id + num_shards, ... (parallel eval of one task)
    # robosuite renders every camera on every control step, but the policy only sees the
    # images of the step before a query (every n_action_steps). By default the camera sensors
    # skip rendering on the other steps; observable timing is untouched, so the images the
    # policy receives come from the very same render call. Video episodes render every step.
    render_every_step: bool = os.environ.get("STARVLA_RENDER_EVERY_STEP", "0") == "1"
    num_shards: int = 1


def resolve_send_state(args: Args) -> bool:
    if args.send_state is not None:
        return args.send_state
    import yaml

    run_dir = Path(args.pretrained_path).resolve().parents[1]
    # config.full.yaml holds every resolved key; config.yaml may omit defaults-equal ones.
    for name in ("config.full.yaml", "config.yaml"):
        cfg_path = run_dir / name
        if cfg_path.is_file():
            vla_cfg = (yaml.safe_load(cfg_path.read_text()) or {}).get("datasets", {}).get("vla_data", {})
            if "include_state" in vla_cfg:
                return vla_cfg["include_state"] not in (False, "False", "false", None)
    raise FileNotFoundError(
        f"cannot infer include_state from {run_dir}/config*.yaml. Pass --args.send-state True/False."
    )


def make_env(task: str, seed: int, extra_kwargs: Optional[dict] = None):
    env_kwargs = dict(
        robots=ROBOT,
        controller_configs=CONTROLLER_CONFIG,
        camera_names=CAMERA_NAMES,
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_depths=False,
        ignore_done=True,
        reward_shaping=False,
        translucent_robot=False,
        seed=seed,
        **EVAL_ENV_KWARGS,
    )
    env_kwargs.update(extra_kwargs or {})
    return robosuite.make(env_name=task, **env_kwargs)


class LazyCameras:
    """Wraps the camera sensors of a robosuite env so they render only while ``on``.

    Off, a sensor returns its last image (never handed to the policy); the observables'
    sampling schedule is unchanged. ``rendered_step`` records the control step of the last
    render so callers can assert that the images they use are fresh. Rewrap after every
    ``env.reset()`` (a hard reset rebuilds the observables).
    """

    def __init__(self, env):
        self.on, self.step, self.rendered_step, self._last = True, 0, -1, {}
        for name, observable in env._observables.items():
            if name.endswith("_image") or name.endswith("_depth"):
                observable._sensor = self._wrap(name, observable._sensor)

    def _wrap(self, name, sensor):
        def lazy_sensor(obs_cache):
            if self.on or name not in self._last:
                self._last[name] = sensor(obs_cache)
                self.rendered_step = self.step
            return self._last[name]

        lazy_sensor.__modality__ = sensor.__modality__  # robosuite reads the modality off the sensor
        return lazy_sensor


def run_episode(env, policy: PolicyWarper, horizon: int, video_frames: Optional[list], render_every_step: bool = False) -> dict:
    t_reset = time.perf_counter()
    env.unset_ep_meta()  # sample a fresh scene (as robomimic's rollout does)
    obs = env.reset()
    t_reset = time.perf_counter() - t_reset
    cams = None if (render_every_step or video_frames is not None) else LazyCameras(env)
    instruction = env.get_ep_meta()["lang"]
    queue: list[np.ndarray] = []
    success, steps, t_policy, t_sim, n_queries = False, 0, 0.0, 0.0, 0
    for steps in range(1, horizon + 1):
        if not queue:
            if cams is not None and steps > 1 and cams.rendered_step != steps - 1:
                raise RuntimeError(f"stale camera images at step {steps} (last render at step {cams.rendered_step})")
            t = time.perf_counter()
            queue = list(policy.step(obs, instruction))
            t_policy += time.perf_counter() - t
            n_queries += 1
        if cams is not None:
            cams.step, cams.on = steps, len(queue) == 1  # render only on the step before the next query
        t = time.perf_counter()
        obs, _, _, _ = env.step(queue.pop(0))
        t_sim += time.perf_counter() - t
        if video_frames is not None:
            video_frames.append(np.concatenate(policy.upright_images(obs), axis=1))
        if env._check_success():
            success = True
            break
    ep_meta = env.get_ep_meta()
    return {
        "success": success,
        "steps": steps,
        "lang": instruction,
        "layout_id": ep_meta.get("layout_id"),
        "style_id": ep_meta.get("style_id"),
        # wall-clock breakdown, to spot eval bottlenecks (policy server vs simulator)
        "time_reset_s": round(t_reset, 2),
        "time_policy_s": round(t_policy, 2),
        "time_sim_s": round(t_sim, 2),
        "num_queries": n_queries,
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    if args.task not in HUMAN50_TASKS:
        raise ValueError(f"unknown task {args.task!r}; choose from {list(HUMAN50_TASKS)}")
    horizon = args.horizon or HUMAN50_TASKS[args.task][1]
    send_state = resolve_send_state(args)
    extra_kwargs = json.loads(args.env_kwargs_json) if args.env_kwargs_json else None

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pretrained_path).with_suffix(".eval") / "robocasa_human50"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    policy = PolicyWarper(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
        image_keys=args.image_keys,
        n_action_steps=args.n_action_steps,
        send_state=send_state,
        num_steps=args.num_steps,
    )
    logging.info("[human50] task=%s horizon=%d send_state=%s image_keys=%s server=%s",
                 args.task, horizon, send_state, args.image_keys, policy.server_metadata)

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"shard_id {args.shard_id} not in [0, {args.num_shards})")
    sharded = args.num_shards > 1
    env_seed = args.seed + 1000 * args.shard_id
    ep_ids = list(range(args.shard_id, args.n_episodes, args.num_shards))
    env = make_env(args.task, env_seed, extra_kwargs)
    episodes, t0 = [], time.time()
    for ep in ep_ids:
        frames = [] if ep < args.n_video_episodes else None
        result = run_episode(env, policy, horizon, frames, args.render_every_step)
        result["episode"] = ep
        episodes.append(result)
        if frames:
            import imageio

            video_dir.mkdir(parents=True, exist_ok=True)
            tag = "success" if result["success"] else "fail"
            imageio.mimsave(video_dir / f"{args.task}_ep{ep:03d}_{tag}.mp4", frames, fps=20)
        n_succ = sum(e["success"] for e in episodes)
        logging.info("[human50] %s ep %d/%d success=%s steps=%d | running SR %.3f (%.0fs) | "
                     "reset %.1fs, policy %.1fs (%.0f ms/query), sim %.1fs (%.1f ms/step)",
                     args.task, len(episodes), len(ep_ids), result["success"], result["steps"],
                     n_succ / len(episodes), time.time() - t0, result["time_reset_s"], result["time_policy_s"],
                     1000 * result["time_policy_s"] / max(result["num_queries"], 1), result["time_sim_s"],
                     1000 * result["time_sim_s"] / max(result["steps"], 1))
    env.close()
    policy.close()

    successes = [e["success"] for e in episodes]
    summary = {
        "task": args.task,
        "success_rate": float(np.mean(successes)),
        "num_success": int(sum(successes)),
        "num_episodes": len(successes),
        "horizon": horizon,
        "n_action_steps": args.n_action_steps,
        "seed": args.seed,
        "env_seed": env_seed,
        "shard": [args.shard_id, args.num_shards],
        "send_state": send_state,
        "image_keys": list(args.image_keys),
        "image_size": list(args.image_size),
        "env_kwargs": {**EVAL_ENV_KWARGS, **(extra_kwargs or {})},
        "checkpoint": args.pretrained_path,
        "wall_time_s": round(time.time() - t0, 1),
        "time_breakdown_s": {k: round(sum(e[f"time_{k}_s"] for e in episodes), 1) for k in ("reset", "policy", "sim")},
        "episodes": episodes,
    }
    out_path = out_dir / (f"{args.task}.shard{args.shard_id}of{args.num_shards}.json" if sharded else f"{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {args.task}: success rate {summary['success_rate']:.3f} ({summary['num_success']}/{len(successes)}) ===")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    tyro.cli(main)

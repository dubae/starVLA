"""LIBERO evaluation client for policies trained in the RoboCasa layout (bench_forget).

Observations are built exactly like the training conversion (robocasa_layout.py) and fed
through the Human-50 PolicyWarper (same per-key sin/cos state, same image resize):
agentview -> left view, blank right view, wrist -> eye-in-hand, frames rotated 180 deg as in
the LIBERO training videos. The first 7 action dims (arm delta pose + gripper) go to the env.

Run inside the LIBERO env, from the StarVLA repo root, with LIBERO on PYTHONPATH:

    python -m examples.experiments.bench_forget.eval_files.libero_eval \
        --args.port 6300 --args.pretrained-path <ckpt.pt> --args.task-ids 0 1
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from examples.experiments.bench_forget.robocasa_layout import (
    LIBERO_BASE_POS,
    LIBERO_BASE_QUAT_XYZW,
    libero_state_to_robocasa,
    robocasa_action_to_libero,
)
from examples.simBenchmarks.Robocasa_human50.eval_files.model2robocasa_human50_interface import PolicyWarper
from examples.simBenchmarks.Robocasa_human50.human50_tasks import STATE_LAYOUT

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6300
    pretrained_path: str = ""
    task_suite_name: str = "libero_object"
    task_ids: Tuple[int, ...] = ()  # empty -> all tasks of the suite
    n_episodes: int = 50  # per task, LIBERO's fixed initial states 0..n-1
    n_action_steps: int = 10
    num_steps_wait: int = 10  # objects settle before the policy acts
    seed: int = 7
    resolution: int = 256  # LIBERO training data renders
    unnorm_key: str = "franka"
    out_dir: Optional[str] = None
    n_video_episodes: int = 1


def _base_pose(env) -> None:
    """The layout assumes the fixed LIBERO mount pose; fail loudly if a scene differs."""
    sim, robot = env.env.sim, env.env.robots[0]
    bid = sim.model.body_name2id(robot.robot_model.root_body)
    pos, wxyz = sim.data.body_xpos[bid], sim.data.body_xquat[bid]
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    if not (np.allclose(pos, LIBERO_BASE_POS, atol=1e-4) and np.allclose(xyzw, LIBERO_BASE_QUAT_XYZW, atol=1e-4)):
        raise RuntimeError(f"robot base pose {pos}, {xyzw} differs from the training layout")


def to_robocasa_obs(obs: dict) -> dict:
    state = libero_state_to_robocasa(obs["robot0_eef_pos"][None], obs["robot0_eef_quat"][None], obs["robot0_gripper_qpos"][None])[0]
    out, start = {}, 0
    for _, key, dim in STATE_LAYOUT:
        out[key] = state[start : start + dim]
        start += dim
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # 180 deg, as in the training videos
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    out["robot0_agentview_left_image"] = agent
    out["robot0_agentview_right_image"] = np.zeros_like(agent)  # blank slot, masked by the model
    out["robot0_eye_in_hand_image"] = wrist
    return out


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = list(args.task_ids) or list(range(suite.n_tasks))
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pretrained_path).with_suffix(".eval") / args.task_suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = PolicyWarper(host=args.host, port=args.port, unnorm_key=args.unnorm_key, n_action_steps=args.n_action_steps,
                          send_state=True, flip_images=False)
    max_steps = MAX_STEPS[args.task_suite_name]

    for tid in task_ids:
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.resolution, camera_widths=args.resolution)
        env.seed(args.seed)
        episodes, t0 = [], time.time()
        for ep in range(args.n_episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            _base_pose(env)
            frames = [] if ep < args.n_video_episodes else None
            success, t, queue = False, 0, []
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            while t < max_steps:
                rc = to_robocasa_obs(obs)
                if frames is not None:
                    frames.append(np.concatenate([rc["robot0_agentview_left_image"], rc["robot0_eye_in_hand_image"]], axis=1))
                if not queue:
                    queue = list(policy.step(rc, task.language))
                obs, _, done, _ = env.step(robocasa_action_to_libero(queue.pop(0)).tolist())
                t += 1
                if done:
                    success = True
                    break
            episodes.append({"episode": ep, "success": success, "steps": t})
            if frames:
                import imageio

                (out_dir / "videos").mkdir(exist_ok=True)
                imageio.mimsave(out_dir / "videos" / f"task{tid:02d}_ep{ep:03d}_{'success' if success else 'fail'}.mp4", frames, fps=20)
            logging.info("[libero] task %d ep %d/%d success=%s steps=%d | running SR %.3f (%.0fs)", tid, ep + 1, args.n_episodes,
                         success, t, np.mean([e["success"] for e in episodes]), time.time() - t0)
        env.close()
        res = {"task_id": tid, "task": task.language, "success_rate": float(np.mean([e["success"] for e in episodes])),
               "num_success": int(sum(e["success"] for e in episodes)), "num_episodes": len(episodes),
               "n_action_steps": args.n_action_steps, "seed": args.seed, "unnorm_key": args.unnorm_key,
               "checkpoint": args.pretrained_path, "wall_time_s": round(time.time() - t0, 1), "episodes": episodes}
        (out_dir / f"task_{tid:02d}.json").write_text(json.dumps(res, indent=2))
        print(f"=== task {tid} ({task.language}): {res['success_rate']:.3f}")
    policy.close()


if __name__ == "__main__":
    tyro.cli(main)

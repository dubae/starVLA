"""Rewrite an IPEC LIBERO LeRobot dataset in the RoboCasa Human-50 layout (see robocasa_layout.py).

    python -m examples.experiments.bench_forget.convert_libero_to_robocasa_layout \
        --src /mnt/nvme/duhoekim/datasets/libero/libero_object_no_noops_1.0.0_lerobot \
        --out playground/Datasets/libero_robocasa_layout/libero_object \
        --robocasa-stats playground/Checkpoints/human50_smolvla_full_50k/dataset_statistics.json

* parquet: 16-D state / 12-D action in the Human-50 order (float64, like the Human-50 converter);
* videos : agentview -> robot0_agentview_left and wrist -> robot0_eye_in_hand are symlinked
           (frames stay as recorded, i.e. rotated 180 deg; the LIBERO eval client rotates the
           same way), robot0_agentview_right is an all-black clip (masked by the model);
* stats  : meta/stats_gr00t.json is computed by the StarVLA loader, then the action dims that
           are constant in LIBERO (base_motion, control_mode) get the RoboCasa statistics, so a
           raw 0 / -1 there normalizes to the same value in both datasets (otherwise min==max
           maps them to 0, which RoboCasa's un-normalization reads as base motion).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from examples.experiments.bench_forget.robocasa_layout import (
    axisangle_to_quat_xyzw,
    libero_action_to_robocasa,
    libero_state_to_robocasa,
)
from examples.simBenchmarks.Robocasa_human50.human50_tasks import CAMERA_NAMES, FPS
from examples.simBenchmarks.Robocasa_human50.train_files.convert_human50_to_lerobot import (
    DATA_PATH,
    VIDEO_PATH,
    build_info,
    build_modality,
    encode_video,
)

SRC_VIDEO = {"robot0_agentview_left": "observation.images.image", "robot0_eye_in_hand": "observation.images.wrist_image"}
BLANK_VIEW = "robot0_agentview_right"
CONSTANT_ACTION_DIMS = slice(7, 12)  # base_motion(4) + control_mode(1)


def convert(src: Path, out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    info = json.loads((src / "meta/info.json").read_text())
    assert info["fps"] == FPS, f"LIBERO fps {info['fps']} != RoboCasa {FPS}"
    h, w = info["features"]["observation.images.image"]["shape"][:2]
    chunks_size = info["chunks_size"]

    episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    total = 0
    blank_cache: dict[int, Path] = {}
    for ep in episodes:
        e = ep["episode_index"]
        chunk = e // chunks_size
        df = pd.read_parquet(src / info["data_path"].format(episode_chunk=chunk, episode_index=e))
        s = np.stack(df["observation.state"].to_numpy()).astype(np.float64)  # eef pos(3) axis-angle(3) gripper qpos(2)
        a = np.stack(df["action"].to_numpy()).astype(np.float64)
        n = len(df)
        state = libero_state_to_robocasa(s[:, :3], axisangle_to_quat_xyzw(s[:, 3:6]), s[:, 6:8])
        action = libero_action_to_robocasa(a)
        frame_index = df["frame_index"].to_numpy().astype(np.int64)
        out_df = pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(action),
                "annotation.human.task_description": df["task_index"].to_numpy().astype(np.int64),
                "next.done": frame_index == n - 1,
                "timestamp": df["timestamp"].to_numpy().astype(np.float32),
                "frame_index": frame_index,
                "episode_index": df["episode_index"].to_numpy().astype(np.int64),
                "index": df["index"].to_numpy().astype(np.int64),
                "task_index": df["task_index"].to_numpy().astype(np.int64),
            }
        )
        p = out / DATA_PATH.format(episode_chunk=chunk, episode_index=e)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(p, index=False)

        for cam, src_key in SRC_VIDEO.items():
            dst = out / VIDEO_PATH.format(episode_chunk=chunk, video_key=f"observation.images.{cam}", episode_index=e)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src / info["video_path"].format(episode_chunk=chunk, video_key=src_key, episode_index=e), dst)
        # all-black clip, one encode per distinct length
        dst = out / VIDEO_PATH.format(episode_chunk=chunk, video_key=f"observation.images.{BLANK_VIEW}", episode_index=e)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if n not in blank_cache:
            blank_cache[n] = out / "meta" / "blank_videos" / f"blank_{n}.mp4"
            encode_video(np.zeros((n, h, w, 3), np.uint8), blank_cache[n], FPS, crf=18)
        os.symlink(blank_cache[n].resolve(), dst)
        total += n

    shutil.copy(src / "meta/episodes.jsonl", out / "meta/episodes.jsonl")
    shutil.copy(src / "meta/tasks.jsonl", out / "meta/tasks.jsonl")
    (out / "meta/modality.json").write_text(json.dumps(build_modality(), indent=4))
    new_info = build_info(len(episodes), total, info["total_tasks"], (h, w))
    new_info["robot_type"] = "Panda (LIBERO, RoboCasa layout)"
    (out / "meta/info.json").write_text(json.dumps(new_info, indent=4))
    (out / "meta/source.json").write_text(json.dumps({"source": str(src), "layout": "examples/experiments/bench_forget/robocasa_layout.py"}, indent=2))
    assert total == info["total_frames"], (total, info["total_frames"])
    print(f"wrote {len(episodes)} episodes / {total} frames -> {out}")


def patch_constant_dim_stats(out: Path, robocasa_stats: Path) -> None:
    """Compute the loader's stats cache for ``out`` and give the LIBERO-constant action dims the
    RoboCasa statistics."""
    from starVLA.dataloader.gr00t_lerobot.datasets import LE_ROBOT_STATS_FILENAME

    cache = out / LE_ROBOT_STATS_FILENAME
    if not cache.is_file():
        raise FileNotFoundError(f"{cache} missing: build the dataloader once (see README) before patching")
    payload = json.loads(cache.read_text())
    rc = json.loads(robocasa_stats.read_text())["new_embodiment"]["action"]
    act = payload["statistics"]["action"]
    for k in ("min", "max", "mean", "std", "q01", "q99"):
        v = np.asarray(act[k], dtype=np.float64)
        v[CONSTANT_ACTION_DIMS] = np.asarray(rc[k], dtype=np.float64)[CONSTANT_ACTION_DIMS]
        act[k] = v.tolist()
    payload["patched_constant_action_dims_from"] = str(robocasa_stats)
    cache.write_text(json.dumps(payload, indent=4))
    print("patched", cache, {k: np.round(act[k][7:12], 3).tolist() for k in ("min", "max")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robocasa-stats", type=Path, help="dataset_statistics.json of a Human-50 run (key new_embodiment)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patch-stats-only", action="store_true")
    args = ap.parse_args()
    if not args.patch_stats_only:
        convert(args.src, args.out, args.overwrite)
    if args.robocasa_stats and args.patch_stats_only:
        patch_constant_dim_stats(args.out, args.robocasa_stats)


if __name__ == "__main__":
    main()

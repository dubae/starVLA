"""Convert RoboCasa (v0.x) Human-50 HDF5 demos to StarVLA-compatible LeRobot v2.1.

Input  (robocasa ``DATASET_BASE_PATH`` layout, see ``download_human50.sh``)::

    <raw_root>/v0.1/single_stage/<category>/<Task>/<date>/demo_gentex_im128_randcams.hdf5

Output (one LeRobot dataset per task, mirroring the RoboCasa365 layout)::

    <out_root>/v0.1/single_stage/<category>/<Task>/<date>/lerobot/
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/observation.images.<camera>/episode_000000.mp4
    └── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, robocasa_env_args.json}

Only the demos listed under the ``50_demos`` filter key are exported (Human-50).
Columns follow the official RoboCasa LeRobot conversion: 16-D ``observation.state``
(base pos/quat, base-relative eef pos/quat, gripper qpos) and the 12-D ``action``
in its native HDF5 / ``env.step()`` order.

Needs only ``h5py``, ``numpy``, ``pandas``, ``pyarrow`` and ``av`` (all present in the
``robocasa`` simulation env). Run from the StarVLA repo root::

    python -m examples.simBenchmarks.Robocasa_human50.train_files.convert_human50_to_lerobot \
        --raw-root playground/Datasets/robocasa_human50_raw \
        --out-root playground/Datasets/robocasa_human50
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd

from examples.simBenchmarks.Robocasa_human50.human50_tasks import (
    ACTION_LAYOUT,
    CAMERA_NAMES,
    FILTER_KEY,
    FPS,
    HDF5_FILENAME,
    HUMAN50_TASKS,
    STATE_LAYOUT,
)

CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def encode_video(frames: np.ndarray, path: Path, fps: int, crf: int) -> None:
    """Encode (T, H, W, 3) uint8 frames to H.264 / yuv420p (same codec as RoboCasa365)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        # keyframe every 2 frames keeps random-access decoding cheap for 128px clips
        stream.options = {"crf": str(crf), "g": "2", "preset": "medium"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_modality() -> dict:
    state, start = {}, 0
    for key, _, dim in STATE_LAYOUT:
        state[key] = {"original_key": "observation.state", "start": start, "end": start + dim}
        start += dim
    action = {key: {"original_key": "action", "start": s, "end": e} for key, s, e in ACTION_LAYOUT}
    video = {cam: {"original_key": f"observation.images.{cam}"} for cam in CAMERA_NAMES}
    annotation = {"human.task_description": {"original_key": "annotation.human.task_description"}}
    return {"state": state, "action": action, "video": video, "annotation": annotation}


def build_info(num_episodes: int, num_frames: int, num_tasks: int, image_hw: tuple[int, int]) -> dict:
    state_dim = sum(dim for _, _, dim in STATE_LAYOUT)
    action_dim = ACTION_LAYOUT[-1][2]
    h, w = image_hw
    video_feature = {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": FPS,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    features = {f"observation.images.{cam}": video_feature for cam in CAMERA_NAMES}
    features.update(
        {
            "observation.state": {"dtype": "float64", "shape": [state_dim]},
            "action": {"dtype": "float64", "shape": [action_dim]},
            "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "total_episodes": num_episodes,
        "total_frames": num_frames,
        "total_tasks": num_tasks,
        "total_videos": num_episodes * len(CAMERA_NAMES),
        "total_chunks": (num_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }


def convert_task(task: str, raw_root: Path, out_root: Path, filter_key: str, crf: int, overwrite: bool) -> str:
    human_path, _ = HUMAN50_TASKS[task]
    hdf5_path = raw_root / human_path / HDF5_FILENAME
    out_dir = out_root / human_path / "lerobot"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"[{task}] missing {hdf5_path}; run download_human50.sh first")
    if (out_dir / "meta" / "info.json").is_file() and not overwrite:
        return f"[{task}] exists, skipped ({out_dir})"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted((d.decode() for d in f["mask"][filter_key][:]), key=_demo_sort_key)
        env_args = json.loads(f["data"].attrs["env_args"])

        task_to_index: dict[str, int] = {}
        episodes, global_index, image_hw = [], 0, None
        for ep_idx, demo in enumerate(demos):
            g = f["data"][demo]
            lang = json.loads(g.attrs["ep_meta"])["lang"]
            task_idx = task_to_index.setdefault(lang, len(task_to_index))

            actions = g["actions"][:].astype(np.float64)
            length = actions.shape[0]
            state = np.concatenate([g["obs"][obs_key][:].astype(np.float64) for _, obs_key, _ in STATE_LAYOUT], axis=1)
            assert state.shape[0] == length, f"[{task}/{demo}] state/action length mismatch"

            chunk = ep_idx // CHUNKS_SIZE
            frame_index = np.arange(length, dtype=np.int64)
            df = pd.DataFrame(
                {
                    "observation.state": list(state),
                    "action": list(actions),
                    "annotation.human.task_description": np.full(length, task_idx, dtype=np.int64),
                    "next.done": frame_index == length - 1,
                    "timestamp": (frame_index / FPS).astype(np.float32),
                    "frame_index": frame_index,
                    "episode_index": np.full(length, ep_idx, dtype=np.int64),
                    "index": np.arange(global_index, global_index + length, dtype=np.int64),
                    "task_index": np.full(length, task_idx, dtype=np.int64),
                }
            )
            parquet_path = out_dir / DATA_PATH.format(episode_chunk=chunk, episode_index=ep_idx)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)

            for cam in CAMERA_NAMES:
                frames = g["obs"][f"{cam}_image"][:]
                assert frames.shape[0] == length, f"[{task}/{demo}] {cam} length mismatch"
                image_hw = frames.shape[1:3]
                encode_video(
                    frames,
                    out_dir / VIDEO_PATH.format(episode_chunk=chunk, video_key=f"observation.images.{cam}", episode_index=ep_idx),
                    FPS,
                    crf,
                )

            episodes.append({"episode_index": ep_idx, "tasks": [lang], "length": length, "source_demo": demo})
            global_index += length

    meta = out_dir / "meta"
    with open(meta / "episodes.jsonl", "w") as fp:
        fp.writelines(json.dumps(ep) + "\n" for ep in episodes)
    with open(meta / "tasks.jsonl", "w") as fp:
        fp.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in task_to_index.items())
    with open(meta / "modality.json", "w") as fp:
        json.dump(build_modality(), fp, indent=4)
    with open(meta / "info.json", "w") as fp:
        json.dump(build_info(len(episodes), global_index, len(task_to_index), image_hw), fp, indent=4)
    # Keep the source env metadata for reference (controller, robot, original scene split).
    with open(meta / "robocasa_env_args.json", "w") as fp:
        json.dump({"source_hdf5": str(hdf5_path), "filter_key": filter_key, "env_args": env_args}, fp, indent=2)
    return f"[{task}] {len(episodes)} episodes, {global_index} frames, {len(task_to_index)} instructions -> {out_dir}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="robocasa DATASET_BASE_PATH holding the HDF5 files")
    parser.add_argument("--out-root", type=Path, required=True, help="StarVLA data_root_dir for the LeRobot datasets")
    parser.add_argument("--tasks", nargs="+", default=list(HUMAN50_TASKS), choices=list(HUMAN50_TASKS))
    parser.add_argument("--filter-key", default=FILTER_KEY, help="HDF5 mask key selecting the demos (Human-50: 50_demos)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF (lower = higher quality)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(convert_task, t, args.raw_root, args.out_root, args.filter_key, args.crf, args.overwrite): t
            for t in args.tasks
        }
        failed = []
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append(futures[fut])
                print(f"[{futures[fut]}] FAILED: {exc!r}", flush=True)
    if failed:
        raise SystemExit(f"conversion failed for: {failed}")


if __name__ == "__main__":
    main()

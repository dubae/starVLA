"""RoboCasa (v0.x) Human-50 (PandaOmron, 24 atomic tasks) — data config and mixtures.

Loaded automatically by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.

Datasets are produced by ``convert_human50_to_lerobot.py`` from the official
``human_im`` HDF5 files (``50_demos`` filter key, 128x128 renders, 20 fps):

* observation.state (16d): base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)
* action (12d, native ``env.step()`` order): eef_pos(3) + eef_rot(3) + gripper_close(1) + base_motion(4) + control_mode(1)
* 3 cameras: robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand

The state/action key names match ``Robocasa_365`` (same PandaOmron convention), but a
separate robot type is used so that normalization statistics are never shared with
RoboCasa365 checkpoints.
"""

import importlib.util
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)

# Shared task table (pure Python), loaded by path because this file is itself loaded by path.
_TASKS_FILE = Path(__file__).resolve().parents[2] / "human50_tasks.py"
_spec = importlib.util.spec_from_file_location("_robocasa_human50_tasks", _TASKS_FILE)
_tasks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tasks)


class PandaOmronRoboCasaHuman50DataConfig:
    """Single-arm PandaOmron used by RoboCasa v0.x (recorded as ``PandaMobile``)."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [f"video.{cam}" for cam in _tasks.CAMERA_NAMES]
    state_keys = [f"state.{key}" for key, _, _ in _tasks.STATE_LAYOUT]
    action_keys = [f"action.{key}" for key, _, _ in _tasks.ACTION_LAYOUT]
    # Per-key dims for PolicyNormProcessor
    action_key_dims = {f"action.{key}": end - start for key, start, end in _tasks.ACTION_LAYOUT}
    state_key_dims = {f"state.{key}": dim for key, _, dim in _tasks.STATE_LAYOUT}
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


class PandaOmronRoboCasaHuman50Chunk50DataConfig(PandaOmronRoboCasaHuman50DataConfig):
    """Same data with 50-step action chunks (SmolVLA / LeRobot default ``chunk_size``)."""

    action_indices = list(range(50))


class PandaOmronRoboCasaHuman50Chunk50FutureDataConfig(PandaOmronRoboCasaHuman50Chunk50DataConfig):
    """Chunk-50 data plus the camera frames at t + 50 (end of the chunk), packed as ``future_image``
    for future-latent (forward-dynamics) losses. Frames past the episode end repeat the last frame."""

    video_indices = [0, 50]

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return config


_ROBOT_TAG = "panda_omron_robocasa_human50"
_ROBOT_TAG_CHUNK50 = "panda_omron_robocasa_human50_chunk50"
_ROBOT_TAG_CHUNK50_FUTURE = "panda_omron_robocasa_human50_chunk50_future50"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TAG: PandaOmronRoboCasaHuman50DataConfig(),
    _ROBOT_TAG_CHUNK50: PandaOmronRoboCasaHuman50Chunk50DataConfig(),
    _ROBOT_TAG_CHUNK50_FUTURE: PandaOmronRoboCasaHuman50Chunk50FutureDataConfig(),
}


def _entries(tasks, robot_tag=_ROBOT_TAG):
    return [(_tasks.lerobot_relpath(t), 1.0, robot_tag) for t in tasks]


_PNP = [t for t in _tasks.HUMAN50_TASKS if t.startswith("PnP")]

DATASET_NAMED_MIXTURES = {
    # Standard Human-50 multi-task setting: all 24 atomic tasks, equal per-task sampling.
    "robocasa_human50_all": _entries(_tasks.HUMAN50_TASKS),
    # 50-step action chunks (framework.action_model.action_horizon: 50)
    "robocasa_human50_all_chunk50": _entries(_tasks.HUMAN50_TASKS, _ROBOT_TAG_CHUNK50),
    # same + camera frames at t + 50 (`future_image`)
    "robocasa_human50_all_chunk50_future50": _entries(_tasks.HUMAN50_TASKS, _ROBOT_TAG_CHUNK50_FUTURE),
    "robocasa_human50_pnp": _entries(_PNP),
    "robocasa_human50_non_pnp": _entries(t for t in _tasks.HUMAN50_TASKS if t not in _PNP),
    # Single-task mixtures, e.g. ``robocasa_human50_OpenDrawer`` (smoke tests / per-task baselines).
    **{f"robocasa_human50_{t}": _entries([t]) for t in _tasks.HUMAN50_TASKS},
}

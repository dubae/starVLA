"""bench_forget: LIBERO-object written in the RoboCasa Human-50 layout (see ../../robocasa_layout.py).

The data config is the Human-50 chunk-50 config itself (same state/action keys, sin/cos state,
per-key min_max action, 50-step chunks, 3 views); only the embodiment tag differs, so the
LIBERO normalization statistics are stored under their own key (``franka``) next to the
RoboCasa ones (``new_embodiment``).
"""

import importlib.util
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

_H50 = Path(__file__).resolve().parents[4] / "simBenchmarks/Robocasa_human50/train_files/data_registry/data_config.py"
_spec = importlib.util.spec_from_file_location("_bench_forget_human50_data_config", _H50)
_h50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h50)


class LiberoRoboCasaLayoutChunk50DataConfig(_h50.PandaOmronRoboCasaHuman50Chunk50DataConfig):
    """LIBERO Panda in the PandaOmron RoboCasa layout; statistics key ``franka``."""

    embodiment_tag = EmbodimentTag.FRANKA


ROBOT_TYPE_CONFIG_MAP = {"libero_robocasa_layout_chunk50": LiberoRoboCasaLayoutChunk50DataConfig()}

DATASET_NAMED_MIXTURES = {
    # data_root_dir: playground/Datasets/libero_robocasa_layout
    "libero_object_robocasa_layout_chunk50": [("libero_object", 1.0, "libero_robocasa_layout_chunk50")],
}

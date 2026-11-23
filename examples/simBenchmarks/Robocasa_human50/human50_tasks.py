"""RoboCasa (v0.x) Human-50 benchmark: the 24 atomic kitchen tasks.

Pure-Python constants shared by the converter, the training data registry and the
evaluation client (no third-party imports, so every environment can load it).

Snapshot of ``robocasa/utils/dataset_registry.py::SINGLE_STAGE_TASK_DATASETS`` at
RoboCasa tag ``v0.2``. ``NavigateKitchen`` is excluded: the RoboCasa paper's atomic
benchmark uses the 24 manipulation tasks below. Each task ships ``human_im``
demonstrations (``demo_gentex_im128_randcams.hdf5``); the ``50_demos`` filter key
selects the Human-50 training split.
"""

# task -> (human_path relative to robocasa DATASET_BASE_PATH, rollout horizon in env steps)
HUMAN50_TASKS = {
    # pick-and-place
    "PnPCounterToCab":       ("v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24", 500),
    "PnPCabToCounter":       ("v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24", 500),
    "PnPCounterToSink":      ("v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25", 700),
    "PnPSinkToCounter":      ("v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2", 500),
    "PnPCounterToMicrowave": ("v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27", 600),
    "PnPMicrowaveToCounter": ("v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26", 500),
    "PnPCounterToStove":     ("v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26", 500),
    "PnPStoveToCounter":     ("v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01", 500),
    # doors
    "OpenSingleDoor":        ("v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24", 500),
    "CloseSingleDoor":       ("v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24", 500),
    "OpenDoubleDoor":        ("v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26", 1000),
    "CloseDoubleDoor":       ("v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29", 700),
    # drawers
    "OpenDrawer":            ("v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03", 500),
    "CloseDrawer":           ("v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30", 500),
    # sink
    "TurnOnSinkFaucet":      ("v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25", 500),
    "TurnOffSinkFaucet":     ("v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25", 500),
    "TurnSinkSpout":         ("v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29", 500),
    # stove
    "TurnOnStove":           ("v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02", 500),
    "TurnOffStove":          ("v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02", 500),
    # coffee
    "CoffeeSetupMug":        ("v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25", 600),
    "CoffeeServeMug":        ("v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01", 600),
    "CoffeePressButton":     ("v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25", 300),
    # microwave
    "TurnOnMicrowave":       ("v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25", 500),
    "TurnOffMicrowave":      ("v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25", 500),
}

HDF5_FILENAME = "demo_gentex_im128_randcams.hdf5"
FILTER_KEY = "50_demos"

# Official evaluation scene protocol (robomimic `robocasa` branch,
# robomimic/scripts/config_gen/config_gen_utils.py): unseen object instances
# (split "B"), five fixed layout/style pairs, no generative textures, fixed cameras.
EVAL_ENV_KWARGS = {
    "obj_instance_split": "B",
    "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
    "generative_textures": None,
    "randomize_cameras": False,
    "layout_ids": None,
    "style_ids": None,
}
EVAL_EPISODES_PER_TASK = 50

# Robot + arm controller exactly as recorded in every Human-50 HDF5 ``env_args``
# (identical across the 24 tasks). RoboCasa v0.2 renames PandaMobile -> PandaOmron and
# converts this legacy OSC config to the robosuite v1.5 composite controller.
ROBOT = "PandaMobile"
CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_ratio_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}

CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
IMAGE_SIZE = 128  # human_im datasets are rendered at 128x128
FPS = 20

# 16-D proprio state, same key layout as the RoboCasa365 LeRobot conversion:
# (lerobot key, hdf5 / env observation key, dim)
STATE_LAYOUT = [
    ("base_position", "robot0_base_pos", 3),
    ("base_rotation", "robot0_base_quat", 4),
    ("end_effector_position_relative", "robot0_base_to_eef_pos", 3),
    ("end_effector_rotation_relative", "robot0_base_to_eef_quat", 4),
    ("gripper_qpos", "robot0_gripper_qpos", 2),
]

# 12-D action, kept in the native HDF5 / env.step() order so predictions can be sent
# to the simulator without reordering: (lerobot key, start, end)
ACTION_LAYOUT = [
    ("end_effector_position", 0, 3),
    ("end_effector_rotation", 3, 6),
    ("gripper_close", 6, 7),
    ("base_motion", 7, 11),
    ("control_mode", 11, 12),
]


def lerobot_relpath(task: str) -> str:
    """LeRobot dataset path relative to the StarVLA ``data_root_dir``."""
    return f"{HUMAN50_TASKS[task][0]}/lerobot"

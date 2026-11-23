"""LIBERO (Panda, fixed base) <-> RoboCasa Human-50 (PandaOmron) observation/action layout.

Both benchmarks drive a Franka arm with the same robosuite OSC_POSE controller (delta pose,
input range [-1, 1], output_max 0.05 m / 0.5 rad, kp 150, 20 Hz), so LIBERO data is written
in the exact RoboCasa layout (``Robocasa_human50/human50_tasks.py``):

  state  (16): base_pos(3) base_quat(4, xyzw) base_to_eef_pos(3) base_to_eef_quat(4, xyzw, w>=0) gripper_qpos(2)
  action (12): eef_pos(3) eef_rot(3) gripper(1, +1 close / -1 open) base_motion(4)=0 control_mode(1)=-1
  views  (3) : robot0_agentview_left <- LIBERO agentview, robot0_agentview_right <- blank
               (masked by framework.smolvla.mask_blank_images), robot0_eye_in_hand <- wrist

The LIBERO robot base sits at world (-0.6, 0, 0) with identity orientation in every
libero_object scene (checked on all 10 tasks), so world-frame deltas equal base-frame deltas.
"""

from __future__ import annotations

import numpy as np

LIBERO_BASE_POS = np.array([-0.6, 0.0, 0.0])
LIBERO_BASE_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])
ROBOCASA_ARM_CONTROL_MODE = -1.0  # Human-50 data: control_mode is -1 (arm) in every frame


def axisangle_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 4) unit quaternion (x, y, z, w)."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, angle, out=np.zeros_like(aa), where=angle > 1e-12)
    return np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], axis=-1)


def canonical_quat(q: np.ndarray) -> np.ndarray:
    """RoboCasa's base_to_eef_quat convention: the hemisphere with w >= 0."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., 3:4] < 0, -q, q)


def libero_state_to_robocasa(eef_pos: np.ndarray, eef_quat_xyzw: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    """World-frame LIBERO proprio (T, ...) -> (T, 16) RoboCasa state (base orientation is identity)."""
    n = eef_pos.shape[0]
    return np.concatenate(
        [
            np.tile(LIBERO_BASE_POS, (n, 1)),
            np.tile(LIBERO_BASE_QUAT_XYZW, (n, 1)),
            eef_pos - LIBERO_BASE_POS,
            canonical_quat(eef_quat_xyzw),
            gripper_qpos,
        ],
        axis=1,
    )


def libero_action_to_robocasa(action: np.ndarray) -> np.ndarray:
    """IPEC LIBERO action (T, 7): delta pos(3), delta axis-angle(3), gripper (1 = open, 0 = close)
    -> (T, 12) RoboCasa action with the robosuite gripper sign (+1 close)."""
    n = action.shape[0]
    gripper_close = 1.0 - 2.0 * action[:, 6:7]
    return np.concatenate(
        [action[:, :6], gripper_close, np.zeros((n, 4)), np.full((n, 1), ROBOCASA_ARM_CONTROL_MODE)], axis=1
    )


def robocasa_action_to_libero(action: np.ndarray) -> np.ndarray:
    """(T, 12) RoboCasa-layout action -> (T, 7) LIBERO env.step action (arm + gripper)."""
    return np.asarray(action)[..., :7]

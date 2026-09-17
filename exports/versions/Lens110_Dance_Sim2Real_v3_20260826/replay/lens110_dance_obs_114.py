"""Lens110 21 关节残差策略的 114 维观测构建器 (实验A: 含 base 线速度)。

观测顺序 (114 维单帧, 与训练 ObservationsCfg 一致):
  [0:3]    base_ang_vel      机身角速度 (body 系, rad/s)
  [3:6]    base_lin_vel      机身线速度 (body 系, m/s)  ← 实验A新增
  [6:9]    gravity_b         世界重力在 body 系投影 = R^T @ [0,0,-1]
  [9:51]   ref_joint_pos/vel 参考关节角 + 速度 (42, USD 顺序)
  [51:72]  joint_pos_rel     关节角 - runtime_default (21, USD 顺序)
  [72:93]  joint_vel_rel     关节速度 (21, USD 顺序)
  [93:114] last_action       上一策略动作 (已裁剪 [-1,1], 21)

执行语义 (21 维残差动作):
  q_des = ref_joint_pos + ACTION_SCALE * clip(action, -1, 1)
"""

from __future__ import annotations

import numpy as np


USD_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

ACTION_SCALE = np.array([
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.1, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15,
], dtype=np.float32)

CLIP_OBS = 5.0
CLIP_ACT = 1.0


def _quat_apply_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R^T @ v, q 为 wxyz, 与 isaaclab quat_apply_inverse 一致。"""
    w, x, y, z = q_wxyz
    xyz = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


def build_actor_observation(
    reference_joint_pos: np.ndarray,
    reference_joint_vel: np.ndarray,
    imu_quat_wxyz: np.ndarray,
    robot_ang_vel_b: np.ndarray,
    robot_lin_vel_b: np.ndarray,
    robot_joint_pos: np.ndarray,
    robot_joint_vel: np.ndarray,
    runtime_default_joint_pos: np.ndarray,
    runtime_default_joint_vel: np.ndarray | None = None,
    previous_raw_action: np.ndarray | None = None,
) -> np.ndarray:
    """按训练顺序组装 114 维观测 (全部 USD 顺序, float32)。"""
    ref_pos = np.asarray(reference_joint_pos, dtype=np.float32)
    ref_vel = np.asarray(reference_joint_vel, dtype=np.float32)
    q = np.asarray(robot_joint_pos, dtype=np.float32)
    qvel = np.asarray(robot_joint_vel, dtype=np.float32)
    default_q = np.asarray(runtime_default_joint_pos, dtype=np.float32)
    default_dq = (
        np.zeros_like(default_q)
        if runtime_default_joint_vel is None
        else np.asarray(runtime_default_joint_vel, dtype=np.float32)
    )
    prev = (
        np.zeros(21, dtype=np.float32)
        if previous_raw_action is None
        else np.clip(np.asarray(previous_raw_action, dtype=np.float32), -CLIP_ACT, CLIP_ACT)
    )

    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_b = _quat_apply_inverse_wxyz(np.asarray(imu_quat_wxyz, dtype=np.float64), gravity_w)

    obs = np.concatenate([
        np.asarray(robot_ang_vel_b, dtype=np.float32),   # 3
        np.asarray(robot_lin_vel_b, dtype=np.float32),   # 3 (实验A新增)
        gravity_b,                                       # 3
        ref_pos, ref_vel,                                # 42
        q - default_q,                                   # 21
        qvel - default_dq,                               # 21
        prev,                                            # 21
    ]).astype(np.float32, copy=False)

    if obs.shape != (114,):
        raise ValueError(f"Invalid actor observation shape: {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("Invalid actor observation: contains NaN/Inf")
    return np.clip(obs, -CLIP_OBS, CLIP_OBS)


def compute_joint_target(
    reference_joint_pos: np.ndarray,
    raw_action: np.ndarray,
) -> np.ndarray:
    """q_des = 参考关节角 + scale * clip(action) (USD 顺序 21)。"""
    action = np.clip(np.asarray(raw_action, dtype=np.float32), -CLIP_ACT, CLIP_ACT)
    return np.asarray(reference_joint_pos, dtype=np.float32) + ACTION_SCALE * action

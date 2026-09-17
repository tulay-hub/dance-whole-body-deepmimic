"""Lens110 21 关节残差策略的 111 维观测构建器 (与 Isaac 训练完全一致)。

观测顺序 (111 维单帧, 与训练 ObservationsCfg 一致):
  [0:3]    base_ang_vel      机身角速度 (body 系, rad/s)
  [3:6]    gravity_b         世界重力在 body 系投影 = R^T @ [0,0,-1]
  [6:27]   ref_joint_pos     参考关节角 (21, USD 顺序, rad)
  [27:48]  ref_joint_vel     参考关节速度 (21, USD 顺序, rad/s)
  [48:69]  joint_pos_rel     关节角 - runtime_default (21, USD 顺序)
  [69:90]  joint_vel_rel     关节速度 - default_vel(0) (21, USD 顺序)
  [90:111] last_action       上一策略动作 (已裁剪到 [-1,1], 21)

执行语义 (21 维残差动作):
  q_des = ref_joint_pos + ACTION_SCALE * clip(action, -1, 1)

注意:
  - runtime_default_joint_pos = 参考动作第 0 帧关节角 (USD 顺序)
  - 训练观测裁剪 clip_observations = 5.0; 动作裁剪 clip_actions = 1.0
  - ONNX 已烘焙 obs clip + running_mean_std 归一化, 输入原始观测即可
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

# 逐关节 action_scale (USD 顺序), 与训练 ActionsCfg 一致
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
    robot_joint_pos: np.ndarray,
    robot_joint_vel: np.ndarray,
    runtime_default_joint_pos: np.ndarray,
    runtime_default_joint_vel: np.ndarray | None = None,
    previous_raw_action: np.ndarray | None = None,
) -> np.ndarray:
    """按训练顺序组装 111 维观测 (全部 USD 顺序, float32)。"""
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
        gravity_b,                                       # 3
        ref_pos, ref_vel,                                # 42
        q - default_q,                                   # 21
        qvel - default_dq,                               # 21
        prev,                                            # 21
    ]).astype(np.float32, copy=False)

    if obs.shape != (111,):
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

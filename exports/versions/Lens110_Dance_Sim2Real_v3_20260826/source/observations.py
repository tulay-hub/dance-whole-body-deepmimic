"""Observation functions for the Lens110 motion-tracking environment.

本模块提供观测项:
  - motion_anchor_ori_b: 参考 anchor 相对机器人 anchor 的朝向 (body 系)
  - official_state_obs:  官方式 45 维/帧状态 (13q + 13dq + 13prev + 3ω + 3姿态)
  - official_future_ref: 精简 81 维前瞻参考 (k/k+10/k+20 x (21关节+6root))

所有参考数据由本工程 lens110_lab 的 MotionCommand 提供 (100Hz npz),
本模块不再直接读 npy。
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from .commands import MotionCommand

# 官方 (infer_zero 实机) 关节顺序 = CSV 列顺序:
#   12 腿 (左 6 + 右 6) + 1 腰 (torso_yaw) + 8 手臂
CSV_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]
_JOINT_MAP_CACHE: dict = {}


def _get_joint_maps(env: ManagerBasedRLEnv) -> tuple[list[int], list[int]]:
    """返回 (csv_to_usd, usd_to_csv) 关节索引映射 (按名字, 模块级缓存)。"""
    robot = env.scene["robot"]
    usd_names = tuple(robot.joint_names)
    if usd_names not in _JOINT_MAP_CACHE:
        csv_to_usd = [usd_names.index(n) for n in CSV_JOINT_ORDER]
        usd_to_csv = [csv_to_usd.index(i) for i in range(len(csv_to_usd))]
        _JOINT_MAP_CACHE[usd_names] = (csv_to_usd, usd_to_csv)
    return _JOINT_MAP_CACHE[usd_names]


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """参考 anchor (骨盆) 相对机器人 anchor 的朝向 (body 系, 取旋转矩阵前两列)。

    从 whole_body_tracking 移植: 让策略知道"参考躯干朝向相对我该转多少"。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def gravity_b_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """世界系重力在 body 系投影 (3 维), 与实机 IMU 同款。

    gravity_b = R^T @ [0,0,-1], 含 roll/pitch 信息, 不含 yaw。
    """
    robot = env.scene["robot"]
    gravity_w = robot.data.GRAVITY_VEC_W
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.unsqueeze(0)
    return quat_apply_inverse(robot.data.root_quat_w, gravity_w.expand(env.num_envs, -1))


def official_state_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """官方 dance 单帧状态观测 (45 维): 13q + 13dq + 13prev + 3角速度 + 3姿态。

    对齐 infer_zero 实机 45 维/帧结构:
      [0:13]  12腿q + 1腰q   (CSV 关节顺序)
      [13:26] 12腿dq + 1腰dq
      [26:39] 12腿prev_action + 1腰prev_action
      [39:42] base 角速度 (body 系)
      [42:45] 世界系重力在 body 系投影 gravity_b (实机 IMU 同款,
              含 roll/pitch 信息, 不含 yaw)
    当前配置 history_length=1 (不进历史); 若按官方 history_length=15
    则拼接成 675 维。
    """
    robot = env.scene["robot"]
    _, usd_to_csv = _get_joint_maps(env)

    jp = robot.data.joint_pos[:, usd_to_csv][:, :13]
    jv = robot.data.joint_vel[:, usd_to_csv][:, :13]
    # 官方模式动作本身就是 13 维 (CSV 顺序: 12腿 + 腰), 无需重排
    act = env.action_manager.action
    ang_vel = robot.data.root_ang_vel_b
    # 姿态: 世界系重力转到 body 系 (3 维, 实机 IMU 同款, 含 roll/pitch, 不含 yaw)
    gravity_w = robot.data.GRAVITY_VEC_W
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.unsqueeze(0)
    att = quat_apply_inverse(robot.data.root_quat_w, gravity_w.expand(env.num_envs, -1))
    return torch.cat([jp, jv, act, ang_vel, att], dim=-1)


def official_future_ref(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """精简 81 维前瞻参考: 3 帧 (k/k+10/k+20) x (21 关节 + 6 root)。

    布局:
      [0:63]   k, k+10, k+20 三帧的 21 关节 (CSV 顺序)
      [63:81]  k, k+10, k+20 三帧的 6 root (euler3 + pos3)

    相比官方 96 维删去了每帧的 quat4 (与 euler 冗余) 和 linvel_x
    (可由 k/k+10/k+20 的 pos 差分近似), 不再是官方 771 结构。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    csv_to_usd, _ = _get_joint_maps(env)
    usd_to_csv = [csv_to_usd.index(i) for i in range(len(csv_to_usd))]

    offsets = torch.tensor([0, 10, 20], device=env.device)
    frames = (command.time_steps.unsqueeze(1) + offsets).clamp(0, motion.joint_pos.shape[0] - 1)  # (N,3)

    jp = motion.joint_pos[frames][:, :, usd_to_csv]  # (N,3,21) CSV 顺序
    aidx = command.motion_anchor_body_index
    pos = motion.body_pos_w[frames, aidx]      # (N,3,3)
    quat = motion.body_quat_w[frames, aidx]    # (N,3,4) 仅用于算 euler
    quat_flat = quat.reshape(-1, 4)
    euler_flat = torch.stack(euler_xyz_from_quat(quat_flat), dim=-1)  # (N*3,3)
    euler = euler_flat.reshape(env.num_envs, 3, 3)
    root = torch.cat([euler, pos], dim=-1)  # (N,3,6) euler3 + pos3

    joints_flat = jp.reshape(env.num_envs, -1)   # 63: 3帧关节
    root_flat = root.reshape(env.num_envs, -1)   # 18: 3帧root
    return torch.cat([joints_flat, root_flat], dim=-1)  # (N,81)

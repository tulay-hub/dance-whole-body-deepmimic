"""Lens110 官方 dance 观测构造 (126 维), 与训练环境完全一致。

  126 = 45 state + 81 future

state (45):
  [0:13]   12腿 q + 腰 q           (CSV 关节顺序, 前 13)
  [13:26]  12腿 dq + 腰 dq
  [26:39]  12腿 prev_action + 腰 prev_action (原始 action, 未缩放)
  [39:42]  base 角速度 (body 系, 与 root_ang_vel_b 一致)
  [42:45]  gravity_b = R^T @ [0,0,-1] (world->body, 含 roll/pitch, 不含 yaw)

future (81):
  3 帧 (k, k+10, k+20) x (21 关节 CSV 顺序 + 6 root euler+pos) = 81

归一化已烘焙进 ONNX (export_onnx.py), 本函数只输出原始 126 维。
"""

import numpy as np


def quat_apply_inverse(q, v):
    """R^T @ v (wxyz), 与 isaaclab quat_apply_inverse 一致。"""
    w, x, y, z = q
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def euler_xyz_from_quat(q):
    """wxyz -> XYZ 欧拉角 (roll, pitch, yaw), 与训练 official_future_ref 一致。"""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sin_pitch) >= 1.0, np.sign(sin_pitch) * np.pi / 2.0, np.arcsin(sin_pitch))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def build_observation(
    q_csv, qvel_csv, last_action, root_quat_wxyz, base_ang_vel_body,
    ref_pos_usd, ref_quat, ref_pos_w, frame, n_frames,
    USD_TO_CSV, CSV_TO_USD, clip_obs=50.0,
):
    """构造 126 维观测。

    Args:
        q_csv: (21,) MuJoCo qpos, CSV 关节顺序
        qvel_csv: (21,) MuJoCo qvel, CSV 关节顺序
        last_action: (13,) 上一步策略输出 (原始 action)
        root_quat_wxyz: (4,) root 四元数
        base_ang_vel_body: (3,) body 系角速度 (MuJoCo qvel[3:6] 直接用)
        ref_pos_usd: (N,21) 参考关节角 USD 顺序
        ref_quat: (N,4) 参考 root 四元数
        ref_pos_w: (N,3) 参考 root 世界位置
        frame: 当前参考帧
        n_frames: 参考总帧数
        USD_TO_CSV / CSV_TO_USD: 关节顺序映射
        clip_obs: 观测裁剪范围 (训练 clip_observations=50)
    """
    q_usd = q_csv[USD_TO_CSV]          # (21,) USD 顺序
    qvel_usd = qvel_csv[USD_TO_CSV]
    q_train = q_usd[USD_TO_CSV][:13]   # 与 robot.data.joint_pos[:, usd_to_csv][:13] 一致
    qvel_train = qvel_usd[USD_TO_CSV][:13]

    gravity_b = quat_apply_inverse(root_quat_wxyz, np.array([0.0, 0.0, -1.0]))
    state = np.concatenate([
        q_train.astype(np.float32),
        qvel_train.astype(np.float32),
        last_action,
        base_ang_vel_body.astype(np.float32),
        gravity_b.astype(np.float32),
    ])

    offsets = np.array([0, 10, 20])
    frames = np.clip(frame + offsets, 0, n_frames - 1)
    joints = ref_pos_usd[frames][:, USD_TO_CSV]   # (3,21) CSV 顺序
    roots = np.stack([
        np.concatenate([euler_xyz_from_quat(ref_quat[f]), ref_pos_w[f]])
        for f in frames
    ])                                            # (3,6)
    future = np.concatenate([joints.reshape(-1), roots.reshape(-1)]).astype(np.float32)

    obs = np.clip(np.concatenate([state, future]), -clip_obs, clip_obs)
    assert obs.shape[0] == 126, obs.shape
    return obs

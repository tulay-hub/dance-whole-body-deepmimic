"""Lens110 舞蹈动作跟踪奖励函数 (移植AMP 全身跟踪奖励)。

本文件的 6 个 ``motion_*`` 奖励函数从 whole_body_tracking (T800 工程) 移植过来,
与 legged_lab 同款机器人舞蹈策略使用的实现一致。已删除旧的关节角模仿奖励
(joint_position_error_exp / joint_velocity_penalty / joint_acceleration_penalty /
feet_contact / joint_velocity_error_exp)。

所有 ``motion_*`` 项都是 exp(-误差/std²) 的核函数奖励, 范围 (0, 1]:
误差为 0 时奖励=1, 误差越大越接近 0。std 越大, 同样的误差惩罚越轻。

依赖: 动作参考由本工程 lens110_lab 的 MotionCommand 提供
(``env.command_manager.get_term("motion")``), 它输出 anchor/body 的位置、朝向、
线/角速度参考, 并与机器人实测值比较。该 MotionCommand 为从
whole_body_tracking 移植后本地化的实现, 本工程不再依赖外部包。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import MotionCommand


def _get_body_indexes(command: MotionCommand, body_names: Sequence[str] | None) -> list[int] | slice:
    """把 body 名字解析成 MotionCommand 里的 body 索引; 为 None 时用全部跟踪 body。"""
    return command.get_body_indexes(body_names)


def _get_adaptive_sigma(env: ManagerBasedRLEnv, key: str | float, error: torch.Tensor):
    """自适应 sigma (2026-08-25, 从 booster_train 移植)。

    - 传入 float 时保持固定 std (原行为);
    - 传入 str 时启用自适应: 用误差的 EMA + 历史最小值生成当前 std。
      训练前期误差大 -> std 自动放宽 (奖励不塌缩成 0);
      后期误差小 -> std 自动收紧 (奖励保持可分辨), 无需人工调 std。

    sigma_update_rate=0.9 对应 EMA 时间常数约 10 个策略步;
    历史最小值保证 std 单调不回升, 奖励随训练逐渐"变严"。
    """
    if isinstance(key, float):
        return key
    sigma_update_rate = 0.9
    if not hasattr(env, "reward_sigmas_ema"):
        env.reward_sigmas_ema = {}
        env.reward_sigmas = {}
    env.reward_sigmas_ema[key] = (
        sigma_update_rate * env.reward_sigmas_ema.get(key, torch.tensor([100.0], device=env.device))
        + (1.0 - sigma_update_rate) * error
    )
    env.reward_sigmas[key] = torch.minimum(
        env.reward_sigmas_ema[key],
        env.reward_sigmas.get(key, torch.tensor([100.0], device=env.device)),
    ).clip(min=1e-8)
    return torch.sqrt(env.reward_sigmas[key])


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """正向奖励: 骨盆(锚点) 在世界系的位置跟踪误差。

    error = ||anchor_pos_w(动作参考) - robot_anchor_pos_w(机器人实际)||²
    reward = exp(-error / std²)

    - 锚点默认为 pelvis, 是全身动作的"根"。
    - 该项负责让机器人整体跟着动作走 (站位/位移/重心高度), std=0.3m。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_feet_world_pos_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float | str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> torch.Tensor:
    """正向奖励: 双脚在世界系的位置跟踪参考双脚位置。

    与 motion_body_pos (相对骨盆) 互补: 该项直接惩罚双脚整体平移 (走路),
    防止"骨盆带着脚一起走"而相对姿势不变的情况。std=0.1m 允许参考本身
    的自然小幅脚部动作 (~8cm), 但 10cm+ 的平移会明显掉分。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    asset_body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    ref_body_ids = _get_motion_body_ids(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, ref_body_ids] - asset.data.body_pos_w[:, asset_body_ids]), dim=-1
    )
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """正向奖励: 骨盆(锚点) 在世界系的朝向跟踪误差。

    error = quat_error_magnitude(动作锚点四元数, 机器人锚点四元数)²
    reward = exp(-error / std²)

    - 让躯干朝向与动作一致, 防止身体歪斜或整体转错方向, std=0.4。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: Sequence[str] | None = None
) -> torch.Tensor:
    """正向奖励: 各跟踪 body 相对锚点的位置跟踪误差。

    error = Σ_body ||body_pos(相对锚点, 动作) - body_pos(相对锚点, 机器人)||²
    reward = exp(-mean(error) / std²)

    - body_pos 用"相对锚点"坐标, 消除整体位移的干扰, 只比四肢相对姿势。
    - body_names 为空时跟踪 MotionCommand 配置里的全部 body (默认脚踝/肘/肩roll),
      是手脚姿势的主要驱动项, std=0.3m。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    per_env_error = error.mean(-1)
    std = _get_adaptive_sigma(env, std, per_env_error.mean())
    return torch.exp(-per_env_error / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: Sequence[str] | None = None
) -> torch.Tensor:
    """正向奖励: 各跟踪 body 相对锚点的朝向跟踪误差。

    error = Σ_body quat_error_magnitude(body_quat(动作), body_quat(机器人))²
    reward = exp(-mean(error) / std²)

    - 让手/脚的朝向也对齐动作, 避免手腕/脚踝姿态错位, std=0.4。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    per_env_error = error.mean(-1)
    std = _get_adaptive_sigma(env, std, per_env_error.mean())
    return torch.exp(-per_env_error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: Sequence[str] | None = None
) -> torch.Tensor:
    """正向奖励: 各 body 世界系线速度跟踪误差。

    error = Σ_body ||body_lin_vel(动作) - body_lin_vel(机器人)||²
    reward = exp(-mean(error) / std²)

    - 让肢体移动速度跟上动作, 减少"位置对但动作迟缓"的拖影, std=1.0 m/s。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    per_env_error = error.mean(-1)
    std = _get_adaptive_sigma(env, std, per_env_error.mean())
    return torch.exp(-per_env_error / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: Sequence[str] | None = None
) -> torch.Tensor:
    """正向奖励: 各 body 世界系角速度跟踪误差。

    error = Σ_body ||body_ang_vel(动作) - body_ang_vel(机器人)||²
    reward = exp(-mean(error) / std²)

    - 让肢体转动速度跟上动作, 与线速度项配合保证动态一致性, std=3.14 rad/s。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    per_env_error = error.mean(-1)
    std = _get_adaptive_sigma(env, std, per_env_error.mean())
    return torch.exp(-per_env_error / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float | str,
    exclude_ankles: bool = False,
) -> torch.Tensor:
    """正向奖励: 关节角位置跟踪误差。

    error = mean((动作关节角 - 机器人关节角)²)
    reward = exp(-error / std²)

    - 直接奖励 21 个关节角与动作一致, 是"动作像不像"的最直接信号。
    - body 级奖励 (位置/朝向) 允许关节角有偏差但 body 位置不变; 该项补上
      关节级精度, 让四肢角度也贴合动作。std 建议 0.5~0.8 rad。
    - exclude_ankles=True 时排除 4 个踝关节 (USD 索引 15/16/19/20):
      脚部要求钉死不动时, 不能再让踝关节去追参考角。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if exclude_ankles:
        keep = [i for i in range(command.joint_pos.shape[-1]) if i not in (15, 16, 19, 20)]
        error = torch.mean(torch.square(command.joint_pos[:, keep] - command.robot_joint_pos[:, keep]), dim=-1)
    else:
        error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """正向奖励: 关节角速度跟踪误差。

    error = mean((动作关节角速度 - 机器人关节角速度)²)
    reward = exp(-error / std²)

    - 位置奖励让姿态"像", 速度奖励让动作节奏也"像", 两者互补。
    - std 建议 1.0~2.0 rad/s (关节速度量级明显大于位置)。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)


# ---------------------------------------------------------------------------
# 脚部接触辅助 (从 T800 whole_body_tracking 移植)
# ---------------------------------------------------------------------------


def _resolve_body_ids(entity, entity_cfg: SceneEntityCfg, body_names: Sequence[str] | None = None):
    if entity_cfg.body_ids != slice(None):
        return entity_cfg.body_ids
    if body_names is None:
        return entity_cfg.body_ids
    cache = getattr(entity, "_wbt_body_id_cache", None)
    if cache is None:
        cache = {}
        setattr(entity, "_wbt_body_id_cache", cache)
    key = tuple(body_names)
    if key not in cache:
        cache[key] = entity.find_bodies(list(body_names), preserve_order=True)[0]
    return cache[key]


def _get_motion_body_ids(command: MotionCommand, body_names: Sequence[str] | None) -> list[int] | slice:
    if body_names is None:
        return slice(None)
    return _get_body_indexes(command, body_names)


def _compute_ref_body_heights(
    env: ManagerBasedRLEnv, command: MotionCommand, body_indexes: list[int] | slice
) -> torch.Tensor:
    ref_positions = command.body_pos_w[:, body_indexes]
    return ref_positions[..., 2] - env.scene.env_origins[:, 2:3]


def robot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_names: Sequence[str],
    contact_time_threshold: float = 1.0e-4,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_body_ids = _resolve_body_ids(contact_sensor, sensor_cfg, body_names)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_body_ids]
    return (contact_time > contact_time_threshold).to(torch.int32)


def ref_contact_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: Sequence[str],
    height_threshold: float = 0.02,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_motion_body_ids(command, body_names)
    if isinstance(body_indexes, slice) or len(body_indexes) != len(body_names):
        raise RuntimeError(f"Unable to resolve all body names in motion command: {body_names}")

    ref_heights = _compute_ref_body_heights(env, command, body_indexes)
    if ref_heights.shape[1] != 2:
        raise ValueError("ref_contact_state expects exactly two foot body names (left, right).")

    left_h = ref_heights[:, 0]
    right_h = ref_heights[:, 1]
    height_diff = torch.abs(left_h - right_h)
    in_contact = torch.ones_like(ref_heights, dtype=torch.int32)
    higher_is_left = left_h > right_h
    diff_mask = height_diff > height_threshold
    in_contact[diff_mask & higher_is_left, 0] = 0
    in_contact[diff_mask & ~higher_is_left, 1] = 0
    return in_contact


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    command_name: str | None = None,
    ref_speed_threshold: float = 0.15,
) -> torch.Tensor:
    """惩罚脚掌在地面滑动 (条件化): 仅当参考脚"应该静止"时惩罚机器人脚滑动。

    原来的实现对接触中的脚一律按速度惩罚, 但参考动作本身有约 10% 的帧
    脚在水平移动 (贴地舞蹈, 峰值 0.79 m/s)。这些帧里"跟参考"必然滑动,
    与 motion_feet_pos 的跟位要求冲突, 策略怎么做都是错。

    现在改为: 参考脚线速度 < ref_speed_threshold 时才惩罚机器人脚滑动,
    参考脚本来就在移动时不再惩罚, 消除"跟位 vs 防滑"的矛盾。
    """
    if len(body_names) != 2:
        raise ValueError("feet_slip expects exactly two foot body names (left, right).")

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_body_ids = _resolve_body_ids(contact_sensor, sensor_cfg, body_names)
    asset_body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    if len(sensor_body_ids) != 2 or len(asset_body_ids) != 2:
        raise RuntimeError(f"Unable to resolve feet_slip body names: {list(body_names)}")

    contacts = torch.norm(contact_sensor.data.net_forces_w[:, sensor_body_ids, :], dim=-1) > 1.0
    foot_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_body_ids, :3], dim=-1)
    penalty = contacts.to(foot_speed.dtype) * foot_speed

    if command_name is not None:
        # 参考脚速度 (MotionLoader 原始世界系线速度, 与 body_names 索引对应)
        command: MotionCommand = env.command_manager.get_term(command_name)
        ref_body_ids = _get_motion_body_ids(command, body_names)
        ref_foot_speed = torch.norm(
            command.motion.body_lin_vel_w[command.time_steps][:, ref_body_ids, :3], dim=-1
        )
        # 只有参考脚"应该静止"时才保留滑动惩罚
        ref_static = (ref_foot_speed < ref_speed_threshold).to(foot_speed.dtype)
        penalty = penalty * ref_static

    return torch.sum(penalty, dim=1)


def feet_still_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    command_name: str = "motion",
    ref_speed_threshold: float = 0.15,
    std: float = 0.05,
) -> torch.Tensor:
    """正向奖励: 参考脚静止时, 机器人接触中的脚保持不动 (防滑奖励)。

    与条件化 feet_slip 互补:
      - 参考脚静止 (速度 < ref_speed_threshold): 机器人脚接触地面且速度低
        则给 exp(-speed²/std²) 奖励, 鼓励"压住脚不乱动";
      - 参考脚移动: 本项返回 0, 不产生任何惩罚/奖励,
        跟位完全交给 motion_feet_pos, 消除"跟位 vs 防滑"的奖励冲突。
    """
    if len(body_names) != 2:
        raise ValueError("feet_still_reward expects exactly two foot body names (left, right).")

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_body_ids = _resolve_body_ids(contact_sensor, sensor_cfg, body_names)
    asset_body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    if len(sensor_body_ids) != 2 or len(asset_body_ids) != 2:
        raise RuntimeError(f"Unable to resolve feet_still body names: {list(body_names)}")

    contacts = torch.norm(contact_sensor.data.net_forces_w[:, sensor_body_ids, :], dim=-1) > 1.0
    foot_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_body_ids, :3], dim=-1)

    # 参考脚速度 (MotionLoader 原始世界系线速度, 与 body_names 索引对应)
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_body_ids = _get_motion_body_ids(command, body_names)
    ref_foot_speed = torch.norm(
        command.motion.body_lin_vel_w[command.time_steps][:, ref_body_ids, :3], dim=-1
    )
    ref_static = (ref_foot_speed < ref_speed_threshold).to(foot_speed.dtype)

    # 接触中且脚速低 → 奖励; 参考移动帧整体置 0
    still_reward = contacts.to(foot_speed.dtype) * torch.exp(-torch.square(foot_speed) / std**2)
    return torch.sum(still_reward * ref_static, dim=1)


def _foot_anchor_state(
    env: ManagerBasedRLEnv,
    pos: torch.Tensor,
    quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """记录每个环境在 episode 开始时的脚底世界位姿锚点 (2026-08-26)。

    - 锚点只在 episode 第 0/1 步 (episode_length_buf <= 1) 记录一次,
      之后整个 episode 不再更新; env 重置后自动重新记录。
    - 约束对象是脚底 (ankle_roll_link) 的世界位置/朝向, 与踝关节角度无关,
      因此踝关节电机可以自由运动来维持脚底位姿。
    """
    if not hasattr(env, "_foot_anchor_pos") or env._foot_anchor_pos is None:
        env._foot_anchor_pos = pos.clone()
        env._foot_anchor_quat = quat.clone()
    ep = env.episode_length_buf
    record = (ep <= 1).unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
    env._foot_anchor_pos = torch.where(record, pos, env._foot_anchor_pos)
    env._foot_anchor_quat = torch.where(record, quat, env._foot_anchor_quat)
    return env._foot_anchor_pos, env._foot_anchor_quat


def foot_anchor_xy_lock_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    sigma_xy: float = 0.01,
) -> torch.Tensor:
    """脚底世界 x/y 位置锁定奖励 (双脚独立, 无 contact gating)。

    reward = Σ_脚 exp(-||foot_xy - anchor_xy||² / sigma_xy²)

    anchor 在 episode 开始 (第 0 帧) 记录, 整个 episode 固定。
    不惩罚为保持脚底位置而发生的髋/膝/踝关节运动。
    """
    if len(body_names) != 2:
        raise ValueError("foot_anchor_xy_lock_reward expects exactly two foot body names.")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    pos = asset.data.body_pos_w[:, body_ids]
    quat = asset.data.body_quat_w[:, body_ids]
    anchor_pos, _ = _foot_anchor_state(env, pos, quat)
    err = torch.sum(torch.square(pos[..., :2] - anchor_pos[..., :2]), dim=-1)
    return torch.exp(-err / sigma_xy**2).sum(dim=-1)


def foot_anchor_z_lock_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    sigma_z: float = 0.01,
) -> torch.Tensor:
    """脚底世界 z (高度) 锁定奖励 (双脚独立, 不允许离地/下沉)。

    reward = Σ_脚 exp(-(foot_z - anchor_z)² / sigma_z²)
    """
    if len(body_names) != 2:
        raise ValueError("foot_anchor_z_lock_reward expects exactly two foot body names.")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    pos = asset.data.body_pos_w[:, body_ids]
    quat = asset.data.body_quat_w[:, body_ids]
    anchor_pos, _ = _foot_anchor_state(env, pos, quat)
    err = torch.square(pos[..., 2] - anchor_pos[..., 2])
    return torch.exp(-err / sigma_z**2).sum(dim=-1)


def foot_anchor_ori_lock_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    sigma_ori: float = 0.03,
) -> torch.Tensor:
    """脚底朝向锁定奖励 (双脚独立, roll/pitch 保持水平, 不允许翻转)。

    reward = Σ_脚 exp(-quat_error² / sigma_ori²)

    anchor 是 episode 开始的脚底四元数 (wxyz); 踝关节 pitch/roll 可以
    自由调节, 只要最终脚底姿态不变。
    """
    if len(body_names) != 2:
        raise ValueError("foot_anchor_ori_lock_reward expects exactly two foot body names.")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    pos = asset.data.body_pos_w[:, body_ids]
    quat = asset.data.body_quat_w[:, body_ids]
    _, anchor_quat = _foot_anchor_state(env, pos, quat)
    err = quat_error_magnitude(quat, anchor_quat) ** 2
    return torch.exp(-err / sigma_ori**2).sum(dim=-1)


def motion_leg_joint_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """正向奖励: 12 个腿部关节角位置跟踪误差 (专项, 不被手臂稀释)。

    左/右 髋p/r/y + 膝 + 踝p/r 都正常跟踪 (电机可以动)。
    脚掌贴地不动由 foot_anchor_xy/z/ori 负责, 不是靠锁死踝关节。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 12 腿关节在 USD 顺序中的索引
    leg_usd_idx = [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
    error = torch.mean(
        torch.square(command.joint_pos[:, leg_usd_idx] - command.robot_joint_pos[:, leg_usd_idx]), dim=-1
    )
    return torch.exp(-error / std**2)


def foot_parallel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    tilt_threshold: float = 0.02,
) -> torch.Tensor:
    """惩罚接触中的脚底与地面不平行 (脚体局部 z 轴偏离竖直超过阈值)。

    只在脚掌接触地面 (接触力 > 1N) 时生效; 倾角 = 脚体局部 z 与世界竖直的夹角,
    超过 tilt_threshold (rad, 默认 0.02≈1.1°) 的部分按超出角度的平方惩罚
    (越翘罚越重)。左右脚分别计算后求和。
    解决策略播放时"撬脚/脚底不贴平": 参考动作已把脚底校平,
    该惩罚让策略在触地时把脚底压平, 而不是靠踝关节乱翘。
    """
    if len(body_names) != 2:
        raise ValueError("foot_parallel_penalty expects exactly two foot body names (left, right).")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_body_ids = _resolve_body_ids(contact_sensor, sensor_cfg, body_names)
    asset_body_ids = _resolve_body_ids(asset, asset_cfg, body_names)
    if len(sensor_body_ids) != 2 or len(asset_body_ids) != 2:
        raise RuntimeError(f"Unable to resolve foot_parallel body names: {list(body_names)}")

    contacts = torch.norm(contact_sensor.data.net_forces_w[:, sensor_body_ids, :], dim=-1) > 1.0
    quat = asset.data.body_quat_w[:, asset_body_ids]  # (N,2,4) wxyz
    z_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(quat[..., :3])
    z_world = quat_apply(quat, z_up)  # (N,2,3)
    tilt = torch.acos(torch.clamp(z_world[..., 2], -1.0, 1.0))  # 脚底倾角 rad
    exceed = torch.clamp(tilt - tilt_threshold, min=0.0)
    return torch.sum(contacts.to(exceed.dtype) * torch.square(exceed), dim=1)


def foot_contact_mismatch(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: Sequence[str],
    height_threshold: float,
    contact_time_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """惩罚脚掌接触状态与参考不一致 (参考脚着地但机器人脚离地, 反之亦然)。"""
    if len(body_names) != 2:
        raise ValueError("foot_contact_mismatch expects exactly two foot body names (left, right).")
    robot_state = robot_contact_state(env, sensor_cfg, body_names, contact_time_threshold)
    ref_state = ref_contact_state(env, command_name, body_names, height_threshold)
    return (robot_state != ref_state).sum(dim=1).to(torch.float32)


def feet_levelness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    transition_end_frame: int = 150,
    window: int = 3,
) -> torch.Tensor:
    """惩罚前置过渡段"结尾帧"双脚前后不水平。

    只在 episode 走到过渡段末尾 (第 transition_end_frame 步附近 window 帧) 生效:
    取左右脚踝在 body 局部坐标系下的前后 (y) 向差的平方, 差越大惩罚越重。
    过渡段中间的脚怎么动不惩罚, 只约束"接舞蹈那一刻"两脚前后对齐。
    """
    if len(body_names) != 2:
        raise ValueError("feet_levelness expects exactly two foot body names (left, right).")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)

    # 只在过渡段结尾帧附近启用
    step = env.episode_length_buf
    mask = ((step >= transition_end_frame - window) & (step <= transition_end_frame)).float()

    # 左右脚相对骨盆在 body 局部系的 y 差
    from isaaclab.utils.math import quat_apply_inverse
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, body_ids]
    feet_pos_b = quat_apply_inverse(root_quat, feet_pos_w - root_pos)
    ydiff = feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1]
    return mask * torch.square(ydiff)


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖励: 未终止时 +1, 鼓励不倒/不提前结束。"""
    return (~env.termination_manager.terminated).float()


def early_termination_penalty(env: ManagerBasedRLEnv, max_penalty: float = 100.0) -> torch.Tensor:
    """早倒惩罚: 非 time_out 终止(倒地)时, 剩余步数比例越大罚越重。

    penalty = -max_penalty * (剩余步数比例) * terminated
      - 刚开局就倒: 剩余比例≈1, 罚 -max_penalty
      - 快跑完才倒: 剩余比例≈0, 罚≈0
      - 正常跑满 (time_out): 不罚

    作用: alive 只奖励"活着", 而本项显式惩罚"倒得早"——
    让策略不仅避免终止, 更优先避免早期倒地 (起步不稳/开头动作)
    丢掉的整段后续奖励。
    """
    terminated = env.termination_manager.terminated
    if not terminated.any():
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    elapsed = env.episode_length_buf.float()
    max_len = float(env.max_episode_length)
    remaining_frac = (max_len - elapsed) / max_len
    return (-max_penalty * remaining_frac * terminated.float()).float()


def motion_anchor_height_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """正向奖励: 骨盆高度跟踪参考动作。

    error = (参考骨盆 z - 机器人骨盆 z)²
    reward = exp(-error / std²)

    - 防倒地: 骨盆塌下去/跳起来都会偏离参考高度, 被惩罚。
    - 防整体漂移: 骨盆高度是"机器人整体位置"的一部分, 跟住参考就不会
      越走越偏。
    - 用动态参考高度而非固定值, 不影响舞蹈本身的骨盆起伏。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[..., 2] - command.robot_anchor_pos_w[..., 2])
    return torch.exp(-error / std**2)


def feet_under_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    max_dist: float = 0.35,
) -> torch.Tensor:
    """惩罚脚踝相对骨盆的水平距离过大 (脚部向外漂移)。

    计算左右脚踝在 body 局部系下的水平 (xy) 距骨盆的距离,
    超过 max_dist 的部分平方惩罚。脚越"走出去"罚越重,
    约束机器人整体不向外漂移。
    """
    if len(body_names) != 2:
        raise ValueError("feet_under_body expects exactly two foot body names (left, right).")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)

    from isaaclab.utils.math import quat_apply_inverse
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, body_ids]
    feet_pos_b = quat_apply_inverse(root_quat, feet_pos_w - root_pos)
    horiz = torch.norm(feet_pos_b[..., :2], dim=-1)  # (N, 2) 脚踝水平距骨盆
    exceed = torch.clamp(horiz - max_dist, min=0.0)
    return torch.sum(torch.square(exceed), dim=-1)


def feet_gap_mismatch(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> torch.Tensor:
    """惩罚机器人双脚水平间距与参考动作的偏差。

    计算左右脚踝水平 (xy) 间距:
      gap = ||左脚_xy - 右脚_xy||
    惩罚 (机器人gap - 参考gap)², 双脚间距越偏离参考罚越重。

    解决"双脚间距随跳舞时长越变越大": 参考间距约 45cm 基本稳定,
    机器人间距被拉回同一水平。
    """
    if len(body_names) != 2:
        raise ValueError("feet_gap_mismatch expects exactly two foot body names (left, right).")
    asset = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg, body_names)

    # 机器人双脚水平间距
    feet_w = asset.data.body_pos_w[:, body_ids, :2]
    robot_gap = torch.norm(feet_w[:, 0] - feet_w[:, 1], dim=-1)

    # 参考双脚水平间距 (MotionCommand body_pos_w, 与 body_names 索引对应)
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_body_ids = _get_motion_body_ids(command, body_names)
    ref_feet_w = command.body_pos_w[:, ref_body_ids, :2]
    ref_gap = torch.norm(ref_feet_w[:, 0] - ref_feet_w[:, 1], dim=-1)

    return torch.square(robot_gap - ref_gap)


def feet_stance_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    desired_time: float = 0.25,
    std: float = 0.08,
) -> torch.Tensor:
    """正奖励 (2026-08-25, 从 booster_train 移植): 脚站稳足够久才抬。

    reward = sum_脚 exp(-max(0, desired_time - contact_time)² / std²)
    脚连续接触地面 >= desired_time 给满奖励; 接触越短奖励越低。
    鼓励"站够再动", 比惩罚"脚动"更平滑, 不会让策略僵住。
    """
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact.data.current_contact_time
    deficit = (desired_time - contact_time).clamp(min=0.0)
    return torch.exp(-torch.square(deficit) / std**2).sum(dim=-1)

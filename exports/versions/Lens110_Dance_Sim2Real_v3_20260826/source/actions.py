"""Lens110 动作项 (从 T800 whole_body_tracking 移植, 已本地化到 lens110_lab)。

本模块提供:
  - ResidualRefJointPositionAction: 残差参考动作 (全身 21 维模式)
  - OfficialDanceJointAction:      官方 dance 动作 (13 维: 12腿+1腰, 手臂开环)

动作参考由本工程 MotionCommand 提供, 不依赖外部 whole_body_tracking 包。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .commands import MotionCommand
from .observations import _get_joint_maps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualRefJointPositionAction(JointAction):
    """残差参考动作: q_des = 参考关节角 + scale * action。

    从 T800 whole_body_tracking 移植。策略只需输出相对参考动作的残差,
    参考关节角由 MotionCommand 提供, 动作幅度天然贴合参考动作,
    避免策略学会"缩水"动作。
    """

    cfg: "ResidualRefJointPositionActionCfg"

    def __init__(self, cfg: "ResidualRefJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name
        # 与共享的启动随机化代码兼容
        self._offset = torch.zeros((env.num_envs, self.action_dim), dtype=torch.float32, device=env.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        command: MotionCommand = self._env.command_manager.get_term(self._command_name)
        target_joint_pos = command.joint_pos + self.processed_actions
        self._asset.set_joint_position_target(target_joint_pos, joint_ids=self._joint_ids)


@configclass
class ResidualRefJointPositionActionCfg(JointActionCfg):
    class_type: type = ResidualRefJointPositionAction
    command_name: str = "motion"


class OfficialDanceJointAction(ActionTerm):
    """官方 dance 动作: 13 维 (12腿 + 1腰), 手臂 8 关节开环参考。

    对齐 infer_zero 实机 ``computeJointDes``:
      q_des[腿 12] = action[0:12] * scale_i + default (CSV 关节顺序)
      q_des[腰 1]  = action[12] * scale_i
      q_des[臂 8]  = MotionCommand 参考关节角 (开环, 不经策略)

    动作维度 13, 顺序 = 官方 CSV 顺序 (12腿 + torso_yaw)。
    scale 支持逐关节 dict (2026-08-20): 按参考动作范围设计,
    避免统一 0.25 覆盖不到 knee/hip_yaw 或统一 0.6 浪费在 hip_roll/ankle_roll。
    """

    cfg: "OfficialDanceJointActionCfg"

    def __init__(self, cfg: "OfficialDanceJointActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        robot = self._asset
        csv_to_usd, _ = _get_joint_maps(env)
        self.leg_usd = torch.tensor(csv_to_usd[:12], dtype=torch.long, device=self.device)
        self.torso_usd = int(csv_to_usd[12])
        self.arm_usd = torch.tensor(csv_to_usd[13:21], dtype=torch.long, device=self.device)
        self._command_name = cfg.command_name
        # 逐关节 scale: CSV 13 关节顺序 (12腿 + 腰)
        csv_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_yaw_joint",
        ]
        if isinstance(cfg.scale, dict):
            self._scale_vec = torch.tensor(
                [cfg.scale.get(n, 0.25) for n in csv_names],
                dtype=torch.float32, device=self.device,
            )
        else:
            self._scale_vec = torch.full((13,), float(cfg.scale), device=self.device)
        self._raw = torch.zeros(env.num_envs, 13, device=self.device)
        self._proc = torch.zeros_like(self._raw)
        self._all_joint_ids = torch.arange(robot.num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 13

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._proc = self._raw * self._scale_vec
        if self.cfg.clip_range is not None:
            self._proc = torch.clamp(self._proc, min=self.cfg.clip_range[0], max=self.cfg.clip_range[1])

    def apply_actions(self):
        command: MotionCommand = self._env.command_manager.get_term(self._command_name)
        default = self._asset.data.default_joint_pos  # (N,21) USD 顺序
        q_des = default.clone()
        q_des[:, self.leg_usd] = self._proc[:, :12] + default[:, self.leg_usd]
        q_des[:, self.torso_usd] = self._proc[:, 12]  # 腰: 官方不加 default
        q_des[:, self.arm_usd] = command.joint_pos[:, self.arm_usd]  # 手臂开环参考
        self._asset.set_joint_position_target(q_des, joint_ids=self._all_joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._proc[env_ids] = 0.0


@configclass
class OfficialDanceJointActionCfg(ActionTermCfg):
    class_type: type = OfficialDanceJointAction
    command_name: str = "motion"
    scale: float | dict[str, float] = 0.25
    clip_range: tuple[float, float] | None = None

"""Lens110 动作参考命令 (MotionCommand, 从 whole_body_tracking 移植并本地化)。

本模块提供:
  - MotionLoader:   加载 100Hz npz 动作参考 (关节/body 位置、朝向、速度)
  - MotionCommand:  环境命令项, 驱动参考帧推进并输出 anchor/body/关节参考,
                    供奖励函数与观测使用
  - MotionCommandCfg: 对应配置类

该实现已移植到 lens110_lab, 本工程不依赖外部 whole_body_tracking 包。
"""

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)


def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    view_shape = [blend.shape[0]] + [1] * a.dim()
    alpha = blend.view(*view_shape)
    return a.unsqueeze(0) * (1.0 - alpha) + b.unsqueeze(0) * alpha


def _quat_slerp_batch(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = torch.sum(q0 * q1, dim=-1)
    q1 = torch.where(dot.unsqueeze(-1) < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1).clamp(-1.0, 1.0)

    linear_mask = torch.abs(dot) > 0.9995
    lerp = _normalize_quat(q0 * (1.0 - blend.unsqueeze(-1)) + q1 * blend.unsqueeze(-1))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    slerp = _normalize_quat(s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1)
    return torch.where(linear_mask.unsqueeze(-1), lerp, slerp)


def _bridge_quat_sequence(last_quat: torch.Tensor, first_quat: torch.Tensor, bridge_frames: int) -> torch.Tensor:
    if bridge_frames <= 0:
        return torch.empty((0,) + last_quat.shape, dtype=last_quat.dtype, device=last_quat.device)
    flat_last = last_quat.reshape(-1, 4)
    flat_first = first_quat.reshape(-1, 4)
    blend = torch.linspace(0.0, 1.0, bridge_frames + 2, device=last_quat.device, dtype=last_quat.dtype)[1:-1]
    bridge = []
    for tau in blend:
        tau_vec = torch.full((flat_last.shape[0],), tau, dtype=last_quat.dtype, device=last_quat.device)
        bridge.append(_quat_slerp_batch(flat_last, flat_first, tau_vec).reshape(last_quat.shape))
    return torch.stack(bridge, dim=0)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        anchor_body_index: int = 0,
        device: str = "cpu",
        min_traj_duration: float | None = None,
        bridge_frames: int = 20,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        fps = data["fps"]
        self.fps = float(np.asarray(fps).reshape(-1)[0])
        motion_tensors = {
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
            "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
            "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
            "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
            "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
        }
        if min_traj_duration is not None and min_traj_duration > 0.0:
            min_frames = int(min_traj_duration * self.fps) + 1
            motion_tensors = self._extend_short_trajectory(
                motion_tensors,
                min_frames=min_frames,
                bridge_frames=bridge_frames,
                source_file=motion_file,
            )
        self.joint_pos = motion_tensors["joint_pos"]
        self.joint_vel = motion_tensors["joint_vel"]
        body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        # Select the tracked bodies once. Repeating this advanced indexing in every
        # reward/observation call creates a fresh tensor and is measurably expensive.
        self.body_pos_w = motion_tensors["body_pos_w"].index_select(1, body_indexes).contiguous()
        self.body_quat_w = motion_tensors["body_quat_w"].index_select(1, body_indexes).contiguous()
        self.body_lin_vel_w = motion_tensors["body_lin_vel_w"].index_select(1, body_indexes).contiguous()
        self.body_ang_vel_w = motion_tensors["body_ang_vel_w"].index_select(1, body_indexes).contiguous()
        # Body positions are static motion data.  Keep the anchor-relative form so
        # heading alignment does not rebuild ``body_pos - anchor_pos`` every step.
        anchor_pos = self.body_pos_w[:, anchor_body_index : anchor_body_index + 1]
        self.body_pos_anchor_rel = (self.body_pos_w - anchor_pos).contiguous()
        self.time_step_total = self.joint_pos.shape[0]

    def _extend_short_trajectory(
        self,
        motion: dict[str, torch.Tensor],
        min_frames: int,
        bridge_frames: int,
        source_file: str,
    ) -> dict[str, torch.Tensor]:
        frame_count = motion["joint_pos"].shape[0]
        if frame_count >= min_frames:
            return motion

        first = {k: v[0] for k, v in motion.items()}
        last = {k: v[-1] for k, v in motion.items()}
        blend = torch.linspace(
            0.0, 1.0, bridge_frames + 2, device=motion["joint_pos"].device, dtype=motion["joint_pos"].dtype
        )[1:-1]

        bridge = {
            "joint_pos": _lerp_tensor(last["joint_pos"], first["joint_pos"], blend),
            "joint_vel": _lerp_tensor(last["joint_vel"], first["joint_vel"], blend),
            "body_pos_w": _lerp_tensor(last["body_pos_w"], first["body_pos_w"], blend),
            "body_quat_w": _bridge_quat_sequence(last["body_quat_w"], first["body_quat_w"], bridge_frames),
            "body_lin_vel_w": _lerp_tensor(last["body_lin_vel_w"], first["body_lin_vel_w"], blend),
            "body_ang_vel_w": _lerp_tensor(last["body_ang_vel_w"], first["body_ang_vel_w"], blend),
        }

        pieces = [{k: v for k, v in motion.items()}]
        total_frames = frame_count
        while total_frames < min_frames:
            if bridge_frames > 0:
                pieces.append(bridge)
                total_frames += bridge_frames
            pieces.append(motion)
            total_frames += frame_count

        result = {key: torch.cat([piece[key] for piece in pieces], dim=0) for key in motion.keys()}
        orig_duration = max(frame_count - 1, 0) / self.fps
        new_duration = max(result["joint_pos"].shape[0] - 1, 0) / self.fps
        print(
            f"[INFO] Extended short motion '{os.path.basename(source_file)}': "
            f"{frame_count} -> {result['joint_pos'].shape[0]} frames "
            f"({orig_duration:.2f}s -> {new_duration:.2f}s, bridge={bridge_frames})"
        )
        return result

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        motion_body_names = self.cfg.motion_body_names if self.cfg.motion_body_names is not None else self.cfg.body_names
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_body_indexes = torch.tensor(
            [motion_body_names.index(name) for name in self.cfg.body_names], dtype=torch.long, device=self.device
        )

        if self.cfg.motion_joint_names is not None:
            self.robot_joint_indexes = torch.tensor(
                self.robot.find_joints(self.cfg.motion_joint_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.robot_joint_indexes = None

        motion_files = self.cfg.motion_files
        if motion_files is None:
            motion_files = [self.cfg.motion_file]
        elif self.cfg.motion_file is not MISSING and self.cfg.motion_file is not None:
            raise ValueError("Set either motion_file or motion_files, not both.")
        if not motion_files or any(not os.path.isfile(path) for path in motion_files):
            raise ValueError(f"Invalid motion_files: {motion_files}")

        self.motions = [
            MotionLoader(
                path,
                self.motion_body_indexes,
                anchor_body_index=self.motion_anchor_body_index,
                device=self.device,
                min_traj_duration=self.cfg.min_traj_duration,
                bridge_frames=self.cfg.bridge_frames,
            )
            for path in motion_files
        ]
        self.motion = self.motions[0]
        self.motion_files = list(motion_files)
        self.motion_lengths = torch.tensor(
            [motion.time_step_total for motion in self.motions], dtype=torch.long, device=self.device
        )
        self.motion_sampling_probs = self._build_sampling_probs(self.cfg.motion_sampling_weights)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.motion_id is not None:
            if not 0 <= self.cfg.motion_id < len(self.motions):
                raise ValueError(f"motion_id must be in [0, {len(self.motions) - 1}].")
            self.motion_ids.fill_(self.cfg.motion_id)
        elif len(self.motions) > 1:
            self.motion_ids[:] = torch.multinomial(
                self.motion_sampling_probs, self.num_envs, replacement=True
            )
        print(
            f"[INFO] Loaded motion pool with {len(self.motions)} motion(s): "
            + ", ".join(os.path.basename(path) for path in self.motion_files)
        )
        self._num_bodies = len(cfg.body_names)
        self._num_motion_joints = self.motion.joint_pos.shape[1]
        self._validate_motion_pool()
        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._active_env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Current-frame caches. Motion data is immutable, so each field is gathered once
        # per environment step and shared by observations, rewards, metrics, and resets.
        self._joint_pos = torch.empty(self.num_envs, self._num_motion_joints, device=self.device)
        self._joint_vel = torch.empty_like(self._joint_pos)
        self._command = torch.empty(self.num_envs, 2 * self._num_motion_joints, device=self.device)
        self._body_pos_w = torch.empty(self.num_envs, self._num_bodies, 3, device=self.device)
        self._body_quat_w = torch.empty(self.num_envs, self._num_bodies, 4, device=self.device)
        self._body_lin_vel_motion_w = torch.empty(self.num_envs, self._num_bodies, 3, device=self.device)
        self._body_ang_vel_motion_w = torch.empty_like(self._body_lin_vel_motion_w)
        self._anchor_pos_w = torch.empty(self.num_envs, 3, device=self.device)
        self._anchor_quat_w = torch.empty(self.num_envs, 4, device=self.device)

        # Reused alignment scratch buffers.  The command is refreshed once per
        # environment step and then shared by all observations/rewards/terminations.
        self._delta_yaw_w = torch.empty(self.num_envs, 4, device=self.device)
        self._aligned_translation_w = torch.empty(self.num_envs, self._num_bodies, 3, device=self.device)

        self.body_pos_aligned_w = torch.zeros(self.num_envs, self._num_bodies, 3, device=self.device)
        self.body_quat_aligned_w = torch.zeros(self.num_envs, self._num_bodies, 4, device=self.device)
        self.body_quat_aligned_w[:, :, 0] = 1.0
        self.body_lin_vel_aligned_w = torch.zeros(self.num_envs, self._num_bodies, 3, device=self.device)
        self.body_ang_vel_aligned_w = torch.zeros(self.num_envs, self._num_bodies, 3, device=self.device)

        self._pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            dtype=torch.float32,
            device=self.device,
        )
        self._velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            dtype=torch.float32,
            device=self.device,
        )
        self._body_index_cache: dict[tuple[str, ...] | None, list[int] | slice] = {None: slice(None)}

        # Compatibility aliases used by the existing rewards, terminations, and observations.
        self.body_pos_relative_w = self.body_pos_aligned_w
        self.body_quat_relative_w = self.body_quat_aligned_w

        self.bin_count = int(self.motion_lengths.max().item() // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return self._command

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_aligned_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_aligned_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._anchor_pos_w

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._anchor_quat_w

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_aligned_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_aligned_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_pos
        return self.robot.data.joint_pos[:, self.robot_joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_vel
        return self.robot.data.joint_vel[:, self.robot_joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def get_body_indexes(self, body_names: Sequence[str] | None) -> list[int] | slice:
        """Resolve tracked motion body names once and reuse their indexes."""
        if body_names is None:
            return slice(None)
        cache_key = tuple(body_names)
        body_indexes = self._body_index_cache.get(cache_key)
        if body_indexes is None:
            requested_names = set(body_names)
            body_indexes = [i for i, name in enumerate(self.cfg.body_names) if name in requested_names]
            self._body_index_cache[cache_key] = body_indexes
        return body_indexes

    def _build_sampling_probs(self, weights: list[float] | None) -> torch.Tensor:
        if weights is None:
            probs = torch.ones(len(self.motions), dtype=torch.float32, device=self.device)
        else:
            if len(weights) != len(self.motions):
                raise ValueError("motion_sampling_weights must have one value per motion file.")
            probs = torch.tensor(weights, dtype=torch.float32, device=self.device)
            if torch.any(probs < 0) or float(probs.sum()) <= 0:
                raise ValueError("motion_sampling_weights must be non-negative and have a positive sum.")
        return probs / probs.sum()

    def _validate_motion_pool(self):
        reference = self.motions[0]
        for index, motion in enumerate(self.motions[1:], start=1):
            if motion.joint_pos.shape[1:] != reference.joint_pos.shape[1:]:
                raise ValueError(f"Motion {index} has incompatible joint_pos shape.")
            if motion.joint_vel.shape[1:] != reference.joint_vel.shape[1:]:
                raise ValueError(f"Motion {index} has incompatible joint_vel shape.")
            for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                if getattr(motion, name).shape[1:] != getattr(reference, name).shape[1:]:
                    raise ValueError(f"Motion {index} has incompatible {name} shape.")
            if abs(motion.fps - reference.fps) > 1e-5:
                raise ValueError(f"Motion {index} has fps={motion.fps}, expected {reference.fps}.")
        for index, motion in enumerate(self.motions):
            for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                if not torch.isfinite(getattr(motion, name)).all():
                    raise ValueError(f"Motion {index} contains non-finite values in {name}.")
            quat_norm = torch.linalg.vector_norm(motion.body_quat_w, dim=-1)
            if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-3, rtol=1e-3):
                raise ValueError(f"Motion {index} contains non-normalized body quaternions.")

    def _motion_groups(self, env_ids: torch.Tensor):
        for motion_id, motion in enumerate(self.motions):
            group_mask = self.motion_ids[env_ids] == motion_id
            if torch.any(group_mask):
                yield motion_id, motion, env_ids[group_mask]

    def _refresh_motion_frame_cache(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._all_env_ids
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        for _, motion, group_env_ids in self._motion_groups(env_ids):
            frame_ids = self.time_steps[group_env_ids]
            self._joint_pos[group_env_ids] = motion.joint_pos[frame_ids]
            self._joint_vel[group_env_ids] = motion.joint_vel[frame_ids]
            self._command[group_env_ids, : self._num_motion_joints] = self._joint_pos[group_env_ids]
            self._command[group_env_ids, self._num_motion_joints :] = self._joint_vel[group_env_ids]

            self._body_pos_w[group_env_ids] = motion.body_pos_w[frame_ids]
            self._body_pos_w[group_env_ids] += self._env.scene.env_origins[group_env_ids, None, :]
            self._body_quat_w[group_env_ids] = motion.body_quat_w[frame_ids]
            self._body_lin_vel_motion_w[group_env_ids] = motion.body_lin_vel_w[frame_ids]
            self._body_ang_vel_motion_w[group_env_ids] = motion.body_ang_vel_w[frame_ids]
            self._anchor_pos_w[group_env_ids] = self._body_pos_w[group_env_ids, self.motion_anchor_body_index]
            self._anchor_quat_w[group_env_ids] = self._body_quat_w[group_env_ids, self.motion_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self.cfg.start_frame_from_beginning:
            self.time_steps[env_ids] = 0
            return
        if len(self.motions) > 1:
            selected_lengths = self.motion_lengths[self.motion_ids[env_ids]]
            self.time_steps[env_ids] = (
                torch.rand(len(env_ids), device=self.device) * (selected_lengths - 1).clamp_min(1)
            ).long()
            entropy = -(self.motion_sampling_probs * (self.motion_sampling_probs + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = entropy / math.log(len(self.motions))
            top_prob, top_id = self.motion_sampling_probs.max(dim=0)
            self.metrics["sampling_top1_prob"][:] = top_prob
            self.metrics["sampling_top1_bin"][:] = top_id.float() / max(len(self.motions) - 1, 1)
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _refresh_aligned_motion_state(
        self,
        env_ids: torch.Tensor | None = None,
        robot_anchor_pos_w: torch.Tensor | None = None,
        robot_anchor_quat_w: torch.Tensor | None = None,
    ):
        if env_ids is None:
            env_ids_tensor = self._all_env_ids
        elif not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids
        if env_ids_tensor.numel() == 0:
            return

        motion_anchor_pos_w = self._anchor_pos_w[env_ids_tensor]
        motion_anchor_quat_w = self._anchor_quat_w[env_ids_tensor]
        if robot_anchor_pos_w is None:
            robot_anchor_pos_w = self.robot_anchor_pos_w[env_ids_tensor]
        if robot_anchor_quat_w is None:
            robot_anchor_quat_w = self.robot_anchor_quat_w[env_ids_tensor]

        delta_yaw_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(motion_anchor_quat_w)))
        self._delta_yaw_w[env_ids_tensor] = delta_yaw_w

        self._aligned_translation_w[env_ids_tensor] = robot_anchor_pos_w[:, None, :]
        # Preserve the original design: horizontal body positions follow the
        # robot anchor, while the reference anchor height remains from motion.
        self._aligned_translation_w[env_ids_tensor, :, 2] = motion_anchor_pos_w[:, None, 2]
        for _, motion, group_env_ids in self._motion_groups(env_ids_tensor):
            group_delta_yaw = self._delta_yaw_w[group_env_ids, None, :].expand(-1, self._num_bodies, -1)
            body_pos_anchor_rel = motion.body_pos_anchor_rel[self.time_steps[group_env_ids]]
            self.body_pos_aligned_w[group_env_ids] = self._aligned_translation_w[group_env_ids] + quat_apply(
                group_delta_yaw, body_pos_anchor_rel
            )
            self.body_quat_aligned_w[group_env_ids] = quat_mul(
                group_delta_yaw, self._body_quat_w[group_env_ids]
            )
            self.body_lin_vel_aligned_w[group_env_ids] = quat_apply(
                group_delta_yaw, self._body_lin_vel_motion_w[group_env_ids]
            )
            self.body_ang_vel_aligned_w[group_env_ids] = quat_apply(
                group_delta_yaw, self._body_ang_vel_motion_w[group_env_ids]
            )

        if self.cfg.validate_targets:
            targets_are_finite = (
                torch.isfinite(self.body_pos_aligned_w[env_ids_tensor]).all()
                & torch.isfinite(self.body_quat_aligned_w[env_ids_tensor]).all()
                & torch.isfinite(self.body_lin_vel_aligned_w[env_ids_tensor]).all()
                & torch.isfinite(self.body_ang_vel_aligned_w[env_ids_tensor]).all()
            )
            if not targets_are_finite:
                raise RuntimeError("MotionCommand produced a non-finite aligned motion target.")

    def _refresh_relative_motion_state(self, env_ids: torch.Tensor | None = None):
        """Compatibility wrapper for callers using the previous method name."""
        self._refresh_aligned_motion_state(env_ids)

    def _pd_stand_reset_would_terminate(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        terminated = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        terminations_cfg = self._env.cfg.terminations

        anchor_pos_cfg = getattr(terminations_cfg, "anchor_pos", None)
        if anchor_pos_cfg is not None:
            threshold = float(anchor_pos_cfg.params["threshold"])
            terminated |= torch.abs(self.anchor_pos_w[env_ids, -1] - self.robot_anchor_pos_w[env_ids, -1]) > threshold

        anchor_ori_cfg = getattr(terminations_cfg, "anchor_ori", None)
        if anchor_ori_cfg is not None:
            threshold = float(anchor_ori_cfg.params["threshold"])
            gravity_vec_w = self.robot.data.GRAVITY_VEC_W
            if gravity_vec_w.ndim > 1:
                gravity_vec_w = gravity_vec_w[env_ids]
            motion_projected_gravity_b = quat_rotate_inverse(self.anchor_quat_w[env_ids], gravity_vec_w)
            robot_projected_gravity_b = quat_rotate_inverse(self.robot_anchor_quat_w[env_ids], gravity_vec_w)
            terminated |= torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > threshold

        return terminated

    def _reset_envs_from_motion(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self._refresh_motion_frame_cache(env_ids)
        root_pos = self.anchor_pos_w[env_ids].clone()
        root_ori = self.anchor_quat_w[env_ids].clone()
        root_lin_vel = self._body_lin_vel_motion_w[env_ids, self.motion_anchor_body_index].clone()
        root_ang_vel = self._body_ang_vel_motion_w[env_ids, self.motion_anchor_body_index].clone()

        rand_samples = sample_uniform(
            self._pose_ranges[:, 0], self._pose_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        rand_samples = sample_uniform(
            self._velocity_ranges[:, 0], self._velocity_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        if self.robot_joint_indexes is None:
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            full_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.robot_joint_indexes]
            clipped_joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            full_joint_pos[:, self.robot_joint_indexes] = clipped_joint_pos
            full_joint_vel[:, self.robot_joint_indexes] = joint_vel
            self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        # The simulator buffers are refreshed only after reset finishes. Use the state just
        # written here so the very first reward observes valid aligned motion targets.
        self._refresh_aligned_motion_state(env_ids, robot_anchor_pos_w=root_pos, robot_anchor_quat_w=root_ori)

    def _reset_envs_from_pd_stand(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return env_ids

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._env.scene.update(self._env.physics_dt)
        self._refresh_motion_frame_cache(env_ids)
        self._refresh_aligned_motion_state(
            env_ids, robot_anchor_pos_w=root_state[:, :3], robot_anchor_quat_w=root_state[:, 3:7]
        )

        return env_ids[self._pd_stand_reset_would_terminate(env_ids)]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(self.motions) > 1 and self.cfg.motion_id is None:
            self.motion_ids[env_ids] = torch.multinomial(
                self.motion_sampling_probs, len(env_ids), replacement=True
            )
        self._adaptive_sampling(env_ids)

        pd_stand_ratio = float(self.cfg.pd_stand_reset_ratio)
        if pd_stand_ratio <= 0.0:
            self._reset_envs_from_motion(env_ids)
            return

        pd_stand_mask = torch.rand(len(env_ids), device=self.device) < pd_stand_ratio
        motion_env_ids = env_ids[~pd_stand_mask]
        pd_stand_env_ids = env_ids[pd_stand_mask]

        self._reset_envs_from_motion(motion_env_ids)
        fallback_env_ids = self._reset_envs_from_pd_stand(pd_stand_env_ids)
        if fallback_env_ids.numel() > 0:
            self._reset_envs_from_motion(fallback_env_ids)

    def _update_command(self):
        self.time_steps += 1
        current_lengths = self.motion_lengths[self.motion_ids]
        env_ids = torch.where(self.time_steps >= current_lengths)[0]
        self._resample_command(env_ids)
        if env_ids.numel() == 0:
            self._refresh_motion_frame_cache()
            self._refresh_aligned_motion_state()
        elif env_ids.numel() < self.num_envs:
            self._active_env_mask.fill_(True)
            self._active_env_mask[env_ids] = False
            active_env_ids = self._all_env_ids[self._active_env_mask]
            self._refresh_motion_frame_cache(active_env_ids)
            self._refresh_aligned_motion_state(active_env_ids)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_aligned_w[:, i], self.body_quat_aligned_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | None = MISSING
    motion_files: list[str] | None = None
    motion_sampling_weights: list[float] | None = None
    motion_id: int | None = None
    """Optional fixed motion index, intended for deterministic playback."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    motion_body_names: list[str] | None = None
    motion_joint_names: list[str] | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    min_traj_duration: float | None = None
    bridge_frames: int = 20

    start_frame_from_beginning: bool = False
    """Fixed-start mode: every episode begins at motion frame 0 instead of a
    failure-weighted random frame. Deterministic and matches 'replay the dance
    from the beginning' behavior."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    pd_stand_reset_ratio: float = 0.0
    validate_targets: bool = False
    """Enable expensive finite-value checks for aligned targets during diagnostics."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

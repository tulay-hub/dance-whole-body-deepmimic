from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


np.set_printoptions(precision=6, suppress=True)

LENS110_TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
]
LENS110_MOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
LENS110_ANCHOR_BODY_NAME = "torso_yaw_link"
LENS110_ROOT_BODY_NAME = "pelvis"
DEFAULT_SIM_DT = 0.003333
DEFAULT_DECIMATION = 10
DEFAULT_STANDING_ROOT_Z = 0.68
DEFAULT_FOOT_CLEARANCE = 0.002
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
STANDING_FOOT_GEOM_NAMES = ["left_ankle_roll_collision", "right_ankle_roll_collision"]
STANDING_JOINT_OVERRIDES = {
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}


@dataclass
class PolicyMetadata:
    joint_names: list[str]
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    observation_names: list[str]
    anchor_body_name: str
    body_names: list[str]
    num_obs: int
    num_actions: int

    @classmethod
    def from_session(cls, session: ort.InferenceSession) -> "PolicyMetadata":
        model_meta = session.get_modelmeta().custom_metadata_map

        def _require(key: str) -> str:
            value = model_meta.get(key)
            if value is None:
                raise KeyError(f"Missing required ONNX metadata field: {key}")
            return value

        def _split_names(key: str) -> list[str]:
            return [item.strip() for item in _require(key).split(",") if item.strip()]

        def _split_floats(key: str) -> np.ndarray:
            return np.asarray([float(item) for item in _require(key).split(",") if item.strip()], dtype=np.float32)

        obs_shape = session.get_inputs()[0].shape
        action_shape = session.get_outputs()[0].shape
        num_obs = int(obs_shape[-1])
        num_actions = int(action_shape[-1])

        return cls(
            joint_names=_split_names("joint_names"),
            joint_stiffness=_split_floats("joint_stiffness"),
            joint_damping=_split_floats("joint_damping"),
            default_joint_pos=_split_floats("default_joint_pos"),
            action_scale=_split_floats("action_scale"),
            observation_names=_split_names("observation_names"),
            anchor_body_name=_require("anchor_body_name"),
            body_names=_split_names("body_names"),
            num_obs=num_obs,
            num_actions=num_actions,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens110 MuJoCo sim2sim for exported BeyondMimic ONNX policies.")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to exported policy.onnx")
    parser.add_argument("--mjcf_path", type=str, required=True, help="Path to the Lens110 MJCF/XML")
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=None,
        help="Unused for Lens110 sim2sim. Kept only for CLI compatibility.",
    )
    parser.add_argument("--motion_file", type=str, required=True, help="Path to the retargeted motion npz")
    parser.add_argument("--sim_dt", type=float, default=DEFAULT_SIM_DT, help="MuJoCo simulation timestep")
    parser.add_argument(
        "--decimation",
        type=int,
        default=DEFAULT_DECIMATION,
        help="Number of MuJoCo steps per policy step",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame in the motion file")
    parser.add_argument("--loop", action="store_true", help="Loop the motion when reaching the end")
    parser.add_argument(
        "--motion_play",
        action="store_true",
        help="Replay motion directly instead of running the policy",
    )
    parser.add_argument(
        "--control_mode",
        choices=("position", "external_pd"),
        default="position",
        help="Use MuJoCo position actuators by default; external_pd keeps the old qfrc_applied path.",
    )
    parser.add_argument(
        "--disable_self_collisions",
        action="store_true",
        help="Disable robot self-collisions for debugging. Default keeps self-collisions enabled like Lens110 training.",
    )
    parser.add_argument(
        "--warmup_seconds",
        type=float,
        default=0.0,
        help="Deprecated compatibility flag. Legacy sim2sim path ignores warmup.",
    )
    return parser.parse_args()


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_inv_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.dot(quat, quat)
    if norm <= 1.0e-12:
        raise ValueError("Cannot invert zero quaternion.")
    return quat_conjugate_wxyz(quat) / norm


def quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    rot = quat_to_matrix_wxyz(quat)
    return rot.T @ vec


def subtract_frame_transforms(
    pos_a: np.ndarray,
    quat_a: np.ndarray,
    pos_b: np.ndarray,
    quat_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rel_pos = quat_rotate_inverse_wxyz(quat_a, pos_b - pos_a)
    rel_quat = quat_mul_wxyz(quat_inv_wxyz(quat_a), quat_b)
    rel_quat /= np.linalg.norm(rel_quat)
    return rel_pos, rel_quat


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


class Lens110Sim2Sim:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy_path = str(Path(args.policy_path).expanduser().resolve())
        self.mjcf_path = str(Path(args.mjcf_path).expanduser().resolve())
        self.motion_file = str(Path(args.motion_file).expanduser().resolve())

        self.motion = np.load(self.motion_file)
        self.motion_joint_pos = self.motion["joint_pos"].astype(np.float32)
        self.motion_joint_vel = self.motion["joint_vel"].astype(np.float32)
        self.motion_body_pos = self.motion["body_pos_w"].astype(np.float32)
        self.motion_body_quat = self.motion["body_quat_w"].astype(np.float32)
        self.motion_body_lin_vel = self.motion["body_lin_vel_w"].astype(np.float32)
        self.motion_body_ang_vel = self.motion["body_ang_vel_w"].astype(np.float32)
        self.motion_num_frames = self.motion_joint_pos.shape[0]
        self.start_frame = min(max(args.start_frame, 0), self.motion_num_frames - 1)

        self.session = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
        self.metadata = PolicyMetadata.from_session(self.session)
        self.session_inputs = self.session.get_inputs()
        self.obs_input_name = self.session_inputs[0].name
        self.time_step_input_name = self.session_inputs[1].name if len(self.session_inputs) > 1 else None

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = args.sim_dt
        self.step_dt = args.sim_dt * args.decimation

        self._validate_training_alignment()
        self._build_joint_mappings()
        self._configure_mujoco_dynamics()
        self._build_motion_body_mappings()
        self._build_sensor_handles()
        self._build_foot_geom_handles()

        self.frame_idx = self.start_frame
        self.last_action = np.zeros(self.metadata.num_actions, dtype=np.float32)
        self.zero_action = np.zeros(self.metadata.num_actions, dtype=np.float32)
        self.target_q_mj = self.default_joint_pos_mj.copy()
        self.paused = False
        self.control_mode = "STANDUP"
        self.latest_obs = None
        self.rl_counter = 0

        self.reset(standing=not self.args.motion_play)

    def _validate_training_alignment(self) -> None:
        if self.metadata.anchor_body_name != LENS110_ANCHOR_BODY_NAME:
            raise ValueError(
                f"Expected Lens110 anchor body '{LENS110_ANCHOR_BODY_NAME}', got '{self.metadata.anchor_body_name}'."
            )
        if self.metadata.body_names != LENS110_TRACKED_BODY_NAMES:
            raise ValueError(
                "Exported policy tracked body_names do not match Lens110 training config.\n"
                f"Expected: {LENS110_TRACKED_BODY_NAMES}\nGot: {self.metadata.body_names}"
            )
        if self.motion_joint_pos.shape[1] != self.metadata.num_actions:
            raise ValueError(
                f"Motion joint count {self.motion_joint_pos.shape[1]} does not match policy action count "
                f"{self.metadata.num_actions}."
            )

    def _build_joint_mappings(self) -> None:
        self.mj_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in range(self.model.nu)
        ]
        if None in self.mj_joint_names:
            raise ValueError("Found unnamed actuators in MJCF; cannot build joint mapping.")
        if len(self.mj_joint_names) != self.metadata.num_actions:
            raise ValueError(
                f"MJCF actuator count {len(self.mj_joint_names)} does not match policy action count "
                f"{self.metadata.num_actions}."
            )

        self.mj_qpos_adr = []
        self.mj_qvel_adr = []
        for joint_name in self.mj_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint '{joint_name}' was not found in the MJCF.")
            self.mj_qpos_adr.append(self.model.jnt_qposadr[joint_id])
            self.mj_qvel_adr.append(self.model.jnt_dofadr[joint_id])

        self.mj_qpos_adr = np.asarray(self.mj_qpos_adr, dtype=np.int32)
        self.mj_qvel_adr = np.asarray(self.mj_qvel_adr, dtype=np.int32)
        self.mj_ctrl_adr = np.arange(len(self.mj_joint_names), dtype=np.int32)
        self.policy_order_in_mj = np.asarray(
            [self.mj_joint_names.index(joint_name) for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.mj_order_in_policy = np.asarray(
            [self.metadata.joint_names.index(joint_name) for joint_name in self.mj_joint_names],
            dtype=np.int32,
        )
        self.default_joint_pos_mj = self.metadata.default_joint_pos[self.mj_order_in_policy]
        self.joint_stiffness_mj = self.metadata.joint_stiffness[self.mj_order_in_policy]
        self.joint_damping_mj = self.metadata.joint_damping[self.mj_order_in_policy]
        self.action_scale_mj = self.metadata.action_scale[self.mj_order_in_policy]
        self.joint_effort_limit_mj = 4.0 * self.action_scale_mj * self.joint_stiffness_mj
        self.standing_joint_pos_mj = self.default_joint_pos_mj.copy()
        for joint_name, joint_pos in STANDING_JOINT_OVERRIDES.items():
            if joint_name not in self.mj_joint_names:
                raise ValueError(f"Standing override joint '{joint_name}' was not found in the MJCF.")
            self.standing_joint_pos_mj[self.mj_joint_names.index(joint_name)] = joint_pos

    def _configure_mujoco_dynamics(self) -> None:
        self.model.dof_damping[self.mj_qvel_adr] = self.joint_damping_mj
        self.model.dof_armature[self.mj_qvel_adr] = 0.01

        actuator_ids = self.mj_ctrl_adr
        self.model.actuator_gainprm[actuator_ids, 0] = self.joint_stiffness_mj
        self.model.actuator_biasprm[actuator_ids, 1] = -self.joint_stiffness_mj
        self.model.actuator_forcelimited[actuator_ids] = 1
        self.model.actuator_forcerange[actuator_ids, 0] = -self.joint_effort_limit_mj
        self.model.actuator_forcerange[actuator_ids, 1] = self.joint_effort_limit_mj
        self._configure_collision_filters()

    def _configure_collision_filters(self) -> None:
        if not self.args.disable_self_collisions:
            return

        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "collision" in geom_name:
                self.model.geom_contype[geom_id] = 2
                self.model.geom_conaffinity[geom_id] = 1

    def _build_motion_body_mappings(self) -> None:
        if len(LENS110_MOTION_BODY_NAMES) != self.motion_body_pos.shape[1]:
            raise ValueError(
                "Motion body count does not match the Lens110 Isaac export body order.\n"
                f"Motion count: {self.motion_body_pos.shape[1]}\n"
                f"Expected count: {len(LENS110_MOTION_BODY_NAMES)}\n"
                f"Expected body names: {LENS110_MOTION_BODY_NAMES}"
            )

        self.motion_body_names = list(LENS110_MOTION_BODY_NAMES)
        self.motion_anchor_idx = self.motion_body_names.index(self.metadata.anchor_body_name)
        self.motion_root_idx = self.motion_body_names.index(LENS110_ROOT_BODY_NAME)
        self.robot_anchor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.metadata.anchor_body_name
        )
        self.robot_root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LENS110_ROOT_BODY_NAME)
        if self.robot_anchor_body_id == -1:
            raise ValueError(f"Anchor body '{self.metadata.anchor_body_name}' was not found in the MJCF.")
        if self.robot_root_body_id == -1:
            raise ValueError(f"Root body '{LENS110_ROOT_BODY_NAME}' was not found in the MJCF.")

    def _build_sensor_handles(self) -> None:
        self.orientation_sensor = "orientation"
        self.position_sensor = "position"
        self.angular_velocity_sensor = "angular-velocity"
        self.linear_velocity_sensor = "linear-velocity"

    def _build_foot_geom_handles(self) -> None:
        self.foot_geom_ids = []
        for geom_name in STANDING_FOOT_GEOM_NAMES:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                raise ValueError(f"Required standing foot geom '{geom_name}' was not found in the MJCF.")
            self.foot_geom_ids.append(geom_id)

    def _set_state_from_motion(
        self,
        frame_idx: int,
        *,
        set_root_velocity: bool,
        set_joint_velocity: bool,
    ) -> None:
        self.data.qpos[0:3] = self.motion_body_pos[frame_idx, self.motion_root_idx]
        self.data.qpos[3:7] = self.motion_body_quat[frame_idx, self.motion_root_idx]
        self.data.qpos[self.mj_qpos_adr] = self.motion_joint_pos[frame_idx, self.mj_order_in_policy]

        self.data.qvel[:] = 0.0
        if set_root_velocity:
            self.data.qvel[0:3] = self.motion_body_lin_vel[frame_idx, self.motion_root_idx]
            self.data.qvel[3:6] = self.motion_body_ang_vel[frame_idx, self.motion_root_idx]
        if set_joint_velocity:
            self.data.qvel[self.mj_qvel_adr] = self.motion_joint_vel[frame_idx, self.mj_order_in_policy]

    def _set_state_to_standing(self, reference_frame_idx: int) -> None:
        root_pos = self.model.qpos0[0:3].copy()
        root_pos[0:2] = self.motion_body_pos[reference_frame_idx, self.motion_root_idx, 0:2]
        root_pos[2] = DEFAULT_STANDING_ROOT_Z

        self.data.qpos[:] = self.model.qpos0.copy()
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = IDENTITY_QUAT_WXYZ
        self.data.qpos[self.mj_qpos_adr] = self.standing_joint_pos_mj
        mujoco.mj_forward(self.model, self.data)
        self._align_standing_feet_to_floor()

    def _get_geom_lowest_z(self, geom_id: int) -> float:
        geom_pos = self.data.geom_xpos[geom_id]
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.model.geom_size[geom_id]
        geom_type = self.model.geom_type[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            support_radius_z = np.abs(geom_mat[2, :]) @ geom_size
            return float(geom_pos[2] - support_radius_z)
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            support_radius_z = (
                abs(geom_mat[2, 0]) * geom_size[0]
                + abs(geom_mat[2, 1]) * geom_size[0]
                + abs(geom_mat[2, 2]) * geom_size[1]
            )
            return float(geom_pos[2] - support_radius_z)

        raise NotImplementedError(f"Unsupported foot geom type: {geom_type}")

    def _align_standing_feet_to_floor(self) -> None:
        lowest_z = min(self._get_geom_lowest_z(geom_id) for geom_id in self.foot_geom_ids)
        self.data.qpos[2] += DEFAULT_FOOT_CLEARANCE - lowest_z
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, standing: bool) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        self.frame_idx = self.start_frame
        self.last_action[:] = 0.0
        self.rl_target_action = self.zero_action.copy()
        self.rl_counter = 0
        if standing:
            self._set_state_to_standing(self.frame_idx)
            self.target_q_mj = self.standing_joint_pos_mj.copy()
        else:
            self._set_state_from_motion(self.frame_idx, set_root_velocity=False, set_joint_velocity=False)
            self.target_q_mj = self.data.qpos[self.mj_qpos_adr].copy()
        self.data.qfrc_applied[:] = 0.0
        self._set_position_targets(self.target_q_mj)
        mujoco.mj_forward(self.model, self.data)
        self.latest_obs = self._build_observation(self.frame_idx)

    def _get_obs_legacy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quat = self.data.sensor(self.orientation_sensor).data.astype(np.float64)
        omega = self.data.sensor(self.angular_velocity_sensor).data.astype(np.float64)
        joint_pos_mj = self.data.qpos[self.mj_qpos_adr].copy()
        joint_vel_mj = self.data.qvel[self.mj_qvel_adr].copy()
        qpos_seq = joint_pos_mj[self.policy_order_in_mj]
        qvel_seq = joint_vel_mj[self.policy_order_in_mj]
        return quat, omega, np.concatenate(
            [
                qpos_seq.astype(np.float32),
                qvel_seq.astype(np.float32),
            ],
            axis=0,
        )

    def _build_observation(self, frame_idx: int) -> np.ndarray:
        _, omega, joint_state = self._get_obs_legacy()
        qpos_seq = joint_state[: self.metadata.num_actions]
        qvel_seq = joint_state[self.metadata.num_actions :]

        motioninput = np.concatenate(
            [self.motion_joint_pos[frame_idx], self.motion_joint_vel[frame_idx]],
            axis=0,
        ).astype(np.float32)
        motionquatcurrent = self.motion_body_quat[frame_idx, self.motion_anchor_idx].astype(np.float64)
        robot_anchor_quat = self.data.xquat[self.robot_anchor_body_id].copy()
        q12 = quat_mul_wxyz(quat_inv_wxyz(robot_anchor_quat), motionquatcurrent)
        motion_anchor_ori_b = quat_to_matrix_wxyz(q12)[:, :2].reshape(-1).astype(np.float32)

        obs = np.concatenate(
            [
                motioninput,
                motion_anchor_ori_b,
                omega.astype(np.float32),
                (qpos_seq - self.metadata.default_joint_pos).astype(np.float32),
                qvel_seq.astype(np.float32),
                self.last_action.astype(np.float32),
            ],
            axis=0,
        )
        if obs.shape[0] != self.metadata.num_obs:
            raise ValueError(
                f"Built observation has size {obs.shape[0]}, but policy expects {self.metadata.num_obs}."
            )
        return obs

    def _infer_action(self, obs: np.ndarray) -> np.ndarray:
        inputs = {self.obs_input_name: obs.reshape(1, -1)}
        if self.time_step_input_name is not None:
            inputs[self.time_step_input_name] = np.array([[self.frame_idx]], dtype=np.float32)
        outputs = self.session.run(None, inputs)
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def _advance_frame(self) -> bool:
        if self.frame_idx + 1 < self.motion_num_frames:
            self.frame_idx += 1
            return True
        if self.args.loop:
            self.frame_idx = self.start_frame
            return True
        return False

    def _set_position_targets(self, joint_pos_mj: np.ndarray) -> None:
        self.data.ctrl[self.mj_ctrl_adr] = joint_pos_mj
        self.data.qfrc_applied[:] = 0.0

    def _set_neutral_position_targets(self) -> None:
        self._set_position_targets(self.data.qpos[self.mj_qpos_adr].copy())

    def _apply_external_joint_torque(self, tau: np.ndarray) -> None:
        self._set_neutral_position_targets()
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.mj_qvel_adr] = tau

    def _prime_policy_target(self) -> None:
        obs = self._build_observation(self.frame_idx)
        self.latest_obs = obs
        next_action = self._infer_action(obs)
        self.last_action = next_action.copy()
        action_mj = self.last_action[self.mj_order_in_policy]
        self.target_q_mj = self.default_joint_pos_mj + action_mj * self.action_scale_mj

    def _step_policy(self) -> bool:
        if self.args.control_mode == "position":
            self._set_position_targets(self.target_q_mj)
        else:
            tau = pd_control(
                self.target_q_mj,
                self.data.qpos[self.mj_qpos_adr].copy(),
                self.joint_stiffness_mj,
                np.zeros_like(self.joint_damping_mj),
                self.data.qvel[self.mj_qvel_adr].copy(),
                self.joint_damping_mj,
            )
            self._apply_external_joint_torque(tau)

        mujoco.mj_step(self.model, self.data)
        self.rl_counter += 1

        if self.rl_counter % self.args.decimation != 0:
            return False

        obs = self._build_observation(self.frame_idx)
        self.latest_obs = obs

        next_action = self._infer_action(obs)
        self.last_action = next_action.copy()
        action_mj = self.last_action[self.mj_order_in_policy]
        self.target_q_mj = self.default_joint_pos_mj + action_mj * self.action_scale_mj

        has_next = self._advance_frame()
        return not has_next

    def _step_motion_play(self) -> tuple[np.ndarray, bool]:
        self.last_action[:] = 0.0
        self._set_state_from_motion(self.frame_idx, set_root_velocity=True, set_joint_velocity=True)
        self._set_neutral_position_targets()
        mujoco.mj_forward(self.model, self.data)

        has_next = self._advance_frame()
        obs = self._build_observation(self.frame_idx)
        self.latest_obs = obs
        return obs, not has_next

    def _hold_standing(self) -> None:
        self.last_action[:] = 0.0
        self._set_state_to_standing(self.frame_idx)
        self.target_q_mj = self.standing_joint_pos_mj.copy()
        self._set_position_targets(self.target_q_mj)
        mujoco.mj_forward(self.model, self.data)
        self.latest_obs = self._build_observation(self.frame_idx)

    def _configure_camera(self, viewer) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = self.robot_root_body_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        viewer.opt.geomgroup[:] = [0, 1, 1, 0, 0, 0]

    def key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except ValueError:
            return

        if key == " ":
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'running'}")
        elif key == "K":
            self.control_mode = "STANDUP"
            print("[INFO] control mode: STANDUP")
        elif key == "L":
            self.reset(standing=False)
            self._prime_policy_target()
            self.control_mode = "RL"
            print("[INFO] control mode: RL (synced to motion start)")
        elif key == "R":
            self.control_mode = "RESET"
            print("[INFO] control mode: RESET")

    def run(self) -> None:
        print(f"[INFO] policy_path: {self.policy_path}")
        print(f"[INFO] mjcf_path:   {self.mjcf_path}")
        print(f"[INFO] motion_file: {self.motion_file}")
        print(f"[INFO] sim_dt: {self.args.sim_dt:.6f}, decimation: {self.args.decimation}")
        print(f"[INFO] control_mode: {self.args.control_mode}")
        print(f"[INFO] self_collisions: {not self.args.disable_self_collisions}")
        if self.args.warmup_seconds > 0.0:
            print(f"[INFO] warmup_seconds flag ignored in legacy mode: {self.args.warmup_seconds:.2f}")
        print(f"[INFO] num_obs: {self.metadata.num_obs}, num_actions: {self.metadata.num_actions}")
        print(f"[INFO] observation_names: {self.metadata.observation_names}")
        print(f"[INFO] motion_body_names: {self.motion_body_names}")
        print(f"[INFO] mj_joint_names: {self.mj_joint_names}")
        print(f"[INFO] policy_joint_names: {self.metadata.joint_names}")
        if not self.args.motion_play:
            print("[INFO] controls: K=standup, L=policy, R=reset, Space=pause")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self._configure_camera(viewer)

            while viewer.is_running():
                step_start = time.time()

                if not self.paused:
                    if self.args.motion_play:
                        _, finished = self._step_motion_play()
                        sleep_dt = self.step_dt
                    elif self.control_mode == "RL":
                        finished = self._step_policy()
                        sleep_dt = self.model.opt.timestep
                    elif self.control_mode == "STANDUP":
                        self._hold_standing()
                        finished = False
                        sleep_dt = self.step_dt
                    elif self.control_mode == "RESET":
                        self.reset(standing=not self.args.motion_play)
                        self.control_mode = "STANDUP"
                        finished = False
                        sleep_dt = self.step_dt
                    else:
                        raise RuntimeError(f"Unknown control mode: {self.control_mode}")

                    if finished:
                        viewer.sync()
                        break

                viewer.sync()

                time_until_next_step = sleep_dt - (time.time() - step_start)
                if time_until_next_step > 0.0:
                    time.sleep(time_until_next_step)


def main() -> None:
    args = parse_args()
    sim = Lens110Sim2Sim(args)
    sim.run()


if __name__ == "__main__":
    main()

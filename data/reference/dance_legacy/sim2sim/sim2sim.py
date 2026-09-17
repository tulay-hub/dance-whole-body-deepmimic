"""Unified sim2sim script supporting multiple robot configurations.

Usage Examples:
    # For HI robot:
    python sim2sim.py --robot hi --motion_file source/motion/hightorque/hi/npz/dance1_subject2.npz \
    --xml_path /path/to/hi.xml --policy_path /path/to/hi_policy.onnx --save_json
    
    # For PI Plus robot:
    python sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/dance1_subject2.npz \
    --xml_path /path/to/pi_plus.xml --policy_path /path/to/pi_plus_policy.onnx --save_json
    
    # With data recording:
    python sim2sim.py --robot nix1 --motion_file motion.npz --policy_path policy.onnx --record
"""

import argparse
import json
import time
import os
import re
import csv
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Simulation parameters
simulation_duration = 200.0
simulation_dt = 0.0033 #0.0016666667
control_decimation = 10

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

# Robot configurations
ROBOT_CONFIGS = {
    
    "nix1": {
        "num_actions": 21,
        "num_obs": 114,
        "reference_body": "torso_link",
        "default_xml": "../ASE/ase/assets/lumos/nix1/mjcf/nix1_joint21_nomesh.xml",  # Must be provided
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 42,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 21,
            "joint_vel": 21,
            "actions": 21
        }
    },
    "nix1_v2": {
        "num_actions": 21,
        "num_obs": 114,
        "reference_body": "torso_link",
        "default_xml": "../ASE/ase/assets/lumos/nix1/mjcf/nix1_v2_joint21.xml",  # Must be provided
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 42,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 21,
            "joint_vel": 21,
            "actions": 21
        }
    },
    "nix2": {
        "num_actions": 21,
        "num_obs": 114,
        "reference_body": "torso_link",
        "default_xml": "../ASE/ase/assets/lumos/nix2/mjcf/nix2_joint21_push_up.xml",  # Must be provided
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 42,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 21,
            "joint_vel": 21,
            "actions": 21
        }
    },
    "lus2_joint21": {
        "num_actions": 21,
        "num_obs": 114,
        "reference_body": "torso_link",
        "default_xml": "../ASE/ase/assets/lumos/lus2/mjcf/lus2_joint21.xml",  # Must be provided
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 42,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 21,
            "joint_vel": 21,
            "actions": 21
        }
    },
    "oli_joint29": {
        "num_actions": 29,
        "num_obs": 154,
        "reference_body": "waist_pitch_link",
        "experiment_name": "oli_flat",
        "default_xml": "../ASE/ase/assets/limxdynamics/oli/mjcf/oli_joint29.xml",  # Must be provided
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 58,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 29,
            "joint_vel": 29,
            "actions": 29
        }
    },
    "lens110": {
        "num_actions": 21,
        "num_obs": 114,
        "reference_body": "torso_yaw_link",
        "experiment_name": "lens110_flat",
        "default_xml_candidates": [
            str((WORKSPACE_ROOT / "TienKung-Lab/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml").resolve()),
            str((WORKSPACE_ROOT / "GMR/assets/lens110_21dof/mjcf/lens110_21dof.xml").resolve()),
        ],
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_yaw_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        ],
        "init_body_index": 0,
        "motion_body_index": 7,
        "observation_structure": {
            "command": 42,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 21,
            "joint_vel": 21,
            "actions": 21
        }
    },
}

ROBOT_CONFIGS["nix1"]["experiment_name"] = "nix1_flat"
ROBOT_CONFIGS["nix1_v2"]["experiment_name"] = "nix1_flat"
ROBOT_CONFIGS["nix2"]["experiment_name"] = "nix2_flat"
ROBOT_CONFIGS["lus2_joint21"]["experiment_name"] = "lus2_flat"


def _resolve_user_path(path_str: str) -> str:
    """Resolve a path provided on CLI while preserving absolute paths."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path).resolve())


def _resolve_motion_file(robot_type: str, motion_file: str) -> str:
    """Allow either a direct path or a bare motion name under output/<robot>."""
    motion_path = Path(motion_file).expanduser()
    if motion_path.is_absolute() or motion_path.suffix == ".npz" or os.sep in motion_file:
        if motion_path.suffix != ".npz":
            motion_path = motion_path.with_suffix(".npz")
        return str(motion_path.resolve())
    return str((REPO_ROOT / "output" / robot_type / f"{motion_file}.npz").resolve())


def _resolve_policy_path(config: dict, load_run: str | None, policy_path: str | None) -> str:
    """Resolve exported ONNX path from either an explicit file or a training run name."""
    if policy_path:
        return _resolve_user_path(policy_path)
    if not load_run:
        raise ValueError("Either --load_run or --policy_path must be provided.")
    experiment_name = config["experiment_name"]
    return str((REPO_ROOT / "logs" / "rsl_rl" / experiment_name / load_run / "exported" / "policy.onnx").resolve())


def _resolve_xml_path(config: dict, xml_path: str | None) -> str:
    """Resolve MuJoCo XML path, preferring an explicit path and falling back to known defaults."""
    if xml_path:
        return _resolve_user_path(xml_path)

    for candidate in config.get("default_xml_candidates", []):
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())

    if "default_xml" in config:
        return config["default_xml"]

    raise FileNotFoundError(f"No default XML path configured for robot: {config}")


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
    
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version)."""
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    if q_vec.ndim == 2:
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    """Calculate relative transformation from frame A to frame B (MuJoCo version)."""
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    
    return rel_pos, rel_quat


def quaternion_conjugate(q):
    """Quaternion conjugate: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Quaternion multiplication: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion."""
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions):
    """Create observation for HI and PI Plus robots."""
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput
    offset += cmd_size
    obs[offset:offset + 6] = motion_ref_ori_b
    offset += 6
    obs[offset:offset + 3] = omega
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq
    offset += num_actions   
    obs[offset:offset + num_actions] = action_buffer
    return obs


class DataRecorder:
    """Class to record simulation data including joint states and torques."""
    
    def __init__(self, robot_type, joint_names, output_dir="simulation_data"):
        self.robot_type = robot_type
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"{robot_type}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data storage
        self.timestamps = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_accelerations = []
        self.joint_torques = []
        self.control_torques = []
        self.base_position = []
        self.base_orientation = []
        
        # Create CSV file for data recording
        self.csv_file = os.path.join(self.output_dir, "joint_data.csv")
        self._create_csv_header()
        
    def _create_csv_header(self):
        """Create CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['timestamp']
            
            # Add joint-specific headers
            for joint in self.joint_names:
                headers.extend([
                    f'{joint}_pos', f'{joint}_vel', f'{joint}_acc', 
                    f'{joint}_torque', f'{joint}_control_torque'
                ])
            
            # Add base pose
            headers.extend(['base_pos_x', 'base_pos_y', 'base_pos_z'])
            headers.extend(['base_quat_w', 'base_quat_x', 'base_quat_y', 'base_quat_z'])
            
            writer.writerow(headers)
    
    def record_frame(self, timestamp, data, control_torques=None):
        """Record data for one simulation frame."""
        self.timestamps.append(timestamp)
        
        # Record joint positions, velocities, accelerations, and torques
        joint_pos = data.qpos[7:7+self.num_joints].copy()
        joint_vel = data.qvel[6:6+self.num_joints].copy()
        joint_acc = data.qacc[6:6+self.num_joints].copy()
        joint_torque = data.qfrc_actuator[6:6+self.num_joints].copy()
        
        self.joint_positions.append(joint_pos)
        self.joint_velocities.append(joint_vel)
        self.joint_accelerations.append(joint_acc)
        self.joint_torques.append(joint_torque)
        
        # Record control torques if provided
        if control_torques is not None:
            self.control_torques.append(control_torques.copy())
        else:
            self.control_torques.append(np.zeros_like(joint_torque))
        
        # Record base pose
        self.base_position.append(data.qpos[0:3].copy())
        self.base_orientation.append(data.qpos[3:7].copy())
        
        # Write to CSV
        self._write_to_csv(timestamp, joint_pos, joint_vel, joint_acc, 
                          joint_torque, self.control_torques[-1])
    
    def _write_to_csv(self, timestamp, joint_pos, joint_vel, joint_acc, joint_torque, control_torque):
        """Write current frame data to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp]
            
            # Add joint data
            for i in range(self.num_joints):
                row.extend([
                    joint_pos[i], joint_vel[i], joint_acc[i], 
                    joint_torque[i], control_torque[i]
                ])
            
            # Add base pose
            row.extend(self.base_position[-1])
            row.extend(self.base_orientation[-1])
            
            writer.writerow(row)
    
    def save_summary(self):
        """Save summary statistics and additional data files."""
        # Convert to numpy arrays for easier analysis
        timestamps = np.array(self.timestamps)
        joint_positions = np.array(self.joint_positions)
        joint_velocities = np.array(self.joint_velocities)
        joint_accelerations = np.array(self.joint_accelerations)
        joint_torques = np.array(self.joint_torques)
        control_torques = np.array(self.control_torques)
        base_position = np.array(self.base_position)
        base_orientation = np.array(self.base_orientation)
        
        # Save numpy arrays
        np.savez(os.path.join(self.output_dir, "simulation_data.npz"),
                timestamps=timestamps,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_accelerations=joint_accelerations,
                joint_torques=joint_torques,
                control_torques=control_torques,
                base_position=base_position,
                base_orientation=base_orientation,
                joint_names=self.joint_names)
        
        # Calculate and save summary statistics
        summary = {
            'simulation_duration': timestamps[-1] if len(timestamps) > 0 else 0,
            'num_frames': len(timestamps),
            'joint_stats': {}
        }
        
        for i, joint_name in enumerate(self.joint_names):
            summary['joint_stats'][joint_name] = {
                'position': {
                    'mean': float(np.mean(joint_positions[:, i])),
                    'std': float(np.std(joint_positions[:, i])),
                    'min': float(np.min(joint_positions[:, i])),
                    'max': float(np.max(joint_positions[:, i]))
                },
                'velocity': {
                    'mean': float(np.mean(joint_velocities[:, i])),
                    'std': float(np.std(joint_velocities[:, i])),
                    'min': float(np.min(joint_velocities[:, i])),
                    'max': float(np.max(joint_velocities[:, i]))
                },
                'torque': {
                    'mean': float(np.mean(joint_torques[:, i])),
                    'std': float(np.std(joint_torques[:, i])),
                    'min': float(np.min(joint_torques[:, i])),
                    'max': float(np.max(joint_torques[:, i]))
                }
            }
        
        # Save summary as JSON
        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO]: Simulation data saved to: {self.output_dir}")
        return summary


def initialize_robot_from_motion_first_frame(d, motionpos, motionquat, motioninputpos, motion_body_idx, joint_xml, joint_seq, joint_pos_array_seq):
    """Initialize robot state from the first frame of motion data."""
    # Get first frame data
    first_frame_idx = 0
    motionpos_first = motionpos[first_frame_idx, motion_body_idx, :]
    motionquat_first = motionquat[first_frame_idx, motion_body_idx, :]
    motioninputpos_first = motioninputpos[first_frame_idx, :]
    
    # Set base position and orientation
    d.qpos[0:3] = motionpos_first  # Base position
    d.qpos[3:7] = motionquat_first  # Base orientation (quaternion)
    
    # Set joint positions - remap from motion sequence order to XML joint order
    if len(motioninputpos_first) == len(joint_seq):
        # Motion data is in joint_seq order, need to remap to XML order
        joint_positions_remapped = np.array([
            motioninputpos_first[joint_seq.index(joint)] for joint in joint_xml
        ])
        d.qpos[7:7+len(joint_xml)] = joint_positions_remapped
    else:
        # If motion data already matches XML order, use directly
        d.qpos[7:7+len(motioninputpos_first)] = motioninputpos_first
    
    # Set velocities to zero for stable start
    d.qvel[:] = 0.0
    
    print(f"[INFO]: Robot initialized from motion first frame")
    print(f"[INFO]: Base position: {motionpos_first}")
    print(f"[INFO]: Base orientation: {motionquat_first}")
    print(f"[INFO]: Joint positions set from motion data")


def run_simulation(
    robot_type: str,
    motion_file: str,
    policy_path: str,
    xml_path: str,
    save_json: bool = False,
    loop: bool = False,
    record: bool = False,
):
    """Run the sim2sim simulation."""
    config = ROBOT_CONFIGS[robot_type]
    print(f"[INFO]: Using robot configuration: {robot_type}")
    print(f"[INFO]: Actions: {config['num_actions']}, Observations: {config['num_obs']}")
    
    # Initialize data recorder if recording is enabled
    recorder = None
    if record:
        recorder = DataRecorder(robot_type, config['joint_names'])
        print(f"[INFO]: Data recording enabled")
    
    # Load motion data
    motion = np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    # number of frames available across all sequences
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    # safe index helper (supports looping)
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1
    
    # Save motion data to JSON if requested
    if save_json:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist()
        }
        # Convert npz path to json path: npz/file.npz -> json/file.json
        import os
        motion_dir = os.path.dirname(motion_file)
        motion_basename = os.path.basename(motion_file)
        
        # Replace 'npz' directory with 'json' directory
        if motion_dir.endswith('/npz') or motion_dir.endswith('\\npz'):
            json_dir = motion_dir[:-3] + 'json'  # Replace last 3 characters 'npz' with 'json'
        else:
            json_dir = motion_dir  # If not in npz directory, use same directory
            
        # Create json directory if it doesn't exist
        os.makedirs(json_dir, exist_ok=True)
        
        # Create json filename
        json_basename = motion_basename.replace('.npz', '.json')
        json_filename = os.path.join(json_dir, json_basename)
        with open(json_filename, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_filename}")
    
    # Load ONNX model and extract metadata
    model = onnx.load(policy_path)
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale = None
    
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    
    # Remap to XML joint order
    joint_xml = config["joint_names"]
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])

    # stiffness_array = np.array([200, 200, 200, 200, 60, 60, 
    #                             200, 200, 200, 200, 60, 60, 
    #                             60, 60, 60, 
    #                             40, 40, 40, 40, 20, 20, 20,
    #                             40, 40, 40, 40, 20, 20, 20])
    # damping_array = np.array([3, 3, 3, 3, 2, 2, 
    #                           3, 3, 3, 3, 2, 2, 
    #                           4, 4, 4, 
    #                           2, 2, 2, 2, 2, 2, 2, 
    #                           2, 2, 2, 2, 2, 2, 2])
    
    print("stiffness_array", stiffness_array)
    print("damping_array", damping_array)
    print("action_scale", action_scale)
    
    # Initialize variables
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Initialize robot state from motion first frame
    motion_body_idx = config.get("motion_body_index", 0)
    init_body_idx = config.get("init_body_index", motion_body_idx)
    initialize_robot_from_motion_first_frame(d, motionpos, motionquat, motioninputpos, 
                                           init_body_idx, joint_xml, joint_seq, joint_pos_array_seq)
    
    # Load policy
    policy = onnxruntime.InferenceSession(policy_path)
    
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0 
    motioninput = np.concatenate((motioninputpos[frame_idx(timestep), :], motioninputvel[frame_idx(timestep), :]), axis=0)
    
    motionposcurrent = motionpos[frame_idx(timestep), motion_body_idx, :]
    motionquatcurrent = motionquat[frame_idx(timestep), motion_body_idx, :]
    
    target_dof_pos = joint_pos_array.copy()

    # Set reference body
    body_name = config["reference_body"]
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:

        # 获取当前相机
        cam = viewer.cam  

        # 设置跟踪相机
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # 跟踪模式
        cam.trackbodyid = 1  # 要跟踪的 body ID（如机器人基座）
        cam.lookat[:] = [0, 0, 0]  # 相机看向的偏移量
        cam.distance = 2.0  # 相机距离
        cam.azimuth = 135 # 水平旋转角度 (0~360°)
        cam.elevation = -10  # 俯仰角度 (-90°~90°)
        cam.trackbodyid = 0  # 跟踪的body id，-1表示不跟踪任何body
        viewer.opt.geomgroup[:] = [0, 1, 1, 1, 0, 0] # 显示碰撞几何体

        start = time.time()
        simulation_time = 0.0
        pbar = tqdm(range(num_frames), desc="Sim2Sim Running")
        
        while viewer.is_running() and time.time() - start < simulation_duration:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # 跟踪模式
            step_start = time.time()

            mujoco.mj_step(m, d)
            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)

            d.ctrl[:] = tau
            counter += 1
            
            # Record data at every simulation step if recording is enabled
            if recorder is not None:
                recorder.record_frame(simulation_time, d, tau)
            simulation_time += simulation_dt
            
            if counter % control_decimation == 0:
                # Update motion data
                idx = frame_idx(timestep)
                motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)
                motionquatcurrent = motionquat[idx, motion_body_idx, :]
                
                # Create observations based on robot type
                offset = 0
                if robot_type in ["nix1", "nix1_v2", "nix2", "lus2_joint21", "oli_joint29", "lens110"]:
                    # HI and PI Plus observation creation
                    robot_quat_w = torch.from_numpy(quat).unsqueeze(0)
                    q01 = quat
                    q02 = motionquatcurrent
                    q10 = quat_inv_np(q01)
                    if q02 is not None:
                        q12 = quat_mul_np(q10, q02)
                    else:
                        q12 = q10
                    mat = matrix_from_quat(torch.from_numpy(q12))
                    motion_ref_ori_b = mat[..., :2].reshape(6)
                    
                    qpos_xml = d.qpos[7:7 + num_actions]
                    qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                    qvel_xml = d.qvel[6:6 + num_actions]
                    qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                    
                    obs = create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions)
                
                # Run policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy.run(['actions'], {
                    'obs': obs_tensor.numpy(),
                    'time_step': np.array([frame_idx(timestep)], dtype=np.float32).reshape(1, 1)
                })[0]
                
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                target_dof_pos = action * action_scale + joint_pos_array_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                
                # advance time step; if not looping and超过序列则保持在末帧
                if loop or timestep + 1 < num_frames:
                    timestep += 1

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            pbar.update(1)
            pbar.n = timestep
            pbar.set_description(f"frame_idx: {timestep}")

    pbar.close()
    
    # Save recorded data if recording is enabled
    if recorder is not None:
        summary = recorder.save_summary()
        print(f"[INFO]: Simulation completed. Recorded {len(recorder.timestamps)} frames.")
        print(f"[INFO]: Data saved to: {recorder.output_dir}")
    else:
        print(f"[INFO]: Simulation completed. Data recording was not enabled.")


def main():
    parser = argparse.ArgumentParser(description="Unified sim2sim script for multiple robots.")
    parser.add_argument("--robot", type=str, choices=["nix1","nix1_v2","nix2","lus2_joint21","oli_joint29","lens110"], required=True,
                        help="Robot type: nix1, nix1_v2, nix2, lus2_joint21, oli_joint29, lens110")
    parser.add_argument("--motion_file", type=str, required=True, 
                        help="Motion npz path, or a bare file name under output/<robot>/")
    parser.add_argument("--load_run", type=str, default=None,
                        help="Training run directory under logs/rsl_rl/<experiment_name>/")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Direct path to exported policy.onnx")
    parser.add_argument("--xml_path", type=str, default=None,
                        help="Direct path to the MuJoCo XML/MJCF file")
    parser.add_argument("--save_json", action="store_true",
                        help="Save motion data to JSON file")
    parser.add_argument("--loop", action="store_true",
                        help="Loop motion/policy when reaching the end of sequence")
    parser.add_argument("--record", action="store_true",
                        help="Record simulation data (joint states, torques, etc.)")
    
    args = parser.parse_args()
    config = ROBOT_CONFIGS[args.robot]
    policy_path = _resolve_policy_path(config, args.load_run, args.policy_path)
    motion_file = _resolve_motion_file(args.robot, args.motion_file)
    xml_path = _resolve_xml_path(config, args.xml_path)

    print(f"[INFO]: Robot: {args.robot}")
    print(f"[INFO]: Motion file: {motion_file}")
    print(f"[INFO]: Policy path: {policy_path}")
    print(f"[INFO]: XML path: {xml_path}")
    print(f"[INFO]: Record data: {args.record}")
    
    run_simulation(args.robot, motion_file, policy_path, xml_path, args.save_json, args.loop, args.record)


if __name__ == "__main__":
    main()

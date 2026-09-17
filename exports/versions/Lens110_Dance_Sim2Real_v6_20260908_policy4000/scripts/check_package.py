#!/usr/bin/env python3
"""Check the v6 Lens110 159D URDF-order deployment package."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "policy_4000" / "policy.onnx"
CKPT_PATH = ROOT / "policy_4000" / "policy.pt"
CONFIG_PATH = ROOT / "config" / "deploy_config_159_urdf.yaml"
MOTION_PATH = ROOT / "motions" / "tangbohushuoDJ_v2_100hz_flatfix.npz"

POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

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

OBS_NAMES = [
    "root_rot_tan_norm", "root_ang_vel_b", "joint_pos", "joint_vel",
    "ref_root_rot_tan_norm", "ref_joint_pos",
]


def main() -> None:
    required = [
        ONNX_PATH, CKPT_PATH, CONFIG_PATH, MOTION_PATH,
        ROOT / "config" / "deploy_config_159_urdf.yaml",
        ROOT / "replay" / "play_lens110_official_159_urdf.py",
        ROOT / "replay" / "mjcf" / "lens110_21dof.xml",
        ROOT / "robot" / "lens110_21dof.urdf",
        ROOT / "run_params" / "agent.yaml",
        ROOT / "run_params" / "env.yaml",
        ROOT / "README.md",
        ROOT / "docs" / "POLICY_CONTRACT.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing files: {missing}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config["observation_dim"] != 159 or config["action_dim"] != 21:
        raise RuntimeError("deploy config dimensions are not 159D input / 21D output")
    if config["observation_names"] != OBS_NAMES:
        raise RuntimeError("deploy config observation order mismatch")
    if config["joint_names"] != POLICY_JOINT_NAMES:
        raise RuntimeError("deploy config policy joint order mismatch")
    if config["action_joint_names"] != POLICY_JOINT_NAMES:
        raise RuntimeError("deploy config action joint order mismatch")
    if config["action_clip"] is not None or config["action_scale"] != 0.25:
        raise RuntimeError("action contract mismatch")
    if len(config["joint_stiffness"]) != 21 or len(config["joint_damping"]) != 21:
        raise RuntimeError("PD arrays must contain 21 entries")

    with np.load(MOTION_PATH) as motion:
        names = [str(name) for name in motion["joint_names"]]
        if names != USD_JOINT_NAMES:
            raise RuntimeError("motion npz is not in the expected Isaac/USD dataset order")
        if motion["joint_pos"].shape[1] != 21:
            raise RuntimeError("motion npz does not contain 21 joints")
        motion_frame_count = motion["joint_pos"].shape[0]

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_info = session.get_inputs()[0]
    out_info = session.get_outputs()[0]
    if in_info.shape != [1, 159] or out_info.shape != [1, 21]:
        raise RuntimeError(f"ONNX shape mismatch: in={in_info.shape}, out={out_info.shape}")
    output = session.run(None, {"obs": np.zeros((1, 159), dtype=np.float32)})[0]
    if output.shape != (1, 21) or not np.isfinite(output).all():
        raise RuntimeError("ONNX smoke test failed")

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    actor_input = tuple(state["model_state_dict"]["actor.0.weight"].shape)
    if actor_input != (512, 159):
        raise RuntimeError(f"checkpoint actor input mismatch: {actor_input}")

    print("PACKAGE OK")
    print(f"  observation_dim: {config['observation_dim']}")
    print(f"  action_dim:      {config['action_dim']}")
    print(f"  onnx:            {in_info.shape} -> {out_info.shape}")
    print(f"  checkpoint:      model_4000.pt, actor input {actor_input}")
    print(f"  motion:          {motion_frame_count} frames, USD dataset order")
    print(f"  policy order:    URDF grouped, {len(POLICY_JOINT_NAMES)} joints")


if __name__ == "__main__":
    main()

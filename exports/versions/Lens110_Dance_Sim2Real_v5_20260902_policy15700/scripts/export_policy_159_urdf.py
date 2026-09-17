"""Export the 161-D USD-order policy as a 159-D URDF-order deployment ONNX.

New deployment contract:
  obs = root_rot_tan_norm(6) + root_ang_vel_b(3)
      + joint_pos_urdf(21) + joint_vel_urdf(21)
      + ref_root_rot_tan_norm(24) + ref_joint_pos_urdf(4*21)
      = 159

The original checkpoint was trained with:
  root_ang_vel_w, foot_contact(2), and Isaac/USD interleaved joint order.
This wrapper keeps that checkpoint untouched and converts the deployment
interface around it.  foot_contact is fixed to zero because the new contract
does not expose it.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import yaml
from tensordict import TensorDict


def _load_actor_critic(ckpt_path: str):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    rsl_rl_path = os.path.join(
        project_root,
        "lens110RL", "lens110", "legged_lab_lbot", "rsl_rl",
    )
    sys.path.insert(0, rsl_rl_path)
    from rsl_rl.modules import ActorCritic  # noqa: E402

    obs = TensorDict({"obs": torch.zeros(1, 161)}, batch_size=[1])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    has_norm = any("normalizer" in k for k in state["model_state_dict"])
    policy = ActorCritic(
        obs=obs,
        obs_groups={"policy": ["obs"], "critic": ["obs"]},
        num_actions=21,
        actor_obs_normalization=has_norm,
        critic_obs_normalization=has_norm,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    return policy, has_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="policy_15700/policy.pt")
    parser.add_argument("--legacy-deploy", default="config/deploy_config_legacy_161.yaml")
    parser.add_argument("--out-onnx", default="policy_15700/policy_159_urdf.onnx")
    parser.add_argument("--out-deploy", default="config/deploy_config_159_urdf.yaml")
    args = parser.parse_args()

    policy, has_norm = _load_actor_critic(args.ckpt)

    # Legacy Isaac/USD interleaved order used by the checkpoint.
    usd_names = [
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
    # New URDF/default grouped order used by deployment.
    urdf_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "torso_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    ]
    urdf_to_usd = torch.tensor([usd_names.index(name) for name in urdf_names], dtype=torch.long)
    # Legacy input construction needs the inverse permutation:
    # legacy_obs[j] = urdf_obs[i] where urdf_names[i] == usd_names[j].
    usd_to_urdf = torch.argsort(urdf_to_usd).to(torch.long)

    class DeploymentPolicy159(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = copy.deepcopy(policy.actor)
            self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
            self.register_buffer("usd_to_urdf", usd_to_urdf)

        @staticmethod
        def rot_tan_to_matrix(root_rot_tan_norm: torch.Tensor) -> torch.Tensor:
            col0 = root_rot_tan_norm[:, 0:3]
            col2 = root_rot_tan_norm[:, 3:6]
            col1 = torch.linalg.cross(col2, col0, dim=1)
            return torch.stack((col0, col1, col2), dim=2)

        def forward(self, obs159: torch.Tensor) -> torch.Tensor:
            root_rot_tan = obs159[:, 0:6]
            root_ang_vel_b = obs159[:, 6:9]
            joint_pos_urdf = obs159[:, 9:30]
            joint_vel_urdf = obs159[:, 30:51]
            ref_root_rot_tan = obs159[:, 51:75]
            ref_joint_pos_urdf = obs159[:, 75:159].reshape(-1, 4, 21)

            root_rot = self.rot_tan_to_matrix(root_rot_tan)
            root_ang_vel_w = torch.bmm(
                root_rot, root_ang_vel_b.unsqueeze(-1)
            ).squeeze(-1)

            joint_pos_usd = joint_pos_urdf[:, self.usd_to_urdf]
            joint_vel_usd = joint_vel_urdf[:, self.usd_to_urdf]
            ref_joint_pos_usd = ref_joint_pos_urdf[:, :, self.usd_to_urdf].reshape(-1, 84)

            # The original checkpoint still expects these two dims.  The new
            # deployment contract does not expose contact, so lock it to zero.
            foot_contact_zero = obs159.new_zeros((obs159.shape[0], 2))
            legacy_obs161 = torch.cat(
                (
                    root_rot_tan,
                    root_ang_vel_w,
                    joint_pos_usd,
                    joint_vel_usd,
                    ref_root_rot_tan,
                    ref_joint_pos_usd,
                    foot_contact_zero,
                ),
                dim=1,
            )
            normalized = self.normalizer(legacy_obs161)
            actions_usd = self.actor(normalized)
            actions_urdf = actions_usd[:, self.urdf_to_usd]
            return actions_urdf

    wrapper = DeploymentPolicy159().eval()
    dummy = torch.zeros(1, 159, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(dummy)
        if out.shape != (1, 21) or not torch.isfinite(out).all():
            raise RuntimeError(f"Wrapper smoke test failed: shape={out.shape}, finite={torch.isfinite(out).all()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_onnx)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out_onnx,
        export_params=True,
        opset_version=17,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )

    with open(args.legacy_deploy, "r", encoding="utf-8") as f:
        legacy = yaml.safe_load(f)
    deploy = dict(legacy)
    deploy["observation_names"] = [
        "root_rot_tan_norm", "root_ang_vel_b", "joint_pos", "joint_vel",
        "ref_root_rot_tan_norm", "ref_joint_pos",
    ]
    deploy["observation_history_lengths"] = [1] * len(deploy["observation_names"])
    deploy["observation_dim"] = 159
    deploy["action_dim"] = 21
    deploy["joint_names"] = urdf_names
    deploy["policy_joint_names"] = urdf_names
    deploy["action_joint_names"] = urdf_names
    deploy["dataset_joint_order"] = "isaac_usd_interleaved_unchanged"
    deploy["training_contract"] = {
        "observation_dim": 161,
        "observation_names": [
            "root_rot_tan_norm", "root_ang_vel_w", "joint_pos", "joint_vel",
            "ref_root_rot_tan_norm", "ref_joint_pos", "foot_contact",
        ],
        "joint_order": "isaac_usd_interleaved_unchanged",
        "changed": False,
    }
    deploy["playback_contract"] = {
        "player": "replay/play_lens110_official_161_aligned.py",
        "onnx": "policy_15700/policy_161_usd_order.onnx",
        "joint_order": "isaac_usd_interleaved_unchanged",
        "changed": False,
    }
    deploy["deployment_onnx_only"] = True
    deploy["action_mode"] = "reference"
    deploy["action_scale"] = 0.25
    deploy["action_clip"] = None
    with open(args.out_deploy, "w", encoding="utf-8") as f:
        yaml.dump(deploy, f, allow_unicode=True, default_flow_style=None, sort_keys=False)

    print(f"checkpoint obs normalization: {has_norm}")
    print(f"exported onnx: {args.out_onnx}")
    print(f"exported deploy: {args.out_deploy}")
    print("new obs order:")
    for i, name in enumerate(deploy["observation_names"]):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()

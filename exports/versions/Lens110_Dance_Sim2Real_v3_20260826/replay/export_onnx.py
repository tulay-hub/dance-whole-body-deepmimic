"""把 rl_games 21 关节策略 checkpoint 导出为带观测归一化的 ONNX。

当前 Lens110 训练 (114 维观测 / 21 维残差动作, 实验A起) 用 rl_games PPO,
checkpoint 里保存的是归一化统计量 + actor MLP 权重。本脚本把它们打包进
一个 ONNX, 供 MuJoCo sim2sim 播放器 (mujoco_sim2sim.py) 直接使用。

观测/动作维度从 checkpoint 自动读取 (支持 111 或 114), 无需手改。

导出内容与训练完全一致:
  1. raw_obs -> clip(-5, 5)                     (wrapper clip_observations=5.0)
  2. (x - running_mean) / sqrt(running_var + 1e-5)
  3. clip(-5, 5)                                (RunningMeanStd 输出裁剪)
  4. actor MLP (111 -> 256 -> 128 -> 64, ELU) -> mu (64 -> 21)

注意: 动作裁剪 (clip_actions=1.0) 不放进 ONNX, 由播放器在缩放前裁剪,
与训练 env.step 的语义一致。

用法 (isaaclab 环境):
    python export_onnx.py \
        --checkpoint logs/rl_games/lens110_lab/<run>/nn/<ckpt>.pth
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn


CLIP_OBS = 5.0
EPS = 1e-5


class DeployActor(nn.Module):
    """obs clip -> RunningMeanStd -> actor MLP (确定性 mean 动作)。"""

    def __init__(
        self,
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        obs_dim: int,
        act_dim: int,
        units=(256, 128, 64),
    ):
        super().__init__()
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_var", obs_var.float())
        layers = []
        in_dim = obs_dim
        for u in units:
            layers += [nn.Linear(in_dim, u), nn.ELU()]
            in_dim = u
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(obs, -CLIP_OBS, CLIP_OBS)
        x = (x - self.obs_mean) / torch.sqrt(self.obs_var + EPS)
        x = torch.clamp(x, -CLIP_OBS, CLIP_OBS)
        return self.mu(self.mlp(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="rl_games .pth checkpoint")
    parser.add_argument("--output", default=None, help="输出 .onnx 路径 (默认与 checkpoint 同名)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    obs_dim = int(sd["running_mean_std.running_mean"].shape[0])
    act_dim = int(sd["a2c_network.mu.weight"].shape[0])

    model = DeployActor(
        obs_mean=sd["running_mean_std.running_mean"],
        obs_var=sd["running_mean_std.running_var"],
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
    # actor_mlp: 0/2/4 是 Linear (中间是 ELU), mu 是输出层
    mlp_sd = {
        k.replace("a2c_network.actor_mlp.", ""): v
        for k, v in sd.items()
        if k.startswith("a2c_network.actor_mlp.")
    }
    mu_sd = {
        k.replace("a2c_network.mu.", ""): v
        for k, v in sd.items()
        if k.startswith("a2c_network.mu.")
    }
    model.mlp.load_state_dict(mlp_sd)
    model.mu.load_state_dict(mu_sd)
    model.eval()

    output = args.output or os.path.splitext(args.checkpoint)[0] + ".onnx"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"exported: {output}")
    print(f"  input obs   (1, {obs_dim})   ->  output actions (1, {act_dim})")

    # 有 onnxruntime 就顺手验证形状/输出
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"obs": np.zeros((1, obs_dim), dtype=np.float32)})[0]
        print(f"onnxruntime verify: out.shape={out.shape}")
        assert out.shape == (1, act_dim)
    except ImportError:
        print("onnxruntime 未安装, 跳过验证")


if __name__ == "__main__":
    main()

# Lens110 Dance Sim2Real v3 (2026-08-26)

当前 21 关节训练配置的快照包, 供 MuJoCo/实机对照。

## 目录

```text
config/      训练环境配置 (lens110_lab_env_cfg.py, lens110.py, PPO yaml)
source/      观测/动作/奖励/命令/终止 源码
motions/     当前训练动作 npz (lens110_amp_100hz_flat.npz)
replay/      21 关节 MuJoCo 播放器 + pth->ONNX 导出脚本 + 111 维观测构建器
```

## 当前训练配置摘要

- 观测: 111 维单帧 (角速度3 + 重力3 + 参考位/速42 + 位/速偏差42 + 上动作21)
- 动作: 21 维残差, `q_des = 参考关节角 + scale * clip(action)`
- 奖励: T800 全套 (anchor/body/joint/leg 跟踪 + 脚部专项) + 脚部钉死惩罚
- 事件: 无; 观测噪声: 无
- PD: 强 PD (髋/膝 120/4, 踝 55/2, 腰 45/1.5, 臂 35/1.2)
- 物理: Isaac 500Hz / 策略 100Hz; MuJoCo 1000Hz / 策略 100Hz

详细顺序见 `ORDERING_REFERENCE.md`。

## 使用

```bash
# pth -> onnx (含观测裁剪 + 归一化)
python replay/export_onnx.py --checkpoint <ckpt.pth>

# MuJoCo 播放 (gmr 环境)
python replay/mujoco_sim2sim.py \
    --onnx <ckpt.onnx> \
    --motion motions/lens110_amp_100hz_flat.npz \
    --mjcf assets/mjcf/lens110_21dof_sim_flatfoot.xml
```

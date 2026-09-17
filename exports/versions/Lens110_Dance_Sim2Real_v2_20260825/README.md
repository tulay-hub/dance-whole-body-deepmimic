# Lens110 舞蹈 Sim2Real 部署包 v2 (2026-08-25)

基于统一 0.25 scale 从头训练的干净版本 (run 2026-08-25_10-45-16,
29000 轮, Isaac 966 分 / 74.6% 跑完, **MuJoCo 完整 36s 不倒**)。

## 与 v1 (42500) 的区别

| | v1 (42500) | **v2 (29000)** |
|---|---|---|
| scale | 统一 0.25 (旧策略续训) | **统一 0.25 (从头干净训练)** |
| 奖励 | 旧配置 | **腿部跟踪 std 0.3 + 脚底平行 -5/0.04** |
| MuJoCo | 36s 完整 | **36s 完整, root_z 更稳 (0.633~0.639)** |
| 训练方式 | 弱 PD 续训 | **从头 + 固定弱 PD + 改进奖励** |

## 包结构

```
Lens110_Dance_Sim2Real_v2_20260825/
├── README.md
├── ORDERING_REFERENCE.md
├── config/robot_humanoid_lens110_config.yaml
├── policy_29000/policy.onnx + policy.pt
├── motions/lens110_dance_100hz.npz
└── replay/
    ├── mujoco_sim2sim_official.py
    ├── lens110_dance_obs.py
    ├── sim2sim_headless_test.py
    ├── mjcf/lens110_21dof_sim_flatfoot.xml
    └── meshes/*.STL
```

## 播放

```bash
python replay/mujoco_sim2sim_official.py \
  --onnx policy_29000/policy.onnx \
  --motion motions/lens110_dance_100hz.npz \
  --mjcf replay/mjcf/lens110_21dof_sim_flatfoot.xml
```

headless 验证:

```bash
python replay/sim2sim_headless_test.py \
  --onnx policy_29000/policy.onnx \
  --motion motions/lens110_dance_100hz.npz \
  --mjcf replay/mjcf/lens110_21dof_sim_flatfoot.xml \
  --duration 36
```

## 关键配置

- 观测 126 (45+81), ONNX 已烘焙归一化;
- 动作: 统一 0.25, 腿 = default + 0.25*clip(a), 腰 = 0.25*a, 臂 = 参考开环;
- PD (播放端): 髋 100/8, 膝 40/5, 踝 20/20, 腰/臂 100/5;
- MuJoCo 1000Hz / 策略 100Hz; Isaac 500Hz / 100Hz; motion 100Hz。

## 训练来源

- run: `logs/rl_games/lens110_lab/2026-08-25_10-45-16`
- 配置: 统一 0.25 scale + 固定弱 PD + 踝/脚底改进奖励 (leg std 0.3, foot_parallel -5/0.04)

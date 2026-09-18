# 跳舞全身奖励结构

任务：`LeggedLab-Isaac--Deepmimic-Lens110-v0`，实现文件：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg.py`。

环境奖励由 7 个参考跟踪项、`alive=+0.2` 和 3 个正则项组成；最大权重是
`ref_track_dof_pos_error_exp=+0.8`。跟踪项使用指数误差，动作平滑使用
`action_rate_l2_scaled`，在 `scale=0.25` 后的关节目标空间计算。`base_contact`、`base_height`、
`bad_orientation`、参考结束和偏差阈值是终止门，不是正奖励。

逐项权重见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)；观测维度和动作语义见
[`docs/INTERFACE_CONTRACTS.md`](../../../docs/INTERFACE_CONTRACTS.md)。

## 训练权重如何理解 / Interpreting training weights

本文按本仓库当前代码说明训练机制；已有策略的复现参数以对应 run 的 `params/env.yaml`、`params/agent.yaml` 和部署配置为准。奖励混合系数、逐项环境奖励权重、优化器 loss 系数、专家样本比例以及课程采样范围是不同概念。

混合系数可以写成 85%/15% 这样的配置比例，但不能代表训练过程中实际累计奖励贡献；单项 reward 的数值范围、门控、控制步长和出现频率都不同。需要实际贡献占比时，应统计同一 run 中每项加权回报，而不是把配置权重归一化成百分比。

Configuration mixing coefficients are not measured reward contributions. Environment weights, optimizer coefficients, expert sampling and curriculum schedules describe different parts of training. Reproduce a saved policy with its own run snapshots.

## 参考动作指导、残差控制与 PPO 系数

DeepMimic 通过明确的参考误差奖励进行专家动作跟踪，训练以参考帧初始化（RSI），再由 PPO 学习参考姿态上的残差修正。全身 H 与侧滚都使用 `ReferenceJointPositionAction`：

```text
q_target = q_reference_current + 0.25 * action
L_PPO = L_clip + 1.0*L_value - 0.005*entropy
clip_actions = None
fixed_action_std = True; init_noise_std = 1.0
action_mean_l2_coef = 0.0
```

当前 runner 未启用动作硬裁剪，因此 `0.25` 是残差比例，不能声称 action 必在 [-1,1] 或 residual 必在 ±0.25 rad。H/159-D 不能混用 checkpoint，原因是输入维度、角速度坐标系和关节顺序不同，而不是参考中心残差公式不同。

参考跟踪的原始权重之和为 1.75：root position 0.15、quaternion 0.15、root linear velocity 0.10、root angular velocity 0.05、key-body position 0.30、DOF position 0.80、DOF velocity 0.10。若只在这七项内部比较系数，比例依次为 8.57%、8.57%、5.71%、2.86%、17.14%、45.71%、5.71%；这仅是配置系数占比，不是含 alive/惩罚后的实际回报占比。H alive=0.20，SideRoll alive=0.05，SideRoll 另有静止段根速度惩罚 -2.0。

当前 PPO 使用 32 rollout steps、5 epochs、4 mini-batches、初始 LR=1e-3（adaptive）、gamma=0.99、lambda=0.95、clip=0.2。动作均值额外 L2 系数为 0；环境 action-rate 权重 -0.001 仍有效，并在 0.25 缩放后的残差空间计算。

源码相对共享框架 `lens110/legged_lab_lbot/`：`source/legged_lab/legged_lab/tasks/locomotion/deepmimic/mdp/actions.py`，`config/lens110/lens110_deepmimic_env_cfg.py`、`lens110_deepmimic_env_cfg_159.py`、`lens110_deepmimic_env_cfg_sideroll.py` 与 `agents/rsl_rl_ppo_cfg.py`（config 相对同一 deepmimic 目录）。

English: H and SideRoll both use reference-centered residual actions. Current runner has no hard action clipping, fixed exploration std=1.0 and action-mean L2 coefficient=0. PPO value/entropy coefficients are 1.0/0.005. Seven tracking weights sum to 1.75, but normalizing those weights only compares configured tracking coefficients; it does not measure reward contribution. H/159-D policies differ in observation/order contracts.

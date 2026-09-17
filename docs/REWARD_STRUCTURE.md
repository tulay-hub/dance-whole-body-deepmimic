# 跳舞全身奖励结构

任务：`LeggedLab-Isaac--Deepmimic-Lens110-v0`，实现文件：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg.py`。

环境奖励由 7 个参考跟踪项、`alive=+0.2` 和 3 个正则项组成；最大权重是
`ref_track_dof_pos_error_exp=+0.8`。跟踪项使用指数误差，动作平滑使用
`action_rate_l2_scaled`，在 `scale=0.25` 后的关节目标空间计算。`base_contact`、`base_height`、
`bad_orientation`、参考结束和偏差阈值是终止门，不是正奖励。

逐项权重见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)；观测维度和动作语义见
[`docs/INTERFACE_CONTRACTS.md`](../../../docs/INTERFACE_CONTRACTS.md)。

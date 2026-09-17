# Lens110 Dance Sim2Real v5 - policy_15700

策略来源:

```text
lens110RL/lens110/legged_lab_lbot/logs/rsl_rl/lens110_deepmimic/2026-09-01_11-59-20/model_15700.pt
```

当前 Isaac 指标:

```text
iteration       15700
reward          53.67
mean episode    32.48s
finish rate     81.5%
root deviation  14.6%
bad orientation 0.0%
base contact    3.3%
```

MuJoCo aligned 验证:

```text
motion: tangbohushuoDJ_v2_100hz_flatfix.npz
duration: 34.5s
player: replay/play_lens110_official_161_aligned.py
result: 完成 3435/3452 帧，root 高度约 0.637~0.647m
```

## 包内容

```text
Lens110_Dance_Sim2Real_v5_20260902_policy15700/
├── policy_15700/policy.onnx           # RSL-RL actor + observation normalizer
├── policy_15700/policy.pt             # 原始 model_15700.pt checkpoint
├── config/deploy_config.yaml          # 关节顺序/PD/action合同/观测合同
├── run_params/agent.yaml              # 原始 PPO/RSL-RL 配置
├── run_params/env.yaml                # 原始环境配置
├── motions/tangbohushuoDJ_v2_100hz_flatfix.npz  # 100Hz 训练/播放动作
├── motions/tangbohushuoDJ_v2_100hz_flatfix.npz  # 100Hz 训练/播放动作
├── replay/play_lens110_official_161_aligned.py # MuJoCo aligned 播放器
├── replay/mjcf/lens110_21dof.xml      # 官方 MuJoCo 模型
├── replay/meshes/                     # MuJoCo 网格
├── robot/lens110_21dof.urdf           # Isaac 训练 URDF，用于质心/惯量/限位对齐
├── docs/lens110_obs_rewards.md        # 观测/奖励说明
└── docs/ORDERING_REFERENCE.md         # 关节/顺序参考
```

## Policy 合同

```text
observation dim: 161
action dim:      21
obs normalization: enabled inside ONNX
action mode:     reference residual
action scale:    0.25
action clip:     null
physics:         500 Hz
policy:          100 Hz
motion:          100 Hz
```

Observation order:

```text
root_rot_tan_norm(6)
+ root_ang_vel_w(3)
+ joint_pos(21)
+ joint_vel(21)
+ ref_root_rot_tan_norm(4*6=24)
+ ref_joint_pos(4*21=84)
+ foot_contact(2)
= 161
```

Important: MuJoCo free joint `data.qvel[3:6]` is body/local angular velocity. Isaac training uses world angular velocity. The aligned player converts it:

```python
root_ang_vel_w = quat_to_rotmat_wxyz(root_q) @ data.qvel[3:6]
```

Action order:

```text
 0 left_hip_pitch_joint        11 left_knee_joint
 1 right_hip_pitch_joint       12 right_knee_joint
 2 torso_yaw_joint             13 left_shoulder_yaw_joint
 3 left_hip_roll_joint         14 right_shoulder_yaw_joint
 4 right_hip_roll_joint        15 left_ankle_pitch_joint
 5 left_shoulder_pitch_joint   16 right_ankle_pitch_joint
 6 right_shoulder_pitch_joint  17 left_elbow_joint
 7 left_hip_yaw_joint          18 right_elbow_joint
 8 right_hip_yaw_joint         19 left_ankle_roll_joint
 9 left_shoulder_roll_joint    20 right_ankle_roll_joint
10 right_shoulder_roll_joint
```

Action semantics:

```text
q_des = reference_joint_pos[current_frame] + 0.25 * action
```

This export uses `action_clip: null`, matching the original `2026-09-01_11-59-20` training config.

## PD

```text
hip/knee:       40 / 5
torso yaw:      100 / 5
ankle pitch:    55 / 5
ankle roll:     55 / 5
arms:           20 / 5
effort:         hip/knee 80 N*m, others 36 N*m
```

## MuJoCo aligned 验证

From package root:

```bash
/home/tulay/miniconda3/envs/gmr/bin/python -u replay/play_lens110_official_161_aligned.py \
  --onnx policy_15700/policy.onnx \
  --deploy config/deploy_config.yaml \
  --motion motions/tangbohushuoDJ_v2_100hz_flatfix.npz \
  --mjcf replay/mjcf/lens110_21dof.xml \
  --foot_collision box
```

Options:

```text
--foot_collision box       official simplified flat foot boxes, default for this package
--foot_collision isaac     use ankle mesh collision from MJCF geoms
--rigid_feet 0.01          optional, foot contact stiffness time constant
--stiff_limits             optional, hard joint limits to reduce soft-limit overshoot
```

Shortcuts in player:

```text
Space pause
[ / ] slow / fast
R restart
Esc close
```

## Notes

- This package is based on `model_15700.pt` from run `2026-09-01_11-59-20`.
- It includes the corrected `root_ang_vel_w` conversion, which was the main reason earlier MuJoCo playback of this family looked unstable.
- The aligned player also aligns MuJoCo inertial parameters from `robot/lens110_21dof.urdf`, clears duplicate passive joint damping, and adds training-like armature by default.
- If a downstream simulator uses body-frame gyro directly, do not feed it as `root_ang_vel_w`; either convert to world frame or retrain using body-frame angular velocity consistently.

## Training / playback / deployment boundary

Training and local MuJoCo playback remain unchanged:

```text
training observation: 161D, root_ang_vel_w + foot_contact, Isaac/USD joint order
playback policy:      policy_15700/policy_161_usd_order.onnx
playback player:      replay/play_lens110_official_161_aligned.py
playback joint order: Isaac/USD joint order
```

The new file below is a deployment/export interface only:

```text
policy_15700/policy_159_urdf.onnx
config/deploy_config_159_urdf.yaml
```

Its external contract is:

```text
input:  159D, root_ang_vel_b, no foot_contact, URDF grouped joint order
output: 21D, URDF grouped joint order
```

The ONNX wrapper internally converts this back to the original 161D checkpoint contract:

```text
root_ang_vel_b -> root_ang_vel_w
foot_contact   -> zeros
URDF order     -> Isaac/USD order
Isaac/USD action order -> URDF action order
```

Do not use `policy_159_urdf.onnx` with the local 161D MuJoCo player. For local playback, continue using `policy_161_usd_order.onnx`.

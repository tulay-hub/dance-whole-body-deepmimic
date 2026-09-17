# Lens110 Dance Sim2Real v6 - policy_4000

本包是 159 维 URDF 顺序策略的原生实机部署包，不是旧 161 维 checkpoint 的 wrapper。

策略来源：

```text
lens110RL/lens110/legged_lab_lbot/logs/rsl_rl/lens110_deepmimic/2026-09-08_14-23-06/model_4000.pt
```

训练任务：

```text
LeggedLab-Isaac--Deepmimic-Lens110-159-v0
num_envs: 4096
observation: 159D
action: 21D
```

## ONNX 部署文件

```text
policy_4000/policy.onnx
policy_4000/policy_159_urdf.onnx
```

两个 ONNX 文件内容相同，前者给部署端使用，后者用于明确命名合同。
`policy.pt` 是原始 PyTorch checkpoint，只用于追溯和再导出。

## 部署合同

```text
input:  [1, 159]
output: [1, 21]
obs normalization: 已烘焙进 ONNX
action mode:       reference residual
action scale:      0.25
action clip:       null
physics:           500 Hz
policy:            100 Hz
motion:            100 Hz
root_ang_vel:      body frame
foot_contact:      removed
```

观测顺序：

```text
  0: root_rot_tan_norm        6
  6: root_ang_vel_b           3
  9: joint_pos_urdf          21
 30: joint_vel_urdf          21
 51: ref_root_rot_tan_norm   24  (4 x 6)
 75: ref_joint_pos_urdf      84  (4 x 21)
159
```

动作顺序与输入关节顺序一致，均为 URDF 默认分组顺序：

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
torso_yaw_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
```

动作语义：

```text
q_des = motion_reference_joint_pos[current_frame] + 0.25 * action
```

动作数据集 `tangbohushuoDJ_v2_100hz_flatfix.npz` 仍保持 Isaac/USD 交叉顺序不变。
实机端或播放端必须在加载动作时把 USD 顺序重排成上面的 URDF 顺序。

## PD

所有 PD 值已经按 URDF 策略顺序写入 `config/deploy_config_159_urdf.yaml`：

```text
hip / knee:       40 / 5
ankle pitch/roll: 55 / 5
torso yaw:        100 / 5
arms:             20 / 5
```

力矩限制：

```text
hip / knee / torso yaw: 80 N*m
ankle / arms:           36 N*m
```

## MuJoCo 本地验证

从包根目录运行：

```bash
python -u replay/play_lens110_official_159_urdf.py \
  --onnx policy_4000/policy.onnx \
  --deploy config/deploy_config_159_urdf.yaml \
  --motion motions/tangbohushuoDJ_v2_100hz_flatfix.npz \
  --mjcf replay/mjcf/lens110_21dof.xml \
  --foot_collision box
```

本包导出后已验证：

```text
ONNX shape: [1,159] -> [1,21]
playback:   3452 / 3452 帧
```

播放器快捷键：

```text
Space 暂停
[ / ] 减速 / 加速
R     重播
Esc   退出
```

## 实机接入注意

1. IMU 陀螺仪必须输入机身系角速度；如果 SDK 只给世界系，需要先用根姿态旋转矩阵转换。
2. 不要给策略输入 `foot_contact`，本模型训练时已经没有这个观测。
3. 关节位置、关节速度、动作输出都必须按 `joint_names` 的顺序映射，禁止依赖电机编号顺序。
4. 参考动作的 4 帧前瞻要按 100 Hz 连续推进，并重排为 URDF 顺序。
5. `action_clip` 为空，不要额外裁剪到 `[-1, 1]`。
6. `run_params/env.yaml` 中可能有 Hydra 记录的本机绝对路径，属于训练记录，不影响部署。

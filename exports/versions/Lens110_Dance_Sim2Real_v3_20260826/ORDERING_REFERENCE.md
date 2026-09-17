# Lens110 21 关节 T800 奖励版 Sim2Real 顺序对照表

当前包对应训练配置 (2026-08-26): 111 维观测 / 21 维残差动作 /
T800 奖励 + 脚部钉死 / 强 PD / 无事件无观测噪声。

## 1. 关节顺序 (USD 顺序 21) —— 训练资产/npz joint_pos/动作

```text
 0 left_hip_pitch_joint      11 left_knee_joint
 1 right_hip_pitch_joint     12 right_knee_joint
 2 torso_yaw_joint           13 left_shoulder_yaw_joint
 3 left_hip_roll_joint       14 right_shoulder_yaw_joint
 4 right_hip_roll_joint      15 left_ankle_pitch_joint
 5 left_shoulder_pitch_joint 16 right_ankle_pitch_joint
 6 right_shoulder_pitch_joint17 left_elbow_joint
 7 left_hip_yaw_joint        18 right_elbow_joint
 8 right_hip_yaw_joint       19 left_ankle_roll_joint
 9 left_shoulder_roll_joint  20 right_ankle_roll_joint
10 right_shoulder_roll_joint
```

## 2. 观测 114 维索引 (实验A: 含 base 线速度)

```text
[0:3]     base_ang_vel (body 系角速度)
[3:6]     base_lin_vel (body 系线速度, 新增)
[6:9]     gravity_b = R^T @ [0,0,-1] (IMU 重力投影)
[9:51]    参考关节角 + 速度 (42, USD)
[51:72]   关节角 - runtime_default (21, USD)
[72:93]   关节速度 (21, USD)
[93:114]  上一动作 (已裁剪 [-1,1], 21)
```

runtime_default_joint_pos = 参考动作第 0 帧关节角 (USD 顺序)。

## 3. 动作 21 维 (残差) + 逐关节 scale

```text
q_des = 参考关节角 + scale_i * clip(action_i, -1, 1)
```

scale (USD 顺序):

```text
 0 L_hip_pitch  0.20   11 L_knee       0.20
 1 R_hip_pitch  0.20   12 R_knee       0.20
 2 torso_yaw    0.10   13 L_sh_yaw     0.10
 3 L_hip_roll   0.10   14 R_sh_yaw     0.10
 4 R_hip_roll   0.10   15 L_ank_pitch  0.12
 5 L_sh_pitch   0.15   16 R_ank_pitch  0.12
 6 R_sh_pitch   0.15   17 L_elbow      0.15
 7 L_hip_yaw    0.10   18 R_elbow      0.15
 8 R_hip_yaw    0.10   19 L_ank_roll   0.08
 9 L_sh_roll    0.15   20 R_ank_roll   0.08
10 R_sh_roll    0.15
```

## 4. PD (训练, Isaac, USD 顺序)

```text
髋 p/r/y: 120/4   膝: 120/4   腰: 45/1.5
踝 p/r:   55/2    肩: 35/1.2  肘: 35/1.2
effort: 髋p/r + 膝 80, 其余 36
```

## 5. 时间轴

```text
Isaac:  physics 500Hz, decimation 5, policy 100Hz, action hold 5
MuJoCo: physics 1000Hz (sim_flatfoot timestep=0.001), policy 100Hz,
        action hold 10
Motion: fps 100, 3552 帧, 整数帧推进
```

## 6. 其他

- 观测裁剪 clip_observations = 5.0; 动作裁剪 clip_actions = 1.0
- 归一化 running_mean_std 烘焙进 ONNX (`export_onnx.py`), 输入原始观测
- npz 字段: `fps, joint_pos(USD21), joint_vel(USD21), body_pos_w(N,7,3),
  body_quat_w(N,7,4), body_lin_vel_w(N,7,3), body_ang_vel_w(N,7,3)`
- body 顺序: 0 pelvis, 1/2 左右踝roll, 3/4 左右肘, 5/6 左右肩roll

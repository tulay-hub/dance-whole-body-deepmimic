# Lens110 Dance Sim2Real v4 (2026-09-01)

策略: L2 训练版 (固定 std=1.0 + 均值 L2 1e-4), run `2026-09-01_15-29-00`, checkpoint 9800。

## 文件

```text
policy_9800/policy.onnx             # 导出策略 (含观测归一化)
policy_9800/policy.pt               # 原始 checkpoint
config/deploy_config.yaml           # 部署配置 (关节顺序/PD/action 语义)
motions/tangbohushuoDJ_v2_100hz_flatfix.npz  # 训练动作 (100Hz, 34.5s)
replay/play_lens110_official_161.py # MuJoCo 播放器 (161 维观测适配版)
replay/mjcf/lens110_21dof.xml       # 官方模型 (500Hz)
replay/meshes/                      # 模型网格
```

## 接口

- 观测: 161 维
  `[root_rot_tan_norm(6) | root_ang_vel(3) | joint_pos(21) | joint_vel(21) |
   ref_root_rot_tan_norm(24) | ref_joint_pos(84) | foot_contact(2)]`
  (关节为 USD 顺序; 脚触地 [左,右])
- 动作: 21 维参考中心残差, `q_des = 参考当前帧关节角 + 0.25 * action`
- 物理: 500Hz / 策略 100Hz
- PD: 腿 40/5, 腰 100/5, 踝 55/5, 臂 20/5 (训练值)

## 关节顺序 (USD/训练)

```text
 0 left_hip_pitch_joint       11 left_knee_joint
 1 right_hip_pitch_joint      12 right_knee_joint
 2 torso_yaw_joint            13 left_shoulder_yaw_joint
 3 left_hip_roll_joint        14 right_shoulder_yaw_joint
 4 right_hip_roll_joint       15 left_ankle_pitch_joint
 5 left_shoulder_pitch_joint  16 right_ankle_pitch_joint
 6 right_shoulder_pitch_joint 17 left_elbow_joint
 7 left_hip_yaw_joint         18 right_elbow_joint
 8 right_hip_yaw_joint        19 left_ankle_roll_joint
 9 left_shoulder_roll_joint   20 right_ankle_roll_joint
10 right_shoulder_roll_joint
```

## MuJoCo 播放

```bash
python replay/play_lens110_official_161.py \
  --onnx policy_9800/policy.onnx \
  --deploy config/deploy_config.yaml \
  --motion motions/tangbohushuoDJ_v2_100hz_flatfix.npz \
  --mjcf replay/mjcf/lens110_21dof.xml
```

播放器已包含的关键修复:
- 初始双脚同时贴地 (脚触地 obs=[1,1])
- 关节目标限位钳制 (与 Isaac 驱动器一致)
- 可选项: `--stiff_limits` (关节限位刚性化, 防止踝 roll 过冲侧翻)
- 快捷键: Space 暂停 | [ / ] 减速/加速 | R 重播 | Esc 退出

## 已知问题

- 该策略在 Isaac 表现正常 (约 78% 完整跳完), 但 MuJoCo 中因动作幅值仍偏大 (±30),
  左右踝/髋 roll 不对称动作导致约 1.5~2s 后摇晃倒地。
- 下一版计划: 均值 L2 惩罚提高到 1e-3 + clip_actions=2.0, 从根本上限制动作幅值。

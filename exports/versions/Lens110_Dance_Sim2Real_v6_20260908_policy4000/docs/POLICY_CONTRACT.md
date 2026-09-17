# Lens110 159D URDF 顺序部署合同

## 模型来源

```text
run:  2026-09-08_14-23-06
task: LeggedLab-Isaac--Deepmimic-Lens110-159-v0
ckpt: model_4000.pt
```

本 checkpoint 是重新训练后的原生 159 维策略，不是旧 161 维模型的接口包装。

## 观测布局

| 起始索引 | 维度 | 名称 | 说明 |
|---:|---:|---|---|
| 0 | 6 | `root_rot_tan_norm` | 当前根姿态 6D 表示 |
| 6 | 3 | `root_ang_vel_b` | 根角速度，机身系 |
| 9 | 21 | `joint_pos` | 当前关节位置，URDF 顺序 |
| 30 | 21 | `joint_vel` | 当前关节速度，URDF 顺序 |
| 51 | 24 | `ref_root_rot_tan_norm` | 参考根姿态，4 帧 × 6 |
| 75 | 84 | `ref_joint_pos` | 参考关节位置，4 帧 × 21，URDF 顺序 |

合计：

```text
6 + 3 + 21 + 21 + 24 + 84 = 159
```

## 动作布局

输出 21 维动作，顺序与 `joint_pos` 相同。执行时：

```text
q_des[i] = ref_joint_pos[current_frame][i] + 0.25 * action[i]
```

不要对 action 附加 `[-1, 1]` clip。

## 关节顺序

```text
0  left_hip_pitch_joint
1  left_hip_roll_joint
2  left_hip_yaw_joint
3  left_knee_joint
4  left_ankle_pitch_joint
5  left_ankle_roll_joint
6  right_hip_pitch_joint
7  right_hip_roll_joint
8  right_hip_yaw_joint
9  right_knee_joint
10 left_ankle_pitch_joint
11 right_ankle_roll_joint
12 torso_yaw_joint
13 left_shoulder_pitch_joint
14 left_shoulder_roll_joint
15 left_shoulder_yaw_joint
16 left_elbow_joint
17 right_shoulder_pitch_joint
18 right_shoulder_roll_joint
19 right_shoulder_yaw_joint
20 right_elbow_joint
```

## 动作数据顺序

`motions/tangbohushuoDJ_v2_100hz_flatfix.npz` 内部仍保持 Isaac/USD 交叉顺序：

```text
0  left_hip_pitch_joint
1  right_hip_pitch_joint
2  torso_yaw_joint
3  left_hip_roll_joint
4  right_hip_roll_joint
5  left_shoulder_pitch_joint
6  right_shoulder_pitch_joint
7  left_hip_yaw_joint
8  right_hip_yaw_joint
9  left_shoulder_roll_joint
10 right_shoulder_roll_joint
11 left_knee_joint
12 right_knee_joint
13 left_shoulder_yaw_joint
14 right_shoulder_yaw_joint
15 left_ankle_pitch_joint
16 right_ankle_pitch_joint
17 left_elbow_joint
18 right_elbow_joint
19 left_ankle_roll_joint
20 right_ankle_roll_joint
```

读取动作数据后必须按名称重排成上面的 URDF 策略顺序。

## 本地播放验证

```text
checkpoint: model_4000.pt
ONNX shape: [1,159] -> [1,21]
playback:   3452 / 3452 帧
```

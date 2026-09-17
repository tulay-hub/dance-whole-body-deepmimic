# 动作 npz 速度字段修复说明 (2026-09-02)

## 修了什么

`motions/tangbohushuoDJ_v2_100hz_flatfix.npz` 的角速度字段被写坏过，本包内已修正：

| 字段 | 修复前 | 修复后 | 说明 |
|---|---|---|---|
| `body_ang_vel_w` 峰值 | 199.75 rad/s | **18.51 rad/s** | 原值集中在第 0~2 帧，且 pelvis/左右踝这些全程不转的刚体也被写了约 200 rad/s |
| `root_ang_vel` | 199.75 rad/s | **0.00 rad/s** | 该动作的 pelvis 朝向全程锁定，真实角速度就是 0 |

修复方式是按"每个刚体自己的姿态四元数序列"重导角速度（先做符号连续化再取轴角导数），
线速度与关节速度一并按位置/角度重算，**姿态数据本身未改动**。

修复后校验（逐通道，存值与姿态导数的最大误差）：

```
pelvis / left_right_ankle_roll_link      误差 0.0            (静止, 应为 0)
left_elbow_link            线 2.46 角 15.64  误差 6e-08 / 5e-07
right_elbow_link           线 2.51 角 18.51  误差 6e-08 / 5e-07
left/right_shoulder_roll   线 0.43 角 14.8/14.9 误差 2e-08 / 5e-07
joint_vel max 10.40 rad/s   与 joint_pos 差分误差 3.6e-06
四元数模长最大偏差 4.1e-08
```

## 对本包使用方的影响

* **实机推理不受影响**：`config/deploy_config.yaml` 的 7 项观测里，
  `root_ang_vel_w` 取的是 IMU 实测，动作参考只用 `ref_joint_pos` 与 `ref_root_rot`，
  都不读 npz 的 `body_ang_vel_w`。
* **本包的 replay 脚本不受影响**：`replay/play_lens110_official_161_aligned.py` 在
  `settle_feet_init()` 里会无条件执行 `data.qvel[:] = 0.0`（12 轮校平循环），
  把第 407 行赋进去的 `ref_ang_vel_w[0]` 覆盖掉。已用坏数据跑过对照，逐帧输出一致。
* **v4 包 (Lens110_Dance_Sim2Real_v4_20260901) 的回放受影响的**：那版脚本是先
  `data.qvel[:] = 0.0`（第 250 行）再 `data.qvel[3:6] = ref_ang_vel_w[0]`（第 252 行），
  且没有 `settle_feet_init`，所以机器人一出生就带约 -199.75 rad/s 的偏航初速。
  v4 的 npz 也已同步成修复后的版本，用 v4 脚本重跑即可。
* 若把这份 npz 再用于训练（`whole_body_tracking` 的 `track_body_ang_vel` 一类奖励会
  直接跟踪 `body_ang_vel_w`），修复前那一项的目标是不可达的常值，会白占权重预算。

## 备份

原始文件在同目录 `tangbohushuoDJ_v2_100hz_flatfix.npz.bak_vel`，未包含在本压缩包内。

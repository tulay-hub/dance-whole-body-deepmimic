<p align="center"><a href="#zh">中文</a> &nbsp;|&nbsp; <a href="#en">English</a></p>
<a id="zh"></a>

# 01 · 跳舞全身（Dance Whole Body）

## 项目定位

这是双足人形机器人当前的全身舞蹈模仿主线：DeepMimic/reference residual policy，完整 21-DOF 输出，当前部署
 取 H 版 `161 -> 21` 契约。

- 训练任务：`LeggedLab-Isaac--Deepmimic-Lens110-v0`
- PLAY 任务：`LeggedLab-Isaac--Deepmimic-Lens110-Play-v0`
- 当前观测：`161`
- 动作：`21`
- 物理/策略频率：`500 Hz / 100 Hz`
- 动作语义：`q_des = reference_joint_pos + 0.25 * clipped_action`

## 训练架构和演示

完整的 DeepMimic/PPO 原理、161 维观测、奖励权重、终止门、导出和真机流程见 [`docs/TRAINING_ARCHITECTURE.md`](docs/TRAINING_ARCHITECTURE.md)。

<video controls width="720" src="docs/media/dance-demo.mp4"></video>

[打开或下载全身舞蹈演示视频](docs/media/dance-demo.mp4)

## 目录

```text
01_dance_whole_body/
├── framework/isaaclab_shared -> framework/isaaclab_shared
├── data/
│   ├── raw/bvh -> ../dance_dataset/raw_bvh/original_bvh
│   ├── processed/retargeted_actions/   # tangbohushuo / fast cars / bangbangbang / legacy set
│   └── training/deepmimic_motion -> shared DeepMimic motion directory
├── exports/
│   ├── packages/                       # 当前可交付舞蹈包
│   ├── versions/                       # v1..v6 版本包和解压目录
│   └── training_exports -> shared runs # checkpoint/play/export 结果入口
├── experiments/deepmimic_runs -> shared DeepMimic logs
└── docs/                               # reward reference 与 checkpoint snapshot
```

## 训练

```bash
./scripts/train.sh \
  --headless --num_envs 1024 --max_iterations 60000
```

单动作/回放前，先确认 `data/training/deepmimic_motion` 对应的 pkl 和配置中的
`motion_data_weights` 同名。需要导出时使用共享框架的
`scripts/rsl_rl/convert_pt_to_onnx_lens110.py`，导出后将 policy、deploy config、motion 和回放说明放进本项目
`exports/`。

## 版本边界

- H 版是当前 IMU + 关节编码器部署主线；F/KeyBody 版是 `251 -> 21` 训练/消融版本。
- 159-D 是独立 URDF 顺序契约，不能加载 H 版 checkpoint。
- `data/raw/bvh` 是共享原始源，处理后的动作才进入本项目 `data/processed`。

奖励逐项说明见 [`docs/REWARD_FRAMEWORKS.md`](docs/REWARD_FRAMEWORKS.md) 的全身舞蹈章节；接口细节见
[`docs/INTERFACE_CONTRACTS.md`](docs/INTERFACE_CONTRACTS.md)。

<a id="en"></a>

## English

This repository contains the whole-body dance imitation project. It uses DeepMimic with reference-motion residual tracking. The current H-line contract uses a 161-dimensional policy-side representation and 21 joint actions; keep the documented joint order and quaternion convention unchanged.

Install the local Isaac Lab environment, then run `./scripts/train.sh --headless --num_envs 1024`. Before exporting a checkpoint, verify the motion-data names, observation/action dimensions, replay result, and deployment configuration. The `data/`, `exports/`, `framework/`, and `docs/` directories keep source motion, training output, and release evidence separate.

Read `docs/REWARD_FRAMEWORKS.md` and `docs/INTERFACE_CONTRACTS.md` before reproducing a run. A successful simulation replay is not real-robot validation.

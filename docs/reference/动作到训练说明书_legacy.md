# Lens110 动作到训练说明书

> 本文档覆盖: 动作从 BVH/pkl 到训练 npz 的完整流程、重要文件与工具位置、
> 训练方法、以及训练产物的播放方式。

---

## 1. 总体流程

```
BVH 源动作 → 重定向 pkl → 100Hz 训练 npz → 训练策略 → checkpoint → ONNX/播放
```

当前项目以 **100Hz 策略 / 500Hz 物理 / 114 维观测 / 21 维动作** 为训练基线
(v29 运行合同结构, 策略频率按需求改为 100Hz)。

---

## 2. 目录结构与重要文件位置

### 2.1 根目录

| 路径 | 说明 |
|---|---|
| `lens110RL/lens110/` | 主工程 (Isaac Lab + 训练) |
| `动作1/ 动作2/ 动作3/` | 各舞蹈动作的 pkl / csv / npz / bvh |
| `资料/` | 参考工程 (legged_lab、实机测试包 v29 等) |

### 2.2 主工程关键文件

| 文件 | 作用 |
|---|---|
| `lens110_lab/scripts/rl_games/train.py` | 训练入口 |
| `lens110_lab/scripts/rl_games/play.py` | 策略播放入口 (Isaac 窗口) |
| `lens110_lab/scripts/rl_games/export_onnx.py` | checkpoint → ONNX 导出 |
| `lens110_lab/source/.../lens110_lab_env_cfg.py` | 环境配置 (观测/动作/奖励/终止/频率) |
| `lens110_lab/source/.../lens110_lab/assets/lens110.py` | 机器人资产 (kp/kd、初始姿态) |
| `lens110_lab/source/.../mdp/rewards.py` | 奖励函数实现 |
| `lens110_lab/source/.../mdp/observations.py` | 观测函数 |
| `lens110_lab/source/.../mdp/actions.py` | 残差参考动作实现 |
| `lens110_lab/source/.../motion/lens110_amp_100hz.npz` | **当前训练用的动作** |
| `lens110_tool.sh` | 动作转换/播放统一工具 (交互式) |
| `lens110_train.sh` | 训练包装脚本 (自动开 TensorBoard) |

### 2.3 训练产物位置

```
lens110RL/lens110/logs/rl_games/lens110_lab/<run时间戳>/
├── nn/                ← checkpoint 所在
│   ├── lens110_lab.pth              ← 最新/最佳
│   ├── last_..._ep_XXXX_rew_XX.pth  ← 每500轮存档
├── params/            ← 训练配置快照 (agent.yaml / env.yaml)
└── summaries/         ← TensorBoard 事件
```

---

## 3. 动作处理工具 (lens110_lab/scripts/bvh_tools/)

所有脚本用 **gmr conda 环境** 运行 (含 mujoco)。

| 工具 | 用法 | 作用 |
|---|---|---|
| `gmr_pkl_to_amp_npz.py` | `python gmr_pkl_to_amp_npz.py --pkl 动作.pkl --out 动作.npz` | pkl → 训练 npz (FK 算 body) |
| `gmr_pkl_to_npy.py` | `python gmr_pkl_to_npy.py --pkl 动作.pkl --out 动作.npy` | pkl → 关节角 npy |
| `npz_to_pkl_csv.py` | `python npz_to_pkl_csv.py --npz 动作.npz --out_pkl a.pkl --out_csv a.csv` | npz → pkl + 28列部署 csv |
| `resample_motion_fps.py` | `python resample_motion_fps.py --in a.npz --out b.npz --fps 100` | 动作重采样到目标 Hz |
| `smooth_motion_pkl.py` | `python smooth_motion_pkl.py --pkl a.pkl --out b.pkl` | 关节角平滑 |
| `limit_motion_speed.py` | `python limit_motion_speed.py --in a.npz --out b.npz --scale 0.5` | 限速 (压缩超速关节幅度, 时长不变) |
| `rebuild_transition.py` | `python rebuild_transition.py --pkl a.pkl --dance_start N --out b.pkl` | 重写起始过渡段 (站姿保持+插值) |
| `pkl_motion_player.py` | `python pkl_motion_player.py --pkl a.pkl` | MuJoCo 播放 pkl (含 root 运动) |
| `mujoco_motion_player.py` | `python mujoco_motion_player.py --npy a.npy` | MuJoCo 播放 npy (仅关节角) |
| `isaac_kinematic_player.py` | 见 `lens110_tool.sh play isaac npz` | Isaac 木偶模式回放 npz |
| `mujoco_sim2sim.py` | `python mujoco_sim2sim.py --onnx policy.onnx --motion a.npz --mjcf lens110_21dof.xml` | MuJoCo 闭环策略验证 |

### 3.1 统一工具 lens110_tool.sh

交互式选择文件, 自动索引所有 `动作*` 目录:

```bash
cd lens110RL/lens110
./lens110_tool.sh                        # 交互模式
./lens110_tool.sh convert pkl npz <pkl>  # pkl → npz
./lens110_tool.sh play mujoco pkl        # 选 pkl 播放
./lens110_tool.sh play isaac policy      # 选 checkpoint 播放
```

---

## 4. 从动作到训练 (标准流程)

### 4.1 准备动作 (以动作1 唐伯虎为例)

```bash
# 1) BVH 重定向 → pkl (GMR-master 流程)
conda activate gmr
python GMR-master/lens110/convert_bvh_to_lens110.py \
    --bvh 动作1/tangbohushuoDJ.bvh \
    --out_dir 动作1 --retarget_fps 120 --csv_fps 50

# 2) pkl → 训练 npz (120Hz)
cd lens110RL/lens110
python lens110_lab/scripts/bvh_tools/gmr_pkl_to_amp_npz.py \
    --pkl 动作1/pkl/tangbohushuoDJ_v2.pkl \
    --out /tmp/dance_120hz.npz

# 3) 重采样到 100Hz (与策略频率一致)
python lens110_lab/scripts/bvh_tools/resample_motion_fps.py \
    --in /tmp/dance_120hz.npz --out /tmp/dance_100hz.npz --fps 100

# 4) 替换训练动作
cp /tmp/dance_100hz.npz \
   lens110_lab/source/lens110_lab/lens110_lab/tasks/manager_based/lens110_lab/motion/lens110_amp_100hz.npz
```

> 可选后处理 (按需): `smooth_motion_pkl.py` 平滑, `limit_motion_speed.py` 限速,
> `rebuild_transition.py` 加重定过渡段。

### 4.2 开始训练

```bash
cd lens110RL/lens110

# 从头训练
./lens110_train.sh --num_envs 1024 --headless --max_iterations 30000

# 从 checkpoint 继续
./lens110_train.sh --num_envs 1024 --headless \
    --checkpoint logs/rl_games/lens110_lab/<run>/nn/lens110_lab.pth \
    --max_iterations 30000
```

`lens110_train.sh` 会自动启动 TensorBoard (localhost:6006)。

### 4.3 训练关键配置 (lens110_lab_env_cfg.py)

| 参数 | 当前值 | 说明 |
|---|---|---|
| 策略频率 | 100Hz (decimation=5 × sim.dt=1/500) | 物理 500Hz |
| 观测 | 114 维 | command 42 + 6 + 3 + 21 + 21 + 21 |
| 动作 | 残差参考 `q_des = 参考 + scale×action` | 21 维 |
| 奖励 | 13 项 | 见 `mdp/rewards.py` |
| episode | 40s | 覆盖 34.5s 纯舞蹈 |

---

## 5. 训练产物播放

### 5.1 播放策略 (Isaac 窗口, 训练效果)

```bash
conda activate isaaclab
cd lens110RL/lens110

TERM=xterm-256color /home/tulay/IsaacLab/isaaclab.sh \
    -p lens110_lab/scripts/rl_games/play.py \
    --task Template-Lens110-Lab-v0 \
    --num_envs 1 \
    --physics_hz 200 \
    --checkpoint logs/rl_games/lens110_lab/<run>/nn/last_..._rew_XX.pth
```

> `--physics_hz 200`: 播放专用轻量物理 (策略仍 100Hz), 避免卡顿。
> 不加则用训练配置 500Hz。

### 5.2 播放参考动作 (MuJoCo, 动作本身)

```bash
conda activate gmr
cd lens110RL/lens110/lens110_lab/scripts/bvh_tools
python pkl_motion_player.py --pkl 动作1/tangbohushuoDJ_v2_100hz.pkl
```

快捷键: 空格=暂停, ↑/↓=调速, R=复位 1x。

### 5.3 导出 ONNX (部署/外部验证)

```bash
conda activate isaaclab
cd lens110RL/lens110
python lens110_lab/scripts/rl_games/export_onnx.py \
    --checkpoint logs/rl_games/lens110_lab/<run>/nn/last_...pth \
    --out policy.onnx
```

### 5.4 MuJoCo sim2sim 闭环验证

```bash
conda activate gmr
python lens110_lab/scripts/bvh_tools/mujoco_sim2sim.py \
    --onnx policy.onnx \
    --motion lens110_lab/source/.../motion/lens110_amp_100hz.npz \
    --mjcf assets/mjcf/lens110_21dof.xml
```

---

## 6. 监控训练

- TensorBoard: http://localhost:6006 (由 `lens110_train.sh` 自动启动)
- 关键标量:
  - `rewards/iter` — 总奖励趋势
  - `Episode/Episode_Reward/motion_joint_position` — 关节跟踪 (接近 1.5 说明像)
  - `Episode/Episode_Reward/motion_global_anchor_pos` — 骨盆位置 (低说明站不稳/漂移)
  - `Episode/Episode_Termination/time_out` — 完整跑完率
  - `Episode/Metrics/motion/error_joint_pos` — 关节误差

## 7. 注意事项

- 训练用 gmr 环境生成 npz, 播放策略用 isaaclab 环境, MuJoCo 播放用 gmr 环境
- checkpoint 每 500 轮保存, 文件名含奖励值, 选最高奖励的播放/继续
- 动作超硬件限速 (10.4 rad/s) 时需先 `limit_motion_speed.py` 限速
- 参考部署模型维度可能与训练不同 (实机 dance 模型 771 输入/13 输出), 部署前需确认观测拼装

# Dance Deployment Package

Lens110 人形机器人舞蹈动作模仿部署包，包含训练好的 ONNX 策略和动作数据。

## 目录结构

```text
/home/ht/workspace/dance
├── data/dance/                    # 动作捕捉数据（.npz 格式）
│   ├── dance1_subject2_grounded.npz
│   └── Shakespeare_s_Bruises.npz
├── policy/                        # 训练好的策略
│   ├── dance1_subject2_lens110_self_collision/
│   │   └── exported/policy.onnx
│   └── Shakespeare_s_Bruises_lens110_self_collision/
│       └── exported/policy.onnx
└── sim2sim/                       # MuJoCo 仿真查看脚本
    └── sim2sim_mujoco.py
```

## 环境要求

- Python 3.8+
- MuJoCo
- ONNX Runtime
- NumPy

安装依赖：

```bash
pip install mujoco onnxruntime numpy
```

## 快速开始

### 1. 准备 MJCF 模型

需要 Lens110 的 MuJoCo XML 文件，例如：

```text
/home/ht/workspace/tienkung_lens110/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml
```

### 2. 运行策略控制模式

运行 `dance1_subject2` 舞蹈：

```bash
cd /home/ht/workspace/dance
python sim2sim/sim2sim_mujoco.py \
  --policy_path policy/dance1_subject2_lens110_self_collision/exported/policy.onnx \
  --mjcf_path /home/ht/workspace/tienkung_lens110/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml \
  --motion_file data/dance/dance1_subject2_grounded.npz
```

运行 `Shakespeare_s_Bruises` 舞蹈：

```bash
cd /home/ht/workspace/dance
python sim2sim/sim2sim_mujoco.py \
  --policy_path policy/Shakespeare_s_Bruises_lens110_self_collision/exported/policy.onnx \
  --mjcf_path /home/ht/workspace/tienkung_lens110/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml \
  --motion_file data/dance/Shakespeare_s_Bruises.npz
```

### 3. 交互控制

程序启动后，在 MuJoCo 查看器窗口中可以使用以下按键：

- **L 键** - 启动策略控制模式，机器人开始执行舞蹈动作
- **K 键** - 切换到站立模式（默认模式）
- **R 键** - 重置到初始状态
- **空格键** - 暂停/继续仿真

**注意**：程序默认处于站立模式，需要按 **L 键** 才会开始执行舞蹈动作。

## 命令行参数

### 必需参数

- `--policy_path` - ONNX 策略文件路径
- `--mjcf_path` - Lens110 MJCF/XML 文件路径
- `--motion_file` - 动作数据 .npz 文件路径

### 可选参数

```bash
--start_frame 100              # 从第 100 帧开始播放（默认 0）
--loop                         # 播放完成后循环（默认不循环）
--motion_play                  # 直接播放动作数据，不使用策略控制
--sim_dt 0.003333              # MuJoCo 仿真时间步长（默认 0.003333）
--decimation 10                # 策略控制频率降采样（默认 10）
--control_mode position        # 控制模式：position 或 external_pd（默认 position）
--disable_self_collisions      # 禁用自碰撞检测（调试用）
```

### 使用示例

**直接播放动作数据**（用于检查动作质量）：

```bash
python sim2sim/sim2sim_mujoco.py \
  --policy_path policy/dance1_subject2_lens110_self_collision/exported/policy.onnx \
  --mjcf_path /home/ht/workspace/tienkung_lens110/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml \
  --motion_file data/dance/dance1_subject2_grounded.npz \
  --motion_play
```

**从指定帧开始循环播放**：

```bash
python sim2sim/sim2sim_mujoco.py \
  --policy_path policy/dance1_subject2_lens110_self_collision/exported/policy.onnx \
  --mjcf_path /home/ht/workspace/tienkung_lens110/legged_lab/assets/model_humanoid_lens110/mjcf/lens110_21dof.xml \
  --motion_file data/dance/dance1_subject2_grounded.npz \
  --start_frame 100 \
  --loop
```

## 策略信息

### dance1_subject2

- **关节数**：21 DOF
- **观测维度**：114
- **动作维度**：21
- **特点**：包含自碰撞检测的舞蹈动作

### Shakespeare_s_Bruises

- **关节数**：21 DOF
- **观测维度**：114
- **动作维度**：21
- **特点**：包含自碰撞检测的舞蹈动作

## 技术细节

### 观测空间（114 维）

- Motion 输入：目标关节位置和速度（42 维）
- Motion 锚点方向：机器人基座相对方向（6 维）
- 基座角速度（3 维）
- 关节位置偏差（21 维）
- 关节速度（21 维）
- 上一步动作（21 维）

### 控制方式

- **position 模式**（默认）：使用 MuJoCo 内置的位置执行器
- **external_pd 模式**：通过外部 PD 控制器计算力矩

### 机器人配置

- **追踪体**：12 个关键 link（pelvis、hip、knee、ankle、torso、shoulder、elbow）
- **锚点**：torso_yaw_link
- **根体**：pelvis
- **站立高度**：0.68 米
- **足部几何体**：left_ankle_roll_collision, right_ankle_roll_collision

## 常见问题

### 1. 找不到 MJCF 文件

确保 `--mjcf_path` 指向正确的 XML 文件路径。如果路径不同，需要修改为你机器上的实际路径。

### 2. 机器人不动

程序默认处于 **站立模式**。在 MuJoCo 查看器窗口中按 **L 键** 启动策略控制。

### 3. 动作不稳定

- 检查是否使用了正确的 motion 文件和 policy 文件配对
- 尝试使用 `--motion_play` 参数直接播放动作数据，确认数据质量
- 确保 MJCF 模型与训练时使用的模型一致

### 4. GPU 警告

```
[W:onnxruntime:Default, device_discovery.cc:283 GetGpuDevices] Failed to detect devices...
```

这是 ONNX Runtime 的 GPU 检测警告，不影响运行。程序会自动使用 CPU 推理。

## 查看所有文件

```bash
cd /home/ht/workspace/dance
find . -maxdepth 4 -type f | sort
```

## 相关资源

- **训练框架**：Isaac Gym
- **机器人平台**：Lens110 (21 DOF 人形机器人)
- **仿真环境**：MuJoCo

# Lens110 舞蹈 Sim2Real 部署包 (2026-08-20)

本包参照 `Lens110_ThreeAction_Walk_Marktime11999_Minimal` 组织, 提供
Lens110 21-DOF 舞蹈策略的 MuJoCo sim2sim 回放与实机部署参考。

## 包结构

```
Lens110_Dance_Sim2Real_20260820/
├── config/robot_humanoid_lens110_config.yaml   # 运行时配置 (PD/动作/时间轴/初始姿态)
├── ORDERING_REFERENCE.md                       # 关节/动作/观测/PD 顺序对照 (sim2real 对照用)
├── policy_42500/
│   ├── policy.onnx                             # 42500 轮策略 (obs 126 -> action 13)
│   └── policy.pt                               # 同策略 PyTorch 权重
├── motions/lens110_dance_100hz.npz             # 舞蹈动作 3552 帧 @100Hz (含 1s 起始过渡)
├── replay/
│   ├── mujoco_sim2sim_official.py              # MuJoCo 播放器 (obs/PD 已修复)
│   ├── lens110_dance_obs.py                    # 126 维观测构造 (独立文件, 含逐维注释)
│   ├── sim2sim_headless_test.py                # headless 泛化测试工具
│   ├── mjcf/lens110_21dof_sim_flatfoot.xml     # MuJoCo 模型 (pelvis 2.3981, knee 2.443)
│   └── meshes/*.STL                            # 网格 (XML 相对引用)
```

## 播放 (MuJoCo)

依赖: `python>=3.10`, `mujoco`, `onnxruntime`, `numpy`

```bash
python replay/mujoco_sim2sim_official.py \
  --onnx policy_42500/policy.onnx \
  --motion motions/lens110_dance_100hz.npz \
  --mjcf replay/mjcf/lens110_21dof_sim_flatfoot.xml
```

快捷键: `Space` 暂停, `[`/`]` 减速/加速, `R` 重播, `Esc` 退出。

无窗口量化测试:

```bash
python replay/sim2sim_headless_test.py \
  --onnx policy_42500/policy.onnx \
  --motion motions/lens110_dance_100hz.npz \
  --mjcf replay/mjcf/lens110_21dof_sim_flatfoot.xml \
  --duration 36
```

## 关键接口 (已从 checkpoint/训练代码确认)

- 观测: **126 维** = 45 state (13q+13dq+13prev+3ω+3gravity_b) + 81 future
  (k/k+10/k+20 帧 x 21 关节 + 6 root)。**归一化已烘焙进 ONNX**,
  播放时无需外部归一化。
- 动作: **13 维 residual**。腿 12 = 帧0 default + 0.25*clip(a,-1,1);
  腰 1 = 0.25*a; 臂 8 = motion reference 开环。
- 控制: MuJoCo 物理 1000Hz / 策略 100Hz (decimation=10);
  Isaac 训练 500Hz / 策略 100Hz (decimation=5); motion 100Hz。
- PD: 髋 100/8, 膝 40/5, 踝 20/20, 腰 100/5, 臂 100/5。

观测构造位于 `replay/lens110_dance_obs.py` 的 `build_observation()`:
state 45 (13q+13dq+13prev+3ω+3gravity_b) + future 81 (3帧 x 21关节+6root),
每个维度都有注释, 供实机 SDK 对照实现。
  髋 100/8 为播放端加固 (2026-08-20): 基线 40/5 时脚距随舞蹈从 0.34m
  逐渐开到 0.925m; 100/8 后全程只到 0.421m 且 36s 不倒 (80/8 为 0.485m;
  120/4 阻尼太低 2.45s 倒, 已排除)。

## 为什么选 42500 轮

弱 PD 续训 (run 2026-08-19_09-48-29, 29000→49000) 中, 42500 是
"Isaac 奖励高 (1090.6) + 倒地率低 (13.5%/8.0%) + MuJoCo 完整 36s 不倒"
三者同时满足的最优点。49000 最终版 MuJoCo 仅 5.5s 即倒 (过拟合 Isaac)。

## 播放器修复记录 (重要)

`mujoco_sim2sim_official.py` 相对历史版本修复了 4 个问题:
1. 观测角速度坐标系 (body 系被误转 world 系);
2. 策略推理频率 10Hz -> 100Hz;
3. 动作 default 硬编码旧帧0 -> 取 motion 帧0;
4. geom margin 0.04 (接触检测距离) -> 恢复默认, margin 导致策略 1.2s 失稳。

## 训练资料位置

- 训练工程: `~/桌面/t800-urkl-tracker/lens110RL/lens110`
- 训练 run: `logs/rl_games/lens110_lab/2026-08-19_09-48-29`
- 动作生成: `lens110_lab/scripts/bvh_tools/make_start_transition.py`
- 监控: `lens110_lab/scripts/bvh_tools/monitor_training.py`
- 完整诊断: `docs/SIM2SIM_DIAGNOSIS_20260819.md`

## 实机部署注意

- 实机 `default_joint_pos` 应为动作帧0 (过渡版), 否则 13 维 residual 的
  baseline 会错位;
- 腰/臂目标: 腰走策略 (0.25*a), 臂走参考开环, 实机需按此实现;
- 建议先跑 headless 36s 验证 (不倒) 再上实机;
- v29 实机包不可信, 以本包配置 + 成功部署包 (ThreeAction) 为对齐基准。

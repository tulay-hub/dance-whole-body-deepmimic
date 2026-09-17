"""MuJoCo sim2sim: 用训练好的 ONNX 策略做闭环推理验证。

与 Isaac 训练环境对齐:
  - 策略 100Hz / 物理 1000Hz (sim_flatfoot 模型默认 timestep=0.001)
  - 观察 114 维: [base_ang_vel(3) | base_lin_vel(3) | gravity_b(3) | command(42) |
    joint_pos_rel(21) | joint_vel_rel(21) | last_action(21)]
  - 动作: q_des = 参考关节角 + scale * action
  - PD: kp/kd 与训练一致 (腿120/4 踝55/2 腰45/1.5 臂35/1.2)
  - default_joint_pos = 参考第 0 帧 (与训练资产一致)
  - ONNX 由 export_onnx.py 导出 (已含 obs 裁剪 + 归一化)

用法 (gmr 环境, 需 onnxruntime):
    python mujoco_sim2sim.py --onnx policy.onnx \
        --motion motion/lens110_amp_100hz_flat.npz \
        --mjcf assets/mjcf/lens110_21dof_sim_flatfoot.xml
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "assets", "mjcf", "lens110_21dof_sim_flatfoot.xml")
)

# 与训练环境一致的逐关节 action_scale (USD 顺序)
ACTION_SCALE = np.array([
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.1, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15,
], dtype=np.float32)

# USD 顺序关节名 (与训练 npz 一致)
USD_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# 训练 kp/kd (USD 顺序, 与资产配置一致)
KP = np.array([
    120, 120, 45, 120, 120, 35, 35, 120, 120, 35, 35, 120, 120, 35, 35, 55, 55, 35, 35, 55, 55,
], dtype=np.float64)
KD = np.array([
    4, 4, 1.5, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 2, 2,
], dtype=np.float64)

# 力矩限制 (USD 顺序, 与训练资产 lens110.py effort_limit_sim 一致)
EFFORT = np.array([
    80, 80, 36, 80, 80, 36, 36, 36, 36, 36, 36, 80, 80, 36, 36, 36, 36, 36, 36, 36, 36,
], dtype=np.float64)

POLICY_HZ = 100.0


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> float:
    """计算指定碰撞几何体 8 个角点在世界系的最低 z (用于贴地修正)。"""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return 0.0
    p = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    sz = model.geom_size[gid]
    low = 1e9
    for sx in (-1, 1):
        for sy in (-1, 1):
            for szz in (-1, 1):
                low = min(low, float((p + mat @ (sz * np.array([sx, sy, szz])))[2]))
    return low


def _quat_rpy_deg(q_wxyz: np.ndarray) -> tuple[float, float]:
    """wxyz 四元数 -> 世界 roll/pitch (度)。"""
    w, x, y, z = q_wxyz
    roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return float(roll), float(pitch)


def flatten_feet_init(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos: dict[str, int],
) -> None:
    """出生时把双脚底校平 (roll/pitch≈0), 解决"出生时脚没贴紧地面"。

    只调整左右踝的 pitch/roll 关节角, 不动其他关节和根姿态;
    用数值雅可比迭代, 左右脚符号自动适配。
    """
    body_name_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    pairs = [
        ("left_ankle_roll_link", "left_ankle_pitch_joint", "left_ankle_roll_joint"),
        ("right_ankle_roll_link", "right_ankle_pitch_joint", "right_ankle_roll_joint"),
    ]
    for bname, pj, rj in pairs:
        if bname not in body_name_to_id or pj not in joint_qpos or rj not in joint_qpos:
            continue
        bid = body_name_to_id[bname]
        pq, rq = joint_qpos[pj], joint_qpos[rj]
        for _ in range(10):
            r0, p0 = _quat_rpy_deg(data.xquat[bid])
            if abs(r0) < 0.05 and abs(p0) < 0.05:
                break
            # 直接比例校正: 踝 pitch/roll 对脚底 pitch/roll 增益≈1:1 (左右同号)
            # 一次最多校正 0.3 rad, 迭代收敛
            droll = float(np.clip(r0 * np.pi / 180.0, -0.3, 0.3))
            dpitch = float(np.clip(p0 * np.pi / 180.0, -0.3, 0.3))
            data.qpos[rq] -= droll
            data.qpos[pq] -= dpitch
            mujoco.mj_forward(model, data)


def quat_to_6d(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 -> 6D 旋转表示 (前两列)。"""
    w, x, y, z = q_wxyz
    # 旋转矩阵
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R[:, :2].flatten()


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_wxyz(q, v):
    """用 wxyz 四元数旋转向量 v (世界->body 用 inv)。"""
    w, x, y, z = q
    t2 = 2 * np.cross(np.array([x, y, z]), v)
    return v + w * t2 + np.cross(np.array([x, y, z]), t2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ref_pos = motion["joint_pos"]  # (N,21) USD 顺序
    ref_vel = motion["joint_vel"]
    ref_quat = motion["body_quat_w"][:, 0]  # (N,4) wxyz, pelvis
    n_frames = ref_pos.shape[0]
    print(f"[sim2sim] 动作 {n_frames} 帧 @{fps:.0f}Hz ({n_frames/fps:.1f}s)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    # 按模型 timestep 计算 decimation, 保证策略 100Hz:
    #   dt=0.001 (1000Hz) -> decimation 10; dt=0.002 (500Hz) -> decimation 5
    decimation = max(1, int(round(1.0 / (POLICY_HZ * model.opt.timestep))))
    print(f"[sim2sim] 模型 timestep={model.opt.timestep}, decimation={decimation} "
          f"(物理 {1.0/model.opt.timestep:.0f}Hz / 策略 {POLICY_HZ:.0f}Hz)")
    # 模型自带 Newton 求解器 (sim_flatfoot: Newton 50, tol 1e-10) 则保留;
    # 普通播放器模型默认 PGS, 强 PD 需要更高精度 -> 提升迭代
    if model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 300
        model.opt.tolerance = 1e-9

    # ---- 动力学 patch (参考 Lens110_ThreeAction 成功部署包 + 官方播放器) ----
    # 接触配置: 脚底参与碰撞, 其余 mesh 不参与自碰撞 (与成功包一致)
    foot_geom_names = [
        "left_ankle_roll_collision", "right_ankle_roll_collision",
        "left_ankle_pitch_collision", "right_ankle_pitch_collision",
    ]
    foot_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names}
    foot_geom_ids.discard(-1)
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname in {"floor", "ground", "plane"} or (gid in foot_geom_ids):
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 15
        else:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        # Isaac 地面材质: static=0.7, dynamic=0.7, combine=average
        model.geom_friction[floor_id] = [0.7, 0.005, 0.0001]

    # 关节: sim_flatfoot 自带 damping/frictionloss/armature=0.01,
    # 对 MuJoCo Newton 求解稳定性是必须的, 保留不清零。
    _has_nonzero_ipos = any(np.abs(model.body_ipos[i]).max() > 1e-8 for i in range(1, model.nbody))
    if not _has_nonzero_ipos and model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            did = model.jnt_dofadr[jid]
            model.dof_damping[did] = 0.0
            model.dof_frictionloss[did] = 0.0
            model.dof_armature[did] = 0.0

    # 运行时把 position actuator 的 kp/kv 改成训练值
    # MuJoCo position actuator: gainprm = [kp, kv, 0], biasprm = [-kp, -kv, 0]
    act_name_to_idx = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                       for i in range(model.nu)}
    # ctrl 索引 -> USD 关节索引 (actuator 按关节名排列, 顺序可能与 USD 不同)
    ctrl_to_usd = np.array([USD_JOINT_NAMES.index(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        for i in range(model.nu)], dtype=int)
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        kp, kd = KP[usd_i], KD[usd_i]
        # 2026-08-26 修复 (对照官方播放器 + Lens110_ThreeAction 成功包):
        # sim_flatfoot 的执行器是 <motor>, 默认 gaintype=FIXED / biastype=NONE,
        # biasprm 会被忽略, 必须显式设成 GAIN_FIXED + BIAS_AFFINE, PD 才生效。
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = kp
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -kp
        model.actuator_biasprm[aidx, 2] = -kd
        # 力矩限制 (与训练资产一致)
        model.actuator_forcelimited[aidx] = 1
        model.actuator_forcerange[aidx] = [-EFFORT[usd_i], EFFORT[usd_i]]
    # 关节点名 -> qpos 索引
    joint_qpos = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
    joint_dof = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_dof[name] = model.jnt_dofadr[i]
    usd_qpos = np.array([joint_qpos[n] for n in USD_JOINT_NAMES])
    usd_dof = np.array([joint_dof[n] for n in USD_JOINT_NAMES])

    # 初始: 参考第 0 帧 (关节角 + pelvis 位置/朝向)
    data.qpos[:3] = motion["body_pos_w"][0, 0]  # pelvis 位置
    data.qpos[3:7] = motion["body_quat_w"][0, 0]  # wxyz
    data.qpos[usd_qpos] = ref_pos[0]
    mujoco.mj_forward(model, data)
    # 出生校平 (flatten_feet_init) 已禁用: 当前策略从"带 3.6° 前倾的第 0 帧"
    # 训练, 出生瞬间把踝关节改平会导致观测/目标不匹配, 播放 22s 倒地。
    # 等训练侧把第 0 帧校平后再启用。
    # 贴地修正: 让脚底最低点贴到 ~0.002m 高度 (与官方播放器一致)
    lowest_z = min(_geom_lowest_z(model, data, g) for g in foot_geom_names)
    data.qpos[2] += 0.002 - lowest_z
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # joint_pos_rel 的基准 = 训练资产 default_joint_pos = 参考第 0 帧
    default_q = ref_pos[0].copy()

    last_action = np.zeros(21, dtype=np.float32)
    frame = 0
    policy_step = 0
    prev_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame < n_frames:
            # --- 策略推理 (每个策略步一次, 100Hz) ---
            # 2026-08-26 修复: 旧实现用 policy_step % decimation 做门控,
            # 导致实际只有 20Hz 推理; 官方播放器同样修复过这个问题。
            q = data.qpos[usd_qpos].copy()
            qvel = data.qvel[usd_dof].copy()
            robot_q_wxyz = data.qpos[3:7].copy()
            # MuJoCo free joint qvel[3:6] 本身就是 body 系角速度 (与训练一致)
            base_ang_vel = data.qvel[3:6].copy()
            # body 系线速度: 世界 qvel[0:3] 转到 body 系 (与训练 base_lin_vel 一致)
            base_lin_vel = quat_rotate_wxyz(
                quat_inv_wxyz(robot_q_wxyz), data.qvel[0:3]
            )
            # 世界重力在 body 系投影 (与实机 IMU 同款)
            gravity_b = quat_rotate_wxyz(
                quat_inv_wxyz(robot_q_wxyz), np.array([0.0, 0.0, -1.0])
            )
            obs = np.concatenate([
                base_ang_vel,                                # 3
                base_lin_vel,                                # 3
                gravity_b,                                   # 3
                ref_pos[frame], ref_vel[frame],              # 42
                q - default_q,                               # 21
                qvel,                                        # 21
                last_action,                                 # 21
            ]).astype(np.float32).reshape(1, -1)
            assert obs.shape[1] == 114, obs.shape
            out = sess.run(None, {"obs": obs})[0][0].astype(np.float32)
            # 与训练 env.step 一致: 动作先裁剪到 [-1,1] 再缩放
            out = np.clip(out, -1.0, 1.0)
            last_action = out
            q_des = ref_pos[frame] + ACTION_SCALE * out
            target_q = q_des

            # --- 物理步 (1000Hz, decimation 次 = 1 策略步) ---
            for _ in range(decimation):
                # position actuator: ctrl = 目标关节角, MuJoCo 内部做 PD
                data.ctrl[:] = target_q[ctrl_to_usd]
                mujoco.mj_step(model, data)

            # 同步画面到 viewer (否则窗口不刷新, 看起来没动)
            viewer.sync()

            frame += 1
            policy_step += 1
            if frame % 100 == 0:
                print(f"[sim2sim] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                      f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[usd_dof]).max():.2f}")

            # 实时速率
            if args.speed > 0:
                dt_target = 1.0 / (POLICY_HZ * args.speed)
                sleep = dt_target - (time.time() - prev_time)
                if sleep > 0:
                    time.sleep(sleep)
                prev_time = time.time()

    print(f"[sim2sim] 完成 {frame} 帧")


if __name__ == "__main__":
    main()

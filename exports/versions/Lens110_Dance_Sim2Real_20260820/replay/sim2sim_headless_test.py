"""MuJoCo headless policy 测试: Test C (policy ON + frame0) / Test D (policy ON + full motion)。

观测/动作接口与 mujoco_sim2sim_official.py 完全一致:
  126 = 45 state (13q+13dq+13prev+3w+3g) + 81 future (3帧 x 21关节+6root)
  onnx 内部已烘焙 running_mean_std 归一化。
"""

import argparse
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

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
CSV_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
USD_TO_CSV = [CSV_JOINT_ORDER.index(n) for n in USD_JOINT_NAMES]
CSV_TO_USD = [USD_JOINT_NAMES.index(n) for n in CSV_JOINT_ORDER]
KP = np.array([
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 100, 100, 20, 20, 100, 100, 20, 20,
], dtype=np.float64)
KD = np.array([
    8, 8, 5, 8, 8, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 20, 20,
], dtype=np.float64)
ACTION_SCALE = 0.25
POLICY_HZ = 100.0


def quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_wxyz(q, v):
    q = q / np.linalg.norm(q)
    qv = np.array([q[0], -q[1], -q[2], -q[3]])
    qv = qv / np.linalg.norm(qv)
    qv = np.array([q[0], -q[1], -q[2], -q[3]])
    return qv


def quat_apply(q, v):
    q = q / np.linalg.norm(q)
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    qq = np.zeros(4)
    def mul(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])
    return mul(mul(q, qv), q_conj)[1:]


def euler_xyz_from_quat(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mjcf", default=os.path.join(PROJ, "assets", "mjcf", "lens110_21dof_sim_flatfoot.xml"))
    ap.add_argument("--motion", required=True)
    ap.add_argument("--fix_frame0", action="store_true")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--viewer", action="store_true", help="开窗口 (对照播放器)")
    ap.add_argument("--player_contact", action="store_true",
                    help="应用播放器的接触配置 (friction 0.7, margin 0.04, 关闭 mesh 自碰撞)")
    ap.add_argument("--player_friction", action="store_true")
    ap.add_argument("--player_margin", action="store_true")
    ap.add_argument("--player_noself", action="store_true")
    ap.add_argument("--hip_kp", type=float, default=None, help="髋 stiffness 覆盖")
    ap.add_argument("--hip_kd", type=float, default=None, help="髋 damping 覆盖")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    ref_pos_usd = motion["joint_pos"]
    ref_quat = motion["body_quat_w"][:, 0]
    ref_pos_w = motion["body_pos_w"][:, 0]
    n_frames = ref_pos_usd.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    kp_arr = KP.copy()
    kd_arr = KD.copy()
    if args.hip_kp is not None or args.hip_kd is not None:
        for usd_i, name in enumerate(USD_JOINT_NAMES):
            if "_hip_" in name:
                if args.hip_kp is not None:
                    kp_arr[usd_i] = args.hip_kp
                if args.hip_kd is not None:
                    kd_arr[usd_i] = args.hip_kd
        print(f"[headless] 髋 PD 覆盖: kp={args.hip_kp or '40'} kd={args.hip_kd or '5'}")
    if args.player_contact or args.player_friction or args.player_margin or args.player_noself:
        foot_names = {"left_ankle_roll_collision", "right_ankle_roll_collision",
                      "left_ankle_pitch_collision", "right_ankle_pitch_collision"}
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname in {"floor", "ground", "plane"} or gname in foot_names:
                if args.player_contact or args.player_margin:
                    model.geom_margin[gid] = 0.04
            else:
                if args.player_contact or args.player_noself:
                    model.geom_conaffinity[gid] = 0
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fid >= 0 and (args.player_contact or args.player_friction):
            model.geom_friction[fid] = [0.7, 0.005, 0.0001]
    decim = max(1, int(round(1.0 / (POLICY_HZ * model.opt.timestep))))
    print(f"[headless] timestep={model.opt.timestep} decim={decim} fix_frame0={args.fix_frame0}")

    act_name_to_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        a = act_name_to_idx[name]
        model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[a, :] = 0.0
        model.actuator_gainprm[a, 0] = kp_arr[usd_i]
        model.actuator_biasprm[a, :] = 0.0
        model.actuator_biasprm[a, 1] = -kp_arr[usd_i]
        model.actuator_biasprm[a, 2] = -kd_arr[usd_i]
        effort = 80.0 if usd_i in (0, 1, 3, 4, 11, 12) else 36.0
        model.actuator_forcelimited[a] = 1
        model.actuator_forcerange[a, 0] = -effort
        model.actuator_forcerange[a, 1] = effort

    joint_qpos = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
                  for i in range(model.njnt)}
    joint_dof = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_dofadr[i]
                 for i in range(model.njnt)}
    csv_qpos = np.array([joint_qpos[n] for n in CSV_JOINT_ORDER])
    csv_dof = np.array([joint_dof[n] for n in CSV_JOINT_ORDER])

    # 初始状态 = 参考帧0 + 贴地
    data.qpos[:3] = ref_pos_w[0]
    data.qpos[3:7] = ref_quat[0]
    for i, name in enumerate(USD_JOINT_NAMES):
        data.qpos[joint_qpos[name]] = ref_pos_usd[0][i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    low = 1e9
    for gname in ("left_ankle_roll_collision", "right_ankle_roll_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        p = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        sz = model.geom_size[gid]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for szz in (-1, 1):
                    low = min(low, float((p + mat @ (sz * np.array([sx, sy, szz])))[2]))
    data.qpos[2] += 0.002 - low
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    default_q_csv = ref_pos_usd[0][CSV_TO_USD].astype(np.float64)
    last_action = np.zeros(13, dtype=np.float32)
    frame = 0
    total_steps = int(args.duration * POLICY_HZ)
    fall_time = None
    ankle_body = {
        n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    viewer = None
    if args.viewer:
        from mujoco import viewer as _viewer

        viewer = _viewer.launch_passive(model, data)
    for step in range(total_steps):
        if args.fix_frame0:
            f = 0
        else:
            f = min(frame, n_frames - 1)
        q_csv = data.qpos[csv_qpos].copy()
        qvel_csv = data.qvel[csv_dof].copy()
        q_usd = q_csv[USD_TO_CSV]
        qvel_usd = qvel_csv[USD_TO_CSV]
        q_train = q_usd[USD_TO_CSV]
        qvel_train = qvel_usd[USD_TO_CSV]
        robot_q_wxyz = data.qpos[3:7].copy()
        # MuJoCo free joint qvel[3:6] 是 body 系角速度, 与训练 root_ang_vel_b 一致
        base_ang_vel = data.qvel[3:6].copy()
        gravity_b = quat_apply(quat_inv_wxyz(robot_q_wxyz), np.array([0.0, 0.0, -1.0]))
        state = np.concatenate([
            q_train[:13].astype(np.float32),
            qvel_train[:13].astype(np.float32),
            last_action,
            base_ang_vel.astype(np.float32),
            gravity_b.astype(np.float32),
        ])
        offsets = np.array([0, 10, 20])
        frames = np.clip(f + offsets, 0, n_frames - 1)
        joints = ref_pos_usd[frames][:, USD_TO_CSV]
        roots = np.stack([
            np.concatenate([euler_xyz_from_quat(ref_quat[ff]), ref_pos_w[ff]])
            for ff in frames
        ])
        future = np.concatenate([joints.reshape(-1), roots.reshape(-1)]).astype(np.float32)
        obs = np.clip(np.concatenate([state, future]), -50.0, 50.0)
        assert obs.shape[0] == 126, obs.shape
        out = sess.run(None, {"obs": obs.reshape(1, -1)})[0][0].astype(np.float32)
        out = np.clip(out, -1.0, 1.0)
        last_action = out
        q_des_csv = default_q_csv.copy()
        q_des_csv[:12] = ACTION_SCALE * out[:12] + default_q_csv[:12]
        q_des_csv[12] = ACTION_SCALE * out[12]
        arm_usd = [USD_JOINT_NAMES.index(n) for n in CSV_JOINT_ORDER[13:21]]
        q_des_csv[13:21] = ref_pos_usd[f][arm_usd]
        for _ in range(decim):
            data.ctrl[:] = q_des_csv
            mujoco.mj_step(model, data)
        t = (step + 1) / POLICY_HZ
        if data.qpos[2] < 0.45 and fall_time is None:
            fall_time = t
        if t in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0) or step == total_steps - 1:
            qw, qx, qy, qz = data.qpos[3:7]
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])
            pw = data.qpos[:3]
            lf = R.T @ (data.xpos[ankle_body["left_ankle_roll_link"]] - pw)
            rf = R.T @ (data.xpos[ankle_body["right_ankle_roll_link"]] - pw)
            print(f"t={t:5.2f}s root_z={data.qpos[2]:.4f} max_jv={np.abs(data.qvel[6:]).max():.2f} "
                  f"脚距={abs(lf[1]-rf[1]):.3f} fall={fall_time if fall_time is not None else '-'}")
        if not args.fix_frame0:
            frame += 1
        if viewer is not None:
            viewer.sync()
    print(f"[headless] 结束 t={total_steps / POLICY_HZ:.1f}s 首次倒地={fall_time}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()

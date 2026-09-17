"""MuJoCo 无头 debug: 测量策略播放时双脚世界位姿是否真的固定。

输出 (左右脚分别):
  - foot x/y drift max  (相对 episode 开始锚点, 目标 <1cm)
  - foot z 范围          (目标 <1cm)
  - foot roll/pitch 范围 (目标 <2°)
  - ankle pitch/roll 关节速度 max/RMS (应明显 >0, 证明踝关节仍正常运动)
  - root pitch/roll 范围
  - 是否倒地/倒地时间

用法 (gmr 环境):
    python debug_foot_anchor_mj.py --onnx <policy.onnx>
"""

from __future__ import annotations

import argparse
import os

import mujoco as mj
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "assets", "mjcf", "lens110_21dof_sim_flatfoot.xml")
)

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
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
ANKLE_JOINTS = ["left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_ankle_pitch_joint", "right_ankle_roll_joint"]

SCALE = np.array([
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.1, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15,
], dtype=np.float32)
KP = np.array([
    120, 120, 45, 120, 120, 35, 35, 120, 120, 35, 35, 120, 120, 35, 35, 55, 55, 35, 35, 55, 55,
], dtype=np.float64)
KD = np.array([
    4, 4, 1.5, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 2, 2,
], dtype=np.float64)
EFFORT = np.array([
    80, 80, 36, 80, 80, 36, 36, 36, 36, 36, 36, 80, 80, 36, 36, 36, 36, 36, 36, 36, 36,
], dtype=np.float64)


def quat_rotate_wxyz(q, v):
    w, x, y, z = q
    t2 = 2 * np.cross(np.array([x, y, z]), v)
    return v + w * t2 + np.cross(np.array([x, y, z]), t2)


def quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rpy_deg(q_wxyz):
    w, x, y, z = q_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(roll), np.degrees(pitch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--max_frames", type=int, default=3552)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    ref_pos, ref_vel = motion["joint_pos"], motion["joint_vel"]
    n = min(len(ref_pos), args.max_frames)

    m = mj.MjModel.from_xml_path(args.mjcf)
    d = mj.MjData(m)
    if m.opt.solver != mj.mjtSolver.mjSOL_NEWTON:
        m.opt.solver = mj.mjtSolver.mjSOL_NEWTON
        m.opt.iterations = 300
        m.opt.tolerance = 1e-9
    dec = max(1, int(round(1.0 / (100.0 * m.opt.timestep))))

    # 接触配置 (同播放器)
    foot_geoms = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, x) for x in
                  ("left_ankle_roll_collision", "right_ankle_roll_collision",
                   "left_ankle_pitch_collision", "right_ankle_pitch_collision")}
    for gid in range(m.ngeom):
        gn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gid)
        if gn in {"floor", "ground", "plane"} or gid in foot_geoms:
            m.geom_contype[gid] = 1
            m.geom_conaffinity[gid] = 15
        else:
            m.geom_contype[gid] = 1
            m.geom_conaffinity[gid] = 0
    fid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
    if fid >= 0:
        m.geom_friction[fid] = [0.7, 0.005, 0.0001]

    act_idx = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    ctrl_to_usd = np.array([USD_JOINT_NAMES.index(mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i))
                            for i in range(m.nu)])
    for i, name in enumerate(USD_JOINT_NAMES):
        a = act_idx[name]
        m.actuator_gaintype[a] = mj.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[a] = mj.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = KP[i]
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_biasprm[a, 1] = -KP[i]
        m.actuator_biasprm[a, 2] = -KD[i]
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = [-EFFORT[i], EFFORT[i]]

    qmap = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i): m.jnt_qposadr[i] for i in range(m.njnt)}
    dmap = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i): m.jnt_dofadr[i] for i in range(m.njnt)}
    usd_qpos = np.array([qmap[x] for x in USD_JOINT_NAMES])
    usd_dof = np.array([dmap[x] for x in USD_JOINT_NAMES])
    body_name_to_id = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    foot_body_ids = np.array([body_name_to_id[x] for x in FOOT_NAMES])
    ankle_dof = np.array([dmap[x] for x in ANKLE_JOINTS])

    d.qpos[:3] = motion["body_pos_w"][0, 0]
    d.qpos[3:7] = motion["body_quat_w"][0, 0]
    d.qpos[usd_qpos] = ref_pos[0]
    mj.mj_forward(m, d)
    gids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, x) for x in
            ("left_ankle_roll_collision", "right_ankle_roll_collision",
             "left_ankle_pitch_collision", "right_ankle_pitch_collision")]
    low = min(min((d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (m.geom_size[g] * np.array([sx, sy, sz])))[2]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)) for g in gids)
    d.qpos[2] += 0.002 - low
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)

    anchors = d.xpos[foot_body_ids].copy()          # (2,3)
    anchor_quats = d.xquat[foot_body_ids].copy()    # (2,4)
    default_q = ref_pos[0].copy()
    last = np.zeros(21, dtype=np.float32)

    drift = np.zeros((n, 2, 3))
    rpy = np.zeros((n, 2, 2))
    ankle_vel = np.zeros((n, 4))
    root_rpy = np.zeros((n, 2))
    root_z = np.zeros(n)
    fail_frame = n
    low_count = 0

    for frame in range(n):
        q = d.qpos[usd_qpos].copy()
        qvel = d.qvel[usd_dof].copy()
        rq = d.qpos[3:7].copy()
        # MuJoCo free joint qvel[3:6] 本身就是 body 系角速度 (与播放器/训练一致)
        ang = d.qvel[3:6].copy()
        lin = quat_rotate_wxyz(quat_inv_wxyz(rq), d.qvel[0:3].copy())
        gb = quat_rotate_wxyz(quat_inv_wxyz(rq), np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate([ang, lin, gb, ref_pos[frame], ref_vel[frame],
                              q - default_q, qvel, last]).astype(np.float32).reshape(1, -1)
        out = sess.run(None, {"obs": obs})[0][0].astype(np.float32)
        out = np.clip(out, -1.0, 1.0)
        last = out
        target = ref_pos[frame] + SCALE * out
        for _ in range(dec):
            d.ctrl[:] = target[ctrl_to_usd]
            mj.mj_step(m, d)

        drift[frame] = d.xpos[foot_body_ids] - anchors
        for k in range(2):
            rpy[frame, k] = quat_to_rpy_deg(d.xquat[foot_body_ids[k]])
        ankle_vel[frame] = d.qvel[ankle_dof]
        root_rpy[frame] = quat_to_rpy_deg(d.qpos[3:7])
        root_z[frame] = d.qpos[2]

        if d.qpos[2] < 0.35:
            low_count += 1
            if low_count >= 20:
                fail_frame = frame
                break
        else:
            low_count = 0

    print(f"跑完帧数: {fail_frame} / {n}  ({fail_frame/100:.1f}s)"
          + ("" if fail_frame == n else "  -> 倒地!"))
    labels = ["LEFT", "RIGHT"]
    for k, lab in enumerate(labels):
        dx = np.abs(drift[:fail_frame, k, 0]).max()
        dy = np.abs(drift[:fail_frame, k, 1]).max()
        dz = np.abs(drift[:fail_frame, k, 2]).max()
        print(f"{lab} foot: drift_x={dx*100:.2f}cm  drift_y={dy*100:.2f}cm  "
              f"drift_z={dz*100:.2f}cm")
        print(f"          roll range={rpy[:fail_frame,k,0].max()-rpy[:fail_frame,k,0].min():.2f}°  "
              f"pitch range={rpy[:fail_frame,k,1].max()-rpy[:fail_frame,k,1].min():.2f}°")
    print(f"ankle pitch/roll vel: max={np.abs(ankle_vel[:fail_frame]).max():.2f} rad/s  "
          f"RMS={np.sqrt((ankle_vel[:fail_frame]**2).mean()):.2f} rad/s")
    print(f"root pitch range={root_rpy[:fail_frame,1].max()-root_rpy[:fail_frame,1].min():.2f}°  "
          f"root roll range={root_rpy[:fail_frame,0].max()-root_rpy[:fail_frame,0].min():.2f}°")


if __name__ == "__main__":
    main()

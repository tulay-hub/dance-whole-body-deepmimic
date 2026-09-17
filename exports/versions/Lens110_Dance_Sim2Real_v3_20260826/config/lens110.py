"""LENS110 机器人资产配置。

定义 ``LENS110_CFG`` (ArticulationCfg)，包括 USD 路径、初始姿态、关节限位与
执行器 PD 参数。供环境配置 ``Lens110LabEnvCfg`` 通过 ``scene.robot`` 引用。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# USD 文件位于 lens110/assets/usd/lens110_21dof.usd，相对本文件向上 5 层到 lens110/ 根。
_HERE = os.path.dirname(os.path.abspath(__file__))
LENS110_USD_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "assets", "usd", "lens110_21dof.usd")
)


LENS110_CFG = ArticulationCfg(
    prim_path="/World/lens110",
    spawn=sim_utils.UsdFileCfg(
        usd_path=LENS110_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 与 legged_lab 同款机器人训练配置一致: 训练时开启自碰撞
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 初始姿态对齐训练参考动作 lens110_amp_100hz_flat.npz 第 0 帧
        # (2026-08-18 起为"缓和前倾 4° -> 帧0"过渡版第 0 帧:
        #  root pitch -4°(原 -6.6°), 踝 pitch +1°补偿, 贴地修正后 root_z=0.6599)
        #   骨盆位置/朝向与关节角都取自参考首帧, 使 reset 后零动作即等于参考开局,
        #   消除"中性站姿 -> 追赶参考开局姿势"的瞬态 (如右腿开局内收的合并动作)。
        # 注意: default_joint_pos 也随之变为参考首帧, 官方动作 q_des[腿] = a*scale + default。
        pos=(0.024599, -0.269941, 0.659900),
        rot=(0.699323, -0.022836, -0.026560, -0.713947),
        joint_pos={
            "left_hip_pitch_joint": 0.205537,
            "right_hip_pitch_joint": 0.213849,
            "torso_yaw_joint": -0.020909,
            "left_hip_roll_joint": 0.158203,
            "right_hip_roll_joint": -0.191155,
            "left_shoulder_pitch_joint": 0.658771,
            "right_shoulder_pitch_joint": 0.629726,
            "left_hip_yaw_joint": 0.293051,
            "right_hip_yaw_joint": -0.304459,
            "left_shoulder_roll_joint": 0.941942,
            "right_shoulder_roll_joint": -1.125977,
            "left_knee_joint": -0.053091,
            "right_knee_joint": -0.068512,
            "left_shoulder_yaw_joint": -0.479678,
            "right_shoulder_yaw_joint": 0.511430,
            "left_ankle_pitch_joint": 0.058394,
            "right_ankle_pitch_joint": 0.076278,
            "left_elbow_joint": -1.131860,
            "right_elbow_joint": -1.114993,
            "left_ankle_roll_joint": -0.177711,
            "right_ankle_roll_joint": 0.199545,
        },
    ),
    actuators={
        # 对齐实机 dance PD (robot_humanoid_lens110_config.yaml control.dance):
        #   髋/膝 40/5 | 踝 pitch/roll 10/10 (实机 upper/lower 5/5 的等价映射)
        #   | 腰 100/5 | 肩/肘 100/5
        #   力矩限制: 髋p/r + 膝 80, 髋yaw/踝/腰/臂 36
        # 弱 PD + 高刚度臂: 对 MuJoCo/实机动力学差异更鲁棒, 手臂开环参考能紧跟动作。
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*knee_joint",
                "torso_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 80.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_yaw_joint": 36.0,
                ".*knee_joint": 80.0,
                "torso_yaw_joint": 36.0,
            },
            stiffness={
                # 2026-08-25: 按强 PD 训练 (旧 run 实证能站住的配置)
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*knee_joint": 120.0,
                "torso_yaw_joint": 45.0,
            },
            damping={
                ".*_hip_pitch_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 4.0,
                ".*knee_joint": 4.0,
                "torso_yaw_joint": 1.5,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch_joint", ".*ankle_roll_joint"],
            effort_limit_sim=36.0,
            # 2026-08-25: 按强 PD 训练
            stiffness=55.0,
            damping=2.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch_joint",
                ".*shoulder_roll_joint",
                ".*shoulder_yaw_joint",
                ".*elbow_joint",
            ],
            effort_limit_sim=36.0,
            # 2026-08-25: 按强 PD 训练
            stiffness=35.0,
            damping=1.2,
        ),
    },
)

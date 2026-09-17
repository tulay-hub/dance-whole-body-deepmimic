# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Lens110 舞蹈动作跟踪环境 (AMP / whole_body_tracking 全身跟踪)。

奖励/观测/终止/动作 scale 对齐 legged_lab 同款机器人舞蹈策略
(dance1_subject2_lens110_self_collision) 的验证配置:
    - 奖励: anchor 位置/朝向、body 位置/朝向、body 线/角速度 各 1.0 + action_rate -0.01
    - 观测: motion 命令 + anchor 朝向 + 机身角速度 + 关节位置/速度 + 上一动作
    - 终止: time_out + anchor 高度/朝向偏差 + 脚/肘离地
    - 动作: 关节位置, 逐关节 scale, use_default_offset=True
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from . import mdp as lens_mdp  # 本工程 MDP (motion/奖励/终止/观测 均已移植到本地)
from lens110_lab.assets.lens110 import LENS110_CFG

_HERE = os.path.dirname(os.path.abspath(__file__))
MOTION_NPZ = os.path.abspath(os.path.join(_HERE, "motion", "lens110_amp_100hz_flat.npz"))

# 跟踪的 body (anchor pelvis 在首位, 与运动数据一致)
TRACKED_BODIES = [
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

@configclass
class Lens110LabSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene with the Lens110 humanoid."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(500.0, 500.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.7,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            ),
        ),
    )
    # robot
    robot: ArticulationCfg = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 全身接触传感器 (对齐 T800: history_length/track_air_time 供
    # foot_contact_mismatch / feet_slip 奖励使用)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """动作: 残差参考关节位置 (从 T800 whole_body_tracking 移植)。

    q_des = 参考关节角 (MotionCommand) + scale * action
    策略只输出残差, 动作幅度天然贴合参考动作, 避免"缩水"动作。
    scale 取厂商运行时契约逐关节官方值。
    """

    joint_pos = lens_mdp.ResidualRefJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        command_name="motion",
        scale={
            # 厂商运行时契约逐关节 action_scale (USD 关节顺序)
            "left_hip_pitch_joint": 0.2,
            "left_hip_roll_joint": 0.1,
            "left_hip_yaw_joint": 0.1,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": 0.12,
            "left_ankle_roll_joint": 0.08,
            "right_hip_pitch_joint": 0.2,
            "right_hip_roll_joint": 0.1,
            "right_hip_yaw_joint": 0.1,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": 0.12,
            "right_ankle_roll_joint": 0.08,
            "torso_yaw_joint": 0.1,
            "left_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.15,
            "left_shoulder_yaw_joint": 0.1,
            "left_elbow_joint": 0.15,
            "right_shoulder_pitch_joint": 0.15,
            "right_shoulder_roll_joint": 0.15,
            "right_shoulder_yaw_joint": 0.1,
            "right_elbow_joint": 0.15,
        },
    )


@configclass
class ObservationsCfg:
    """观测: 111 维单帧 (T800 奖励替换后维度保持不变)。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 2026-08-27 实验A: 111 -> 114, 增加 base_lin_vel (body 系线速度),
        # 让策略能感知整体漂移并主动回正。
        #   [0:3]   base_ang_vel (body 系角速度)
        #   [3:6]   base_lin_vel (body 系线速度, 新增)
        #   [6:9]   gravity_b (世界重力在 body 系投影)
        #   [9:51]  command = ref_joint_pos(21) + ref_joint_vel(21)
        #   [51:72] joint_pos - default (21)
        #   [72:93] joint_vel - default (21)
        #   [93:114] last_action (21)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        gravity_b = ObsTerm(func=lens_mdp.gravity_b_obs)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # T800: 不加观测噪声
            self.enable_corruption = False
            self.concatenate_terms = True
            # 单帧 114 维 (与 build_actor_observation 一致)
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """motion 命令: 加载舞蹈动作全身状态, 提供 anchor/body 参考。"""

    motion = lens_mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=MOTION_NPZ,
        motion_id=0,
        resampling_time_range=(1e9, 1e9),
        anchor_body_name="pelvis",
        body_names=TRACKED_BODIES,
        pose_range={},
        velocity_range={},
        joint_position_range=(-0.05, 0.05),
        start_frame_from_beginning=True,
    )
    # ↑ start_frame_from_beginning=True: 每个 episode 固定从舞蹈第 0 帧起步,
    #   不再从随机中间帧开始 (随机帧 + 大抖动会导致起步即倒)。
    #   joint_position_range 从默认 ±0.52 收到 ±0.05, 起步更干净。


@configclass
class EventCfg:
    """T800 旧 run: 无域随机化事件。"""

    pass

@configclass
class RewardsCfg:
    """奖励: T800 (whole_body_tracking) 旧 run 配置 (2026-08-25_10-45-16 提取)。

    这是之前能正常跳过舞蹈的官方 13 维模式所用奖励:
      anchor pos/ori 1.0/1.0 + anchor_height 2.0 + feet_pos 2.0 +
      body 4 项 1.0 + joint 3.0 + leg_joint 15.0 +
      平滑 -1 + 越限 -10 + 非脚碰撞 -0.1 + 脚部专项 (接触错配/站稳/平行/间距)
      + alive 0.1 + 早倒惩罚 (最大 -100)。
    """

    motion_global_anchor_pos = RewTerm(
        func=lens_mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_anchor_height = RewTerm(
        func=lens_mdp.motion_anchor_height_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.1},
    )
    motion_global_anchor_ori = RewTerm(
        func=lens_mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=lens_mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3,
            # 只跟踪手部, 排除脚: 脚由 feet_planted / foot_parallel 负责钉死
            "body_names": [
                "left_elbow_link", "right_elbow_link",
                "left_shoulder_roll_link", "right_shoulder_roll_link",
            ],
        },
    )
    motion_body_ori = RewTerm(
        func=lens_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.4,
            "body_names": [
                "left_elbow_link", "right_elbow_link",
                "left_shoulder_roll_link", "right_shoulder_roll_link",
            ],
        },
    )
    motion_body_lin_vel = RewTerm(
        func=lens_mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": [
                "left_elbow_link", "right_elbow_link",
                "left_shoulder_roll_link", "right_shoulder_roll_link",
            ],
        },
    )
    motion_body_ang_vel = RewTerm(
        func=lens_mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": [
                "left_elbow_link", "right_elbow_link",
                "left_shoulder_roll_link", "right_shoulder_roll_link",
            ],
        },
    )
    motion_joint_position = RewTerm(
        func=lens_mdp.motion_joint_position_error_exp,
        weight=3.0,
        # 踝关节正常跟舞 (电机可以动), 脚底贴地由 foot_anchor_* 负责
        params={"command_name": "motion", "std": 0.5},
    )
    motion_leg_joint_position = RewTerm(
        func=lens_mdp.motion_leg_joint_position_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.3},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    # --- 脚底世界位姿锁定 (2026-08-26, 按用户目标重写) ---
    # 约束对象 = 脚底 (ankle_roll_link) 的世界位置/朝向, 与踝关节角度无关;
    # 锚点在 episode 开始记录, 全程固定; 无 contact gating;
    # 踝关节电机保持自由, 只为维持脚底位姿而运动。
    foot_anchor_xy = RewTerm(
        func=lens_mdp.foot_anchor_xy_lock_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "sigma_xy": 0.01,
        },
    )
    foot_anchor_z = RewTerm(
        func=lens_mdp.foot_anchor_z_lock_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "sigma_z": 0.01,
        },
    )
    foot_anchor_ori = RewTerm(
        func=lens_mdp.foot_anchor_ori_lock_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "sigma_ori": 0.03,
        },
    )
    alive = RewTerm(func=lens_mdp.alive, weight=0.1)
    early_termination = RewTerm(
        func=lens_mdp.early_termination_penalty,
        weight=1.0,
        params={"max_penalty": 100.0},
    )

@configclass
class TerminationsCfg:
    """终止: 超时 + anchor 高度/朝向偏差 (对齐 T800, 无 ee_body 终止)。

    T800 只有 anchor_pos/anchor_ori 两个终止项。去掉脚/肘 z 偏差终止,
    避免策略因怕触发终止而做保守小动作。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=lens_mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.40},
    )
    # ↑ 阈值 0.25 -> 0.40: 起始过渡段(站立->走两步)骨盆难免有偏差,
    #   太严导致策略频繁重置学不到内容。放宽后让策略有机会走完过渡段。
    anchor_ori = DoneTerm(
        func=lens_mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    # ↑ 阈值 1.0 -> 0.8 (约 47° -> 37°): 放宽到 1.0 后策略会利用过大的倾斜
    #   容差, 躯干越倾越远最终连累骨盆高度超差 (anchor_pos 终止率升至 97%);
    #   恢复 0.8 让倾斜状态尽早判死, 防止"倾斜->塌陷"的级联退化。


@configclass
class Lens110LabEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Lens110 motion-tracking environment."""

    # --- PD 退火课程 (2026-08-18) ---
    # 训练初期用强腿 PD (旧 run 实证能站住的 120/55), 按训练轮次线性退火
    # 到弱腿 PD (40/20), 让策略先学会站稳再适应目标动力学。
    pd_anneal_enabled: bool = True
    pd_anneal_max_epochs: int = 30000      # 由 train.py 覆盖为实际 max_epochs
    pd_anneal_horizon_length: int = 32     # 由 train.py 覆盖为 agent cfg horizon_length
    pd_anneal_start_ratio: float = 0.33    # 前 1/3 保持强 PD
    pd_anneal_end_ratio: float = 0.66      # 中 1/3 线性退火, 后 1/3 保持弱 PD
    pd_anneal_update_every: int = 100      # 每 100 次 env.step 调用更新一次
    # --- 分段 stand PD (2026-08-18) ---
    # 舞蹈准备段(开头站姿)与结束静止段用成功部署包 stand 高 PD,
    # 中间舞蹈主体用退火 PD; 边界 50 帧平滑过渡。
    # 2026-08-18 咨询结果: 别人未使用分段 PD, 该思路留存但默认关闭
    pd_stand_segments: bool = False
    pd_stand_start_frames: int = 650       # 过渡 100 + 原准备段 550 帧
    pd_stand_end_frame: int = 3250         # 结束静止段起点
    pd_stand_blend_frames: int = 50        # 边界过渡帧数 (0.5s)

    # Scene settings
    scene: Lens110LabSceneCfg = Lens110LabSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # 策略 100Hz, 物理 500Hz (5 substeps per policy)
        # 对齐 v29 运行合同结构, 按需求: policy=100Hz, control=500Hz
        self.decimation = 5
        self.episode_length_s = 40
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 500
        # 渲染间隔 = 2 个策略步: 避免窗口播放时 vsync(60fps) 拖慢策略步进
        # (训练 headless 不渲染, 此设置不影响训练; 播放更接近 100Hz 节奏)
        self.sim.render_interval = self.decimation * 2


@configclass
class OfficialObservationsCfg:
    """官方式观测: 771 = 675 (45 维/帧 x 15 历史) + 96 (3 帧前瞻参考)。

    对齐 infer_zero L110_dance_qpg_policy_0303.onnx 的输入结构:
      - state 组 (45x15=675): 13q + 13dq + 13prev + 3角速度 + 3姿态
      - future 组 (96): k/k+10/k+20 三帧 x (21 关节 + 11 root)
    """

    @configclass
    class StateCfg(ObsGroup):
        official_state = ObsTerm(
            func=lens_mdp.official_state_obs,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # 实验: 关闭历史 (history=1 -> state 45 维, 总观测 45+96=141)
            # 注意: 这不是官方 771 结构, 仅用于对比"无历史"训练效果
            self.history_length = 1

    @configclass
    class FutureCfg(ObsGroup):
        official_future = ObsTerm(
            func=lens_mdp.official_future_ref,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1  # 96 (不进历史)

    state: StateCfg = StateCfg()
    future: FutureCfg = FutureCfg()


@configclass
class OfficialActionsCfg:
    """官方式动作: 13 维 (12腿 + 1腰), 手臂 8 关节开环参考。"""

    joint_pos = lens_mdp.OfficialDanceJointActionCfg(
        asset_name="robot",
        command_name="motion",
        # 2026-08-21: 统一全关节 0.25 (与 42500 相同动作语义),
        # 从头重新训练干净版本, 保留当前踝/脚底改进奖励。
        scale=0.25,
    )


@configclass
class Lens110LabOfficialEnvCfg(Lens110LabEnvCfg):
    """官方式 dance 环境: 观测 771 / 动作 13 / 手臂开环 (sim2real 对齐)。

    继承现有 Lens110LabEnvCfg, 只覆盖观测与动作; 场景/命令/奖励/终止不变。
    与现有 21 维全身模式共存, 通过 gym task 名切换。
    """

    observations: OfficialObservationsCfg = OfficialObservationsCfg()
    actions: OfficialActionsCfg = OfficialActionsCfg()

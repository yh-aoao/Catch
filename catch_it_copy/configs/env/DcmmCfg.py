import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
XML_ARM_PATH = "urdf/xarm6_right.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
distance_thresh = 0.25

## Define the initial joint positions of the arm and the hand
arm_joints = np.array([
   0.0, 0.0, -0.0, 3.07, 2.25, -1.5
])

# roll 模式初始臂姿：回退到默认值，通过 _apply_wrist_flip 在 reset 后执行翻转动作
# 直接修改关节值（如 j5=0.0/-1.0）会导致 MuJoCo 零范数四元数 / 运动学奇点
roll_arm_joints = np.array([
    0.0, 0.0, -0.0, 3.07, 2.25, -1.5
])

hand_joints = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

## Define the reward weights
reward_weights = {
    "r_base_pos": 0.0,
    "r_ee_pos": 10.0,
    "r_precision": 10.0,
    "r_orient": 1.0,
    "r_touch": {
        'Tracking': 5,
        'Catching': 0.1
    },
    "r_constraint": 1.0,
    "r_stability": 20.0,
    "r_ctrl": {
        'base': 0.2,
        'arm': 1.0,
        'hand': 0.2,
    },
    "r_collision": -10.0,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1.5,              # in m/s
  'max_angular_speed': 4.8,             # in rad/s
  'max_speed_cmd': 10.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}

## Define IK
ik_config = {
    "solver_type": "QP", 
    "ps": 0.001, 
    "λΣ": 12.5, 
    "ilimit": 100, 
    "ee_tol": 1e-4
}

# Define the Randomization Params
## Wheel Drive
k_drive = np.array([0.75, 1.25])
## Wheel Steer
k_steer = np.array([0.75, 1.25])
## Arm Joints
k_arm = np.array([0.75, 1.25])
## Hand Joints
k_hand = np.array([0.75, 1.25])
## Object Shape and Size
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.5, 0.75])
## Observation Noise
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.01
k_obs_hand = 0.01
## Actions Noise
k_act = 0.025
## Action Delay
act_delay = {
    'base': [1,],
    'arm': [1,],
    'hand': [1,],
}

## 弹跳模式物理参数
# 弹性系数 COR → dampratio = (1-COR)*damp_scale
# COR=0.80 → dampratio=0.10, 弹跳 5~6 次
bounce_restitution = np.array([0.78, 0.78])
# 接触时间常数（秒）
bounce_solref_timeconst = np.array([0.008, 0.008])
# solref 阻尼比缩放系数
bounce_damp_scale = 0.50
# 自由关节阻尼（空气阻力）
bounce_joint_damping = np.array([0.00015, 0.00015])
# 地面摩擦系数（滑动、扭转、滚动）
bounce_friction = np.array([[0.50, 0.05, 0.015], [0.50, 0.05, 0.015]])
# 小球质量（kg）
bounce_mass = np.array([0.05, 0.05])
# 小球半径（m）
bounce_radius = np.array([0.04, 0.04])
# 初始释放高度（m）
bounce_init_height = np.array([0.90, 0.90])
# 初始水平速度（m/s）
bounce_init_speed = np.array([1.0, 1.0])
# 初始竖直速度（m/s），负=向下
bounce_init_vz = np.array([-0.2, -0.2])

## ==================== Roll 模式专用参数 ====================
## Roll 模式采用"掌心朝上舀球"策略（palm-up scooping）：
##   手放在地面上，掌心朝上，手指半闭合形成"栅栏"，
##   小球沿地面滚过来进入手掌，然后收紧手指抓住。
##   这种策略不需要复杂的手腕翻转，运动学上更自然。

# ---- 台面参数 ----
# 台面顶部高度（m），球心 = roll_table_height + radius + 0.002
# arm_base 在 z≈0.40，台面提高到 0.65 让球在臂的舒适工作区间内（比肩高 ~29cm）
roll_table_height = 0.55
roll_table_size = np.array([1.2, 1.6, 0.02])  # 半尺寸: x, y, z (宽 2.4m, 长 3.2m, 厚 4cm)
roll_table_pos = np.array([0.0, 2.5, 0.53])   # 台面中心世界坐标 (z=0.53, 顶部=0.55)

# ---- 跟踪阶段（Tracking）奖励权重 ----
# XY 即时奖励权重（高斯型，距离越近分越高）
roll_w_xy = 1.0
# XY 奖励的衰减参数（m），距离超过此值奖励下降明显
roll_sigma_xy = 0.45
# 靠近奖励权重（鼓励每一步更接近球）
roll_w_approach = 5.0
# 高度奖励权重（鼓励末端位于球同一高度，舀球策略不需要在上方）
roll_w_h = 0.4
# 高度奖励的衰减参数（m）
roll_sigma_h = 0.10
# 高度偏移（m），舀球策略中手与球同高（0=球面高度）
roll_height_offset = 0.0
# 掌心朝向球体奖励权重（Tracking 任务）。鼓励 link6 Y 轴（掌心法线）指向球体
# 这比固定掌心朝上更通用：掌心水平迎向滚来的球，运动学上完全可行
roll_w_palm_face = 8.0
# 掌心朝向球体额外加成（当掌心方向与球方向夹角 < arccos(roll_palm_face_cos) 时触发）
roll_w_palm_face_bonus = 3.0
# 掌心朝向球体奖励权重（Catching 任务 tracking 阶段）
roll_w_palm_face_c = 0.5
# 掌心朝向球体额外加成（Catching 任务 tracking 阶段）
roll_w_palm_face_c_bonus = 0.3
# 手指方向奖励权重（Tracking 任务）
# 鼓励手指指向球来的方向，形成拦截姿态
roll_w_finger_dir = 4.0
# 手指方向奖励额外加成（当手指 XY 方向与目标方向夹角 < 25° 时触发）
roll_w_finger_dir_bonus = 1.5
# 手指方向余弦阈值（cos(25°) ≈ 0.906），超过此值触发额外加成
roll_finger_dir_cos = 0.906
# 协同改善加成权重：当同一策略步内手掌和手指方向双向改善时额外奖励
roll_w_coordinated_improvement = 1.0

# ---- 跟踪失败判定阈值 ----
# XY 距离阈值（m），用于 tracking→grasping 阶段切换和靠近判定
roll_tracking_xy_thresh = 0.03
# Z 方向距离阈值（m），用于 tracking→grasping 阶段切换
roll_tracking_z_thresh = 0.03
# 掌心朝向球体的余弦阈值（cos(角度)），大于此值视为掌心朝球。0.94 ≈ 20° 以内
roll_palm_face_cos = 0.94
# 无靠近检测：最近 K 个策略步内没有靠近则判定失败
roll_no_approach_K = 10
# 无靠近检测：episode 开始后多少秒内不触发（给模型初始反应时间）
roll_no_approach_grace = 0.35
# 掌心朝向检测：episode 开始后多少秒内不触发（给模型调整掌心的时间）
roll_palm_grace = 0.8
# 无靠近检测：K 步内靠近量小于此值视为"没有靠近"（m）
roll_no_approach_eps = 0.002

# ---- 抓取阶段（Grasping）判定条件 ----
# 抓取成功：球的线速度低于此阈值视为"稳定"（m/s）
roll_catch_v_thresh = 0.05
# 抓取成功：连续低速的控制步数累加到该值即判定成功
roll_catch_N_control = 5
# 抓取失败：手掌接触球的最大等待步数，超时即判定失败
roll_catch_wait_steps = 20
# 抓取成功：MCP 关节平均屈曲超过此值才算真正抓取（rad），防止球停在手掌上就算成功
roll_catch_finger_thresh = 0.3
# 跟踪阶段手指预置 MCP 屈曲角（rad），0=全开，0.6=半闭合形成"栅栏"
roll_hand_ready_mcp = 0.6
# 跟踪阶段手指预置 DIP/指尖屈曲角（rad）
roll_hand_ready_dip = 0.3
# 跟踪阶段拇指预置屈曲角（rad）
roll_hand_ready_thumb = 0.3
# 跟踪阶段手指动作缩放系数（0=完全固定，0.3=允许模型微调）
roll_hand_action_scale = 0.3

# roll 模式固定底座（仅训练臂+手，排除底座协同问题）
roll_fix_base = True

# ---- 桌面高度奖励（新增）----
# 桌面高度锚点（m），手在桌面上方这个高度范围内获得奖励
roll_table_anchor_z = 0.59
# 桌面高度奖励权重
roll_w_table_h = 2.0
# 桌面高度奖励衰减参数（m），离桌面越远奖励越低
roll_sigma_table_h = 0.08
# 桌面下方惩罚权重（手绝不能穿到桌面下方）
roll_w_below_table_penalty = -5.0

# =============================================================================
# Bounce 模式专用参数
# 弹跳模式与滚动模式的物理特性完全不同（3D 轨迹 vs 贴地滚动），
# 因此需要独立的奖励权重和判定阈值。
# =============================================================================

# ---- Bounce Tracking 奖励权重 ----
bounce_w_3d = 1.0              # 3D 位置奖励权重（替代 roll_w_xy）
bounce_sigma_3d = 0.45         # 3D 距离衰减参数（m），与 roll_sigma_xy 保持一致
bounce_w_approach = 5.0        # 3D 靠近增量奖励权重
bounce_w_palm_face = 8.0       # 掌心朝向球体奖励权重（Tracking，替代固定掌心朝下）
bounce_w_palm_face_bonus = 3.0 # 掌心朝向球体额外加成（夹角 < 20° 时触发）
bounce_palm_face_cos = 0.94    # 掌心朝向球体的余弦阈值（cos(20°) ≈ 0.94）
bounce_w_finger_dir = 8.0      # 手指方向奖励权重
bounce_w_finger_dir_bonus = 3.0# 手指方向额外加成
bounce_finger_dir_cos = 0.906  # 手指方向余弦阈值（cos(25°) ≈ 0.906）
bounce_w_vel_match = 2.0       # 速度匹配奖励权重
bounce_sigma_vel = 0.3         # 速度匹配衰减参数（m/s）
bounce_w_coordinated = 2.0     # 协同改善加成（手掌+手指方向同时改善）

# ---- Bounce 阶段切换阈值 ----
bounce_tracking_xy_thresh = 0.03   # XY 距离阈值（m）
bounce_tracking_z_thresh = 0.03    # Z 距离阈值（m）
bounce_palm_face_cos_stage = 0.90  # 掌心朝向球体余弦阈值（阶段切换用，cos(25.8°) ≈ 0.90）

# ---- Bounce 终止判定 ----
bounce_no_approach_K = 10          # 无靠近检测：最近 K 个策略步
bounce_no_approach_grace = 0.35    # 无靠近检测：宽限期（s）
bounce_no_approach_eps = 0.002     # 无靠近检测：最小靠近量（m），基于 3D 距离

# ---- Bounce 抓取阶段判定 ----
bounce_catch_v_thresh = 0.05       # 抓取成功：球速低于此值（m/s）
bounce_catch_N_control = 5         # 抓取成功：连续低速步数
bounce_catch_wait_steps = 20       # 抓取失败：最大等待步数
bounce_catch_finger_thresh = 0.3   # 抓取成功：MCP 关节平均屈曲超过此值才算真正抓取（rad）
bounce_catch_require_palm_contact = True  # 抓取成功必须手掌接触球
bounce_hand_ready_mcp = 0.5        # 跟踪阶段手指预置 MCP 屈曲角（rad），0=全开
bounce_hand_ready_dip = 0.2        # 跟踪阶段手指预置 DIP/指尖屈曲角（rad）
bounce_hand_ready_thumb = 0.2      # 跟踪阶段拇指预置屈曲角（rad）
bounce_hand_action_scale = 0.3     # 跟踪阶段手指动作缩放系数（0=完全固定）

# 灵巧手碰撞几何体摩擦系数（减缓小球弹开）
# [滑动摩擦, 扭转摩擦, 滚动摩擦]，默认 MuJoCo 为 [1.0, 0.005, 0.0001]
# 增大滑动摩擦可让小球接触手掌时消耗更多动能，减少弹飞
bounce_hand_friction = np.array([2.0, 0.5, 0.1])

# =============================================================================
# Throw_Basket 模式专用参数（抛球入篮）
# 小球初始在手掌中，机械臂执行抛掷动作将球投向篮筐。
# =============================================================================

# ---- 篮筐参数 ----
# 篮筐中心世界坐标（m），从 arm_base 前方约 1.8m、高 0.9m 处
basket_center = np.array([0.0, 2.2, 0.9])
# 篮筐半径（m），定义一个圆形目标区域
basket_radius = 0.15
# 篮筐高度（m），从篮筐中心向下的深度
basket_depth = 0.3
# 篮筐颜色 RGBA
basket_rgba = [1.0, 0.4, 0.0, 0.7]

# ---- 抛球物理参数 ----
# 小球半径（m）
basket_ball_radius = 0.04
# 小球质量（kg）
basket_ball_mass = 0.05
# 持球时长（s），球在手中稳定后再抛出
basket_hold_duration = 0.3
# ---- 抛球入篮奖励权重 ----
# 球到篮筐中心的 3D 距离奖励（高斯型）
basket_w_dist = 10.0
# 球到篮筐距离衰减参数（m）
basket_sigma_dist = 0.3
# 入篮成功奖励（球进入篮筐区域，即距离 < basket_radius）
basket_w_score = 50.0
# 靠近奖励权重（鼓励球向篮筐移动）
basket_w_approach = 5.0
# 篮筐上方奖励权重（鼓励球从上方接近篮筐）
basket_w_above = 2.0
# 控制惩罚权重（惩罚底座/臂动作突变）
basket_w_ctrl_base = 0.1
basket_w_ctrl_arm = 0.5
basket_w_ctrl_hand = 0.1

# ---- 抛球入篮终止判定 ----
# 球落地则判定失败
basket_floor_z = 0.0
# 球飞出太远判定失败（篮筐中心距离超过此值）
basket_fail_dist = 2.5
# 最大 episode 时长（s）
basket_max_time = 4.0

# 固定底座（True=只训练臂+手，False=底盘也参与）
basket_fix_base = True

## Define PID params for wheel drive and steering.
# driving
Kp_drive = 5
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -200
ulim_drive = 200
# steering
Kp_steer = 50.0
Ki_steer = 2.5
Kd_steer = 7.5
llim_steer = -50
ulim_steer = 50

## Define PID params for the arm and hand. 
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = np.array([4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      1e-1, 1e-1, 1e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      1e-2, 1e-2, 2e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0
hand_mask = np.array([1, 0, 1, 1,
                      1, 0, 1, 1,
                      1, 0, 1, 1,
                      0, 1, 1, 1])
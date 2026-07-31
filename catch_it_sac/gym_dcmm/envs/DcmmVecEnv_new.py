"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the RL environment for the DCMM task
       DCMM任务的强化学习环境实现（移动底盘+机械臂+机械手的抓取/跟踪任务）
"""
import os, sys
# 添加上级目录和 gym_dcmm 目录到系统路径（解决模块导入问题）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 添加到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加到 gym_dcmm 目录
import argparse
import math
print(os.getcwd())  # 打印当前工作目录（调试用）
print("sys.path:", sys.path[:5])  # 打印前 5 个路径
import configs.env.DcmmCfg as DcmmCfg  # 导入环境配置参数（物理参数/奖励权重等）
import cv2 as cv  # 图像处理
import numpy as np
import mujoco  # Mujoco仿真核心库
import mujoco.viewer  # Mujoco可视化界面
import gymnasium as gym  # 强化学习环境标准库
from gymnasium import spaces  # 定义观测/动作空间
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM  # DCMM机器人Mujoco封装类
from gym_dcmm.utils.ik_pkg.ik_base import IKBase  # 逆运动学基础类
import copy
from termcolor import colored  # 终端颜色输出
from decorators import *  # 自定义装饰器（未展示）
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer  # Mujoco渲染器
from gym_dcmm.utils.util import *  # 工具函数（如坐标转换/四元数处理）
import xml.etree.ElementTree as ET  # XML解析（修改Mujoco模型）
from scipy.spatial.transform import Rotation as R  # 旋转变换
from collections import deque  # 双端队列（存储历史数据）

# os.environ['MUJOCO_GL'] = 'egl'  # 设置Mujoco渲染后端（注释掉则用默认）
np.set_printoptions(precision=8)  # 设置numpy输出精度（8位小数）

# ===================== 全局变量（键盘控制）=====================
paused = True  # 仿真暂停标志
cmd_lin_y = 0.0  # 底盘线速度y轴指令
cmd_lin_x = 0.0  # 底盘线速度x轴指令
cmd_ang = 0.0  # 底盘角速度指令
trigger_delta = False  # 机械臂位置增量触发
trigger_delta_hand = False  # 机械手位置增量触发

def env_key_callback(keycode):
    """
    Mujoco可视化界面的键盘回调函数（手动控制机器人）
    Args:
        keycode: 按键编码
    """
    print("chr(keycode): ", (keycode))
    global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
    if keycode == 265: # 上方向键（AKA: up）
        cmd_lin_y += 1
        print("up %f" % cmd_lin_y)
    if keycode == 264: # 下方向键（AKA: down）
        cmd_lin_y -= 1
        print("down %f" % cmd_lin_y)
    if keycode == 263: # 左方向键（AKA: left）
        cmd_lin_x -= 1
        print("left: %f" % cmd_lin_x)
    if keycode == 262: # 右方向键（AKA: right）
        cmd_lin_x += 1
        print("right %f" % cmd_lin_x) 
    if keycode == 52: # 数字键4（AKA: 4）- 左转
        cmd_ang -= 0.2
        print("turn left %f" % cmd_ang)
    if keycode == 54: # 数字键6（AKA: 6）- 右转
        cmd_ang += 0.2
        print("turn right %f" % cmd_ang)
    if chr(keycode) == ' ': # 空格键（AKA: space）- 暂停/继续
        if paused: paused = not paused
    if keycode == 334: # 小键盘+（AKA + (on the numpad)）- 机械臂位置增加
        trigger_delta = True
        delta_xyz = 0.1
    if keycode == 333: # 小键盘-（AKA - (on the numpad)）- 机械臂位置减少
        trigger_delta = True
        delta_xyz = -0.1
    if keycode == 327: # 小键盘7（AKA 7 (on the numpad)）- 机械手位置增加
        trigger_delta_hand = True
        delta_xyz_hand = 0.2
    if keycode == 329: # 小键盘9（AKA 9 (on the numpad)）- 机械手位置减少
        trigger_delta_hand = True
        delta_xyz_hand = -0.2

class DcmmVecEnv(gym.Env):
    """
    DCMM机器人强化学习环境类（继承gym.Env标准接口）
    支持两种任务：Tracking（跟踪）/Catching（抓取）
    核心功能：仿真步进、观测获取、奖励计算、环境重置、渲染可视化
    """
    # 渲染模式元数据（支持的渲染类型）
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    
    """
    初始化参数说明：
    Args:
        render_mode: str - 渲染模式，可选"rgb_array"(RGB图像)/"depth_array"(深度图)/"depth_rgb_array"
        render_per_step: bool - 是否每仿真步都渲染
        viewer: bool - 是否显示Mujoco可视化窗口
        imshow_cam: bool - 是否显示相机图像窗口
        object_eval: bool - 是否使用评估用物体（真实网格模型）
        camera_name: str - 相机名称列表（如["top", "wrist"]）
        object_name: str - 目标物体名称
        env_time: float - 环境最大运行时间（秒）
        steps_per_policy: int - 每个策略动作对应的仿真步数
        img_size: tuple - 图像尺寸（高,宽）
        device: str - 计算设备（如"cuda:0"/"cpu"）
        print_obs/reward/ctrl/info/contacts: bool - 是否打印对应信息（调试用）
    """
    def __init__(
        self,
        task="tracking",
        render_mode="depth_array",
        render_per_step=False,
        viewer=False,
        imshow_cam=False,
        object_eval=False,
        camera_name=["top", "wrist"],
        object_name="object",
        env_time=2.5,
        steps_per_policy=20,
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
        object_motion="throw" # 新增的参数用来判断物体运动类型（throw/roll）
    ):
        # 任务合法性检查（仅支持Tracking/Catching）
        if task not in ["Tracking", "Catching"]:
            raise ValueError("Invalid task: {}".format(task))
        # 渲染模式合法性检查
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # 基础配置赋值
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy  # 每个动作的仿真步数
        self.render_per_step = render_per_step
        
        # 打印配置（调试用）
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        self.object_motion = object_motion # 初始化目标物体运动类型
        
        # 初始化DCMM机器人Mujoco实例
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        # self.Dcmm.show_model_info()  # 打印模型信息（调试用，注释掉）
        
        # 计算仿真帧率（1/(策略步长*仿真时间步)）
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)
        
        # 物体随机化参数
        self.random_mass = 0.25  # 物体随机质量
        self.object_static_time = 0.75  # 物体初始静止时间
        self.object_throw = False  # 物体是否已抛出
        self.object_train = True  # 是否为训练模式（训练/评估）
        if object_eval: self.set_object_eval()  # 评估模式设置
        
        # 重置物体参数（随机化质量/形状/阻尼）
        if self.object_motion == "throw":
            self.Dcmm.model_xml_string = self._reset_object_throw()
        else:
            self.Dcmm.model_xml_string = self._reset_object_roll()
        # 重新加载修改后的XML模型
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)
        
        # 获取几何ID（用于碰撞检测）
        self.hand_start_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        print("self.hand_start_id: ", self.hand_start_id)
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')  # 地面ID
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)  # 物体ID
        self.base_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base')  # 底盘ID

        # 设置相机配置（从DcmmCfg读取宽度/高度）
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        # 初始化Mujoco渲染器
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )
        
        # 启动Mujoco可视化窗口（如果开启）
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            # 启动被动模式viewer（支持键盘回调）
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            # 修改相机视角（俯视）
            self.Dcmm.viewer.cam.lookat[0:2] = [0, 1]  # 注视点
            self.Dcmm.viewer.cam.distance = 5.0  # 距离
            self.Dcmm.viewer.cam.azimuth = 180  # 方位角
            # self.viewer.cam.elevation = -1.57  # 仰角（注释掉）
        else: self.Dcmm.viewer = None

        # ===================== 定义观测空间（gym.spaces.Dict）=====================
        # 观测空间维度说明：总计44维
        # - base: 2维（底盘线速度）
        # - arm: 3+4+3+6=16维（末端位置/四元数/线速度/关节角度）
        # - hand: 12维（机械手关节角度，根据mask筛选）
        # - object: 3+3=6维（物体位置/线速度）
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15  # 机械手可动关节索引
        self.observation_space = spaces.Dict(
            {
                "base": spaces.Dict({
                    "v_lin_2d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),  # 底盘2D线速度
                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),  # 末端3D位置
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),    # 末端四元数
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32), # 末端3D线速度
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),  # 机械臂关节角度（带限位）
                }),
                "hand": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices]),
                                   high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices]),
                                   dtype=np.float32),  # 机械手关节角度（带限位）
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),  # 物体3D位置
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32), # 物体3D线速度
                    ## TODO: to be determined - 物体形状特征（待实现）
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
            }
        )
        
        # ===================== 定义动作空间（gym.spaces.Dict）=====================
        # 动作空间维度说明：总计2+4+12=18维
        # - base: 2维（底盘线速度指令）
        # - arm: 4维（末端位置增量）
        # - hand: 12维（机械手关节角度指令）
        # 底盘动作范围
        base_low = np.array([-4, -4])
        base_high = np.array([4, 4])
        # 机械臂动作范围（末端位置增量）
        arm_low = -0.025*np.ones(4)
        arm_high = 0.025*np.ones(4)
        # 机械手动作范围
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])

        # 获取初始末端位置和物体位置
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        # 定义动作空间
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),  # 底盘动作
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32),    # 机械臂动作
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),  # 机械手动作
            }
        )
        
        # 动作延迟缓冲区（模拟硬件通信延迟）
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),
        }
        
        # 合并动作空间上下限（用于标准化）
        self.actions_low = np.concatenate([base_low, arm_low, hand_low])
        self.actions_high = np.concatenate([base_high, arm_high, hand_high])

        # 计算不同任务的观测/动作维度
        self.obs_dim = get_total_dimension(self.observation_space)  # 总观测维度（44）
        self.act_dim = get_total_dimension(self.action_space)      # 总动作维度（18）
        self.obs_t_dim = self.obs_dim - 12 - 6  # 跟踪任务观测维度（18）：减去机械手12+关节角度6
        self.act_t_dim = self.act_dim - 12      # 跟踪任务动作维度（6）：减去机械手12
        self.obs_c_dim = self.obs_dim - 6       # 抓取任务观测维度（30）：减去关节角度6
        self.act_c_dim = self.act_dim           # 抓取任务动作维度（18）
        print("##### Tracking Task \n obs_dim: {}, act_dim: {}".format(self.obs_t_dim, self.act_t_dim))
        print("##### Catching Task \n obs_dim: {}, act_dim: {}\n".format(self.obs_c_dim, self.act_c_dim))

        # ===================== 环境状态初始化 =====================
        self.arm_limit = True  # 机械臂是否在关节限位内
        self.terminated = False  # 环境终止标志（碰撞/失败）
        self.start_time = self.Dcmm.data.time  # 环境起始时间
        self.catch_time = self.Dcmm.data.time - self.start_time  # 抓取成功时间
        self.reward_touch = 0  # 接触奖励
        self.reward_stability = 0  # 稳定抓取奖励
        self.env_time = env_time  # 环境最大运行时间
        self.stage_list = ["tracking", "grasping"]  # 任务阶段（跟踪/抓取）
        self.stage = self.stage_list[0]  # 默认阶段：tracking
        self.steps = 0  # 环境步数计数器

        self.prev_ctrl = np.zeros(18)  # 上一步控制指令
        self.init_ctrl = True  # 控制初始化标志
        self.vel_init = False  # 速度初始化标志
        self.vel_history = deque(maxlen=4)  # 速度历史队列

        # 环境信息字典（存储关键指标）
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),  # 末端到物体距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] - 
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),  # 底盘到物体距离
            "env_time": self.Dcmm.data.time - self.start_time,  # 环境运行时间
            "imgs": {}  # 渲染图像
        }
        
        # 碰撞接触信息
        self.contacts = {
            "object_contacts": np.array([]),  # 物体接触的几何ID
            "hand_contacts": np.array([]),    # 机械手接触的几何ID
        }

        # 物体初始状态
        self.object_q = np.array([1, 0, 0, 0])  # 物体四元数（初始无旋转）
        self.object_pos3d = np.array([0, 0, 1.5])  # 物体初始位置
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])  # 物体初始速度
        self.step_touch = False  # 本步是否接触到物体

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))  # 图像存储数组

        # 随机化参数（PID/观测/动作噪声）
        self.k_arm = np.ones(6)    # 机械臂PID随机系数
        self.k_drive = np.ones(4)  # 底盘驱动PID随机系数
        self.k_steer = np.ones(4)  # 底盘转向PID随机系数
        self.k_hand = np.ones(1)   # 机械手PID随机系数
        # 观测/动作噪声系数（从配置文件读取）
        self.k_obs_base = DcmmCfg.k_obs_base
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act

    def set_object_eval(self):
        """设置为评估模式（使用真实网格物体，而非简单几何形状）"""
        self.object_train = False

    def update_render_state(self, render_per_step):
        """更新渲染状态（是否每步渲染）"""
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        """更新任务阶段（tracking/grasping）"""
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def _get_contacts(self):
        """
        获取Mujoco仿真中的碰撞接触信息
        Returns:
            dict: 物体/机械手/底盘的接触几何ID
        """
        # 获取接触的几何ID
        geom_ids = self.Dcmm.data.contact.geom
        geom1_ids = self.Dcmm.data.contact.geom1
        geom2_ids = self.Dcmm.data.contact.geom2
        
        ## 获取机械手的接触点
        geom1_hand = np.where((geom1_ids < self.object_id) & (geom1_ids >= self.hand_start_id))[0]
        geom2_hand = np.where((geom2_ids < self.object_id) & (geom2_ids >= self.hand_start_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        
        ## 获取物体的接触点
        geom1_object = np.where((geom1_ids == self.object_id))[0]
        geom2_object = np.where((geom2_ids == self.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        
        ## 获取底盘的接触点
        geom1_base = np.where((geom1_ids == self.base_id))[0]
        geom2_base = np.where((geom2_ids == self.base_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom_ids[geom1_base][:,1]
        if geom2_base.size != 0:
            contacts_geom2 = geom_ids[geom2_base][:,0]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        
        # 打印接触信息（调试用）
        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("hand_contacts: ", hand_contacts)
            print("base_contacts: ", base_contacts)
            
        return {
            "object_contacts": object_contacts,
            "hand_contacts": hand_contacts,
            "base_contacts": base_contacts
        }

    def _get_base_vel(self):
        """
        计算底盘的相对线速度（相对于底盘自身坐标系）
        Returns:
            np.array: [vx, vy] 底盘自身坐标系下的线速度
        """
        # 获取底盘偏航角（从四元数转换）
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        # 全局坐标系下的底盘速度
        global_base_vel = self.Dcmm.data.qvel[0:2]
        # 坐标变换：全局→底盘自身
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        return np.array([base_vel_x, base_vel_y])

    def _get_relative_ee_pos3d(self):
        """
        计算机械臂末端相对于底盘的3D位置
        Returns:
            np.array: [x, y, z] 末端相对位置
        """
        # 获取底盘偏航角
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        # 2D位置坐标变换（全局→底盘）
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body("link6").xpos[0:2], 
                                base_yaw)
        # Z轴为绝对高度差
        return np.array([x, y, 
                         self.Dcmm.data.body("link6").xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])

    def _get_relative_ee_quat(self):
        """
        计算机械臂末端相对于底盘的四元数
        Returns:
            np.array: [x, y, z, w] 相对四元数
        """
        quat = relative_quaternion(self.Dcmm.data.body("base_link").xquat, self.Dcmm.data.body("link6").xquat)
        return np.array(quat)

    def _get_relative_ee_v_lin_3d(self):
        """
        计算机械臂末端相对于底盘的3D线速度
        Returns:
            np.array: [vx, vy, vz] 相对线速度
        """
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]
        global_ee_v_lin = self.Dcmm.data.body("link6").cvel[3:6]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        # 2D速度坐标变换
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        # TODO: 真实场景中需通过位置差分估计速度
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2]-base_vel[2]])
    
    def _get_relative_object_pos3d(self):
        """
        计算目标物体相对于底盘的3D位置
        Returns:
            np.array: [x, y, z] 物体相对位置
        """
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])

    def _get_relative_object_v_lin_3d(self):
        """
        计算目标物体相对于底盘的3D线速度
        Returns:
            np.array: [vx, vy, vz] 物体相对线速度
        """
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]
        global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])

    def _get_obs(self):
        """
        获取环境观测（带噪声）
        Returns:
            dict: 观测字典（base/arm/hand/object）
        """
        # 获取相对位置
        ee_pos3d = self._get_relative_ee_pos3d()
        obj_pos3d = self._get_relative_object_pos3d()
        
        # 初始化历史位置
        if self.init_pos:
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False
        
        # 添加观测噪声（模拟传感器误差）
        obs = {
            "base": {
                "v_lin_2d": self._get_base_vel() + np.random.normal(0, self.k_obs_base, 2),
            },
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),
                "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),  # 差分法计算速度
                "joint_pos": np.array(self.Dcmm.data.qpos[15:21]) + np.random.normal(0, self.k_obs_arm, 6),
            },
            "hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, 12),
            "object": {
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),
                # "v_lin_3d": self._get_relative_object_v_lin_3d() + np.random.normal(0, self.k_obs_object, 3),
                "v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),  # 差分法计算速度
            },
        }
        
        # 更新历史位置
        self.prev_ee_pos3d = ee_pos3d
        self.prev_obj_pos3d = obj_pos3d
        
        # 打印观测（调试用）
        if self.print_obs:
            print("##### print obs: \n", obs)
            
        return obs

    def _get_hand_obs(self):
        """
        获取机械手观测（仅返回可动关节，共12维）
        Returns:
            np.array: 12维机械手关节角度
        """
        # print("full hand: ", self.Dcmm.data.qpos[21:37])
        hand_obs = np.zeros(12)
        # 拇指关节（后3个）
        hand_obs[9] = self.Dcmm.data.qpos[21+13]
        hand_obs[10] = self.Dcmm.data.qpos[21+14]
        hand_obs[11] = self.Dcmm.data.qpos[21+15]
        # 其他三根手指（每根3个关节）
        hand_obs[0] = self.Dcmm.data.qpos[21]
        hand_obs[1:3] = self.Dcmm.data.qpos[(21+2):(21+4)]
        hand_obs[3] = self.Dcmm.data.qpos[21+4]
        hand_obs[4:6] = self.Dcmm.data.qpos[(21+6):(21+8)]
        hand_obs[6] = self.Dcmm.data.qpos[21+8]
        hand_obs[7:9] = self.Dcmm.data.qpos[(21+10):(21+12)]
        return hand_obs
    
    def _get_info(self):
        """
        获取环境信息（关键指标）
        Returns:
            dict: 环境时间/末端到物体距离/底盘到物体距离
        """
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        
        # 打印信息（调试用）
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
            
        return {
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
        }
    
    def update_target_ctrl(self):
        """更新目标控制指令到延迟缓冲区（模拟硬件延迟）"""
        self.action_buffer["base"].append(copy.deepcopy(self.Dcmm.target_base_vel[:]))
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))

    def _get_ctrl(self):
        """
        将动作空间指令转换为Mujoco控制指令（带动作噪声）
        Returns:
            np.array: 30维控制指令（底盘转向4+驱动4+机械臂6+机械手16）
        """
        # 底盘速度控制
        mv_steer, mv_drive = self.Dcmm.move_base_vel(self.action_buffer["base"][0]) # 8维
        # 机械臂PID控制
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[15:21], self.Dcmm.data.time) # 6维
        # 机械手PID控制
        mv_hand = self.Dcmm.hand_pid.update(self.action_buffer["hand"][0], self.Dcmm.data.qpos[21:37], self.Dcmm.data.time) # 16维
        
        # 合并控制指令
        ctrl = np.concatenate([mv_steer, mv_drive, mv_arm, mv_hand], axis=0)
        
        # 添加动作噪声（模拟执行器误差）
        ctrl *= np.random.normal(1, self.k_act, 30)
        
        # 打印控制指令（调试用）
        if self.print_ctrl:
            print("##### ctrl:")
            print("mv_steer: {}, \nmv_drive: {}, \nmv_arm: {}, \nmv_hand: {}\n".format(mv_steer, mv_drive, mv_arm, mv_hand))
            
        return ctrl

    def _reset_object_throw(self):
        """
        随机化目标物体参数（质量/阻尼/形状/尺寸）
        Returns:
            str: 修改后的Mujoco XML模型字符串
        """
        # 解析XML模型字符串
        root = ET.fromstring(self.Dcmm.model_xml_string)

        # 找到物体body节点
        object_body = root.find(".//body[@name='object']")
        if object_body is not None:
            # 随机化物体质量
            inertial = object_body.find("inertial")
            if inertial is not None:
                self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])
                inertial.set("mass", str(self.random_mass))
            
            # 随机化物体阻尼
            joint = object_body.find("joint")
            if joint is not None:
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
                joint.set("damping", str(random_damping))
            
            # 随机化物体形状/尺寸（训练模式）或网格模型（评估模式）
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                object_id = np.random.choice([0, 1, 2, 3, 4])
                if self.object_train:
                    # 训练模式：使用简单几何形状
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])
                else:
                    # 评估模式：使用真实网格模型
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        
        # 转换回XML字符串
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str

    def random_object_pose_throw(self):
        """
        随机化物体初始位姿和速度（增加训练多样性）
        """
        # 随机位置
        x = np.random.rand() - 0.5 # x: (-0.5, 0.5)
        y = 2.2 + 0.3 * np.random.rand() # y: (2.2, 2.5)
        # 随机高度（低/高两种模式）
        low_factor = False if np.random.rand() < 0.5 else True
        if low_factor: height = 0.7 + 0.3 * np.random.rand()# 低高度: (0.7, 1.0)
        else: height = 1.0 + 0.6 * np.random.rand() # 高高度: (1.0, 1.6)
        
        # 随机速度
        r_vel = 1 + np.random.rand() # 速度大小: (1, 2)
        alpha_vel = math.pi * (np.random.rand()*1/6 + 5/12) # 速度角度: (5/12π, 7/12π)
        v_lin_x = r_vel * math.cos(alpha_vel) # x方向速度: (-0.0, -0.5)
        v_lin_y = - r_vel * math.sin(alpha_vel) # y方向速度: (-0.5, -1.0)
        v_lin_z = 0.5 * np.random.rand() + 2.0 # z方向速度: (2.0, 2.5)
        
        # 速度修正（根据位置调整）
        if y > 2.25: v_lin_y -= 0.4
        if height < 1.0: v_lin_z += 1
        
        # 更新物体位姿和速度
        self.object_pos3d = np.array([x, y, height])
        self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        
        # 随机静止时间
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])
        
        # 随机旋转四元数
        r_obj_quat = R.from_euler('xyz', [0, np.random.rand()*1*math.pi, 0], degrees=False)
        self.object_q = r_obj_quat.as_quat()
    
    
    # def _reset_object_roll(self):

    #     """
    #     固定目标物体为【球体 sphere】，随机化球体参数（质量/阻尼/尺寸）
    #     Returns:
    #         str: 修改后的Mujoco XML模型字符串
    #     """
    #     # 解析XML模型字符串
    #     root = ET.fromstring(self.Dcmm.model_xml_string)

    #     # 找到物体body节点
    #     object_body = root.find(".//body[@name='object']")
    #     if object_body is not None:
    #         # 随机化物体质量
    #         inertial = object_body.find("inertial")
    #         if inertial is not None:
    #             # 修复：正确随机质量范围
    #             self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[1])
    #             inertial.set("mass", str(self.random_mass))
            
    #         # 随机化物体阻尼
    #         joint = object_body.find("joint")
    #         if joint is not None:
    #             random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
    #             joint.set("damping", str(random_damping))
            
    #         # 固定为球体 + 随机球体半径
    #         geom = object_body.find(".//geom[@name='object']")
    #         if geom is not None:
    #             # ==============================================
    #             # 核心修改：强制固定为球体，不再随机形状
    #             # ==============================================
    #             geom.set("type", "sphere")
                
    #             # 球体专用尺寸随机（半径 0.035~0.045m）
    #             sphere_size_range = DcmmCfg.object_size["sphere"]
    #             random_radius = np.random.uniform(sphere_size_range[0,0], sphere_size_range[0,1])
    #             geom.set("size", f"{random_radius:.5f}")
                
    #             # 确保评估模式也不会变成网格模型
    #             if "mesh" in geom.attrib:
    #                 del geom.attrib["mesh"]
            
    #         # 可选：给球体添加摩擦（滚动更真实）
    #         geom = object_body.find(".//geom[@name='object']")
    #         if geom is not None:
    #             geom.set("friction", "1.0 0.1 0.1")
        
    #     # 转换回XML字符串
    #     xml_str = ET.tostring(root, encoding='unicode')
        
    #     return xml_str
    def _reset_object_roll(self):
        root = ET.fromstring(self.Dcmm.model_xml_string)
        object_body = root.find(".//body[@name='object']")
        
        if object_body is not None:
            # 质量
            inertial = object_body.find("inertial")
            if inertial is not None:
                self.random_mass = np.random.uniform(0.05, 0.2)
                inertial.set("mass", str(self.random_mass))
            
            # 阻尼：滚动必须小阻尼
            joint = object_body.find("joint")
            if joint is not None:
                joint.set("damping", "0.0001")
                joint.set("armature", "0.001")  # 让旋转更丝滑

            # 球体 + 正确摩擦（滚动关键）
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                geom.set("type", "sphere")
                r = np.random.uniform(0.03, 0.05)
                geom.set("size", f"{r:.4f}")
                geom.set("friction", "1.2 0.1 0.1")  # 高滑动摩擦
                geom.set("solimp", "0.9 0.95 0.01")
                geom.set("solref", "0.02 1")

            if "mesh" in geom.attrib:
                del geom.attrib["mesh"]

        xml_str = ET.tostring(root, encoding='unicode')
        return xml_str

    def random_object_pose_roll(self):
        """
        【球体专用】贴地初始化并按 v≈ωr 设置滚动
        方向：-45° ~ 15° 之间随机（相对于正前方）
        """
        x = np.random.uniform(-0.25, 0.25)
        y = np.random.uniform(2.1, 2.6)

        radius = float(self.Dcmm.model.geom_size[self.object_id][0])
        z = radius + 0.002

        # ===================== 速度大小（可自己调快慢）=====================
        speed = np.random.uniform(1.55, 2.05) 

        # ===================== 核心：方向角度限制在 -45° ~ 15° =====================
        # 角度单位：弧度
        min_angle = -30.0   # 最左
        max_angle = 30.0    # 最右
        angle_deg = np.random.uniform(min_angle, max_angle)
        heading = np.radians(angle_deg)  # 转弧度

        # 根据角度计算速度方向
        vx = speed * math.sin(heading)
        vy = -speed * math.cos(heading)
        vz = 0.0

        # 滚动角速度（物理正确）
        wx = -vy / max(radius, 1e-4)
        wy = vx / max(radius, 1e-4)
        wz = 0.0

        self.object_pos3d = np.array([x, y, z])
        self.object_vel6d = np.array([vx, vy, vz, wx, wy, wz])
        self.object_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.object_static_time = 0.0



    def random_PID(self):
        """
        随机化PID控制器参数（增加模型鲁棒性）
        """
        # 生成随机系数
        self.k_arm = np.random.uniform(0, 1, size=6)
        self.k_drive = np.random.uniform(0, 1, size=4)
        self.k_steer = np.random.uniform(0, 1, size=4)
        self.k_hand = np.random.uniform(0, 1, size=1)
        
        # 重置PID控制器（应用随机系数）
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])
        self.Dcmm.steer_pid.reset(self.k_steer*(DcmmCfg.k_steer[1]-DcmmCfg.k_steer[0])+DcmmCfg.k_steer[0])
        self.Dcmm.drive_pid.reset(self.k_drive*(DcmmCfg.k_drive[1]-DcmmCfg.k_drive[0])+DcmmCfg.k_drive[0])
        self.Dcmm.hand_pid.reset(self.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])
    
    def random_delay(self):
        """
        随机化动作延迟（模拟硬件通信延迟）
        """
        # 随机设置缓冲区最大长度
        self.action_buffer["base"].set_maxlen(np.random.choice(DcmmCfg.act_delay['base']))
        self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        self.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        
        # 清空缓冲区
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()

    def _reset_simulation(self):
        """
        重置Mujoco仿真状态
        """
        # 重置仿真数据
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        
        # 重置控制指令
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        
        # 设置机械臂/机械手初始关节角度
        self.Dcmm.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[21:37] = DcmmCfg.hand_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        
        # 设置物体初始位置（默认）
        #self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])
        
        # 不在这里设置空中位置，避免覆盖 roll 模式的贴地位置
        # if self.object_motion != "roll":
        #     self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])


        # 随机化物体位姿（根据运动类型）
        if self.object_motion == "roll":
            self.random_object_pose_roll()
            init_vel = self.object_vel6d[:]
        else:
            self.random_object_pose_throw()
            init_vel = np.zeros(6)
        
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=init_vel)
        
        # 随机化重力
        # self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        self.Dcmm.model.opt.gravity[2] = -9.81
        
        # 随机化PID参数
        self.random_PID()
        
        # 随机化动作延迟
        self.random_delay()
        
        # 前向动力学计算（更新模型状态）
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)

    def reset(self):
        """
        重置环境（符合 gym.Env 标准接口）
        Returns:
            tuple: (observation, info) 初始观测和信息
        """
        # 重置仿真
        self._reset_simulation()
        
        # 重置状态标志
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False  # 统一初始化为 False，表示还未抛出/开始滚动
        self.steps = 0
        
        # 重置时间
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time

        # 重置目标控制指令
        self.Dcmm.target_base_vel = np.array([0.0, 0.0, 0.0])
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        
        # 重置奖励和阶段
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0

        # 重置信息字典
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
        }
        
        # 获取初始观测和信息
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d = self._get_relative_object_pos3d()
        observation = self._get_obs()
        info = self._get_info()
        
        # 渲染初始图像
        imgs = self.render()
        info['imgs'] = imgs
        
        # 记录控制参数（用于调试/分析）
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))

        return observation, info

    def norm_ctrl(self, ctrl, components):
        '''
        计算控制指令的归一化范数（用于控制平滑度惩罚）
        Input: 
            ctrl: dict - 控制指令字典
            components: list - 需要计算的组件（base/arm/hand）
        Return: 
            norm: float - 归一化范数
        '''
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component] for component in components])
        return np.linalg.norm(ctrl_array)

    def compute_reward(self, obs, info, ctrl):
        '''
        计算强化学习奖励（核心函数）
        奖励组成：
        - 位置奖励（末端/底盘到物体）
        - 精度奖励（末端到物体距离）
        - 姿态奖励（末端与物体速度方向）
        - 接触奖励（成功接触物体）
        - 稳定奖励（抓取后物体稳定）
        - 碰撞惩罚（底盘/机械臂碰撞）
        - 约束惩罚（关节超出限位）
        - 控制惩罚（动作突变）
        '''
        rewards = 0.0
        
        ## 1. 位置奖励（鼓励末端/底盘靠近物体）
        reward_base_pos = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]
        reward_ee_pos = (self.info["ee_distance"] - info["ee_distance"]) * DcmmCfg.reward_weights["r_ee_pos"]
        reward_ee_precision = math.exp(-50*info["ee_distance"]**2) * DcmmCfg.reward_weights["r_precision"]

        ## 2. 碰撞惩罚（发生碰撞则扣分）
        reward_collision = 0
        if self.contacts['base_contacts'].size != 0:
            reward_collision = DcmmCfg.reward_weights["r_collision"]
        
        ## 3. 约束惩罚（关节超出限位则扣分）
        reward_constraint = 0 if self.arm_limit else -1
        reward_constraint *= DcmmCfg.reward_weights["r_constraint"]

        ## 4. 接触奖励（成功接触物体则加分）
        if self.step_touch:
            if not self.reward_touch:
                self.catch_time = self.Dcmm.data.time - self.start_time
            self.reward_touch = DcmmCfg.reward_weights["r_touch"][self.task]
        else:
            self.reward_touch = 0

        ## 5. 分任务/分阶段计算总奖励
        if self.task == "Catching":
            reward_orient = 0
            reward_ee_height = 0
            reward_intercept = 0
            
            ### 跟踪阶段（tracking）
            if self.stage == "tracking":
                ## 控制惩罚（惩罚机械手动作突变）
                reward_ctrl = - self.norm_ctrl(ctrl, {"hand"})
                
                if self.object_motion == "throw":
                    ## 姿态奖励（末端z轴与物体速度方向对齐）
                    rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                    local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
                    hand_z_axis = np.array([0, 0, 1])
                    reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
                else:
                    ## 滚动模式姿态奖励：手部 z 轴朝下 (World -Z)
                    rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                    hand_z_in_base = rotation_matrix[:, 2]
                    target_z_in_base = np.array([0, 0, -1])
                    reward_orient = max(0, cos_angle_between_vectors(hand_z_in_base, target_z_in_base)) * DcmmCfg.reward_weights["r_orient"]
                    
                    ## 滚动模式高度奖励：靠近地面但不贴地
                    ee_height = self.Dcmm.data.body("link6").xpos[2]
                    obj_radius = float(self.Dcmm.model.geom_size[self.object_id][0])
                    target_height = obj_radius + 0.12 # 目标高度：球半径 + 12cm 左右，留出抓取空间
                    reward_ee_height = math.exp(-30 * (ee_height - target_height)**2) * 2.0 

                    ## 滚动模式拦截奖励：鼓励手在球的滚动路径前方
                    # 获取球的速度方向（XY平面）
                    obj_v = obs["object"]["v_lin_3d"][:2]
                    obj_v_norm = np.linalg.norm(obj_v)
                    if obj_v_norm > 0.1:
                        obj_v_dir = obj_v / obj_v_norm
                        # 向量：球 -> 手
                        rel_pos_xy = obs["arm"]["ee_pos3d"][:2] - obs["object"]["pos3d"][:2]
                        # 计算球到手的向量在球速度方向上的投影
                        projection = np.dot(rel_pos_xy, obj_v_dir)
                        # 如果投影为正，说明手在球的前方
                        # 我们希望手在球前方一段距离（例如 0.2m - 0.5m）
                        reward_intercept = math.exp(-5 * (projection - 0.3)**2) * 2.0

                ## 总奖励
                rewards = reward_base_pos + reward_ee_pos + reward_orient + reward_ee_height + reward_intercept + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
                
                # 打印奖励（调试用）
                if self.print_reward:
                    if reward_constraint < 0:
                        print("ctrl: ", ctrl)
                    print("### print reward (tracking)")
                    print("ee_pos: {:.3f}, orient: {:.3f}, height: {:.3f}, intercept: {:.3f}, ctrl: {:.3f}, touch: {:.3f}".format(
                        reward_ee_pos, reward_orient, reward_ee_height, reward_intercept, reward_ctrl, self.reward_touch
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
            ### 抓取阶段（grasping）
            else:
                ## 控制惩罚（惩罚底盘/机械臂动作突变）
                reward_ctrl = - self.norm_ctrl(ctrl, {"base", "arm"})
                ## 姿态奖励（满值）
                reward_orient = DcmmCfg.reward_weights["r_orient"]
                ## 稳定奖励（抓取后物体稳定时间越长，奖励越高）
                if self.reward_touch:
                    self.reward_stability = (info["env_time"] - self.catch_time) * DcmmCfg.reward_weights["r_stability"]
                else:
                    self.reward_stability = 0.0
                ## 总奖励
                rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint \
                        + self.reward_touch + self.reward_stability
                
                # 打印奖励（调试用）
                if self.print_reward:
                    print("##### print reward (grasping)")
                    print("reward_touch: {}, \nreward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, \n".format(
                        self.reward_touch, reward_ee_pos, reward_ee_precision, reward_orient
                    ) + "reward_stability: {:.3f}, reward_collision: {:.3f}, \nreward_ctrl: {:.3f}, reward_constraint: {:.3f}".format(
                        self.reward_stability, reward_collision, reward_ctrl, reward_constraint
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
        ## 跟踪任务（Tracking）
        elif self.task == 'Tracking':
            reward_ee_height = 0
            reward_intercept = 0
            ## 控制惩罚（惩罚底盘/机械臂动作突变）
            reward_ctrl = - self.norm_ctrl(ctrl, {"base", "arm"})
            
            if self.object_motion == "throw":
                ## 姿态奖励（末端z轴与物体速度方向对齐）
                rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
                hand_z_axis = np.array([0, 0, 1])
                reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
            else:
                ## 滚动模式姿态奖励：手部 z 轴朝下 (World -Z)
                rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                hand_z_in_base = rotation_matrix[:, 2]
                target_z_in_base = np.array([0, 0, -1])
                reward_orient = max(0, cos_angle_between_vectors(hand_z_in_base, target_z_in_base)) * DcmmCfg.reward_weights["r_orient"]
                
                ## 滚动模式高度奖励：靠近地面但不贴地
                ee_height = self.Dcmm.data.body("link6").xpos[2]
                obj_radius = float(self.Dcmm.model.geom_size[self.object_id][0])
                target_height = obj_radius + 0.12 # 目标高度
                reward_ee_height = math.exp(-30 * (ee_height - target_height)**2) * 2.0 

                ## 滚动模式拦截奖励
                obj_v = obs["object"]["v_lin_3d"][:2]
                obj_v_norm = np.linalg.norm(obj_v)
                if obj_v_norm > 0.1:
                    obj_v_dir = obj_v / obj_v_norm
                    rel_pos_xy = obs["arm"]["ee_pos3d"][:2] - obs["object"]["pos3d"][:2]
                    projection = np.dot(rel_pos_xy, obj_v_dir)
                    reward_intercept = math.exp(-5 * (projection - 0.3)**2) * 2.0

            ## 总奖励
            rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ee_height + reward_intercept + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
            
            # 打印奖励（调试用）
            if self.print_reward:
                if reward_constraint < 0:
                    print("ctrl: ", ctrl)
                print("### print reward (tracking task)")
                print("ee_pos: {:.3f}, precision: {:.3f}, orient: {:.3f}, height: {:.3f}, intercept: {:.3f}, ctrl: {:.3f}, touch: {:.3f}".format(
                    reward_ee_pos, reward_ee_precision, reward_orient, reward_ee_height, reward_intercept, reward_ctrl, self.reward_touch
                ))
                print("total reward: {:.3f}\n".format(rewards))
        else:
            raise ValueError("Invalid task: {}".format(self.task))
        
        return rewards

    def _step_mujoco_simulation(self, action_dict):
        """
        执行Mujoco仿真步进（核心函数）
        Args:
            action_dict: dict - 动作指令字典（base/arm/hand）
        """
        ## 设置底盘目标速度
        self.Dcmm.target_base_vel[0:2] = action_dict['base']
        
        ## 机械臂逆运动学求解（末端位置增量→关节角度）
        action_arm = np.concatenate((action_dict["arm"], np.zeros(3)))
        result_QP, _ = self.Dcmm.move_ee_pose(action_arm)
        if result_QP[1]:
            self.arm_limit = True
            self.Dcmm.target_arm_qpos[:] = result_QP[0]
        else:
            # print("IK Failed!!!")
            self.arm_limit = False
        
        ## 设置机械手目标关节角度
        self.Dcmm.action_hand2qpos(action_dict["hand"])
        
        ## 更新目标动作到延迟缓冲区
        self.update_target_ctrl()
        
        ## 重置接触标志
        self.step_touch = False
        
        ## 执行多个仿真步（每个策略动作对应多个仿真步）
        for i in range(self.steps_per_policy):
            ## 更新控制指令
            self.Dcmm.data.ctrl[:-1] = self._get_ctrl()
            
            ## 渲染（如果开启）
            if self.render_per_step:
                img = self.render()
            
            ## ===================== 物体运动逻辑（区分 throw / roll 模式）=====================
            if self.object_motion == "throw":
                # ========== 抛射模式（原来逻辑不变）==========
                if self.Dcmm.data.time - self.start_time < self.object_static_time:
                    self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                                velocity=np.zeros(6))
                    self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
                elif not self.object_throw:
                    self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                                velocity=self.object_vel6d[:])
                    self.Dcmm.data.ctrl[-1] = 0.0
                    self.object_throw = True

            else:
                # ========== 滚动模式：只在第一次设置速度，之后完全自由滚动 ==========
                self.Dcmm.data.ctrl[-1] = 0.0  # 无外力

                # 只在第一次进入时设置初始滚动速度，之后绝不覆盖！
                if not self.object_throw:
                    self.Dcmm.set_throw_pos_vel(
                        pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                        velocity=self.object_vel6d[:]
                    )
                    self.object_throw = True  # 只设一次！





            ## 执行 Mujoco 仿真步
            mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)
            mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)

            ## 更新接触信息
            self.contacts = self._get_contacts()
            
            ## 碰撞检测（底盘碰撞则终止）
            if self.contacts['base_contacts'].size != 0:
                self.terminated = True
            
            ## 物体接触检测（判断是否成功抓取/跟踪）
            mask_coll = self.contacts['object_contacts'] < self.hand_start_id
            mask_finger = self.contacts['object_contacts'] > self.hand_start_id
            mask_hand = self.contacts['object_contacts'] >= self.hand_start_id
            mask_palm = self.contacts['object_contacts'] == self.hand_start_id
            
            # 判断是否接触到物体
            if self.step_touch == False:
                if self.task == "Catching" and np.any(mask_hand):
                    self.step_touch = True
                elif self.task == "Tracking" and np.any(mask_palm):
                    self.step_touch = True
            
            # # 判断是否失败（物体掉落/碰撞）
            # if not self.terminated:
            #     if self.task == "Catching":
            #         self.terminated = np.any(mask_coll)
            #     elif self.task == "Tracking":
            #         self.terminated = np.any(mask_coll) or np.any(mask_finger)
            # ===================== 失败终止条件（区分 throw / roll 模式）=====================
            if not self.terminated:
                if self.task == "Catching":
                    if self.object_motion == "throw":
                        # 抛射模式：碰到地面/障碍物 = 失败
                        self.terminated = np.any(mask_coll)
                    else:
                        # ===================== 滚动模式（正确逻辑）=====================
                        # 滚动模式：碰到地面 = 完全正常！不终止！
                        # 只有 滚出桌子 才终止！
                        
                        obj_x = self.Dcmm.data.body(self.object_name).xpos[0]
                        obj_y = self.Dcmm.data.body(self.object_name).xpos[1]
                        obj_z = self.Dcmm.data.body(self.object_name).xpos[2]
                        obj_radius = float(self.Dcmm.model.geom_size[self.object_id][0])

                        out_left_right = abs(obj_x) > 1.2
                        out_forward_back = obj_y < 1.0 or obj_y > 4.0
                        fall_down = obj_z < max(0.01, 0.5 * obj_radius)

                        self.terminated = out_left_right or out_forward_back or fall_down

                elif self.task == "Tracking":
                    self.terminated = np.any(mask_coll) or np.any(mask_finger)

            
            # 失败则提前终止
            if self.terminated:
                break

    def step(self, action):
        """
        环境步进（符合gym.Env标准接口）
        Args:
            action: dict - 动作指令字典
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        
        # 执行仿真步进
        self._step_mujoco_simulation(action)
        
        # 获取新的观测和信息
        obs = self._get_obs()
        info = self._get_info()
        
        # 抓取任务阶段切换（跟踪→抓取）
        if self.task == 'Catching':
            if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                self.stage = "grasping"
            elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
                self.terminated = True
        
        # 计算奖励
        reward = self.compute_reward(obs, info, action)
        
        # 更新信息字典
        self.info["base_distance"] = info["base_distance"]
        self.info["ee_distance"] = info["ee_distance"]
        
        # 渲染图像
        imgs = self.render()
        info['imgs'] = imgs
        
        # 记录控制参数
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        
        # 判断是否截断（任务完成/超时）
        if self.task == "Catching":
            if info["env_time"] > self.env_time:
                truncated = True
            else: truncated = False
        elif self.task == "Tracking":
            if self.step_touch:
                truncated = True
            else: truncated = False
        
        terminated = self.terminated
        done = terminated or truncated
        
        # 测试模式下完成后重置（注释掉则保持最终状态）
        if done:
            # self.reset()
            pass
        
        return obs, reward, terminated, truncated, info

    def preprocess_depth_with_mask(self, rgb_img, depth_img, 
                                   depth_threshold=3.0, 
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        """
        深度图像预处理（添加掩码和噪声，模拟真实传感器）
        Args:
            rgb_img: np.array - RGB图像
            depth_img: np.array - 深度图像
            depth_threshold: float - 深度阈值
            num_white_points_range: tuple - 噪声点数量范围
            point_size_range: tuple - 噪声点大小范围
        Returns:
            tuple: (处理后的深度图, 掩码内平均深度)
        """
        # RGB掩码（筛选红色区域）
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        
        # 深度掩码（筛选有效深度）
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        
        # 组合掩码
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        
        # 应用掩码到深度图
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        
        # 计算掩码内平均深度
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        
        # 添加随机噪声点（模拟传感器噪声）
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        
        # 生成噪声点掩码
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        
        # 应用噪声点
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        """
        环境渲染（生成RGB/深度图像）
        Returns:
            np.array: 渲染图像数组
        """
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        
        # 遍历所有相机
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                # 人类可视化模式
                self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.render_mode != "depth_rgb_array":
                # RGB/深度图模式
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                # 显示RGB图像
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # 深度图转换（0-1→真实米数）
                elif self.render_mode == "depth_array":
                    img = self.Dcmm.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            else:
                # RGB+深度图模式
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )   
                # 深度图转换和预处理
                img_depth = self.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            
            # 同步viewer（如果开启）
            if self.Dcmm.viewer != None: 
                self.Dcmm.viewer.sync()
        
        # 返回深度图（depth_rgb_array模式）
        if self.render_mode == "depth_rgb_array":
            imgs = imgs_depth
        
        return imgs

    def close(self):
        """关闭环境（释放资源）"""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def run_test(self):
        """
        测试模式（键盘手动控制机器人）
        """
        global cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
        self.reset()
        action = np.zeros(18)
        
        # 无限循环（手动控制）
        while True:
            # 动作指令赋值（从键盘全局变量）
            action[0:2] = np.array([cmd_lin_x, cmd_lin_y])
            
            # 机械臂位置增量
            if trigger_delta:
                print("delta_xyz: ", delta_xyz)
                action[2:6] = np.array([delta_xyz, delta_xyz, delta_xyz, delta_xyz])
                trigger_delta = False
            else:
                action[2:6] = np.zeros(4)
            
            # 机械手位置增量
            if trigger_delta_hand:
                print("delta_xyz_hand: ", delta_xyz_hand)
                action[6:18] = np.ones(12)*delta_xyz_hand
                trigger_delta_hand = False
            else:
                action[6:18] = np.zeros(12)
            
            # 转换为动作字典
            base_tensor = action[:2]
            arm_tensor = action[2:6]
            hand_tensor = action[6:18]
            actions_dict = {
                'arm': arm_tensor,
                'base': base_tensor,
                'hand': hand_tensor
            }
            
            # 执行环境步进
            observation, reward, terminated, truncated, info = self.step(actions_dict)
            if terminated or truncated:
                self.reset()
            if terminated or truncated:
                self.reset()

# ===================== 主函数（测试入口）=====================

# ===================== 主函数（测试入口）=====================

if __name__ == "__main__":
    # 切换工作目录到上上级目录
    # 解决不同层级目录下运行时的模块导入路径问题
    os.chdir('../../')
    
    # 创建命令行参数解析器，用于接收运行时的参数输入
    # description参数用于说明这个解析器的用途（DcmmVecEnv环境的参数配置）
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    
    # 添加--viewer命令行参数：是否打开Mujoco可视化窗口
    # action='store_true'表示只要输入该参数，值就为True，否则为False
    # help参数是该参数的说明文档，用于-h/--help时展示
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    
    # 添加--imshow_cam命令行参数：是否显示相机图像窗口
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    
    # 添加--object_motion命令行参数：物体运动类型（滚动/投掷）
    parser.add_argument('--object_motion', type=str, default="throw", help="object motion type (roll/throw)")
    
    # 解析命令行传入的参数，将结果存储在args对象中
    args = parser.parse_args()
    
    # 打印解析后的参数（调试用），方便确认参数是否正确传入
    print("args: ", args)
    
    # 实例化DCMM机器人强化学习环境对象
    # 参数说明：
    # task='Catching'          - 任务类型为抓取任务（可选Tracking跟踪任务）
    # object_name='object'     - 目标物体在Mujoco模型中的名称
    # render_per_step=False    - 不每仿真步都渲染（提升运行速度）
    # print_reward=False       - 不打印奖励信息
    # print_info=False         - 不打印环境信息（如距离、时间等）
    # print_contacts=False     - 不打印碰撞接触信息
    # print_ctrl=False         - 不打印控制指令信息
    # print_obs=False          - 不打印观测信息
    # camera_name = ["top"]    - 使用名为"top"的相机进行渲染
    # render_mode="rgb_array"  - 渲染模式为RGB图像数组
    # imshow_cam=args.imshow_cam - 是否显示相机图像窗口（由命令行参数控制）
    # viewer = args.viewer     - 是否打开Mujoco可视化窗口（由命令行参数控制）
    # object_eval=False        - 使用训练模式的物体（简单几何形状，非真实网格模型）
    # env_time = 2.5           - 环境最大运行时间2.5秒
    # steps_per_policy=20      - 每个策略动作对应20个Mujoco仿真步
    env = DcmmVecEnv(task='Catching', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=1,
                    object_motion=args.object_motion)
    
    # 运行环境的测试模式（键盘手动控制机器人）
    # 该模式下可以通过键盘方向键/数字键控制机器人底盘、机械臂和机械手
    env.run_test()
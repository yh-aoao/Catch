"""计算 throw_force 的初始臂姿 IK
目标：掌心朝上（手背朝下）、手指朝前下抓球，便于朝前扔
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial.transform import Rotation as R
import mujoco
from configs.env import DcmmCfg
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM

dcmm = MJ_DCMM(viewer=False, object_name='object', object_eval=False)

# 目标手位置（arm_base 前方偏下，抓球高度）
# arm_base 在 y≈0.12, z≈0.40；手放在前方 0.35m、高度 0.5m
target_pos = np.array([0.0, 0.45, 0.45])

# 目标姿态：掌心朝上(Y=[0,0,1])、手指朝前下(Z 朝前偏下)
# 构造旋转矩阵：X=Y×Z
palm_normal = np.array([0.0, 0.0, 1.0])   # 掌心朝上 = 手背朝下
finger_dir = np.array([0.0, 0.8, -0.6])    # 手指朝前下
finger_dir = finger_dir / np.linalg.norm(finger_dir)
x_axis = np.cross(palm_normal, finger_dir)
x_axis = x_axis / np.linalg.norm(x_axis)
y_axis = np.cross(finger_dir, x_axis)  # 重新正交化，保证掌心法线正确
R_target = np.column_stack([x_axis, y_axis, finger_dir])
target_quat = R.from_matrix(R_target).as_quat()  # [x,y,z,w]

print("目标位置:", target_pos)
print("目标姿态矩阵:")
print(np.round(R_target, 3))
print("目标四元数[x,y,z,w]:", np.round(target_quat, 4))

# 用 IK 求解（从默认臂姿开始）
dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
mujoco.mj_fwdPosition(dcmm.model_arm, dcmm.data_arm)

result = dcmm.ik_arm_solve(target_pos, target_quat)
if result[1]:
    joints = result[0]
    print("\nIK 成功！臂关节角:")
    print(np.round(joints, 4))
    print("\n复制到 DcmmCfg.py 的 throw_force_arm_joints:")
    print(f"throw_force_arm_joints = np.array({np.round(joints, 4).tolist()})")
else:
    print("\nIK 失败：目标姿态不可达（可能超出关节限位）")
    # 尝试只解位置，姿态用默认
    print("尝试只解位置...")
    result2 = dcmm.ik_arm_solve(target_pos, dcmm.data_arm.body('link6').xquat)
    if result2[1]:
        print("位置可达，关节角:", np.round(result2[0], 4))
    else:
        print("位置也不可达")

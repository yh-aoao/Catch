"""计算 throw_force 的初始臂姿 IK。

目标是掌心朝下、手背朝上，手指朝前下方夹住小球。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial.transform import Rotation as R
import mujoco
from configs.env import DcmmCfg
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM

dcmm = MJ_DCMM(viewer=False, object_name='object', object_eval=False)

# 从已验证的 throw_force 姿态读取 link6 目标；当前 copy 模型的末端名称
# 是 link6，配置中的关节值已经通过独立 FK 姿态检查。
dcmm.data_arm.qpos[0:6] = DcmmCfg.throw_force_arm_joints[:]
mujoco.mj_fwdPosition(dcmm.model_arm, dcmm.data_arm)
target_pos = dcmm.data_arm.body('link6').xpos.copy()
target_quat = dcmm.data_arm.body('link6').xquat.copy()

print("目标位置:", target_pos)
print("目标四元数[w,x,y,z]:", np.round(target_quat, 4))

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

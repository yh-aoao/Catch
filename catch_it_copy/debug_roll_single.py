"""
单环境复现脚本：用 gym.make（非 vector env）直接 step roll 环境。
单环境下异常直接显示，不会被 async vector env 吞掉。
用法（GPU 机器上，catch_it_copy 目录下）：
    python3 debug_roll_single.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MUJOCO_GL'] = 'egl'  # 必须在 import gym_dcmm（内部 import mujoco）之前

import gymnasium as gym
import gym_dcmm
import numpy as np

print("=== 创建单环境 roll Tracking ===")
env = gym.make(
    "gym_dcmm/DcmmVecWorld-v0",
    task="Tracking", camera_name=["top"],
    render_per_step=False, render_mode="rgb_array",
    object_name="object", img_size=[112, 112],
    imshow_cam=False, viewer=False,
    print_obs=False, print_info=False,
    print_reward=False, print_ctrl=False,
    print_contacts=False, object_eval=False,
    env_time=2.5, steps_per_policy=20,
    object_motion="roll",
)

print("=== reset ===")
obs, info = env.reset()
print("reset OK, obs keys:", list(obs.keys()))

action = {
    'base': np.zeros((2,), dtype=np.float32),
    'arm': np.zeros((6,), dtype=np.float32),
    'hand': np.zeros((12,), dtype=np.float32),
}

print("=== step 循环 ===")
for step in range(30):
    obs, r, term, trunc, info = env.step(action)
    print(f"step {step}: r={r}, term={term}, trunc={trunc}")
    if term or trunc:
        print("episode ended")
        break

print("=== DONE，没有崩溃 ===")
env.close()

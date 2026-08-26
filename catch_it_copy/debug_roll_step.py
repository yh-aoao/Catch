"""
最小复现脚本：绕过 PPO 直接 step roll 环境，暴露 env.step 的真正异常。

用法（GPU 机器上，catch_it_copy 目录下）：
    python3 debug_roll_step.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import gym_dcmm
import numpy as np

print("=== 创建 roll Tracking 环境 ===")
env = gym.make_vec(
    "gym_dcmm/DcmmVecWorld-v0", num_envs=1,
    task="Tracking", camera_name=["top"],
    render_per_step=False, render_mode="rgb_array",
    object_name="object",
    img_size=[112, 112],
    imshow_cam=False, viewer=False,
    print_obs=False, print_info=False,
    print_reward=False, print_ctrl=False,
    print_contacts=False, object_eval=False,
    env_time=2.5, steps_per_policy=20,
    object_motion="roll",
)

# 模仿训练时的 action：base(2) + arm(6) + hand(12)，全零
action = {
    'base': np.zeros((1, 2), dtype=np.float32),
    'arm': np.zeros((1, 6), dtype=np.float32),
    'hand': np.zeros((1, 12), dtype=np.float32),
}

for ep in range(5):
    print(f"\n=== episode {ep}: reset ===")
    obs, info = env.reset()
    print("reset OK, obs keys:", list(obs.keys()))
    for step in range(30):
        obs, r, term, trunc, info = env.step(action)
        print(f"  step {step}: r={r}, term={term}, trunc={trunc}")
        if np.any(term) or np.any(trunc):
            print(f"  episode ended at step {step}")
            break

print("\n=== DONE，没有崩溃 ===")
env.close()

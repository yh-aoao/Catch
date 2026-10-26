import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import gym_dcmm
import numpy as np

for motion in ["roll", "throw_basket", "bounce"]:
    print(f"\n========== {motion} ==========")
    env = gym.make("gym_dcmm/DcmmVecWorld-v0",
        task="Tracking", camera_name=["top"], render_per_step=False,
        render_mode="rgb_array", object_name="object", img_size=[112,112],
        imshow_cam=False, viewer=False, print_obs=False, print_info=False,
        print_reward=False, print_ctrl=False, print_contacts=False,
        object_eval=False, env_time=2.5, steps_per_policy=20,
        object_motion=motion)
    action = {'base': np.zeros(2), 'arm': np.zeros(6), 'hand': np.zeros(12)}
    for ep in range(5):
        obs, info = env.reset()
        for step in range(100):
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        print(f"  ep{ep}: 结束于 step {step}, term={term}, trunc={trunc}")
    env.close()

print("\nALL DONE")
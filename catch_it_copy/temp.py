import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=True)

print("[0] script start", flush=True)

import numpy as np
print("[1] numpy imported", flush=True)

from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
print("[2] DcmmVecEnv imported", flush=True)

print("[3] creating env", flush=True)
env = DcmmVecEnv(
    task="Tracking",
    object_motion="roll",
    object_name="object",
    camera_name=[],
    render_mode=None,   # 先改成 None，排除渲染器/相机问题
    viewer=False,
    imshow_cam=False,
    render_per_step=False,
    steps_per_policy=20,
    print_contacts=True,
)
print("[4] env created", flush=True)

print("[5] resetting", flush=True)
obs, info = env.reset()
print("[6] reset done", flush=True)

action = {"base": np.zeros(2), "arm": np.zeros(4), "hand": np.zeros(12)}

for i in range(5):
    print(f"[7] before step {i}", flush=True)
    obs, r, terminated, truncated, info = env.step(action)
    print(
        i,
        "reward=", r,
        "terminated=", terminated,
        "truncated=", truncated,
        "reason=", env.terminated_reason,
        "contacts=", env.contacts["object_contacts"],
        "floor_id=", env.floor_id,
        "step_touch=", env.step_touch,
        flush=True,
    )
    if terminated or truncated:
        break

env.close()
print("[8] done", flush=True)
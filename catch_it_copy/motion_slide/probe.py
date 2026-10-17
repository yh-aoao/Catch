"""Zero-action Slide scene check. This does not test a trained catching policy."""
import argparse
import json
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--config')
    args = parser.parse_args()
    from .environment import SlideEnv
    env = SlideEnv(task='Tracking', slide_config=args.config, slide_log=True,
                   viewer=False, render_mode=None, render_per_step=False, imshow_cam=False,
                   object_eval=False, steps_per_policy=20, env_time=5.)
    window = None
    try:
        for episode in range(args.episodes):
            env.reset(seed=episode)
            if args.viewer and window is None:
                import mujoco.viewer
                window = mujoco.viewer.launch_passive(env.Dcmm.model, env.Dcmm.data)
            print(json.dumps({'episode': episode, 'initial_position': env.Dcmm.data.qpos[37:40].tolist(),
                              'initial_velocity': env.Dcmm.data.qvel[36:42].tolist()}), flush=True)
            while True:
                _, _, done, timeout, info = env.step(dict(base=np.zeros(2), arm=np.zeros(6), hand=np.zeros(12)))
                if window is not None:
                    if not window.is_running():
                        return
                    window.sync()
                    time.sleep(.04)
                if done or timeout:
                    print(json.dumps({'departed': bool(env.departed), 'reason': info['terminated_reason'],
                                      'final_position': env.Dcmm.data.qpos[37:40].tolist()}), flush=True)
                    break
    finally:
        if window is not None:
            window.close()
        env.close()


if __name__ == '__main__':
    main()

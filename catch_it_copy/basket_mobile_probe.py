"""MuJoCo task feasibility probe. The diagnostic driver is NOT a trained policy."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(episodes, transfer_controls=False, parking_stage=False):
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    results = []
    for seed in range(episodes):
        np.random.seed(seed)
        env = DcmmVecEnv(task='Catching', object_motion='throw_basket',
            basket_arm_control='joint_velocity', basket_reward_version='basket_mobile_v1', viewer=False,
            basket_mobile_options=(dict(position_tolerance=.07, speed_tolerance=.06,
                settle_seconds=.30, brake_after_unlock=True, approach_braking=True,
                parking_only=parking_stage) if transfer_controls else dict(parking_only=parking_stage)))
        try:
            obs, _ = env.reset(seed=seed)
            target = env.basket_center.copy()
            target_body = env.Dcmm.model.body('basket_target').pos.copy()
            initial_position = env.Dcmm.data.body('arm_base').xpos.copy()
            assert env.obs_c_dim == 51 and env.act_c_dim == 20 and obs['mobile'].shape == (16,)
            reward_total = 0.
            for _ in range(250):
                # This PD-like parking reference is only a feasibility diagnostic.
                # Deliberately nonzero arm/finger requests verify navigation masks.
                parking = env._basket_parking_state()
                obs, reward, terminated, truncated, info = env.step(dict(
                    base=parking['desired_velocity'], arm=np.full(6, .025), hand=np.full(12, .15)))
                reward_total += reward
                np.testing.assert_array_equal(env.basket_center, target)
                np.testing.assert_array_equal(env.Dcmm.model.body('basket_target').pos, target_body)
                if env.basket_mobile_state['unlocked'] or terminated or truncated:
                    if parking_stage and env.basket_mobile_state['unlocked']:
                        assert terminated and info['success'] and info['terminated_reason'] == 'parking_score'
                        assert not info['safe_basket_success']
                    break
                np.testing.assert_array_equal(env.Dcmm.target_arm_qpos, env.basket_carry_arm)
                np.testing.assert_array_equal(env.Dcmm.target_hand_qpos, env.basket_carry_hand)
            control = info['basket_control']
            results.append(dict(seed=seed, target=target.tolist(),
                displacement=float(np.linalg.norm(env.Dcmm.data.body('arm_base').xpos[:2]-initial_position[:2])),
                parking_success=info['parking_success'], reason=info.get('terminated_reason', 'running'),
                sim_seconds=info['env_time'], parking_distance=control['parking_distance'],
                speed=control['base_speed'], reward=reward_total,
                robot_collision=info['robot_dangerous_collision'],
                launch_success=info['launch_success']))
        finally:
            env.close()
    return dict(kind='scripted_navigation_feasibility_not_RL_success', episodes=results,
                parked=sum(r['parking_success'] for r in results), count=len(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--report', type=Path, default=Path('/tmp/basket_mobile_probe.json'))
    parser.add_argument('--transfer-controls', action='store_true')
    parser.add_argument('--parking-stage', action='store_true')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    report = run(args.episodes, args.transfer_controls, args.parking_stage)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Diagnostic parking: {}/{}; report={}'.format(report['parked'], report['count'], args.report))
    raise SystemExit(0 if report['parked'] == report['count'] else 1)

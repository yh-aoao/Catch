"""Check physical holding and delayed action opening; not a throwing policy test."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(episodes, hold_min, hold_max):
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    rows=[]
    for seed in range(episodes):
        np.random.seed(123+seed)
        env=DcmmVecEnv(task='Catching', object_motion='throw_basket',
            basket_arm_control='joint_velocity', basket_reward_version='basket_mobile_v1',
            basket_mobile_options=dict(hold_only=True, fixed_throw_probability=1.,
                hold_time_range=[hold_min,hold_max], collision_free_spawn=True,
                hold_position=True, brake_after_unlock=True, settle_contact_grace=.03,
                post_ready_seconds=2.5))
        try:
            env.reset(seed=seed)
            initial=env.Dcmm.data.qpos[:2].copy()
            depth=min((float(c.dist) for c in env.Dcmm.data.contact
                       if env.object_id in (c.geom1,c.geom2)),default=0.)
            assert depth>=0., 'Reset object contact is penetrating'
            for _ in range(250):
                _,_,terminated,truncated,info=env.step(dict(base=np.ones(2)*.5,
                    arm=np.ones(6)*.025,hand=np.ones(12)*.15))
                np.testing.assert_array_equal(env.Dcmm.target_base_vel[:2],np.zeros(2))
                np.testing.assert_array_equal(env.Dcmm.target_arm_qpos,env.basket_carry_arm)
                np.testing.assert_array_equal(env.Dcmm.target_hand_qpos,env.basket_carry_hand)
                assert env.Dcmm.data.ctrl[-1]==0., 'Unexpected ball actuation'
                if env.basket_mobile_state['unlocked'] or terminated or truncated:
                    break
            state=env.basket_mobile_state
            row=dict(seed=seed,clear_spawn=env.basket_spawn_check,initial_depth=depth,
                requested_wait=state['hold_until'],ready_time=state.get('ready_time'),
                held_ball_speed=state.get('held_ball_speed_at_ready'),opened=bool(state['unlocked']),
                base_displacement=float(np.linalg.norm(env.Dcmm.data.qpos[:2]-initial)),
                reason=info.get('terminated_reason'))
            if state['unlocked']:
                assert row['ready_time']>=row['requested_wait']
                assert row['held_ball_speed']<env.basket_cfg.mobile_hold_ball_speed_tolerance
            rows.append(row)
        finally:
            env.close()
    return dict(kind='physical_holding_feasibility_not_RL_success',episodes=rows,
                opened=sum(row['opened'] for row in rows),count=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=15)
    parser.add_argument('--hold-min',type=float,default=.3)
    parser.add_argument('--hold-max',type=float,default=2.)
    parser.add_argument('--report',type=Path,default=Path('/tmp/basket_hold_probe.json'))
    args=parser.parse_args()
    if args.episodes<=0:parser.error('--episodes must be positive')
    result=run(args.episodes,args.hold_min,args.hold_max)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Held until ready: {}/{}; report={}'.format(result['opened'],result['count'],args.report))
    raise SystemExit(0 if result['opened']==result['count'] else 1)

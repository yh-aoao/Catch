"""Controlled physical carry-pose screening. No training and no ball attachment."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from gym_dcmm.utils.basket_grip import palm_coordinates


def candidates():
    flexes=[(.6,.5,.5),(.8,.7,.7),(1.,.9,.9),(.6,.9,.9)]
    positions=[(0.,.03,.07),(.015,0.,.075),(.015,.025,.075),(.015,.05,.075)]
    result=[]
    for angle,flex,pos in itertools.product([0.,10.,20.],flexes,positions):
        result.append(dict(name='p%02d'%len(result),hold_arm_offset_deg=[0.,0.,0.,0.,angle,0.],
            hold_hand_flex=list(flex),hold_thumb_flex=[.4,.4,.4],hold_ball_local=list(pos)))
    return result


def extended_candidates():
    arms=[[0.,-20.,0.,0.,0.,0.],[0.,-40.,0.,0.,0.,0.],
          [0.,0.,-20.,0.,0.,0.],[0.,0.,-40.,0.,0.,0.],
          [0.,-20.,0.,0.,20.,0.],[0.,-30.,0.,0.,20.,0.],
          [0.,-40.,0.,0.,10.,0.]]
    result=[]
    for arm,flex,pos in itertools.product(arms,[(.6,.5,.5),(.8,.7,.7),(1.,.9,.9)],
                                         [(0.,.03,.07),(.015,0.,.075),(.015,.025,.075)]):
        result.append(dict(name='e%02d'%len(result),hold_arm_offset_deg=arm,
            hold_hand_flex=list(flex),hold_thumb_flex=[.4,.4,.4],hold_ball_local=list(pos)))
    return result


def make_env():
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    return DcmmVecEnv(task='Catching',object_motion='throw_basket',
        basket_arm_control='joint_velocity',basket_reward_version='basket_mobile_v1')


def configure(env,candidate,wait):
    from gym_dcmm.utils.basket_mobile import make_config
    import configs.env.DcmmCfg as base
    options=dict(hold_only=True,fixed_throw_probability=1.,hold_time_range=[wait,wait],
        collision_free_spawn=True,hold_position=True,brake_after_unlock=True,
        settle_contact_grace=.03,post_ready_seconds=2.5)
    options.update({k:v for k,v in candidate.items() if k!='name'})
    env.basket_cfg=make_config(base,options)


def sample(env):
    import mujoco
    mujoco.mj_forward(env.Dcmm.model,env.Dcmm.data)
    data=env.Dcmm.data
    return dict(time=float(data.time),local=palm_coordinates(data,data.qpos[37:40]).tolist(),
        world=data.qpos[37:40].tolist(),speed=float(np.linalg.norm(data.qvel[36:39])),
        hand=env._get_hand_obs().tolist())


def hold_trial(env,candidate,seed=0,seconds=2.):
    configure(env,candidate,4.)
    np.random.seed(123+seed)
    obs,_=env.reset(seed=123+seed)
    trace=[sample(env)];done=False
    for _ in range(int(np.ceil(seconds/.04))):
        obs,_,terminated,truncated,info=env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
        trace.append(sample(env))
        if terminated or truncated:
            done=True;break
    after=[x for x in trace if x['time']>=.3]
    late=[x for x in trace if x['time']>=seconds-.4]
    position=np.array([x['local'] for x in after])
    last=np.array([x['local'] for x in late])
    return dict(candidate=candidate,seed=seed,retained=not done,
        reason=info.get('terminated_reason'),robot_collision=info['robot_dangerous_collision'],
        spawn=env.basket_spawn_check,final_local=trace[-1]['local'],
        wait_spread=float(np.linalg.norm(np.ptp(position,axis=0))) if len(position) else None,
        late_spread=float(np.linalg.norm(np.ptp(last,axis=0))) if len(last)>1 else None,
        final_speed=trace[-1]['speed'],trace=trace)


class ThrowActor:
    def __init__(self,filename):
        import torch
        from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
        from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
        checkpoint=torch.load(filename,map_location='cpu')
        state=checkpoint['model'];self.width=state['actor_mlp.mlp.0.weight'].shape[1]
        if self.width not in (35,51) or state['mu.weight'].shape[0]!=18:
            raise ValueError('Grip diagnostic requires a fixed/held arm18 actor')
        units=[v.shape[0] for k,v in state.items() if k.startswith('actor_mlp.') and k.endswith('.weight')]
        self.model=ActorCritic(dict(actor_units=units,actions_num=18,input_shape=(self.width,),
            separate_value_mlp=any(k.startswith('value_mlp.') for k in state))).eval()
        self.model.load_state_dict(state,strict=True)
        self.normalizer=RunningMeanStd((self.width,)).eval()
        self.normalizer.load_state_dict(checkpoint['running_mean_std'],strict=True)

    def action(self,obs):
        import torch
        parts=[obs['base']['v_lin_2d'],obs['arm']['ee_pos3d'],obs['arm']['ee_quat'],
            obs['arm']['ee_v_lin_3d'],obs['object']['pos3d'],obs['object']['v_lin_3d'],
            obs['hand'],obs['basket']['rel_pos3d'],obs['basket']['target_rel_pos2d']]
        if self.width==51:parts.append(obs['mobile'])
        raw=np.concatenate(parts)
        if self.width==35:raw[-2:]=[0.,.2]
        with torch.no_grad():
            return self.model.act_inference(dict(obs=self.normalizer(torch.tensor(raw[None],dtype=torch.float32)))).clamp(-1,1).numpy()[0]


def record_macro(filename):
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    actor=ThrowActor(filename);np.random.seed(123)
    env=DcmmVecEnv(task='Catching',object_motion='throw_basket',basket_arm_control='joint_velocity',
        basket_fixed_base_training=True,basket_reward_version='basket_aim_v1')
    actions=[]
    try:
        obs,_=env.reset(seed=123)
        for _ in range(60):
            action=actor.action(obs);actions.append(action.tolist())
            obs,_,t,tr,_=env.step(dict(base=np.zeros(2),arm=action[:6]*.025,hand=action[6:]*.15))
            if t or tr:break
    finally:env.close()
    return np.asarray(actions)


def throw_trial(env,candidate,wait,seed,actor=None,macro=None,opening=.04):
    configure(env,candidate,wait)
    np.random.seed(123+seed);obs,_=env.reset(seed=123+seed)
    hold_trace=[sample(env)]
    for _ in range(230):
        obs,_,t,tr,info=env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
        hold_trace.append(sample(env))
        if env.basket_mobile_state['unlocked'] or t or tr:break
    opened=bool(env.basket_mobile_state['unlocked']);local_at_ready=sample(env)['local']
    if opened:
        for i in range(100):
            if actor is not None:action=actor.action(obs)
            else:
                action=np.zeros(18);action[:6]=macro[min(i,len(macro)-1),:6]
                if i*.04>=opening:
                    mask=[0,2,3,4,6,7,8,10,11,13,14,15]
                    action[6:]=np.clip(-env.Dcmm.target_hand_qpos[mask]/.15,-1,1)
            obs,_,t,tr,info=env.step(dict(base=np.zeros(2),arm=action[:6]*.025,hand=action[6:]*.15))
            if t or tr:break
    state=env.basket_simple_state;release=state.get('candidate',{})
    last=env.basket_last_contact or {}
    return dict(candidate=candidate,seed=seed,requested_wait=wait,
        actual_ready=env.basket_mobile_state.get('ready_time'),local_at_ready=local_at_ready,
        controller='actor' if actor is not None else 'common_arm_macro_opening',opening=opening,
        launch=info['launch_success'],safe=info['safe_launch_success'],hit=info['safe_basket_success'],
        robot_collision=info['robot_dangerous_collision'],distance=state.get('best_distance',0.),
        release_velocity=release.get('velocity',np.zeros(3)).tolist(),
        release_local=last.get('palm_local'),
        aim_error=info['basket_control']['aim_error'],reason=info.get('terminated_reason'),hold_trace=hold_trace)


def screen(report,extended=False):
    env=make_env();rows=[]
    try:
        for candidate in (extended_candidates() if extended else candidates()):
            try:
                row=hold_trial(env,candidate)
            except ValueError as exc:
                row=dict(candidate=candidate,retained=False,error=str(exc))
            rows.append(row)
            print('HOLD',candidate['name'],row['retained'],
                  row.get('final_local'),row.get('wait_spread'),flush=True)
            report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps(rows,indent=2))
    finally:
        env.close()
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('/tmp/basket_grip_screen.json'))
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    result=screen(args.report,args.extended)
    good=[r for r in result if r['retained'] and not r.get('robot_collision')]
    print('Retained without dangerous robot collision:',len(good),'/',len(result))

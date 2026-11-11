"""Replay audited launch states through Track and Catch; read-only diagnosis."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'catch_it_copy'))
from gym_dcmm.envs.object_benchmark import make_object_benchmark
from ycb_objects.evaluate_trained_balls import PolicyEnvironment,batched
from gym_dcmm.utils.roll_rewards import hand_collision_ids
from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_track import PPO_Track
from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
OUT=ROOT/'analysis_artifacts/roll_launch_audit_20261111'

def run(job):
    name,stage=job;torch.set_num_threads(1)
    audit=json.loads((OUT/f'{name}_dr.json').read_text())
    cfg=OmegaConf.load(ROOT/'analysis_artifacts/all_targets_catch_setup_20261104/roll_resolved.yaml')
    cfg.test=True;cfg.num_envs=1;cfg.train.ppo.num_actors=1;cfg.rl_device='cpu';cfg.device_id=-1
    cfg.checkpoint_tracking=cfg.checkpoint_catching=''
    cfg.task='Tracking' if stage=='track' else 'Catching_TwoStage'
    checkpoint=ROOT/('11-6/roll_track_best_reward_25.16.pth' if stage=='track' else '11-11/roll_catch_best_reward_222.64.pth')
    agent=None;rows=[]
    for slot in range(4):
        env=make_object_benchmark(object_benchmark='roll_v1',benchmark_catalog=OUT/'catalog_used.json',
            benchmark_options={'domain_randomization':'physical_dr_v1'},object_motion='roll',
            task='Tracking' if stage=='track' else 'Catching',test_object=name,viewer=False,
            render_mode=None,object_eval=False,benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=62000+slot)
        try:
            wrapper=PolicyEnvironment(env)
            if agent is None:
                cls=PPO_Track if stage=='track' else PPO_Catch_TwoStage
                agent=cls(wrapper,str(OUT/f'policy_{name}_{stage}'),cfg);agent.restore_test(str(checkpoint));agent.set_eval()
            else:agent.env=wrapper
            tracker=None;original=env._get_contacts
            def contacts():
                result=original()
                if tracker is not None:
                    model,data=env.Dcmm.model,env.Dcmm.data
                    ids=tracker['object_ids'];hands=tracker['hands'];hand=False;table=False
                    for c in data.contact:
                        a,b=int(c.geom1),int(c.geom2)
                        if (a in ids)==(b in ids):continue
                        other=b if a in ids else a
                        hand=hand or other in hands;table=table or other==env.table_geom_id
                    if hand and tracker['first_hand'] is None:tracker['first_hand']=float(data.time)
                    tracker['hand_on_table']=tracker['hand_on_table'] or bool(hand and table)
                return result
            env._get_contacts=contacts
            selected=[r for r in audit['rows'] if r['slot']==slot][:5]
            for launch in selected:
                tracker=None;env.reset(seed=launch['seed'])
                initial=launch['initial']
                env.object_pos3d=np.asarray(initial['position']);env.object_q=np.asarray(initial['quaternion']);env.object_vel6d=np.asarray(initial['velocity'])
                env.Dcmm.set_throw_pos_vel(pose=np.r_[env.object_pos3d,env.object_q],velocity=env.object_vel6d)
                mujoco.mj_forward(env.Dcmm.model,env.Dcmm.data);obs=env._get_obs()
                bid=env.Dcmm.model.body('object').id
                tracker=dict(object_ids=set(np.flatnonzero(env.Dcmm.model.geom_bodyid==bid).tolist()),
                    hands=set(hand_collision_ids(env.Dcmm.model,env.hand_start_id).tolist()),first_hand=None,hand_on_table=False)
                actual_depart=None;distance_at_passive_depart=None
                with torch.inference_mode():
                    for step in range(200):
                        actions=agent.model_act({'obs':agent.obs2tensor(batched(obs))},inference=True)['actions'].clamp(-1,1)
                        actions=torch.nn.functional.pad(actions,(0,agent.full_action_dim-actions.shape[1]))
                        command=agent.action2dict(actions)
                        obs,reward,term,trunc,info=env.step({k:v[0] for k,v in command.items()})
                        if info['roll_eval'].get('departed') and actual_depart is None:actual_depart=float(info['env_time'])
                        if launch['departure'] is not None and distance_at_passive_depart is None and info['env_time']>=launch['departure']:
                            distance_at_passive_depart=float(info['ee_distance'])
                        if term or trunc:break
                rows.append(dict(slot=slot,seed=launch['seed'],passive_valid=launch['launch_valid'],
                    passive_departure=launch['departure'],actual_departure=actual_depart,
                    first_hand_contact=tracker['first_hand'],hand_on_table=tracker['hand_on_table'],
                    hand_distance_at_passive_departure=distance_at_passive_depart,
                    success=bool(info['roll_eval_success']),reason=info.get('terminated_reason'),
                    final_eval=info['roll_eval']))
        finally:env.close()
    if agent is not None:agent.writer.close()
    result=dict(object=name,stage=stage,episodes=len(rows),successes=sum(r['success'] for r in rows),
        passive_valid=sum(r['passive_valid'] for r in rows),hand_contact=sum(r['first_hand_contact'] is not None for r in rows),
        hand_on_table=sum(r['hand_on_table'] for r in rows),rows=rows,checkpoint=str(checkpoint))
    (OUT/f'policy_{name}_{stage}.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='rows'}

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,(n,s)) for n in ('apple','lemon') for s in ('track','catch')]):
            print(json.dumps(f.result()),flush=True)

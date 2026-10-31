"""Read-only timestep experiment; never changes project physics defaults."""
import json,time,hashlib
from pathlib import Path
import mujoco,numpy as np,torch
from omegaconf import OmegaConf
from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from ycb_objects.evaluate_trained_balls import PolicyEnvironment,batched
from ycb_objects.generalization_probe import ContactRecorder
ROOT=Path('/home/isee/Catch/catch_it_copy')
OUT=ROOT.parent/'analysis_artifacts/bounce_viewer_speed_20261029'
RUN=ROOT/'outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32'
MATERIAL=ROOT/'ycb_objects/calibration_data/balls.three_ball_training_v1.json'
DTS=[.000125,.00025,.0005,.001]
BALLS=['default','ycb:tennis_ball','ycb:baseball']
torch.set_num_threads(1)

def make(ball):
 return make_bounce_physical_v2(task='Catching',object_motion='bounce',test_object=ball,
  bounce_launch='baseball_intercept_v1',bounce_material_file=MATERIAL,
  viewer=False,render_mode=None,object_eval=False,env_time=2.5,steps_per_policy=20)

def aslist(a):return ' '.join(str(float(x)) for x in np.asarray(a).ravel())

def isolated(s,dt):
 xml=f'''<mujoco><option timestep="{dt}" integrator="RK4" gravity="0 0 -9.81" cone="pyramidal"/>
 <worldbody><geom name="floor" type="plane" size="5 5 .1" solref=".02 1"/>
 <body name="object" pos="{aslist(s['position'])}"><freejoint/>
 <inertial pos="0 0 0" mass="{s['mass']}" diaginertia="{aslist(s['inertia'])}"/>
 <geom name="ball" type="sphere" size="{s['radius']}" mass="0" solref=".02 1"/>
 </body></worldbody><contact><pair geom1="floor" geom2="ball" condim="4"
 friction="{aslist(s['pair_friction'])}" solref="{aslist(s['pair_solref'])}"
 solimp="{aslist(s['pair_solimp'])}"/></contact></mujoco>'''
 m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qvel[:]=s['velocity']
 m.pair_solreffriction[0]=s['pair_solreffriction']
 mujoco.mj_forward(m,d)
 initialenergy=.5*s['mass']*float(np.dot(d.qvel[:3],d.qvel[:3]))+.5*float(np.dot(s['inertia'],d.qvel[3:]**2))
 bound=s['position'][2]+initialenergy/(s['mass']*9.81)
 hit=False;left=False;before_vz=float(d.qvel[2]);peak=None;impact=None;minz=float(d.qpos[2]);started=time.perf_counter()
 for i in range(round(2./dt)):
  mujoco.mj_step(m,d);z=float(d.qpos[2]);minz=min(minz,z)
  if d.ncon and not hit:
   impact=dict(time=float(d.time),position=d.qpos[:3].tolist(),velocity=d.qvel[:3].tolist());hit=True
  if hit and not d.ncon:left=True
  if left and before_vz>0 and d.qvel[2]<=0 and peak is None:
   peak=dict(time=float(d.time),position=d.qpos[:3].tolist(),velocity=d.qvel[:3].tolist())
  before_vz=float(d.qvel[2])
 elapsed=time.perf_counter()-started
 return dict(dt=dt,first_impact=impact,first_peak=peak,min_z=minz,
  max_penetration=max(0.,s['radius']-minz),energy_height_bound=bound,
  passive_energy_violation=bool(peak and peak['position'][2]>bound+.005),wall_seconds=elapsed,
  simulation_duration=2.,real_time_factor=2./elapsed)

result=dict(assumptions='Same material, launch, mass/inertia, zero armature/damping; isolated sphere-floor RK4; 2ms control and40ms policy preserved in closed-loop runs. Candidates are previews, not certified equivalent.',
 baseline_dt=DTS[0],candidate_dts=DTS[1:],mujoco_version=mujoco.__version__,isolated=[],closed_loop=[])
for ball in BALLS:
 env=make(ball)
 try:
  for seed in range(3):
   env.reset(seed=seed);m=env.Dcmm.model;bid=m.body('object').id;p=m.pair('bounce_v2_ground').id
   s=dict(ball=ball,seed=seed,position=env.object_pos3d.tolist(),velocity=env.object_vel6d.tolist(),
    radius=float(m.geom_size[env.object_id,0]),mass=float(m.body_mass[bid]),inertia=m.body_inertia[bid].copy(),
    pair_friction=m.pair_friction[p].copy(),pair_solref=m.pair_solref[p].copy(),pair_solimp=m.pair_solimp[p].copy(),pair_solreffriction=m.pair_solreffriction[p].copy())
   cases=[isolated(s,dt) for dt in DTS];ref=cases[0]
   for c in cases:
    c.update(ball=ball,seed=seed)
    if c['first_peak'] and ref['first_peak']:
     c['peak_height_error_m']=abs(c['first_peak']['position'][2]-ref['first_peak']['position'][2])
     c['peak_time_error_s']=abs(c['first_peak']['time']-ref['first_peak']['time'])
     c['peak_xy_error_m']=float(np.linalg.norm(np.asarray(c['first_peak']['position'][:2])-ref['first_peak']['position'][:2]))
   result['isolated'].extend(cases)
 finally:env.close()
for dt in DTS:
 rs=[r for r in result['isolated'] if r['dt']==dt]
 print('ISOLATED',dt,'max-height-error',max(r.get('peak_height_error_m',0) for r in rs),'max-xy-error',max(r.get('peak_xy_error_m',0) for r in rs),'energy violations',sum(r['passive_energy_violation'] for r in rs),flush=True)
(OUT/'dt_probe.json').write_text(json.dumps(result,indent=2))

cfg=OmegaConf.load(RUN/'resolved_config.yaml');cfg.test=True;cfg.num_envs=1;cfg.train.ppo.num_actors=1;cfg.rl_device='cpu';cfg.device_id=-1;cfg.viewer=False;cfg.checkpoint_tracking='';cfg.checkpoint_catching=''
ckpt=RUN/'nn/bounce_catch_best_reward_464.31.pth';weights=torch.load(ckpt,map_location='cpu')
result['checkpoint']=str(ckpt);result['checkpoint_sha256']=hashlib.sha256(ckpt.read_bytes()).hexdigest()
with torch.inference_mode():
 for ball in BALLS:
  env=make(ball);agent=PPO_Catch_TwoStage(PolicyEnvironment(env),'/tmp/bounce_dt_probe_agent',cfg)
  try:
   agent.restore_test(str(ckpt));agent.model.load_state_dict(weights['model'],strict=True);agent.set_eval()
   for seed in range(2):
    reftrace=None;refsuccess=None
    for dt in DTS:
     obs,_=env.reset(seed=seed);env.Dcmm.model.opt.timestep=dt;env._bounce_physics_substeps=round(.002/dt);env.fps=25
     recorder=ContactRecorder(env);orig=env._get_contacts
     def contacts():
      value=orig();recorder.update();return value
     env._get_contacts=contacts
     trace=[];started=time.perf_counter()
     for k in range(70):
      a=agent.model_act({'obs':agent.obs2tensor(batched(obs))},inference=True)['actions'].clamp(-1,1)
      command=agent.action2dict(a)
      obs,reward,t,u,info=env.step({key:value[0] for key,value in command.items()})
      trace.append(env.Dcmm.data.qpos[37:40].tolist())
      if t or u:break
     wall=time.perf_counter()-started;env._get_contacts=orig
     if reftrace is None:reftrace=np.asarray(trace);refsuccess=bool(info['success'])
     common=min(len(trace),len(reftrace));error=float(np.max(np.linalg.norm(np.asarray(trace)[:common]-reftrace[:common],axis=1)))
     row=dict(ball=ball,seed=seed,dt=dt,substeps=round(.002/dt),steps=len(trace),
      success=bool(info['success']),success_same_as_reference=bool(info['success'])==refsuccess,
      hand_contact=recorder.had_hand,max_contiguous_hand_contact_s=recorder.max_hand_duration,
      final_ball=trace[-1],max_common_policy_position_error_m=error,
      reason=info.get('terminated_reason') or env.terminated_reason or 'timeout',
      simulated_seconds=float(info['env_time']),wall_seconds=wall,wall_ms_per_policy=wall/len(trace)*1000,
      real_time_factor=float(info['env_time'])/wall)
     result['closed_loop'].append(row)
     (OUT/'dt_probe.json').write_text(json.dumps(result,indent=2))
     print('CLOSED',json.dumps(row),flush=True)
  finally:agent.writer.close();env.close()
result['summary']=[]
for dt in DTS:
 a=[r for r in result['isolated'] if r['dt']==dt]
 b=[r for r in result['closed_loop'] if r['dt']==dt]
 result['summary'].append(dict(dt=dt,
  isolated_max_peak_height_error_m=max(r.get('peak_height_error_m',0) for r in a),
  isolated_max_peak_xy_error_m=max(r.get('peak_xy_error_m',0) for r in a),
  isolated_max_peak_time_error_s=max(r.get('peak_time_error_s',0) for r in a),
  passive_energy_violations=sum(r['passive_energy_violation'] for r in a),
  closed_loop_cases=len(b),changed_outcomes=sum(not r['success_same_as_reference'] for r in b),
  max_common_policy_position_error_m=max(r['max_common_policy_position_error_m'] for r in b),
  mean_wall_ms_per_policy=float(np.mean([r['wall_ms_per_policy'] for r in b])),
  mean_real_time_factor=float(np.mean([r['real_time_factor'] for r in b]))))
result['complete']=True
(OUT/'dt_probe.json').write_text(json.dumps(result,indent=2))
print('COMPLETE',flush=True)

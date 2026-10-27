import numpy as np,torch,json
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.utils.basket_teacher import FrozenThrowTeacher
teacher=FrozenThrowTeacher.from_file('/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth')
results=[]
for seed in range(10):
 np.random.seed(seed)
 e=DcmmVecEnv(task='Catching',object_motion='throw_basket',basket_arm_control='joint_velocity',basket_reward_version='basket_mobile_v1')
 e.basket_cfg.basket_track_position_tolerance=.04;e.basket_cfg.basket_track_speed_tolerance=.04;e.basket_cfg.mobile_settle_seconds=.35
 obs,_=e.reset(seed=seed)
 for j in range(250):
  ready=e.basket_mobile_state['unlocked']
  base=np.zeros(2) if ready else e._basket_parking_state()['desired_velocity']
  a=np.zeros(18)
  if ready:
   raw=np.concatenate([obs['base']['v_lin_2d'],obs['arm']['ee_pos3d'],obs['arm']['ee_quat'],obs['arm']['ee_v_lin_3d'],obs['object']['pos3d'],obs['object']['v_lin_3d'],obs['hand'],obs['basket']['rel_pos3d'],obs['basket']['target_rel_pos2d']])
   a=teacher.actions(torch.tensor(raw[None],dtype=torch.float32)).numpy()[0]
  obs,r,t,tr,info=e.step(dict(base=base,arm=a[:6]*.025,hand=a[6:]*.15))
  if t or tr:break
 results.append(dict(seed=seed,success=info['safe_basket_success'],reason=info.get('terminated_reason'),control=info['basket_control']))
 print('RESULT',seed,info['safe_basket_success'],info.get('terminated_reason'),info['basket_control']['aim_error'],flush=True)
 e.close()
open('/tmp/teacher_physics_probe.json','w').write(json.dumps(results,indent=2))

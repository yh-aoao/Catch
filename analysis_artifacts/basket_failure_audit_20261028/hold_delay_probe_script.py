import numpy as np,torch,json
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.utils.basket_teacher import FrozenThrowTeacher
teacher=FrozenThrowTeacher.from_file('/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth')
rows=[]
for delay in [0.,.1,.3,1.,2.]:
 for seed in range(5):
  np.random.seed(123+seed)
  e=DcmmVecEnv(task='Catching',object_motion='throw_basket',basket_fixed_base_training=True,basket_arm_control='joint_velocity',basket_reward_version='basket_aim_v1')
  o,_=e.reset(seed=seed);hold=e.Dcmm.data.qpos[15:21].copy()
  for j in range(150):
   if e.Dcmm.data.time < delay:
    e._basket_joint_probe_target=hold;a=np.zeros(18)
   else:
    if hasattr(e,'_basket_joint_probe_target'):del e._basket_joint_probe_target
    raw=np.concatenate([o['base']['v_lin_2d'],o['arm']['ee_pos3d'],o['arm']['ee_quat'],o['arm']['ee_v_lin_3d'],o['object']['pos3d'],o['object']['v_lin_3d'],o['hand'],o['basket']['rel_pos3d'],o['basket']['target_rel_pos2d']])
    a=teacher.actions(torch.tensor(raw[None],dtype=torch.float32)).numpy()[0]
   o,r,t,tr,i=e.step(dict(base=np.zeros(2),arm=a[:6]*.025,hand=a[6:]*.15))
   if t or tr:break
  rows.append(dict(delay=delay,seed=seed,success=i['safe_basket_success'],reason=i.get('terminated_reason'),control=i['basket_control']))
  e.close()
 print('HOLD',delay,sum(x['success'] for x in rows if x['delay']==delay),'/5',flush=True)
open('/tmp/basket_hold_delay_probe.json','w').write(json.dumps(rows,indent=2))

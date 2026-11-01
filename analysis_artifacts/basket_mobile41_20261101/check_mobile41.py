import os,sys,json
os.chdir('/home/isee/Catch/catch_it_copy');sys.path.insert(0,os.getcwd())
import numpy as np,torch,gymnasium as gym
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from types import SimpleNamespace as N
OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.utils.basket_throw_policy import FrozenThrowPolicy
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.utils.basket_fingers import make_options as finger_options
from gym_dcmm.utils.basket_carry import make_options as carry_options
with initialize_config_dir(config_dir=os.getcwd()+'/configs',version_base='1.1'):
 c=compose(config_name='basket_mobile41')
kwargs={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in ['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control','basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options','basket_finger_reward','basket_finger_options']}
kwargs.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_mobile_options=OmegaConf.to_container(c.basket_mobile),basket_log=False)
if '--fixed' in sys.argv:
 kwargs['basket_mobile_options'].update(target_x_range=[0.,0.],target_y_range=[1.05,1.05])
env=DcmmVecEnv(**kwargs)
source=torch.load(c.basket_throw_checkpoint,map_location='cpu',weights_only=False)
policy=FrozenThrowPolicy(source,finger_options(c.basket_finger_options),carry_options(c.basket_carry_options),'cpu')
records=[]
try:
 for seed in range(10):
  obs,info=env.reset(seed=seed)
  for step in range(350):
   raw=PPO_Catch_OneStage.obs2tensor(N(device='cpu'),{k:({a:np.expand_dims(b,0) for a,b in v.items()} if isinstance(v,dict) else np.expand_dims(v,0)) for k,v in obs.items()})
   arm=policy.actions(raw).numpy()[0]
   park=env._basket_parking_state(); drive=park['error_world']*1.5
   drive*=min(1.,.25/max(np.linalg.norm(drive),1e-9))
   obs,r,done,trunc,info=env.step(dict(base=drive,arm=arm[:6]*.025,hand=arm[6:]*.15))
   if done or trunc:
    records.append(dict(seed=seed,steps=step+1,reason=info.get('terminated_reason'),success=bool(info.get('safe_basket_success')),parking=bool(info.get('parking_success')),qualified=bool(info.get('qualified_release')),control=info.get('basket_control'))); print(seed,records[-1]['reason'],records[-1]['parking'],records[-1]['qualified'],flush=True);break
finally: env.close()
path='/home/isee/Catch/analysis_artifacts/basket_mobile41_20261101';os.makedirs(path,exist_ok=True)
with open(path+('/fixed_smoke.json' if '--fixed' in sys.argv else '/navigation_smoke.json'),'w') as f:json.dump(records,f,indent=2)

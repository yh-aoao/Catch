"""Exercise real CLI resolution, initialization and spawned physics; never optimize."""
import os,sys
from pathlib import Path
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).parent
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

def check_train(config,output,resume=None,initialize_from=None):
 import copy,torch,numpy as np
 from basket_three_stage.policy import Policy
 from basket_three_stage.pool import Pool
 from basket_three_stage.run import package,validate_initialization
 model=Policy()
 if initialize_from:
  saved=torch.load(initialize_from,map_location='cpu',weights_only=False)
  validate_initialization(saved,config);model.load_state_dict(saved['model'],strict=True)
 assert all(p.requires_grad for p in model.parameters())
 options=copy.deepcopy(config['options']);options['evaluation']=True;options['episode_seconds']=.04;options['hold_max_seconds']=.04
 pool=Pool(2,options,123)
 try:
  for _ in range(3):
   with torch.inference_mode():actions=model.act(torch.as_tensor(pool.obs)).numpy()
   obs,reward,term,trunc,infos,final=pool.step(actions)
   assert obs.shape==final.shape==(2,66)
   assert np.isfinite(obs).all() and np.isfinite(reward).all()
   assert (term|trunc).all()
   import json
   json.dumps(infos)
   assert all('reset_sampling' in i and 'next_reset_sampling' in i for i in infos)
 finally:pool.close()
 torch.save(package(model,config,0),output/'untrained_smoke.pth')
 print('PASS CLI',config['options']['curriculum_stage'],'2 workers, load/reset/inference; NO training',flush=True)

def main():
 import basket_three_stage.run as runner
 runner.train=check_train
 previous=None
 for i,stage in enumerate(('hold','throw','mobile'),1):
  output=OUT/('final_cli_stage'+str(i))
  sys.argv=['check_cli','train','--config',f'basket_three_stage/configs/stage{i}_{stage}.json','--output',str(output),'--device','cpu','--num-envs','2']
  if previous:sys.argv+=['--initialize-from',str(previous)]
  runner.main();previous=output/'untrained_smoke.pth'
if __name__=='__main__':main()

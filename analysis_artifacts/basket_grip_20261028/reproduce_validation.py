import json,itertools
from pathlib import Path
from basket_grip_sweep import make_env,hold_trial,throw_trial,ThrowActor,record_macro
root=Path('/home/isee/Catch/analysis_artifacts/basket_grip_20261028');screen=json.loads((root/'screen.json').read_text());choices=[x['candidate'] for x in screen if x['candidate']['name'] in ['p00','p04']]
actor=ThrowActor('/home/isee/Catch/catch_it_copy/outputs/BasketHold_03_06/2026-10-28/11:38:10/nn/throw_basket_catch_best_reward_-3.42.pth');macro=json.loads((root/'common_macro.json').read_text())
import numpy as np
macro=np.asarray(macro)
env=make_env();result=dict(holds=[],throws=[],common_control=[])
try:
 for c,seed in itertools.product(choices,range(10)):
  r=hold_trial(env,c,seed);result['holds'].append(r)
  print('VALIDATE_HOLD',c['name'],seed,r['retained'],r['wait_spread'],flush=True)
 for c,wait,seed in itertools.product(choices,[.4,1.,2.],range(10)):
  r=throw_trial(env,c,wait,seed,actor=actor);result['throws'].append(r)
  print('VALIDATE_THROW',c['name'],wait,seed,r['hit'],round(r['distance'],3),r['reason'],flush=True)
  (root/'validation.json').write_text(json.dumps(result,indent=2))
 for c,seed in itertools.product(choices,range(5)):
  r=throw_trial(env,c,1.,seed,macro=macro,opening=.0);result['common_control'].append(r)
  print('VALIDATE_COMMON',c['name'],seed,r['safe'],round(r['distance'],3),flush=True)
  (root/'validation.json').write_text(json.dumps(result,indent=2))
finally:env.close()

"""Read-only diagnosis of Roll stable interval resets."""
import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,'/home/isee/Catch/catch_it_copy')
import gym_dcmm.utils.roll_evaluation as evaluation
advance=evaluation.advance_catch;publish=evaluation.publish
records=[]
def observe(state,contact,clear,near,speed,dt):
 counts=state.setdefault('_probe_stability',{})
 for key,value in {'ticks':True,'contact':contact,'clear_near':clear and near,'contact_clear_near':contact and clear and near,'slow_contact_clear_near':contact and clear and near and speed<=.15,'fast_contact_clear_near':contact and clear and near and speed>.15,'gap_clear_near':not contact and clear and near}.items():
  counts[key]=counts.get(key,0)+int(value)
 if state.get('stable_duration',0)>0:
  reason='speed' if speed>.15 else 'not_clear' if not clear else 'not_near' if not near else 'long_contact_gap' if not contact and state.get('gap',0)+dt>.08 else None
  if reason:counts['stable_break_'+reason]=counts.get('stable_break_'+reason,0)+1
 return advance(state,contact,clear,near,speed,dt)
def record(env,info,terminated,truncated):
 publish(env,info,terminated,truncated)
 if terminated or truncated:records.append(dict(counts=env._roll_eval.get('_probe_stability'),max_stable_duration=env._roll_eval.get('max_stable_duration'),success=info['roll_eval_success']))
evaluation.advance_catch=observe;evaluation.publish=record
spec=importlib.util.spec_from_file_location('main_eval','/home/isee/Catch/analysis/evaluate_catch_balls_20261102.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.main()
out=Path(sys.argv[sys.argv.index('--output')+1]);(out/'stability_probe.json').write_text(json.dumps(records,indent=2))

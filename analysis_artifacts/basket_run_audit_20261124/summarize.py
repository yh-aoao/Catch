import json,collections,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/isee/Catch/catch_it_copy');OUT=Path(__file__).parent
results=[]
for base in ['basket_autonomous','basket_three_stage']:
 for file in sorted((ROOT/base).rglob('metrics.jsonl')):
  p=file.parent;lines=[json.loads(x) for x in file.read_text().splitlines() if x.strip()];cfgfile=p/'resolved_config.json';cfg=json.loads(cfgfile.read_text()) if cfgfile.exists() else {}
  print('\nRUN',p.relative_to(ROOT),'metrics',len(lines));print('last',lines[-1])
  opt=cfg.get('options',{});print('protocol', {k:opt.get(k) for k in ['training_stage','curriculum_stage','reward_version','arm_velocity_protocol','release_protocol','sampling_probabilities']},'envs',cfg.get('num_envs'))
  evaluations=[]
  for f in sorted(p.glob('eval_*.json')):
   e=json.loads(f.read_text());r=e.get('records',[])
   extra=dict(file=f.name,episodes=e['episodes'],successes=e['successes'],success_rate=e['success_rate'],reasons=e['reasons'])
   extra['groups']={}
   for group in sorted({str(x.get('reset_source_task','unspecified')) for x in r}):
    a=[x for x in r if str(x.get('reset_source_task','unspecified'))==group]
    extra['groups'][group]=dict(n=len(a),successes=sum(x.get('success',x.get('safe_basket_success',False)) for x in a),reasons=dict(collections.Counter(x['terminated_reason'] for x in a)))
   extra['contact_at_end']=sum(x.get('hand_contact',False) for x in r)
   extra['released']=sum(x.get('released',False) for x in r)
   extra['purposeful_launch']=sum(x.get('purposeful_launch',False) for x in r)
   extra['mean_steps']=float(np.mean([x['steps'] for x in r])) if r else 0
   extra['mean_reward']=float(np.mean([x['reward'] for x in r])) if r else 0
   keys=set().union(*(x.get('reward_totals',{}).keys() for x in r))
   extra['mean_terms']={k:float(np.mean([x.get('reward_totals',{}).get(k,0) for x in r])) for k in keys}
   evaluations.append(extra)
   print(f.name,extra['success_rate'],extra['reasons'],'groups',extra['groups'] if base=='basket_three_stage' else '', 'contact',extra['contact_at_end'],'released',extra['released'])
  hashesfile=p/'source_hashes.json';matches={}
  if hashesfile.exists():
   for name,digest in json.loads(hashesfile.read_text()).items():
    q=ROOT/name
    if q.exists():matches[name]=hashlib.sha256(q.read_bytes()).hexdigest()==digest
  results.append(dict(run=str(p.relative_to(ROOT)),last_metrics=lines[-1],config=cfg,evaluations=evaluations,current_source_matches=matches))
  if evaluations:print('LATEST MEAN TERMS',evaluations[-1]['mean_terms'])
(OUT/'summary.json').write_text(json.dumps(results,indent=2))

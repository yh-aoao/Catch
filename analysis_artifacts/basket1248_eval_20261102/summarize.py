from pathlib import Path
import json,numpy as np,collections,hashlib
OUT=Path('/home/isee/Catch/analysis_artifacts/basket1248_eval_20261102')
summary={}
def stats(x):
 x=[float(v) for v in x if v is not None and np.isfinite(v)]
 return dict(mean=float(np.mean(x)),median=float(np.median(x)),min=min(x),max=max(x)) if x else None
for name in ('original','tennis_ball','baseball'):
 records=json.loads((OUT/(name+'.json')).read_text());summary[name]={}
 for mode in ('mean','sample'):
  rows=[r for r in records if r['mode']==mode];candidates=[r['candidate'] for r in rows if r['candidate']]
  terms=sorted(set(k for r in rows for k in r['terms']))
  crossing=[]
  for r in rows:
   c=r['candidate']
   if not c:continue
   p=np.array(c['position']);v=np.array(c['velocity']);g=np.array([0,0,-9.81]);n=np.array([0,-np.sin(np.deg2rad(25)),np.cos(np.deg2rad(25))]);center=np.array([0,1.05,.55])
   roots=np.roots([.5*g@n,v@n,(p-center)@n])
   ts=[float(t.real) for t in roots if abs(t.imag)<1e-8 and t.real>0 and (v+g*t.real)@n<0]
   if ts:
    t=min(ts);point=p+v*t+.5*g*t*t
    crossing.append(dict(point=point.tolist(),error=float(np.linalg.norm(point-center)),dx=float(point[0]),dy=float(point[1]-center[1])))
  nearest=[min(np.linalg.norm(np.array(t['ball'])-np.array([0,1.05,.55])) for t in r['trace']) for r in rows]
  summary[name][mode]=dict(n=len(rows),success=sum(r['success'] for r in rows),launch=sum(r['launch'] for r in rows),
   reasons=dict(collections.Counter(r['reason'] for r in rows)),reward=stats([r['reward'] for r in rows]),
   time=stats([r['control']['policy_seconds'] for r in rows]),
   release_policy_time=stats([r['control']['release_policy_seconds'] for r in rows]),
   release_forward_speed=stats([r['control']['release_forward_speed'] for r in rows]),
   release_vx=stats([c['velocity'][0] for c in candidates]),release_vy=stats([c['velocity'][1] for c in candidates]),
   release_vz=stats([c['velocity'][2] for c in candidates]),release_y=stats([c['position'][1] for c in candidates]),
   release_z=stats([c['position'][2] for c in candidates]),predicted_error=stats([r['control']['aim_error'] for r in rows]),
   predicted_cross_dx=stats([x['dx'] for x in crossing]),predicted_cross_dy=stats([x['dy'] for x in crossing]),
   nearest_policy_sample_distance=stats(nearest),
   arm_forward=stats([r['control']['arm_forward'] for r in rows]),
   proximal_speed=stats([r['control']['proximal_forward_speed'] for r in rows]),
   reward_terms={k:float(np.mean([r['terms'].get(k,0.) for r in rows])) for k in terms},
   crossing_rows=crossing)
summary['checkpoint_sha256']=hashlib.sha256(Path('/home/isee/Catch/11-2/throw_basket_catch_best_reward_12.48.pth').read_bytes()).hexdigest()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ('original','tennis_ball','baseball'):
 for mode in ('mean','sample'):
  r=summary[name][mode]
  print(name,mode,'SR',r['success'],'/',r['n'],'launch',r['launch'],'reason',r['reasons'])
  print({k:round(r[k]['mean'],4) if r[k] else None for k in ['reward','release_policy_time','release_forward_speed','release_vx','release_vy','release_vz','release_y','release_z','predicted_error','predicted_cross_dx','predicted_cross_dy','arm_forward','proximal_speed']})
  print('terms',{k:round(v,3) for k,v in r['reward_terms'].items()})

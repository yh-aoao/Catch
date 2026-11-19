from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
assert (p/'COMPLETE').is_file()
spec=json.loads((p/'run_spec.json').read_text());root=p.parents[1]/'catch_it_copy'
for name,h in spec['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
models=spec['models'];catalog=json.loads((p/'catalog_used.json').read_text());totals=[];pairs={}
for m in models:
 successes=stable=0;n=0
 for obj in catalog[m['motion']]:
  folder=p/m['name']/obj;d=json.loads((folder/'summary.json').read_text())
  rows=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
  assert len(rows)==spec['episodes']==d['episodes']
  assert len({(r['slot'],r['episode']) for r in rows})==spec['episodes']
  for slot in range(spec['instances']):
   draws=[r['physics']['randomization'] for r in rows if r['slot']==slot]
   assert len(draws)==spec['episodes']//spec['instances']
   assert all(x==draws[0] for x in draws)
  pairs[m['name'],obj]={(r['slot'],r['episode']):(r['episode_seed'],r['initial'],r['physics']['randomization']) for r in rows}
  successes+=d['successes'];n+=d['episodes'];stable+=d['stable_hold_episodes'] or 0
 totals.append(dict(name=m['name'],episodes=n,successes=successes,success_rate=successes/n,stable_hold_rate=stable/n if m['motion']=='roll' else None))
for motion in ('roll','bounce'):
 for obj in catalog[motion]:assert pairs[motion+'_best_1119',obj]==pairs[motion+'_last_1119',obj],(motion,obj)
(p/'VALIDATION.json').write_text(json.dumps(dict(completed=True,paired_best_last_initial_states_and_physical_draws=True,creation_time_parameters_fixed_across_resets=True,source_hashes_match=True,totals=totals),indent=2))
print(json.dumps(totals,indent=2))

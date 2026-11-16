from pathlib import Path
import json
import numpy as np
BASE=Path(__file__).resolve().parent;OUT=BASE/'results'
assert (OUT/'COMPLETE').exists(),'Evaluation is not complete'
models=json.loads((BASE/'models.json').read_text());catalog=json.loads((OUT/'catalog_used.json').read_text())
rows=[];episodes={};validation=[]
for model in models:
 for obj in catalog[model['motion']]:
  case=OUT/model['name']/obj;summary=json.loads((case/'summary.json').read_text());r=[json.loads(s) for s in (case/'episodes.jsonl').read_text().splitlines()]
  assert len(r)==20 and summary['completed'] and sum(x['success'] for x in r)==summary['successes']
  assert len({(x['slot'],x['episode']) for x in r})==20
  assert all(sum(x['slot']==i for x in r)==5 for i in range(4))
  assert len({x['physics']['randomization']['size_scale'] for x in r})==20
  for x in r:
   d=x['physics']['randomization'];assert d['version']=='physical_dr_episode_v1' and d['sampling']=='episode_reset'
   for k,lo,hi in [('size_scale',.95,1.05),('mass_scale',.9,1.1),('friction_scale',.85,1.15),('rolling_scale',.8,1.2),('damping_scale',.9,1.1)]:assert lo<=d[k]<=hi
  failed=[x for x in r if not x['success']]
  row={k:summary[k] for k in ('name','motion','object','episodes','successes','success_rate','stable_hold_episodes','stable_hold_rate','wilson_95_interval')}
  row['failure_reasons']={reason:sum(x['reason']==reason for x in failed) for reason in sorted({x['reason'] for x in failed})}
  if model['motion']=='bounce':
   row['failed_with_hand_contact']=sum(bool(x['bounce_diagnostics'].get('bounce_grasp_had_contact',False)) for x in failed)
  rows.append(row);episodes[model['name'],obj]={(x['slot'],x['episode']):x for x in r}
  validation.append(dict(model=model['name'],object=obj,episodes=20,physical_draws=20,quota_per_slot=5))
for obj in catalog['bounce']:
 a=episodes['bounce_best_1_57',obj];b=episodes['bounce_last',obj]
 for k,x in a.items():
  y=b[k];assert x['episode_seed']==y['episode_seed'];assert x['physics']['randomization']==y['physics']['randomization']
  np.testing.assert_array_equal(x['initial']['qpos'],y['initial']['qpos']);np.testing.assert_array_equal(x['initial']['qvel'],y['initial']['qvel'])
totals=[]
for m in models:
 rr=[r for r in rows if r['name']==m['name']];n=sum(r['episodes'] for r in rr);k=sum(r['successes'] for r in rr)
 totals.append(dict(name=m['name'],episodes=n,successes=k,success_rate=k/n,stable_hold_rate=sum(r['stable_hold_episodes'] for r in rr)/n if m['motion']=='roll' else None))
(BASE/'validated_summary.json').write_text(json.dumps(rows,indent=2));(BASE/'totals.json').write_text(json.dumps(totals,indent=2))
(BASE/'VALIDATION.json').write_text(json.dumps(dict(cases=19,episodes=380,completed=True,per_episode_draws_verified=True,bounce_models_initial_states_and_physics_identical=True,cases_checked=validation),indent=2))
lines=['# 2026-11-16 Catch 小样本评估','','每目标20回合，4个同步环境各5回合；每回合重新采样尺寸、质量与接触参数（physical_dr_episode_v1）。每目标核验20组不同物理参数。固定确定性策略，发射基种子41000；保留所有失败回合。','','|模型|目标|成功次数|成功率|Roll稳定持球率|失败原因|','|---|---|---:|---:|---:|---|']
for r in rows:
 stable=f"{r['stable_hold_rate']:.1%}" if r['stable_hold_rate'] is not None else '—'
 lines.append(f"|{r['name']}|{r['object']}|{r['successes']}/20|{r['success_rate']:.1%}|{stable}|{r['failure_reasons']}|")
lines+=['','|模型|成功次数|总体成功率|Roll稳定持球率|','|---|---:|---:|---:|']
for r in totals:
 stable=f"{r['stable_hold_rate']:.1%}" if r['stable_hold_rate'] is not None else '—'
 lines.append(f"|{r['name']}|{r['successes']}/{r['episodes']}|{r['success_rate']:.1%}|{stable}|")
lines+=['','Roll主指标为回合结束时region_retention_v5：发生真实手接触并离桌、无外部支撑，物体最后仍在手部保留区域；不要求低相对速度。稳定持球是回合中曾连续满足真实接触、低相对速度等条件约0.3秒，不保证最后仍稳定。','Bounce采用controlled_hold_v1：离地、无外部支撑、多部位接触及低相对速度等条件连续成立约0.3秒。Roll与Bounce的成功判定不同，不宜直接横向比较。','Bounce best和last已核验所有对应回合的初始位置、速度、物理参数完全一致；同步环境沿用全局随机数，后续噪声不保证严格逐步配对。','每目标20回合，单回合对应5个百分点；Wilson区间见validated_summary.json，不能把小差异认定为确定提升。','现有历史固定参数/每环境采样结果与本次逐回合采样的分布不同，不应仅按数字直接判断训练方案优劣。','原始配置、checkpoint及源文件哈希见results/run_spec.json；每回合初始状态、终止原因及物理参数见results/模型名/目标名/episodes.jsonl。未启动训练。']
(BASE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(totals,indent=2))

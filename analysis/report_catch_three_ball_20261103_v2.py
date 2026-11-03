from pathlib import Path
from collections import Counter
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'analysis_artifacts/catch_three_ball_20261103_v2'
results=[];comparison=[]
for task in ('roll','bounce'):
 for ball in ('original','tennis_ball','baseball'):
  p=out/(task+'_best_'+ball)
  assert (p/'COMPLETE').is_file(),p
  rows=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()]
  assert len(rows)==100 and [r['seed'] for r in rows]==list(range(100))
  result=json.loads((p/'summary.json').read_text());result['failed_after_contact']=sum(not r['success'] and r['hand_contact'] for r in rows)
  if task=='roll':
   result['stable_and_success']=sum(bool(r['roll_stable_hold']) and r['success'] for r in rows)
   result['failure_region']=dict(Counter((r['roll_evaluation'] or {}).get('region_unmet') or 'other' for r in rows if not r['success']))
  else:result['failed_some_valid_hold']=sum(not r['success'] and r['bounce_diagnostics']['bounce_grasp_max_hold_seconds']>0 for r in rows)
  results.append(result)
  references=[('11-2',ROOT/'analysis_artifacts/catch_three_ball_20261102'/(task+'_'+ball))]
  references.extend(('11-3 '+v,ROOT/'analysis_artifacts/catch_best_last_20261103'/(task+'_'+v+'_'+ball)) for v in ('best','last'))
  for label,ref in references:
   old=[json.loads(x) for x in (ref/'episodes.jsonl').read_text().splitlines()]
   initial_equal=sum(a['initial']==b['initial'] for a,b in zip(old,rows));assert initial_equal==100
   comparison.append(dict(task=task,ball=ball,reference=label,initial_equal=initial_equal,old_successes=sum(r['success'] for r in old),new_successes=result['successes'],improved=sum(not a['success'] and b['success'] for a,b in zip(old,rows)),regressed=sum(a['success'] and not b['success'] for a,b in zip(old,rows))))
notes=json.loads((out/'evaluation_notes.json').read_text())
for name,m in notes['models'].items():
 assert hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']
 for ball in ('original','tennis_ball','baseball'):
  assert json.loads((out/(name+'_'+ball)/'provenance.json').read_text())['checkpoint_sha256']==m['sha256']
status=json.loads((out/'run_status.json').read_text());assert len(status)==6 and all(r['exit']==0 for r in status)
(out/'summary_all.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(out/'historical_paired_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
with (out/'summary.csv').open('w') as f:
 fields=['task','object','episodes','successes','success_rate','stable_hold_episodes','hand_contact_episodes','failed_after_contact']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
lines=['# 11-3-2 Catch模型三球评估','',
'2026-11-03：Roll269.97、Bounce9.40，各球100局，种子0–99，共600局。原PPO确定性推理、严格加载actor/RMS，失败全部计入分母。与11-2、11-3 best/last逐球100/100初始状态一致。',
'目录没有服务器原始resolved_config.yaml；沿用前两轮重建配置，若服务器改过环境或动作参数需补原配置复核。Bounce checkpoint确认controlled_hold_v1、439万步附近、联合训练不冻结。',
'Roll主成功=最终区域保持且排除失败，稳定持球另列（曾达到相对速度<=0.15m/s持续0.3秒）。Bounce主成功=连续0.3秒真实多区域受控保持，成功立即终止；两任务口径不同。','',
'|任务|原始球|网球|棒球|','|---|---:|---:|---:|']
for task in ('roll','bounce'):
 rs=[r for r in results if r['task']==task]
 lines.append('|'+task+'|'+'|'.join(str(r['successes'])+'%' for r in rs)+'|')
 if task=='roll':lines.append('|Roll曾稳定持球|'+'|'.join(str(r['stable_hold_episodes'])+'%' for r in rs)+'|')
lines+=['','## 诊断','']
for r in results:
 lines.append(f"- {r['task']} {r['object']}：触手{r['hand_contact_episodes']}/100；失败但曾触手{r['failed_after_contact']}；终止原因{r['reasons']}。")
 if r['task']=='roll':lines.append(f"  主要成功且曾稳定持球{r['stable_and_success']}局；失败末态区域诊断{r['failure_region']}。")
 else:lines.append(f"  失败中曾满足短暂有效保持{r['failed_some_valid_hold']}局。")
lines+=['','## 历史对照','', '|任务|球|参照模型|旧成功率|本次成功率|','|---|---|---|---:|---:|']
for c in comparison:lines.append(f"|{c['task']}|{c['ball']}|{c['reference']}|{c['old_successes']}%|{c['new_successes']}%|")
lines+=['','比例有抽样误差，95% Wilson区间见summary_all.json。相同种子既用于比较也用于选模型，不是独立保留验收集。reward文件名不代表抓取成功率；未修改训练奖励或控制代码。各子目录包含模型/配置/源码哈希、真实物理参数和逐回合记录。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'COMPLETE').write_text('600 episodes, all exit 0, all model hashes unchanged and historical initial states identical.\n')
print('\n'.join(lines[:13]))

"""Summarize paired best/last evaluation without changing success definitions."""
from pathlib import Path
from collections import Counter
import json,csv
ROOT=Path(__file__).resolve().parents[1]/'analysis_artifacts/catch_best_last_20261103'
results=[];episodes={}
for task in ('roll','bounce'):
 for variant in ('best','last'):
  for ball in ('original','tennis_ball','baseball'):
   name=f'{task}_{variant}_{ball}';p=ROOT/name
   assert (p/'COMPLETE').exists(),name
   data=[json.loads(line) for line in (p/'episodes.jsonl').read_text().splitlines()]
   assert len(data)==100 and [r['seed'] for r in data]==list(range(100))
   episodes[name]=data
   r=json.loads((p/'summary.json').read_text());r.update(model=task+'_'+variant,variant=variant,ball=ball)
   r['failed_after_contact']=sum(not x['success'] and x['hand_contact'] for x in data)
   r['no_contact']=sum(not x['hand_contact'] for x in data)
   if task=='roll':
    r['stable_and_success']=sum(x['success'] and bool(x['roll_stable_hold']) for x in data)
    r['failure_region']=dict(Counter((x['roll_evaluation'] or {}).get('region_unmet') or 'other' for x in data if not x['success']))
   else:
    r['failed_some_valid_hold']=sum(not x['success'] and x['bounce_diagnostics']['bounce_grasp_max_hold_seconds']>0 for x in data)
   results.append(r)
comparisons=[]
for task in ('roll','bounce'):
 for ball in ('original','tennis_ball','baseball'):
  best=episodes[f'{task}_best_{ball}'];last=episodes[f'{task}_last_{ball}']
  c=dict(task=task,ball=ball,initial_equal=sum(a['initial']==b['initial'] for a,b in zip(best,last)),last_wins=sum(not a['success'] and b['success'] for a,b in zip(best,last)),best_wins=sum(a['success'] and not b['success'] for a,b in zip(best,last)))
  assert c['initial_equal']==100,c
  comparisons.append(c)
(ROOT/'summary_all.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(ROOT/'paired_comparison.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
fields=['task','variant','ball','episodes','successes','success_rate','stable_hold_episodes','hand_contact_episodes','failed_after_contact','no_contact']
with (ROOT/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
lines=['# 2026-11-03 Roll / Bounce best与last对照','',
'每模型/球100回合、种子0–99，共1200回合。原PPO确定性推理，严格加载actor与RMS，所有失败计入分母。best/last对应100/100初始状态完全相同。',
'训练服务器未提供原始resolved_config.yaml，沿用11-02已验证的重建配置；若服务器覆盖过速度、发射、手部动作尺度或奖励选项，需原配置复核。Bounce checkpoint确认controlled_hold_v1且track_lr_scale=1、freeze_track_steps=0、freeze_track_normalization=false。',
'Roll主要成功=region_retention_v5（结束时区域保持且排除失败）；曾稳定持球另列（0.3秒、相对速度<=0.15m/s，不要求结束时仍稳定）。Bounce主要成功=连续0.3秒真实多区域接触受控保持。两任务不能直接横向比较。','',
'|任务|模型|原始球|网球|棒球|三球合计|',
'|---|---|---:|---:|---:|---:|']
for task in ('roll','bounce'):
 for variant in ('best','last'):
  rs=[r for r in results if r['model']==task+'_'+variant]
  lines.append(f"|{task}|{variant}|"+'|'.join(str(r['successes'])+'%' for r in rs)+f"|{sum(r['successes'] for r in rs)}/300|")
lines+=['','## 稳定性及失败诊断','']
for r in results:
 lines.append(f"- {r['model']} {r['ball']}：触手{r['hand_contact_episodes']}/100；失败且曾触手{r['failed_after_contact']}；终止原因{r['reasons']}。")
 if r['task']=='roll':lines.append(f"  曾稳定持球{r['stable_hold_episodes']}%，其中主要成功且曾稳定{r['stable_and_success']}%；失败末态区域{r['failure_region']}。")
 else:lines.append(f"  失败中曾满足短暂有效保持{r['failed_some_valid_hold']}局。")
lines+=['','## 模型选择','',
'best_reward仅按训练奖励选择，不能保证真实抓取更好；last只是最后保存。按三球独立成功率、稳定性和最弱球种共同选择。小差异存在抽样误差，95% Wilson区间见summary_all.json。本次用于模型选择的种子集并非另留的最终独立验收集。',
'训练期间success为探索策略及滑动窗口，不能与这里确定性独立回合直接比较。Roll累计日志也不等于当前模型表现。',
'完整配置、物理参数、模型/源码哈希、逐局数据均在各模型/球子目录。未修改训练奖励和控制。','',
'## Viewer','',
'```bash','conda activate dcmm','cd /home/isee/Catch/catch_it_copy',
'bash scripts/view_roll_bounce_20261103.sh roll original best',
'bash scripts/view_roll_bounce_20261103.sh roll original last',
'bash scripts/view_roll_bounce_20261103.sh bounce original best',
'bash scripts/view_roll_bounce_20261103.sh bounce original last','```',
'original可替换为tennis_ball/baseball。Bounce先预计算10局，再1倍速回放，不是在线交互控制。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
status=json.loads((ROOT/'run_status.json').read_text());assert len(status)==12 and all(x['exit']==0 for x in status)
(ROOT/'COMPLETE').write_text('1200 episodes, 12 conditions, seeds 0-99, all exits 0; paired initial states identical.\n')
print('\n'.join(lines[:14]))
diagnosis=json.loads((ROOT/'roll_stability_diagnosis.json').read_text())
with (ROOT/'REPORT.md').open('a') as f:
 f.write('\n## Roll稳定性补查\n\n')
 f.write('best原始球种子0–9完整物理采样；增加只读记录后，逐回合reward/final/success与主测试完全一致。\n\n')
 f.write(json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n')
 f.write('\n在离桌离地且位于手区、真实触手的2664个采样中，2119个相对速度大于0.15m/s（约79.5%），10局中观察到72次低速稳定区间因相对速度超限而中断；最长稳定区间0.046秒。该小样本支持接触后速度控制是问题，不证明所有球或所有回合均为同一原因。\n')
 f.write('\n核对Roll环境DcmmVecEnv_roll_645edc4.py、physical_v1、配置DcmmCfg_roll_645edc4.py、评价roll_evaluation.py与11-02已测版本哈希一致，当前稳定性差异不是这些代码变化造成。\n')

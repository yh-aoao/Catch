import json,csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'analysis_artifacts/catch_three_ball_20261102'
rows=[]
for task in ('roll','bounce'):
 for ball in ('original','tennis_ball','baseball'):
  p=ROOT/(task+'_'+ball)
  assert (p/'COMPLETE').exists(),p
  data=[json.loads(s) for s in (p/'episodes.jsonl').read_text().splitlines()]
  assert len(data)==100 and [r['seed'] for r in data]==list(range(100))
  s=json.loads((p/'summary.json').read_text())
  result=dict(s, task=task, ball=ball)
  failures=[r for r in data if not r['success']]
  result['failed_after_hand_contact']=sum(r['hand_contact'] for r in failures)
  result['no_hand_contact']=sum(not r['hand_contact'] for r in data)
  if task=='roll':
   result['final_hold_episodes']=sum(bool(r['roll_final_hold']) for r in data)
   result['stable_hold_successes']=sum(r['success'] and bool(r['roll_stable_hold']) for r in data)
   result['failed_region_reasons']={}
   for r in failures:
    state=r['roll_evaluation'] or {}
    reason='not_clear' if not state.get('clear') else state.get('region_unmet') or 'other'
    result['failed_region_reasons'][reason]=result['failed_region_reasons'].get(reason,0)+1
  else:
   result['failure_max_valid_hold_seconds']=[r['bounce_diagnostics']['bounce_grasp_max_hold_seconds'] for r in failures]
   result['failures_with_some_valid_hold']=sum(v>0 for v in result['failure_max_valid_hold_seconds'])
   result['success_terminal_relative_speed_median']=statistics.median(r['bounce_diagnostics']['bounce_grasp_state']['relative_speed'] for r in data if r['success'])
   result['success_thumb_contact']=sum(r['success'] and r['bounce_diagnostics']['bounce_grasp_state']['thumb_contact'] for r in data)
   result['terminal_step_joint_violation_max']=max(r['bounce_diagnostics']['bounce_grasp_joint_violation'] for r in data)
  rows.append(result)
(ROOT/'summary_all.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
with (ROOT/'summary.csv').open('w') as f:
 fields=['task','ball','episodes','successes','success_rate','hand_contact_episodes','stable_hold_episodes','failed_after_hand_contact','no_hand_contact']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
lines=['# 2026-11-02 Roll / Bounce Catch三球测试','',
'每个条件100回合，种子0–99，原PPO确定性推理，严格加载actor及归一化参数。仅评估，不修改训练环境。',
'配置缺失：11-2只有权重；按此前equal-v2命令重建配置。若服务器自行覆盖过发射、速度、控制或奖励选项，需要提供原resolved_config.yaml复核。',
'Roll：region_retention_v5，最终区域保持并排除失败；稳定持球是另列的曾达到低相对速度0.3秒指标。Bounce：controlled_hold_v1，连续0.3秒多区域真实接触受控保持；成功立即结束，不证明更长时间保持。两任务成功率不直接横向比较。','',
'|任务|球|成功/100|触手/100|曾稳定保持/100（Roll）|失败但曾触手|',
'|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['task']}|{r['ball']}|{r['successes']}|{r['hand_contact_episodes']}|{r['stable_hold_episodes'] if r['task']=='roll' else '—'}|{r['failed_after_hand_contact']}|")
lines+=['','## 诊断','']
for r in rows:
 lines.append(f"- {r['task']} {r['ball']}：终止原因 {r['reasons']}；未触手{r['no_hand_contact']}局；失败但曾触手{r['failed_after_hand_contact']}局。")
 if r['task']=='roll':lines.append(f"  最终区域保持{r['final_hold_episodes']}局，其中正式成功且曾稳定保持{r['stable_hold_successes']}局。失败末态诊断{r['failed_region_reasons']}。")
 else:lines.append(f"  失败中曾满足过有效保持{r['failures_with_some_valid_hold']}局；成功末态相对速度中位数{r['success_terminal_relative_speed_median']:.4f}m/s；成功末态拇指参与{r['success_thumb_contact']}局。末控制步最大关节越限{r['terminal_step_joint_violation_max']:.6f}rad，仅末步统计，不代表全程峰值。")
lines+=['','## 解读与限制','',
'Roll主要口径不要求低速度或最短保持时间，不能将区域保持成功等同于稳定抓握。优先检查接触后速度匹配、包球与保持，再考虑改奖励。',
'Bounce失败可区分完全未触手与触手后无法持续保持。当前成功终止在0.3秒；长期保持及搬运鲁棒性未测。没有hand_joint_limit终止也不能单独证明所有手指姿态正常。',
'每种球100回合，比例存在抽样误差，95% Wilson区间见summary_all.json。训练日志含随机探索及不同统计窗口，不能当作独立测试结果。',
'完整逐回合数据、物理参数、模型/源码哈希见各子目录provenance.json、episodes.jsonl、source_hashes.json。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
probes=json.loads((ROOT/'hand_probe_summary.json').read_text())
comparison=json.loads((ROOT/'roll_comparison_to_252.json').read_text())
with (ROOT/'REPORT.md').open('a') as file:
 file.write('\n## 手部轨迹补查\n\n每种任务/球另查种子0–9共60局，逐回合success/reward/final与主测试完全一致，确认新增记录未改变轨迹。\n\n')
 for r in probes:
  file.write(f"- {r['condition']}：全程最大实际关节越限{r['max_actual_joint_violation']:.4f}rad；普通指尖相对掌面最小高度{r['min_ordinary_tip_height']:.4f}m。\n")
  if 'contact_failure_fraction_overlapping' in r:file.write(f"  接触采样中不满足项（可重叠，含正常初始碰撞阶段，不等同于失败原因占比）：{r['contact_failure_fraction_overlapping']}。\n")
 file.write('\nBounce补查无旧视频数弧度严重越限，但有约1.2–1.4cm指尖短时低于局部掌面；不能据此称手指姿态问题完全消失。Roll约0.20–0.23rad实际越限值得改善控制。\n')
 file.write('\n## 与历史Roll252同种子对照\n\n')
 for r in comparison:file.write(f"- {r['ball']}：旧252为{r['old252_successes']}%，新250为{r['new250_successes']}%；100/100初始状态完全一致。改善{r['improved_cases']}局、退化{r['regressed_cases']}局。仅区域保持指标比较，非稳定抓握。\n")
 file.write('\n## 后续建议\n\nRoll优先改善触球后的速度匹配、多指包球和稳定保持，并验证目标限幅/PID约束；保留当前版本做对照。不要仅提高接触或靠近奖励，也不要只提高区域保持数字。Bounce先保留当前约束与成功阈值，训练/评估更新的last与best_success候选；关注手掌接近速度及普通手指/拇指共同包球。若另行修改奖励，可考虑接触阶段速度匹配和持续有效包球，避免强制三指同角。\n')

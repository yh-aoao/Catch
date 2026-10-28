import json
from pathlib import Path
root=Path('/home/isee/Catch/analysis_artifacts/bounce_track_evaluation_20261028_v1')
catch=Path('/home/isee/Catch/analysis_artifacts/trained_ball_evaluation_20261028_v1')
lines=['# Bounce Track 与 Catch 对照（2026-10-28）','','每种球每个任务 100 局，种子 0–99，确定性推理；当前本地代码、保存配置的物理条件、固定球类。', '', 'Track：bounce_track_best_reward_19.31.pth；Catch：bounce_catch_best_reward_288.46.pth。','','| 球类 | Track 接触成功 | Catch 当前成功判定 | Catch 曾接触手 | Catch ball_left |','|---|---:|---:|---:|---:|']
all_rows={}
for name,label in [('original','原始球'),('tennis_ball','网球'),('baseball','棒球')]:
 d=root/name
 assert (d/'COMPLETE').exists()
 rows=[json.loads(x) for x in (d/'episodes.jsonl').read_text().splitlines()]
 assert len(rows)==100 and sorted(x['seed'] for x in rows)==list(range(100))
 s=json.loads((d/'summary.json').read_text());c=json.loads((catch/('bounce_catch_'+name)/'summary.json').read_text())
 assert s['successes']==sum(x['task_success'] for x in rows)
 all_rows[name]={'track':s,'catch':c}
 lines.append(f"| {label} | {s['successes']}% | {c['successes']}% | {c['hand_contact_episodes']}% | {c['reasons'].get('ball_left',0)}/100 |")
lines += ['', 'Track 成功仅要求手部接触且未失败，接触即结束；Catch 要到达 2.5 秒评估时限、未失败、处于 grasping 阶段且末端距球小于 0.25 米。这不是严格稳定抓握判定。两种成功率不能视为同一指标。', '', 'Catch 中三种球各有 99/100 局最终处于 grasping 阶段；ball_left 失败中，原始球 69/71、网球 58/60、棒球 73/90 局曾发生手部接触。主要可见瓶颈为接近/接触后的持续控制，但这些统计不能单独证明具体成因（闭手时机、相对速度、接触力等需额外诊断）。', '', '第二阶段优化整个模型，跟踪分支未冻结；已比较两份权重，actor_mlp_t、mu_t 均发生变化。独立 Track 的成功率不是 Catch 内跟踪分支的独立成功率。', '', 'Track 与 Catch 使用不同历史环境类；共享环境文件与训练保存的源代码哈希存在差异。本结果代表当前本地环境，不能证明完全复现训练机历史环境。', '', '本轮仅扩展评估工具支持 bounce_track，未修改奖励、物理、训练模型。']
(root/'summary_all.json').write_text(json.dumps(all_rows,indent=2))
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

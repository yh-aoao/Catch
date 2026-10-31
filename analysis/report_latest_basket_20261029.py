"""Validate and report the completed fixed-target Basket evaluation."""
from pathlib import Path
import json
import shutil
import statistics
import hashlib

p=Path('/home/isee/Catch/analysis_artifacts/latest_three_task_evaluation_20261029_v1/basket')
assert (p/'COMPLETE').exists()
rows=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()]
s=json.loads((p/'summary.json').read_text())
meta=json.loads((p/'provenance.json').read_text())
meta['rng_protocol']='Before every episode: np.random.seed(seed), random.seed(seed), torch.manual_seed(seed), env.reset(seed=seed); seeds 0 through 99 inclusive'
(p/'provenance.json').write_text(json.dumps(meta,indent=2))
assert len(rows)==100 and sorted(r['seed'] for r in rows)==list(range(100))
assert sum(r['success'] for r in rows)==s['successes']
assert all(r['max_ball_force']==0 for r in rows)
assert all(r['success']==(r['reason']=='basket_score') for r in rows)
forward=[r['simple_state'].get('carry',{}).get('forward',0.) for r in rows]
speed=[r['simple_state'].get('carry',{}).get('proximal_speed',0.) for r in rows]
frontier=[r['simple_state'].get('carry',{}).get('frontier',0.) for r in rows]
d=dict(carry_last_forward_m=dict(min=min(forward),median=statistics.median(forward),max=max(forward),mean=statistics.mean(forward)),
 carry_ever_forward_6cm_episodes=sum(x>=1.-1e-9 for x in frontier),
 carry_last_proximal_speed_ge_0p3_episodes=sum(x>=.3 for x in speed),
 policy_hand_contact_seconds_mean=statistics.mean(r['contact_total_seconds'] for r in rows),
 physical_preparation_seconds=dict(min=min(r['initial']['preparation']['seconds'] for r in rows),max=max(r['initial']['preparation']['seconds'] for r in rows)),
 failed_seeds=[r['seed'] for r in rows if not r['success']])
(p/'diagnostics_summary.json').write_text(json.dumps(d,indent=2))
source=Path('/home/isee/Catch/analysis/evaluate_basket_model_20261029.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['evaluator_sha256']
shutil.copy2(source,p/'EVALUATOR_SNAPSHOT.py')
shutil.copy2('/tmp/basket_latest_eval_20261029.log',p/'evaluation.log')
lo,hi=s['wilson_95_interval']
report=f'''# Basket 指定模型评估（2026-10-29）

模型：`{meta['checkpoint']}`。

种子 0–99，各局同时设置全局 NumPy、Python、Torch 和环境 Gym seed；CPU、确定性均值动作、原 PPO 归一化/动作缩放，严格加载全部权重，100 局全部正常结束。

| 指标 | 结果 |
|---|---:|
| 实际 basket_score / 主成功率 | {s['successes']}/100 = {100*s['success_rate']:.0f}% |
| Wilson 95%置信区间 | {100*lo:.2f}–{100*hi:.2f}% |
| 几何命中篮圈 / 安全入篮 | {s['basket_hits']} / {s['safe_basket_successes']} |
| 已确认真实出手 / 安全出手 | {s['confirmed_launches']} / {s['safe_launches']} |
| 危险机器人碰撞 | {s['dangerous_robot_collision_episodes']} |
| 平均策略回合仿真时间 | {s['mean_policy_sim_seconds']:.3f} s |
| 平均奖励 | {s['mean_reward']:.3f} |

终止原因：`{s['reasons']}`。

训练协议是 `basket_aim_v1 + fixed_base + fixed_physical + aim_carry_v1`。实际目标固定在 `{meta['basket_center']}` m，篮圈半径 `{meta['basket_radius']}` m、球半径 `{meta['ball_radius']}` m、倾角 `{meta['basket_tilt_deg']}` 度；这是固定近目标投篮评估，不是随机远目标或移动底盘泛化评估。

成功需经过真实手部接触和确认自由飞行，球穿过扣除球半径后的有效倾斜篮圈开口，再观察0.3秒后无危险机器人碰撞。没有调用任何eval-only pose/hand/probe覆盖，固定手指参数来自训练本身。reset采用训练时的无外力物理稳定准备（{d['physical_preparation_seconds']['min']:.3f}–{d['physical_preparation_seconds']['max']:.3f}s），策略期间检查球actuator控制始终为0。

Carry前移6cm属于奖励目标，不是这版basket_score硬门槛。最后带球前移量中位数约 `{d['carry_last_forward_m']['median']:.4f}` m，最大约 `{d['carry_last_forward_m']['max']:.4f}` m；达到过6cm奖励目标的 `{d['carry_ever_forward_6cm_episodes']}/100` 局。近端手臂最后朝前速度≥.3m/s为 `{d['carry_last_proximal_speed_ge_0p3_episodes']}/100`。高投篮率不代表已学会明显的长距离带球前移。策略阶段手接触平均 `{d['policy_hand_contact_seconds_mean']:.4f}`s，不包括reset物理准备。

该训练目录未保存源码hash provenance，不能证明当前本地所有源文件与训练机历史文件逐字相同；模型的协议/动作布局/carry参数已与当前入口严格匹配，所有当前源码、模型和配置哈希见provenance.json。逐局记录见episodes.jsonl，复现脚本快照见EVALUATOR_SNAPSHOT.py。

正式统计仅来自本目录basket；相邻basket_stream_seeded_precheck为种子核查前的连续随机流预检查，basket_startup_failed为序列化启动失败且未产生评估局，不计入最终结果。
'''
(p/'REPORT.md').write_text(report)
print(json.dumps(dict(summary=s,diagnostics=d),indent=2))

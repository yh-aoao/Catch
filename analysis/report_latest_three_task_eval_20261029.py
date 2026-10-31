"""Validate all 700 episodes and write the latest three-model evaluation report."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/home/isee/Catch')
OUT=ROOT/'analysis_artifacts/latest_three_task_evaluation_20261029_v1'
OLD=ROOT/'analysis_artifacts/trained_ball_evaluation_20261028_v1'
NAMES=[task+'_'+ball for task in ['roll_track','bounce_catch']
       for ball in ['original','tennis_ball','baseball']]+['basket']
summary={}
paired={}
for name in NAMES:
    path=OUT/name
    assert (path/'COMPLETE').exists(),name
    rows=[json.loads(line) for line in (path/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==100 and sorted(r['seed'] for r in rows)==list(range(100)),name
    assert len({r['seed'] for r in rows})==100
    s=json.loads((path/'summary.json').read_text())
    key='success' if name=='basket' else 'task_success'
    assert sum(bool(r[key]) for r in rows)==s['successes'],name
    assert s['episodes']==100 and s['success_rate']==s['successes']/100
    summary[name]=s
    if name!='basket':
        previous=[json.loads(line) for line in (OLD/name/'episodes.jsonl').read_text().splitlines()]
        old={r['seed']:r for r in previous}
        paired[name]=dict(previous_successes=sum(r['task_success'] for r in previous),
            current_successes=s['successes'],
            initial_states_exactly_matching=sum(r['initial']==old[r['seed']]['initial'] for r in rows),
            newly_successful=sum(r['task_success'] and not old[r['seed']]['task_success'] for r in rows),
            newly_failed=sum(not r['task_success'] and old[r['seed']]['task_success'] for r in rows))

(OUT/'summary_all.json').write_text(json.dumps(summary,indent=2))
(OUT/'previous_model_comparison.json').write_text(json.dumps(paired,indent=2))
with (OUT/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['case','episodes','successes','success_rate','ci95_low','ci95_high','raw_environment_successes','hand_contact_episodes','continuous_hand_contact_0p3s','success_with_contact_0p3s'])
    writer.writeheader()
    for name,s in summary.items():
        writer.writerow(dict(case=name,episodes=100,successes=s['successes'],success_rate=s['success_rate'],
            ci95_low=s['wilson_95_interval'][0],ci95_high=s['wilson_95_interval'][1],
            raw_environment_successes=s.get('environment_successes',s['successes']),
            hand_contact_episodes=s.get('hand_contact_episodes'),continuous_hand_contact_0p3s=s.get('continuous_hand_contact_0p3s'),
            success_with_contact_0p3s=s.get('successes_with_continuous_contact_0p3s')))

lines=['# 最新 Bounce、Roll、Basket 模型评估（2026-10-29）','','指定模型每个条件100局、独立种子0–99，共700局。CPU、无viewer，确定性均值动作、保存的归一化、原始动作适配，模型权重严格加载。Bounce/Roll固定所选球的标称物理参数，无训练mixed模式的额外质量/半径抖动。','','## 主要结果','','| 任务 | 球类/设置 | 成功/100 | 95% Wilson区间 |','|---|---|---:|---:|']
labels={'original':'原始球','tennis_ball':'网球','baseball':'棒球'}
for task,label in [('roll_track','Roll Track 离桌拦截'),('bounce_catch','Bounce Catch 旧版proximity')]:
    for ball,ball_label in labels.items():
        s=summary[task+'_'+ball];lo,hi=s['wilson_95_interval']
        lines.append('| {} | {} | {} | {:.1f}%–{:.1f}% |'.format(label,ball_label,s['successes'],100*lo,100*hi))
s=summary['basket'];lo,hi=s['wilson_95_interval']
lines.append('| Basket 投篮 | 固定近目标、训练球 | {} | {:.1f}%–{:.1f}% |'.format(s['successes'],100*lo,100*hi))
lines+=['','','## 指标边界','','Roll主指标为PPO使用的roll_eval_success：球完全离开桌子前缘、无桌面/地面支撑后，手部曾完成拦截。不要求稳定抓住，且这个事件一旦发生后续失败也可能保留为真。原始info.success是另一套判定，三球分别为64/100、54/100、65/100；不是主指标。Track接触后常结束，不能用接触时长为0认定它的Catch能力。','','Bounce模型464.31的保存配置没有controlled_hold_v1，实际评估legacy旧PPO/旧BounceEnv。主指标为bounce_catch_proximity_v2：到达2.5秒时限、未失败、处于grasping且末端距球<0.25m；不是严格受控离地保持。','','| Bounce球类 | 曾接触手 | 任意连续手接触≥0.3s | 主成功且连续接触≥0.3s | ball_left失败 |','|---|---:|---:|---:|---:|']
for ball,label in labels.items():
    s=summary['bounce_catch_'+ball]
    lines.append('| {} | {} | {} | {} | {} |'.format(label,s['hand_contact_episodes'],s['continuous_hand_contact_0p3s'],s['successes_with_continuous_contact_0p3s'],s['reasons'].get('ball_left',0)))
lines+=['','','辅助接触指标来自2ms控制tick，不验证法向力、离地、多指受控或关节正常，也不是每125us物理子步连续测量；不可当实机稳定抓握率。','','Basket主结果为真实自由飞行穿过篮圈、再经过0.3秒安全跟随的basket_score。篮圈中心[0,1.05,0.55]m、半径0.20m、倾角25°，使用训练时固定底盘/物理准备/固定手指设置，没有额外评估辅助投球或粘球。100次确认出手，97次真实命中与安全入篮，危险机器人碰撞0；失败为not_a_throw1、ball_on_floor2。此结果不代表随机远目标泛化。','','Carry前移6cm仅是奖励目标，并非该版成功硬门槛。100局达到过6cm的次数为0；最后前移中位约1.29cm、最大约3.21cm。因此97%不代表完整Carry动作达标。详见[basket报告](basket/REPORT.md)。','','## 与上次指定模型对照','','两次Bounce/Roll保存的配置和物理provenance逐字一致，当前物理/环境路径与训练历史一致，推理loop相同。各球100局的初始状态逐局比较见previous_model_comparison.json。下表为同条件描述性对照，不对小幅差异作显著性承诺。','','| 任务/球类 | 上次模型 | 本次模型 | 变化百分点 |','|---|---:|---:|---:|']
for task,label in [('roll_track','Roll'),('bounce_catch','Bounce')]:
    for ball,ball_label in labels.items():
        p=paired[task+'_'+ball];delta=p['current_successes']-p['previous_successes']
        lines.append('| {} {} | {}% | {}% | {:+d} |'.format(label,ball_label,p['previous_successes'],p['current_successes'],delta))
lines+=['','','上次为Roll23.86、Bounce288.46，本次为Roll27.44、Bounce464.31。两项均提升，但Bounce主指标仍仅约四成，持续控制依然需要改善。','','## 来源与复现','','[source_audit.json](source_audit.json)记录训练版本核对：BounceEnv、RollTrackingEnv及相关PPO/配置/机器人/物理组件与训练历史匹配，Bounce工厂变化仅为新版opt-in接线，默认legacy行为保留。Basket训练目录没有源码指纹，代表当前本地对应保存配置的评估，无法证明全部历史依赖逐字相同。','','每个正式子目录保存provenance.json、episodes.jsonl、summary.json、COMPLETE。Basket首轮连续随机流预检96%保存在basket_stream_seeded_precheck，已排除；正式basket为逐局同时重设NumPy/Python/Torch/Gym种子的97%。basket_startup_failed也不计入700局。','','使用的模型：']
for name in ['roll_track_original','bounce_catch_original','basket']:
    p=json.loads((OUT/name/'provenance.json').read_text())
    lines.append('- `{}`，SHA256 `{}`。'.format(p['checkpoint'],p['checkpoint_sha256']))
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
script=ROOT/'catch_it_copy/ycb_objects/evaluate_trained_balls.py'
shutil.copy2(script,OUT/'BALL_EVALUATOR_SNAPSHOT.py')
shutil.copy2(ROOT/'catch_it_copy/ycb_objects/generalization_probe.py',OUT/'CONTACT_RECORDER_SNAPSHOT.py')
for name in NAMES[:-1]:
    log=Path('/tmp')/('latest_eval_'+name+'_20261029.log')
    if log.exists():shutil.copy2(log,OUT/name/'run.log')
(OUT/'COMPLETE').write_text('700 canonical episodes verified\n')
print(json.dumps({k:v['successes'] for k,v in summary.items()}))
print('initial states paired',json.dumps({k:v['initial_states_exactly_matching'] for k,v in paired.items()}))

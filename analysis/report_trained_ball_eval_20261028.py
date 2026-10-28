"""Summarize completed fixed-object tests without changing their criteria."""
import csv
import json
from pathlib import Path

ROOT=Path('/home/isee/Catch/analysis_artifacts/trained_ball_evaluation_20261028_v1')
LABELS={'original':'原始参考球','tennis_ball':'网球','baseball':'棒球'}


def main():
    records=[]
    for task in ('roll_track','bounce_catch'):
        for obj in LABELS:
            folder=ROOT/(task+'_'+obj)
            if not (folder/'COMPLETE').exists():
                raise SystemExit('Evaluation still running: '+str(folder))
            data=json.loads((folder/'summary.json').read_text())
            assert data['episodes']==100
            episodes=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
            assert len(episodes)==100 and {r['seed'] for r in episodes}==set(range(100))
            assert sum(r['task_success'] for r in episodes)==data['successes']
            provenance=json.loads((folder/'provenance.json').read_text())
            records.append(dict(data,task=task,object=obj,
                checkpoint=provenance['checkpoint'],checkpoint_sha256=provenance['checkpoint_sha256'],
                criterion=provenance['criterion'],provenance=str(folder/'provenance.json')))
    report=dict(episodes_per_condition=100,total_episodes=600,seed_start=0,seed_end=99,
        inference='Deterministic policy mean with saved observation normalization; clamped/padded/scaled exactly as PPO test',
        material_hash_matches_training=dict(roll_track=True,bounce_catch=True),results=records,
        actual_factory_overrides=dict(ball_mode='fixed',sampling_weights=None,viewer=False,render_mode=None,
                                      episode_seconds=2.5,policy_seconds=.04),
        limitations=['Roll Track measures interception, not holding. Interception can precede a later failure.',
                     'Bounce existing success criterion is terminal proximity/stage/horizon, not verified grasp.',
                     'Contact duration is sampled at 2ms controller ticks. >=0.3s is supplementary, not proof of stable grasp.',
                     '100 seeds per fixed nominal ball; not exhaustive unseen-material or real-robot testing.',
                     'evaluation_config in each provenance adapts the saved PPO configuration; fixed ball/object overrides are applied directly in the factory.',
                     'Physical factory/material hashes match training records. Shared DcmmVecEnv.py whole-file hash differs; exact historical shared-module replay is not established. These results evaluate current local code.'])
    comparison=ROOT/'training_source_comparison.json'
    if comparison.exists():
        report['training_shared_source_comparison']=json.loads(comparison.read_text())
    (ROOT/'summary_all.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    columns=['task','object','episodes','successes','success_rate','environment_successes',
             'hand_contact_episodes','continuous_hand_contact_0p3s','successes_with_continuous_contact_0p3s']
    with (ROOT/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    lines=['# 新模型逐球评估（2026-10-28）','',
        '每个条件 100 回合，共 600 回合，seed=0–99。使用新模型 best checkpoint，CPU 无 Viewer，2.5 s 回合上限、40 ms 策略周期。',
        '保留训练物理/材质/发射配置，Roll 初速 0.8–1.3 m/s；固定测试每种球，不按训练比例混抽。两份材质文件 SHA256 与训练记录一致。','',
        '|任务|球类|现有主判据成功|95% Wilson 区间|环境原始 success 标志|手接触回合|连续手接触≥0.3s|主判据成功且连续接触≥0.3s|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        ci=r['wilson_95_interval']
        lines.append(f"|{r['task']}|{LABELS[r['object']]}|{r['successes']}/100 ({100*r['success_rate']:.0f}%)|{100*ci[0]:.1f}–{100*ci[1]:.1f}%|{r['environment_successes']}/100|{r['hand_contact_episodes']}/100|{r['continuous_hand_contact_0p3s']}/100|{r['successes_with_continuous_contact_0p3s']}/100|")
    lines+=['','## 判据和限制','',
        '- Roll Track 的主判据是 roll_eval_success：球完整越过桌前沿、无桌/地支撑时，曾被手接触拦截。不是完整抓取，也不要求之后继续持稳。',
        '- Roll 环境原始 success 更侧重掌心触球且无失败，与主判据不相同；两列同时保留。',
        '- Bounce 主判据是 bounce_catch_proximity_v2：到达回合时限、无失败终止、grasping 阶段、ee_distance<0.25 m。不能称为严格持球成功率。',
        '- 持续接触按 2 ms 控制子步检测，≥0.3 s 只是补充证据，并非力闭合/末端稳持判据。Roll Track 触球即可能结束，持续接触不适合评价其策略。',
        '- 权重严格加载，未允许尺寸不匹配的部分加载；调用原 PPO 的 obs2tensor/model_act(inference=True)/action2dict，未训练或更改权重。',
        '- 每个子目录保存逐回合 JSONL、模型和配置/材质 SHA256、实际物理参数、终止原因、首末状态和源码 SHA256。','',
        '物理工厂与材质文件 SHA256 与训练记录一致；共享 DcmmVecEnv.py 的整文件 SHA256 不同，未找到对应历史源码备份，故这些是当前本地代码下的评估，不声称逐字复现训练机历史共享模块。',
        '原训练配置中的 mixed 字段保留在 PPO 配置快照中；实际工厂明确按当前球固定测试，不混合采样。','',
        '## 终止原因','']
    for r in records:
        lines.append(f"- {r['task']} / {LABELS[r['object']]}：`{json.dumps(r['reasons'],ensure_ascii=False)}`")
    lines+=['','## 模型','']
    for task in ('roll_track','bounce_catch'):
        r=next(r for r in records if r['task']==task)
        lines.extend([f"- {task}: `{r['checkpoint']}`",f"  SHA256: `{r['checkpoint_sha256']}`"])
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()

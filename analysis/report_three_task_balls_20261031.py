"""Summarize the fixed 900-episode, three-task/three-ball evaluation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis_artifacts/three_task_three_ball_success_20261031_v1'
TASKS = ('roll', 'bounce', 'basket')
BALLS = ('original', 'tennis_ball', 'baseball')
task_names = {'roll':'Roll Catch 252.54', 'bounce':'Bounce Catch 464.31', 'basket':'Basket 38.38'}
ball_names = {'original':'原始球', 'tennis_ball':'网球', 'baseball':'棒球'}
records = []
summaries = {}
for task in TASKS:
    for ball in BALLS:
        key = task+'_'+ball
        folder = OUT/'roll_metric_correction'/key if task == 'roll' else OUT/key
        assert (folder/'COMPLETE').is_file(), 'Incomplete: '+key
        episodes = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
        assert len(episodes) == 100 and [r['seed'] for r in episodes] == list(range(100))
        s = json.loads((folder/'summary.json').read_text())
        p = json.loads((folder/'provenance.json').read_text())
        assert s['successes'] == sum(r['success'] for r in episodes)
        assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest() == p['checkpoint_sha256']
        summaries[key] = s
        records.append(dict(task=task, ball=ball, model=task_names[task], episodes=s['episodes'],
            successes=s['successes'], success_rate=s['success_rate'],
            legacy_successes=s.get('legacy_successes', s['successes']),
            stable_hold_episodes=s.get('stable_hold_episodes') if task == 'roll' else None,
            ci95_low=s['wilson_95_interval'][0], ci95_high=s['wilson_95_interval'][1],
            hand_contact_episodes=s['hand_contact_episodes'],
            continuous_hand_contact_0p3s=s['contact_0p3s'],
            success_with_contact_0p3s=s['success_with_contact_0p3s'],
            preparation_failures=s['preparation_failures'], criterion=p['criterion']['name'],
            checkpoint=p['checkpoint'], checkpoint_sha256=p['checkpoint_sha256']))

with (OUT/'summary.csv').open('w') as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
(OUT/'summary_all.json').write_text(json.dumps(summaries, indent=2))

lines = ['# 三任务三球成功率（2026-10-31，更正Roll统计字段）', '',
    '**更正：此前Roll的0/100读取了旧字段info.success，不能代表当前Viewer成功率。现已按原PPO使用的roll_eval_success重跑三球各100局。只修正评估字段，没有修改环境、物理、策略、奖励或成功阈值。旧结果保存在superseded_legacy_roll_report中，原roll_*目录的0%仅作为legacy指标对照。**', '',
    '每个组合100局，逐局设置 NumPy/Python/Torch/Gym seed=0–99，共900局。使用真实模型和保存的归一化参数、原PPO确定性动作适配，关闭Viewer。初始化失败计入分母，不重抽、不丢弃。', '',
    '| 任务与模型 | 原始球 | 网球 | 棒球 |', '| --- | ---: | ---: | ---: |']
for task in TASKS:
    values = [f"{summaries[task+'_'+ball]['successes']}%（{summaries[task+'_'+ball]['successes']}/100）" for ball in BALLS]
    lines.append('| '+task_names[task]+' | '+' | '.join(values)+' |')

lines += ['', 'Roll三球的手接触回合数分别为'+
    '、'.join(str(summaries['roll_'+ball]['hand_contact_episodes'])+'/100' for ball in BALLS)+
    '。手接触、当前区域持球指标、旧低速指标和稳定持球诊断分别统计，不可互相替代。',
    'Basket网球与棒球的初始化失败数分别为'+
    str(summaries['basket_tennis_ball']['preparation_failures'])+'和'+
    str(summaries['basket_baseball']['preparation_failures'])+
    '，初始化成功与后续投篮成功分开统计。']

roll = json.loads((OUT/'roll_metric_correction/roll_original/provenance.json').read_text())['criterion']
lines += ['', '## 成功判据与测试条件', '',
    '- Roll：使用当前physical_v1、初速0.8–1.3m/s、legacy奖励、关闭桌前等待硬约束。主指标是roll_eval_success（region_retention_v5），与正式PPO的terminal_metrics相同：真实手接触启动持球状态；球已离桌、无桌地接触且回合结束仍在掌心附近区域，失败终止覆盖成功。掌心区域为水平半径12cm，高度相对掌心[-2cm, 球半径+12cm]。主指标不要求连续低速或固定保持时长，也不要求每个瞬间都持续接触。',
    f"- Roll旧指标info.success/roll_legacy_success：相对手速≤{roll['legacy_speed_limit']}m/s且连续{roll['legacy_consecutive_samples']}个策略步，仍为三球0/100。roll_stable_hold是另一项独立诊断：真实接触、近手、无桌地接触、速度≤0.15m/s累计0.30s（符合条件的短间隙可容许）。不能用旧指标或此诊断替代主成功率。",
    '- Bounce：physical_v2、legacy抓取、baseball_intercept_v1发射。成功为到达2.5s时限、未失败、处于grasping且末端距离<0.25m；该数值不等于严格稳定握持率。保留原0.125ms物理步长。',
    '- Basket：38.38保存配置、固定底盘、自然接触托球、物理自由飞行入篮与安全跟随。原始球和棒球初始手姿[0.8,0.7,0.7]；网球按给用户的测试命令改为[1.0,0.9,0.9]。网球这一列含初始手姿适配，不是完全相同初始条件下的泛化。',
    '- 三球固定各自尺寸/质量/惯量和接触参数。网球与棒球为解析碰撞球＋YCB外观。参数是项目中的近似先验。',
    '- 此前Roll的82%/81%/84%来自27.44的Tracking模型，衡量拦截，不能作为252.54 Catch的当前区域持球成功率。', '',
    '## 接触诊断（各100局）', '',
    '手接触和0.3s连续接触是辅助诊断，不代替成功判据。Basket托球准备阶段不计入下表接触时长。', '',
    '| 任务/球 | 至少手接触一次 | 连续手接触≥0.3s | 成功且有≥0.3s接触 | 准备失败 |',
    '| --- | ---: | ---: | ---: | ---: |']
for task in TASKS:
    for ball in BALLS:
        s = summaries[task+'_'+ball]
        lines.append(f"| {task_names[task]} / {ball_names[ball]} | {s['hand_contact_episodes']} | {s['contact_0p3s']} | {s['success_with_contact_0p3s']} | {s['preparation_failures']} |")
lines += ['', '## 终止原因', '']
for task in TASKS:
    for ball in BALLS:
        lines.append(f"- {task_names[task]} / {ball_names[ball]}：`{json.dumps(summaries[task+'_'+ball]['reasons'], ensure_ascii=False)}`")
lines += ['', '## 可复核文件', '',
    '- summary.csv：九个组合的成功数、成功率和95% Wilson区间。',
    '- Roll更正数据在roll_metric_correction/roll_*；Bounce/Basket沿用原对应目录。目录内有episodes.jsonl逐局记录、provenance.json模型哈希/配置/物理参数、source_hashes.json代码哈希、summary.json汇总。',
    '- run_status.json：本次启动命令、退出状态与耗时。smoke_*为独立入口检查，不计入900局。',
    '- 评估脚本：analysis/evaluate_three_task_balls_20261031.py。汇总脚本：analysis/report_three_task_balls_20261031.py。', '',
    '有限样本的0/100不代表真实成功概率严格为零；95% Wilson区间上界约3.7%。三个任务的成功定义不同，不宜直接横向排名。', '']
(OUT/'REPORT.md').write_text('\n'.join(lines))
(OUT/'COMPLETE').write_text('900 episodes verified\n')
print('\n'.join(lines[:12]))

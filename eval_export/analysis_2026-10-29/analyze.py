"""Validate the exported evaluations and regenerate summary/report/figure."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT.parent
TASKS = ('throw', 'roll', 'bounce')
NAMES = {
    'future': ('C1', '未来运动预测'),
    'belief': ('C2', '视觉与历史融合'),
    'spatial': ('C3', '空间定位'),
    'contact': ('C4', '接触辅助预测'),
    'camera_gate': ('C5', '位置辅助＋相机门控'),
}


def main():
    summaries = []
    contracts = {task: [] for task in TASKS}
    for path in sorted(SOURCE.glob('*_eval_*.json')):
        match = re.fullmatch(r'(.+)_eval_(\d+)', path.stem)
        model, iteration = match[1], int(match[2])
        data = json.loads(path.read_text())
        assert set(data) == set(TASKS)
        task_summary = {}
        for task in TASKS:
            item = data[task]
            records = item['records']
            assert item['complete'] and len(records) == item['episodes'] == item['requested_episodes'] == 300
            assert sum(bool(r['success']) for r in records) == item['successes']
            assert abs(item['success_rate'] - item['successes']/item['episodes']) < 1e-12
            assert dict(Counter(r['reason'] for r in records)) == item['termination_counts']
            contract = {key: item[key] for key in ('success_metric', 'success_definition',
                        'evaluation_revision', 'seed', 'stage', 'camera', 'policy_role')}
            contracts[task].append(contract)
            task_summary[task] = {key: item[key] for key in ('episodes', 'successes', 'success_rate',
                'wilson_95', 'success_metric', 'success_definition', 'termination_counts')}
        records = data['roll']['records']
        successful = [r for r in records if r['success']]
        roll = {key: sum(bool(r['diagnostics'].get(key, False)) for r in records)
                for key in ('roll_caught_once', 'roll_final_hold', 'roll_stable_hold', 'success')}
        roll.update(success_contact_at_end=sum(bool(r['diagnostics']['roll_eval']['contact']) for r in successful),
                    success_relative_speed_mean=statistics.mean(r['diagnostics']['roll_hold']['relative_speed'] for r in successful),
                    success_contact_fraction_mean=statistics.mean(r['diagnostics']['roll_eval']['contact_fraction'] for r in successful),
                    success_end_reasons=dict(Counter(r['reason'] for r in successful)),
                    legacy_truncated_rate=data['roll']['legacy_truncated_rate'])
        rates = [task_summary[t]['success_rate'] for t in TASKS]
        summaries.append(dict(model=model, code=NAMES[model][0], name=NAMES[model][1],
                              iteration_from_filename=iteration, source=path.name,
                              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tasks=task_summary,
                              macro_success=sum(rates)/3, minimum_success=min(rates),
                              task_spread=max(rates)-min(rates),
                              total_successes=sum(task_summary[t]['successes'] for t in TASKS),
                              total_episodes=sum(task_summary[t]['episodes'] for t in TASKS), roll_diagnostics=roll))
    assert len(summaries) == 5
    assert all(all(contract == values[0] for contract in values) for values in contracts.values())
    summaries.sort(key=lambda item: item['macro_success'], reverse=True)
    result = dict(date='2026-10-29', contract_consistent=True, contracts={t: v[0] for t,v in contracts.items()},
                  records_validated=4500, iteration_source='export filename; checkpoint metadata not supplied', models=summaries)
    (OUTPUT/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_report(summaries)
    plot(summaries)
    for row in summaries:
        print(row['model'], row['iteration_from_filename'],
              [round(100*row['tasks'][t]['success_rate'], 2) for t in TASKS],
              'mean', round(100*row['macro_success'], 2), 'min', round(100*row['minimum_success'], 2))


def write_report(rows):
    lines = ['# 五组辅助蒸馏成功率分析（2026-10-29）', '',
        '依据eval_export中的五份评估JSON，逐回合核验共4500条记录。原始文件未修改。', '',
        '## 评估条件与口径', '',
        '- 全部为Catch阶段visual_student，seed=123，每任务300回合且complete=true。',
        '- Top/Wrist双深度相机，均为160×160，深度范围及EGL后端字段一致。',
        '- Throw取success/throw_official；Roll取roll_eval_success/region_retention_v5；',
        '  Bounce取success/bounce_catch_proximity_v2。不能用Roll原success或truncated替代。',
        '- 四份文件名标注10000轮，Contact文件名标注6000轮，Contact属于预算不同的中间结果。',
        '- 导出文件不含checkpoint迭代/哈希、完整训练配置或resume_audit，无法独立确认best/last选择、',
        '  训练并行数、完整采样预算及恢复中接受的源码变动。下述排名是当前导出结果排名。', '',
        '## 主成功率', '',
        '| 模型 | 文件标注轮次 | Throw | Roll | Bounce | 三任务均值 | 最低任务 | 成功总数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [f"{100*row['tasks'][t]['success_rate']:.2f}%" for t in TASKS]
        lines.append(f"| {row['code']} {row['model']} | {row['iteration_from_filename']} | {' | '.join(values)} | "
                     f"{100*row['macro_success']:.2f}% | {100*row['minimum_success']:.2f}% | {row['total_successes']}/900 |")
    lines += ['', '各任务回合数相同，所以三任务均值也等于成功总数/900；未混合不同成功定义计算逐任务指标。', '',
        '## 结果解释', '',
        '1. Belief均值72.00%，Future为71.56%，只差4/900个成功（0.44个百分点），应视为接近。',
        '   Belief的Throw为80.33%、Roll为64.67%；Future的Bounce为76.33%，比Belief高5.33个百分点。',
        '2. Contact仅6000轮，但Roll达67.67%，是本组最高；三任务最低值67.33%，也是本组最高。',
        '   三任务极差只有2.67个百分点，属于最均衡的当前结果；不能据此保证续训到10000一定上升。',
        '3. Camera gate的Bounce达到83.67%，本组最高，但Throw61.67%、Roll55.67%，均值67.00%。',
        '   高Bounce表现与另外两任务的损失并存，不能将该模块称为全面提升。',
        '4. Spatial均值68.00%，本次没有超过Belief/Future，Roll57.33%是其主要短板。',
        '5. 除Contact外，四组最低任务均为Roll；多任务效果的主要短板仍在Roll。', '',
        '## 单任务95% Wilson区间', '',
        '以下直接使用原JSON中的区间，不将它们当作训练多seed稳定性估计。', '',
        '| 模型 | Throw | Roll | Bounce |', '|---|---:|---:|---:|']
    for row in rows:
        cells = [f"{100*row['tasks'][t]['wilson_95'][0]:.1f}～{100*row['tasks'][t]['wilson_95'][1]:.1f}%" for t in TASKS]
        lines.append(f"| {row['model']} | {' | '.join(cells)} |")
    lines += ['', '单任务只有300回合，若干个百分点的差距仍存在采样不确定性。',
        '原worker每回合显式重设seed，回合长度和步内随机数消耗本身不会造成后续reset采样序列漂移；此前将它们列为不能配对的直接原因需要更正。',
        '旧导出没有初始状态记录，且仍需核对源码和环境构造参数，不应仅按记录序号认定严格配对。新的1500回合多种子评估已记录并核验初始状态，详见[新报告](../../analysis_artifacts/distill_best_multiseed_20261029_v1/report/REPORT.md)。', '',
        '## Roll保持质量', '',
        '| 模型 | 曾进入区域保持状态 | 主成功 | 原环境success | 低速稳定保持roll_stable_hold |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        r = row['roll_diagnostics']
        lines.append(f"| {row['model']} | {r['roll_caught_once']}/300 | {row['tasks']['roll']['successes']}/300 | "
                     f"{r['success']}/300 | {r['roll_stable_hold']}/300 |")
    lines += ['',
        '- 五组roll_stable_hold均为0，原环境success也均为0；主成功标志不是这两个标志。',
        '- 当前区域保持口径允许已有接触启动保持、随后无持续接触但球仍在手附近；主成功不要求低瞬时相对速度。',
        '- 本地对应实现将低速稳定保持作为单独诊断，使用低相对速度、接触/短间断及持续时间条件。',
        '  JSON未附服务器源码哈希，因此正式报告前仍应核对实际源码及视频，不根据单个标志断言物理抓握质量。',
        '- 本组所有Roll主成功都以timeout结束，符合末端区域保持的统计方式；timeout本身不能算成功。',
        '- 例如Contact的truncated率是81.00%，主成功为67.67%，不能把前者报告成抓取成功率。',
        '- Camera gate曾进入保持状态271/300，主成功仅167/300；Spatial为276/300→172/300。',
        '  这些计数提示需要关注接触后的保持/丢失，而不仅是能否接近球。', '',
        '## 下一步', '',
        '1. 先将Contact续训到10000，再与同预算模型比较；最好另保留各模型6000轮结果作横向对照。',
        '2. 综合方向优先复验Belief与Future，均衡/最差任务方向优先复验Contact。',
        '3. 用同预算B6位置辅助基线、相同评价口径和多个训练seed验证模块收益；本目录没有B0/B6/教师对照。',
        '4. 单独评估Roll的区域保持、低速稳定保持及视频表现，保留原指标，不为得到更高成功率临时改阈值。', '',
        '## 原始来源', '']
    for row in rows:
        lines.append(f"- [{row['source']}](../{row['source']})，SHA256：`{row['source_sha256']}`。")
    lines += ['', '![各任务成功率与95%区间](success_rates.png)', '',
              '结构化汇总：`summary.json`；重新生成：在dcmm环境运行本目录`analyze.py`。', '']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
    labels = [f"{r['model'].replace('_', ' ').title()} ({r['iteration_from_filename']//1000}k)" for r in rows]
    y = np.arange(len(rows))
    for ax, task, color in zip(axes, TASKS, ['#3478a3', '#d68b28', '#409482']):
        values = np.array([100*r['tasks'][task]['success_rate'] for r in rows])
        lower = np.array([100*r['tasks'][task]['wilson_95'][0] for r in rows])
        upper = np.array([100*r['tasks'][task]['wilson_95'][1] for r in rows])
        ax.barh(y, values, height=.58, color=color, alpha=.88)
        ax.errorbar(values, y, xerr=[values-lower, upper-values], fmt='none',
                    ecolor='#23313b', capsize=3, linewidth=1)
        for i, value in enumerate(values):
            ax.text(upper[i]+1.4, i, f'{value:.1f}%', va='center', fontsize=10)
        ax.set_xlim(0, 101)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('Success rate (%)')
        ax.set_title(task.title(), fontsize=13, fontweight='bold')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dfe5e9', linewidth=.7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle('Five auxiliary distillation variants — Catch evaluation', fontsize=14, fontweight='bold', y=.98)
    fig.text(.5, .025, '300 episodes per task; seed 123; whiskers: 95% Wilson intervals. Contact: 6k iterations; others: 10k.',
             ha='center', fontsize=9, color='#4b5862')
    fig.tight_layout(rect=[0, .065, 1, .94])
    fig.savefig(OUTPUT/'success_rates.png', dpi=180)
    fig.savefig(OUTPUT/'success_rates.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()

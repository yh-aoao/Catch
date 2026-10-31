"""Compare Contact best/last under one shared environment seed (1800 episodes).

Requires all six canonical jobs to be complete. This is a conditional comparison
of two fixed checkpoints, not an estimate of variation across environment seeds.
No simulator, Torch or GPU is imported.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import report_distill_multiseed_20261029 as stats


ROOT = Path('/home/isee/Catch')
BASE = ROOT/'analysis_artifacts/contact_best_last_20261030_v1'
SEED = 20261029
EPISODES = 300
TASKS = ('throw', 'roll', 'bounce')
LABELS = ('best', 'last')
ITERATIONS = {'best': 6000, 'last': 10000}
CHECKPOINT_SHA = {
    'best': '381bb98d3605b0396f8cffa025900171da23cbed05d425bf3a0c1f239d42bec3',
    'last': 'b520de40e69fc4ef79ef522f5caa27f3dda9ffda912723ffc8344f48eaa2ce8b'}


def load_job(folder, label, task, audit_path, audit):
    if not (folder/'COMPLETE').exists():
        raise RuntimeError('Unfinished canonical job: '+str(folder))
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    expected = ('contact', label, task, ITERATIONS[label])
    actual = (protocol.get('variant'), protocol.get('checkpoint_label'),
              protocol.get('task'), protocol.get('iteration'))
    if actual != expected:
        raise RuntimeError('Wrong checkpoint, task or iteration: '+str(folder))
    if protocol['checkpoint_sha256'] != CHECKPOINT_SHA[label] or protocol['checkpoint_sha256'] != audit[label]['checkpoint_sha256']:
        raise RuntimeError('Wrong checkpoint SHA256: '+str(folder))
    if protocol['source_audit_sha256'] != stats.sha(audit_path):
        raise RuntimeError('Source audit changed: '+str(audit_path))
    if protocol['policy_batch_size'] != 1 or not protocol['deterministic_algorithms'] or protocol['tf32']:
        raise RuntimeError('Canonical single-sample inference settings changed: '+str(folder))
    if protocol['seeds'] != [SEED] or protocol['episodes_per_seed'] != EPISODES:
        raise RuntimeError('Expected one shared seed and 300 episodes per task: '+str(folder))
    rows_path = folder/f'seed_{SEED}.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len(rows) != EPISODES or [row['episode_id'] for row in rows] != list(range(EPISODES)):
        raise RuntimeError('Need 300 unique contiguous episode IDs: '+str(rows_path))
    task_id = TASKS.index(task)
    for row in rows:
        reset_seed = (SEED+1000000+task_id*10000+row['episode_id']*100003) % (2**32)
        if row['seed'] != SEED or row['reset_seed'] != reset_seed:
            raise RuntimeError('Wrong episode/reset seed: '+str(rows_path))
        metric = stats.METRICS[task][0]
        if type(row['success']) is not bool or metric not in row['diagnostics'] or bool(row['diagnostics'][metric]) != row['success']:
            raise RuntimeError('Success differs from original terminal metric: '+str(rows_path))
        if stats.metric_version(row) != stats.METRICS[task][1]:
            raise RuntimeError('Success metric version changed: '+str(rows_path))
        if not row['initial'].get('physical_hash') or not row['initial'].get('depth_sha256'):
            raise RuntimeError('Missing initial-state hashes: '+str(rows_path))
    summary_path = folder/'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['protocol'] != protocol or summary['episodes'] != EPISODES or summary['successes'] != sum(row['success'] for row in rows):
        raise RuntimeError('Evaluator summary differs from raw canonical episodes: '+str(folder))
    return dict(protocol=protocol, rows=rows, input_sha256={str(path): stats.sha(path)
                for path in (protocol_path, rows_path, summary_path)})


def pair_rows(best, last):
    """Describe last-best only when captured initial physical states match."""
    counts = Counter()
    visual_counts = Counter()
    differences, visual_differences, mismatches = [], [], []
    for left, right in zip(best['rows'], last['rows']):
        if (left['seed'], left['episode_id']) != (right['seed'], right['episode_id']):
            raise RuntimeError('Cannot pair different seed/episode IDs')
        episode = left['episode_id']
        if left['initial']['physical_hash'] != right['initial']['physical_hash']:
            mismatches.append(dict(episode_id=episode, kind='physical'))
            continue
        key = ('both_success' if left['success'] and right['success'] else
               'both_failure' if not left['success'] and not right['success'] else
               'last_only' if right['success'] else 'best_only')
        counts[key] += 1
        differences.append(int(right['success'])-int(left['success']))
        if left['initial']['depth_sha256'] == right['initial']['depth_sha256']:
            visual_counts[key] += 1
            visual_differences.append(differences[-1])
        else:
            mismatches.append(dict(episode_id=episode, kind='initial_depth'))
    keys = ('both_success', 'both_failure', 'best_only', 'last_only')
    physical_matches = len(differences)
    return dict(direction='last_minus_best', common_episode_ids=len(best['rows']),
        physical_matches=physical_matches, physical_mismatches=len(best['rows'])-physical_matches,
        initial_depth_matches=len(visual_differences),
        initial_depth_mismatches_with_matching_physics=physical_matches-len(visual_differences),
        last_minus_best=statistics.mean(differences) if differences else None,
        same_depth_last_minus_best=statistics.mean(visual_differences) if visual_differences else None,
        discordant_pairs=counts['best_only']+counts['last_only'],
        contingency={key: counts[key] for key in keys},
        same_depth_contingency={key: visual_counts[key] for key in keys},
        runtime_versions_equal=best['protocol']['versions'] == last['protocol']['versions'],
        runtime_versions={'best': best['protocol']['versions'], 'last': last['protocol']['versions']},
        mismatch_records=mismatches)


def preference(best, last):
    if abs(last-best) < 1e-12:
        return 'tie'
    return 'last' if last > best else 'best'


def aggregate(jobs, audit):
    data = dict(complete=True, episodes=1800, episodes_per_checkpoint=900,
                episodes_per_task=300, environment_seed=SEED, n_environment_seeds=1,
                comparison_direction='last_minus_best', models={}, comparisons={}, input_sha256={},
                notes=[
                    'One environment evaluation seed, not independent training repeats.',
                    'Wilson intervals condition on this sampled environment; no seed-level t interval is reported.',
                    'The original training best criterion is the minimum success rate over the three tasks.',
                    'Raw all-episode rate differences and matched-subset differences are separate quantities.',
                    'Roll region retention is distinct from stable holding.',
                    'The stopped five-seed evaluation and earlier pilots are excluded.'])
    for label in LABELS:
        tasks = {}
        for task in TASKS:
            job = jobs[label, task]
            row = stats.summarize_rows(job['rows'])
            row['protocol'] = job['protocol']
            row['environment_draw'] = dict(
                radii=sorted({episode['initial']['radius'] for episode in job['rows']}),
                masses=sorted({episode['initial']['object_mass'] for episode in job['rows']}))
            tasks[task] = row
            data['input_sha256'].update(job['input_sha256'])
        rates = {task: row['rate'] for task, row in tasks.items()}
        minimum = min(rates.values())
        item = audit[label]
        data['models'][label] = dict(checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256'],
            iteration=ITERATIONS[label], tasks=tasks,
            successes=sum(row['successes'] for row in tasks.values()), episodes=900,
            mean_task_success_rate=statistics.mean(rates.values()), minimum_task_success_rate=minimum,
            minimum_tasks=[task for task, rate in rates.items() if rate == minimum],
            source_manifest_files=len(item['files']),
            matching_source_manifest_files=sum(row['matching'] for row in item['files']),
            source_exceptions=item.get('exceptions', []),
            additional_historical_assets=len(item.get('additional_required_assets', [])))
    for task in TASKS:
        pairing = pair_rows(jobs['best', task], jobs['last', task])
        pairing['all_episode_rate_difference'] = data['models']['last']['tasks'][task]['rate']-data['models']['best']['tasks'][task]['rate']
        data['comparisons'][task] = pairing
    best, last = data['models']['best'], data['models']['last']
    paired_values = [data['comparisons'][task]['last_minus_best'] for task in TASKS]
    data['comparison_summary'] = dict(
        original_selection_metric='minimum_task_success_rate',
        minimum_task_preference=preference(best['minimum_task_success_rate'], last['minimum_task_success_rate']),
        minimum_task_rate_difference=last['minimum_task_success_rate']-best['minimum_task_success_rate'],
        mean_task_preference=preference(best['mean_task_success_rate'], last['mean_task_success_rate']),
        mean_task_rate_difference=last['mean_task_success_rate']-best['mean_task_success_rate'],
        matched_mean_task_difference=statistics.mean(paired_values) if all(value is not None for value in paired_values) else None,
        best_tasks_won=[task for task, row in data['comparisons'].items() if row['all_episode_rate_difference'] < 0],
        last_tasks_won=[task for task, row in data['comparisons'].items() if row['all_episode_rate_difference'] > 0],
        tied_tasks=[task for task, row in data['comparisons'].items() if row['all_episode_rate_difference'] == 0])
    return data


def describe_preference(value):
    return {'best': 'best（6000轮）较高', 'last': 'last（10000轮）较高', 'tie': '两者相同'}[value]


def make_report(data, output):
    models = data['models']
    comparison = data['comparison_summary']
    if models['best']['minimum_tasks'] == models['last']['minimum_tasks']:
        minimum_note = '本次两模型最低成功率对应的任务相同：'+', '.join(task.title() for task in models['best']['minimum_tasks'])+'。'
    else:
        minimum_note = '两模型的最差任务不同，该最低成功率差值不是同一任务的配对差。'
    lines = ['# Contact best 与 last 单种子评估（2026-10-30）', '',
        '**完整评估：1800回合。** 两模型分别在Throw、Roll、Bounce测试300回合；共用环境种子20261029。', '',
        f"`student_best.pth` 是Contact第6000轮，`student_last.pth` 是同结构Contact第10000轮。按原训练best选择口径“三任务最差成功率”比较，本次是 **{describe_preference(comparison['minimum_task_preference'])}**；按三任务平均比较，**{describe_preference(comparison['mean_task_preference'])}**。两种口径须分别看待。", '',
        '这是一个环境种子下两份具体权重的条件对照，不能据此判断跨环境种子、跨训练种子的稳定优劣。文件名best反映原训练评估的选择结果，并不保证在新的环境种子下始终胜出。', '',
        '## 三项任务成功率', '',
        '|任务|best 6000 成功/回合|best 成功率及条件95% Wilson区间|last 10000 成功/回合|last 成功率及条件95% Wilson区间|last−best（百分点）|',
        '|---|---:|---|---:|---|---:|']
    for task in TASKS:
        left, right = models['best']['tasks'][task], models['last']['tasks'][task]
        lines.append(f"|{task.title()}|{left['successes']}/{left['episodes']}|{stats.pct(left['rate'])} {stats.interval(left['conditional_wilson95'])}|{right['successes']}/{right['episodes']}|{stats.pct(right['rate'])} {stats.interval(right['conditional_wilson95'])}|{stats.pp(data['comparisons'][task]['all_episode_rate_difference'])}|")
    lines += ['', '|模型|三任务成功总数/900|三任务等权平均|最差任务成功率|最差任务|', '|---|---:|---:|---:|---|']
    for label in LABELS:
        row = models[label]
        lines.append(f"|Contact {label} {row['iteration']}轮|{row['successes']}/{row['episodes']}|{stats.pct(row['mean_task_success_rate'])}|{stats.pct(row['minimum_task_success_rate'])}|{', '.join(task.title() for task in row['minimum_tasks'])}|")
    lines += ['', f"三任务平均差值（last−best）为 **{stats.pp(comparison['mean_task_rate_difference'])}个百分点**；最差任务成功率差值为 **{stats.pp(comparison['minimum_task_rate_difference'])}个百分点**。{minimum_note}", '',
        '每项任务的Wilson区间仅近似反映固定当前采样环境参数后的回合抽样误差；同一构造种子下球的半径、质量等可能固定，不是300组独立物理参数。只有一个环境种子，本报告不计算环境种子层面的t区间，也不把900回合混合区间当作跨任务泛化区间。', '',
        '## 同一初始状态下的配对成败', '',
        '仅对种子、回合编号及捕获的初始物理哈希一致的回合比较。差值方向为last−best；last单独成功数多于best单独成功数时，匹配子集中的last成功率较高。', '',
        '|任务|物理匹配/共同编号|初始深度逐位一致|双方成功|双方失败|仅best成功|仅last成功|配对差值（百分点）|运行库版本一致|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for task in TASKS:
        row = data['comparisons'][task]
        counts = row['contingency']
        lines.append(f"|{task.title()}|{row['physical_matches']}/{row['common_episode_ids']}|{row['initial_depth_matches']}/{row['physical_matches']}|{counts['both_success']}|{counts['both_failure']}|{counts['best_only']}|{counts['last_only']}|{stats.pp(row['last_minus_best'])}|{'是' if row['runtime_versions_equal'] else '否，详见JSON'}|")
    lines += ['', '物理一致不代表深度浮点数组逐位一致。图像一致子集的成败列联表和差值、所有不匹配编号、运行库版本保存在summary.json。图像一致子集可能存在选择偏差，不替代全部正式回合。若存在物理不匹配，配对差值只描述匹配子集，可能不同于上方全部300回合的原始成功率差。', '',
        '配对discordant counts（仅一方成功的数量）用于展示这300回合的具体胜负，不代表不同环境参数下的重复证据；不把本次一个seed的差值解释成普遍优势。', '',
        '## Roll 的区域保持与稳定抓持', '',
        'Roll主指标为 `roll_eval_success / region_retention_v5`，与 `roll_stable_hold` 是不同判据。区域保持成功不能直接当作可靠的稳定抓球成功。', '',
        '|模型|指标|成功/已报告回合|比例|', '|---|---|---:|---:|']
    for label in LABELS:
        for name, row in models[label]['tasks']['roll']['auxiliary'].items():
            lines.append(f"|{label} {ITERATIONS[label]}轮|{name}|{row['successes']}/{row['reported']}|{stats.pct(row['rate'])}|")
    lines += ['', '|任务|模型|本次记录的半径集合（米）|质量集合（千克）|', '|---|---|---|---|']
    for task in TASKS:
        for label in LABELS:
            row = models[label]['tasks'][task]['environment_draw']
            radii = ', '.join(f'{value:.6f}' for value in row['radii'])
            masses = ', '.join(f'{value:.6f}' for value in row['masses'])
            lines.append(f'|{task.title()}|{label}|{radii}|{masses}|')
    lines += ['', '## 文件、协议与评估边界', '']
    for label in LABELS:
        model = models[label]
        lines += [f"- {label}：`{model['checkpoint']}`，iteration={model['iteration']}，SHA256 `{model['checkpoint_sha256']}`。",
            f"  源清单哈希一致 {model['matching_source_manifest_files']}/{model['source_manifest_files']}，另核验 {model['additional_historical_assets']} 个历史资源。"]
        for exception in model['source_exceptions']:
            lines.append('  源码例外：'+json.dumps(exception, ensure_ascii=False))
    lines += ['', '使用独立历史源码、原网络、归一化器、动作适配、两路深度与原任务判定；策略推理batch为1。没有重新训练或更改权重，仿真物体真值不进入策略。原worker逐回合重新设定随机种子，本次以实际初始状态哈希核验配对。', '',
        '任务判据：Throw = `success / throw_official`；Roll = `roll_eval_success / region_retention_v5`；Bounce = `success / bounce_catch_proximity_v2`。本次不是新YCB目标泛化、实机测试或跨训练seed的算法排名。', '',
        '先前已停止的5组种子Contact测试、旧300回合导出、其他模型评估、试跑均不计入这1800回合。完整输入哈希、终止原因、条件区间和配对记录见 [summary.json](summary.json)。', '',
        '![Contact best与last三任务成功率](success_rates.png)', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def make_plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    colors = {'best': '#398877', 'last': '#D17936'}
    x = np.arange(3)
    for index, label in enumerate(LABELS):
        rows = [data['models'][label]['tasks'][task] for task in TASKS]
        rates = np.array([row['rate']*100 for row in rows])
        lower = np.array([row['conditional_wilson95'][0]*100 for row in rows])
        upper = np.array([row['conditional_wilson95'][1]*100 for row in rows])
        positions = x+(index-.5)*.32
        axis.bar(positions, rates, width=.29, color=colors[label], alpha=.82,
                 label=f'Contact {label} ({ITERATIONS[label]} iterations)',
                 yerr=[rates-lower, upper-rates], capsize=4,
                 error_kw=dict(ecolor='#3B454A', lw=1.1))
        for position, rate, top in zip(positions, rates, upper):
            axis.text(position, top+1.8, f'{rate:.2f}%', ha='center', fontsize=10, color=colors[label])
    axis.set_xticks(x, [task.title() for task in TASKS])
    axis.set_ylim(0, 108)
    axis.set_ylabel('Success rate (%)')
    axis.set_title('Contact best vs last: one shared environment seed')
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)
    axis.legend(loc='upper right', fontsize=9)
    fig.text(.5, .015, 'Seed 20261029; 300 episodes/task/model; error bars: conditional 95% Wilson intervals',
             ha='center', fontsize=9, color='#4D555B')
    fig.tight_layout(rect=(0, .045, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(output/f'success_rates.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=BASE/'formal')
    parser.add_argument('--audit', type=Path, default=BASE/'source_audit.json')
    parser.add_argument('--output-root', type=Path, default=BASE/'report')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    jobs = {(label, task): load_job(args.input_root/f'{label}_{task}', label, task, args.audit, audit)
            for label in LABELS for task in TASKS}
    data = aggregate(jobs, audit)
    if sum(len(job['rows']) for job in jobs.values()) != 1800:
        raise RuntimeError('Need exactly 1800 canonical episodes')
    data['report_script_sha256'] = stats.sha(__file__)
    data['shared_statistics_script_sha256'] = stats.sha(stats.__file__)
    data['input_sha256'][str(args.audit)] = stats.sha(args.audit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'summary.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    make_report(data, args.output_root)
    make_plot(data, args.output_root)
    print(json.dumps(dict(complete=True, episodes=1800,
                          report=str(args.output_root/'REPORT.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()

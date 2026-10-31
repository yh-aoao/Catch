"""Summarize the complete Contact 6k evaluation and compare archived 10k policies.

Only canonical formal directories are read. No simulator, Torch or GPU is used.
Five seeds are environment evaluation repeats, not independent policy training.
"""
import argparse
import json
from pathlib import Path
import statistics

import report_distill_multiseed_20261029 as stats


ROOT = Path('/home/isee/Catch')
BASE = ROOT/'analysis_artifacts/contact_multiseed_20261030_v1'
REFERENCE = ROOT/'analysis_artifacts/distill_best_multiseed_20261029_v1'
TASKS = stats.TASKS
SEEDS = stats.SEEDS
VARIANTS = ('contact', 'belief', 'future')
LABELS = {'contact': 'Contact best (6k)', 'belief': 'Belief best (10k)',
          'future': 'Future last (10k)'}
CONTACT_SHA = '381bb98d3605b0396f8cffa025900171da23cbed05d425bf3a0c1f239d42bec3'


def load_job(folder, variant, task, audit_path, audit):
    """Validate exact episode counts, metric contracts and archived provenance."""
    if not (folder/'COMPLETE').exists():
        raise RuntimeError('Unfinished evaluation: '+str(folder))
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    iteration = 6000 if variant == 'contact' else 10000
    if (protocol['variant'], protocol['task'], protocol['iteration']) != (variant, task, iteration):
        raise RuntimeError('Wrong model, task or training iteration: '+str(folder))
    if protocol['policy_batch_size'] != 1 or not protocol['deterministic_algorithms'] or protocol['tf32']:
        raise RuntimeError('Inference protocol differs from canonical single-sample settings')
    if tuple(protocol['seeds']) != SEEDS or protocol['episodes_per_seed'] != 300:
        raise RuntimeError('Expected the canonical 5 seeds x 300 episodes: '+str(folder))
    if protocol['checkpoint_sha256'] != audit[variant]['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash differs from source audit: '+str(folder))
    if variant == 'contact' and protocol['checkpoint_sha256'] != CONTACT_SHA:
        raise RuntimeError('Not the Contact checkpoint supplied for this evaluation')
    if protocol['source_audit_sha256'] != stats.sha(audit_path):
        raise RuntimeError('Source audit changed: '+str(audit_path))
    inputs = {str(protocol_path): stats.sha(protocol_path)}
    streams = {}
    task_id = TASKS.index(task)
    for seed in SEEDS:
        path = folder/f'seed_{seed}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 300 or [row['episode_id'] for row in rows] != list(range(300)):
            raise RuntimeError('Need 300 unique, contiguous episode IDs: '+str(path))
        for row in rows:
            expected_reset = (seed+1000000+task_id*10000+row['episode_id']*100003) % (2**32)
            if row['seed'] != seed or row['reset_seed'] != expected_reset:
                raise RuntimeError('Unexpected episode seed: '+str(path))
            if type(row['success']) is not bool:
                raise RuntimeError('Success must be a boolean: '+str(path))
            if stats.metric_version(row) != stats.METRICS[task][1]:
                raise RuntimeError('Different success metric version: '+str(path))
            metric = stats.METRICS[task][0]
            if metric not in row['diagnostics'] or bool(row['diagnostics'][metric]) != row['success']:
                raise RuntimeError('Recorded result differs from terminal success metric: '+str(path))
            if not row['initial'].get('physical_hash') or not row['initial'].get('depth_sha256'):
                raise RuntimeError('Missing initial state or depth hash: '+str(path))
        streams[seed] = rows
        inputs[str(path)] = stats.sha(path)
    summary_path = folder/'summary.json'
    saved = json.loads(summary_path.read_text())
    if saved['episodes'] != 1500 or saved['successes'] != sum(row['success'] for rows in streams.values() for row in rows):
        raise RuntimeError('Raw episode counts disagree with evaluation summary: '+str(folder))
    if saved['protocol'] != protocol:
        raise RuntimeError('Summary protocol mismatch: '+str(folder))
    inputs[str(summary_path)] = stats.sha(summary_path)
    return dict(protocol=protocol, streams=streams, source_files=inputs)


def summarize_job(job):
    streams = job['streams']
    rows = [row for seed in SEEDS for row in streams[seed]]
    result = stats.summarize_rows(rows)
    result['per_seed'] = {str(seed): stats.summarize_rows(streams[seed]) for seed in SEEDS}
    result['seed_rate'] = stats.seed_statistics((result['per_seed'][str(seed)]['rate'] for seed in SEEDS), rate=True)
    result['environment_draws'] = {str(seed): dict(
        radii=sorted({row['initial']['radius'] for row in streams[seed]}),
        masses=sorted({row['initial']['object_mass'] for row in streams[seed]})) for seed in SEEDS}
    result['protocol'] = job['protocol']
    return result


def rename_pairing(value):
    """The shared pairing helper computes right-left; use neutral Contact labels."""
    names = {'future_minus_belief': 'contact_minus_reference',
             'same_depth_future_minus_belief': 'same_depth_contact_minus_reference',
             'future_only': 'contact_only', 'belief_only': 'reference_only'}
    if isinstance(value, dict):
        return {names.get(key, key): rename_pairing(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_pairing(item) for item in value]
    return value


def paired(reference, contact):
    result = rename_pairing(stats.paired_task(reference, contact))
    result['comparison_direction'] = 'contact_minus_reference'
    result['runtime_versions_equal'] = reference['protocol']['versions'] == contact['protocol']['versions']
    result['reference_runtime_versions'] = reference['protocol']['versions']
    result['contact_runtime_versions'] = contact['protocol']['versions']
    result['success_metric_versions_equal'] = True  # load_job validates every row.
    return result


def aggregate(jobs, contact_audit, reference_audit):
    data = dict(complete=True, contact_episodes=4500, reference_episodes=9000,
                episodes_per_seed_per_task=300, seeds=list(SEEDS),
                comparison_direction='contact_minus_reference', models={}, comparisons={},
                input_sha256={}, notes=[
                    'Contact is iteration 6000, whereas archived Belief and Future are iteration 10000.',
                    'These are the supplied fixed policies, not an equal-training-budget algorithm ranking.',
                    'The five seeds are evaluation/environment seeds, not independent training repeats.',
                    'No old 300-episode export, pilot or discarded run is pooled into canonical samples.',
                    'Wilson intervals are conditional episode-level descriptions; approximate five-seed t intervals are exploratory.',
                    'Roll region retention and stable holding are separate success measures.'])
    for variant in VARIANTS:
        audit = contact_audit if variant == 'contact' else reference_audit
        tasks = {task: summarize_job(jobs[variant, task]) for task in TASKS}
        per_seed_macro = {str(seed): statistics.mean(tasks[task]['per_seed'][str(seed)]['rate'] for task in TASKS)
                          for seed in SEEDS}
        item = audit[variant]
        data['models'][variant] = dict(checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256'],
            iteration=item['iteration'], tasks=tasks, per_seed_macro=per_seed_macro,
            macro=stats.seed_statistics(per_seed_macro.values(), rate=True),
            source_exceptions=item.get('exceptions', []), source_manifest_files=len(item['files']),
            matching_source_manifest_files=sum(row['matching'] for row in item['files']),
            additional_historical_assets=len(item.get('additional_required_assets', [])))
        for task in TASKS:
            data['input_sha256'].update(jobs[variant, task]['source_files'])
    for reference in ('belief', 'future'):
        tasks = {task: paired(jobs[reference, task], jobs['contact', task]) for task in TASKS}
        macro_differences = {str(seed): statistics.mean(tasks[task]['per_seed'][str(seed)]['contact_minus_reference']
                               for task in TASKS)
                            for seed in SEEDS if all(tasks[task]['per_seed'][str(seed)]['contact_minus_reference'] is not None
                                                    for task in TASKS)}
        data['comparisons'][reference] = dict(tasks=tasks,
            macro=dict(per_seed=macro_differences, seed_difference=stats.seed_statistics(macro_differences.values())))
    return data


def make_report(data, output):
    contact = data['models']['contact']
    lines = ['# Contact 模型多种子成功率评估（2026-10-30）', '',
        '**完整评估：Contact新增4500回合，每任务1500回合（5组环境种子×300）。**', '',
        f"根目录 `student_best.pth` 的元数据为 `distillation_aux_contact_v1`、iteration={contact['iteration']}。它与之前 `distill_best_ckpt/contact_student_best.pth` 的SHA256相同，都是同一份6000轮best检查点；best文件可以选自训练结束前的轮次。", '',
        '## Contact 成功率', '',
        '|任务|成功/回合|成功率|条件回合级95% Wilson区间|5种子最小–最大|种子间标准差|',
        '|---|---:|---:|---|---|---:|']
    for task in TASKS:
        row = contact['tasks'][task]
        seed = row['seed_rate']
        lines.append(f"|{task.title()}|{row['successes']}/{row['episodes']}|{stats.pct(row['rate'])}|{stats.interval(row['conditional_wilson95'])}|{stats.pct(seed['minimum'])}–{stats.pct(seed['maximum'])}|{stats.pct(seed['sample_sd'])}|")
    lines += ['', f"三任务等权平均为 **{stats.pct(contact['macro']['mean'])}**；5组环境种子均值的近似95% t区间为 {stats.interval(contact['macro']['t95'])}。", '',
        'Wilson区间只近似描述固定这些环境参数后的回合抽样误差；物体半径、质量等可能在环境构造时固定，因此1500回合并非1500组独立物理参数。5种子的t区间仅作探索性描述，不能代表不同训练随机种子的算法方差。', '',
        '## 每组环境种子', '',
        '|评估种子|Throw|Roll|Bounce|三任务平均|', '|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        rates = [contact['tasks'][task]['per_seed'][str(seed)]['rate'] for task in TASKS]
        lines.append('|'+str(seed)+'|'+'|'.join(stats.pct(rate) for rate in rates)+f"|{stats.pct(contact['per_seed_macro'][str(seed)])}|")
    lines += ['', '## 与前次两份模型对照', '',
        '**Contact为6000轮，Belief和Future为10000轮。下面是这些具体权重的描述性对照，不是等训练轮次的算法排名。** 前次两模型的9000回合读取原始正式记录重新核验，没有加入旧300回合导出。', '',
        '|任务|Contact best 6k|Belief best 10k|Future last 10k|', '|---|---:|---:|---:|']
    for task in TASKS:
        lines.append('|'+task.title()+'|'+'|'.join(stats.pct(data['models'][variant]['tasks'][task]['rate']) for variant in VARIANTS)+'|')
    lines.append('|三任务平均|'+'|'.join(stats.pct(data['models'][variant]['macro']['mean']) for variant in VARIANTS)+'|')
    lines += ['', '配对差值方向统一为 **Contact − 对照模型**，单位为百分点。只纳入相同种子、相同回合编号且捕获的初始物理哈希一致的回合。深度哈希另行核验，不将物理一致直接等同于视觉输入逐位一致。', '',
        '|对照模型|任务|物理匹配/共同编号|初始深度逐位一致|有效种子数|配对差值（种子等权）|种子层面近似95% t区间|运行库版本一致|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for reference in ('belief', 'future'):
        comparison = data['comparisons'][reference]
        for task in TASKS:
            row = comparison['tasks'][task]
            diff = row['seed_difference']
            lines.append(f"|{LABELS[reference]}|{task.title()}|{row['physical_matches']}/{row['common_episode_ids']}|{row['initial_depth_matches']}/{row['physical_matches']}|{diff['n_seeds']}|{stats.pp(diff['mean'])}|{stats.interval(diff['t95'], True)}|{'是' if row['runtime_versions_equal'] else '否，详见JSON'}|")
        diff = comparison['macro']['seed_difference']
        lines.append(f"|{LABELS[reference]}|三任务平均|—|—|{diff['n_seeds']}|{stats.pp(diff['mean'])}|{stats.interval(diff['t95'], True)}|—|")
    lines += ['', '若某种子没有物理匹配回合，它不进入配对差值均值；三任务平均仅取三个任务均有匹配回合的种子。部分匹配时，各任务差值以各自匹配子集为分母，不能直接等同于全部回合成功率之差。有效种子数另列。', '',
        '相同初始深度子集的差值、每种子差值、成败列联表、所有不匹配编号和运行库版本字典均保存在 summary.json。视觉一致子集可能具有选择偏差，不能替代全部正式回合。配对区间跨0时，当前5组环境不足以支持稳定的模型优劣结论；即使不跨0，也不能据此判断等训练预算或跨训练seed的算法优劣。', '',
        '## Roll 的区域保持与稳定抓持', '',
        '主指标沿用 `roll_eval_success / region_retention_v5`。它衡量区域保持，必须与更严格的 `roll_stable_hold` 区分，不能直接写成可靠的稳定抓球成功率。', '',
        '|指标|Contact 成功/已报告回合|比例|', '|---|---:|---:|']
    for name, row in contact['tasks']['roll']['auxiliary'].items():
        lines.append(f"|{name}|{row['successes']}/{row['reported']}|{stats.pct(row['rate'])}|")
    lines += ['', '|评估种子|Roll半径（米）|Roll质量（千克）|', '|---:|---|---|']
    for seed, row in contact['tasks']['roll']['environment_draws'].items():
        radii = ', '.join(f'{value:.6f}' for value in row['radii'])
        masses = ', '.join(f'{value:.6f}' for value in row['masses'])
        lines.append(f'|{seed}|{radii}|{masses}|')
    lines += ['', '## 来源与边界', '',
        f"Contact检查点SHA256：`{contact['checkpoint_sha256']}`。源清单文件 {contact['matching_source_manifest_files']}/{contact['source_manifest_files']} 哈希一致，另核验 {contact['additional_historical_assets']} 个历史资源。完整来源见上级目录 source_audit.json。", '',
        '每次策略推理batch固定为1，保留原网络、归一化器、动作适配、两路深度相机、物理和成功判定；没有重新训练或改动权重。短测、废弃试跑和旧300回合导出均未计入Contact4500回合。原worker逐回合重设seed，本次配对依据实际记录的物理与图像哈希核验。', '',
        'Throw使用 `success / throw_official`；Roll使用 `roll_eval_success / region_retention_v5`；Bounce使用 `success / bounce_catch_proximity_v2`。这些是原任务成功定义，本次不是实机测试、新YCB物体泛化测试或严格统一抓持判据实验。', '',
        '两个对照模型的历史源码例外沿用前次审计：Future有一个未被导入的训练启动器占位文件，其余有效来源详见[前次报告](../../distill_best_multiseed_20261029_v1/report/REPORT.md)。本地运行库与原训练服务器可能不同；跨日期对照的实际版本核验见上表及JSON。', '',
        '逐回合输入文件哈希、完整统计与配对记录见 [summary.json](summary.json)。', '',
        '![各模型成功率与种子波动](success_by_seed.png)', '',
        '![Contact相对于对照模型的配对差值](paired_differences.png)', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def make_plots(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    colors = {'contact': '#398877', 'belief': '#336DA8', 'future': '#D17936'}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for axis, task in zip(axes, TASKS):
        for i, variant in enumerate(VARIANTS):
            row = data['models'][variant]['tasks'][task]
            values = [row['per_seed'][str(seed)]['rate']*100 for seed in SEEDS]
            axis.bar(i, row['rate']*100, width=.5, color=colors[variant], alpha=.18)
            axis.scatter(i+np.linspace(-.1, .1, 5), values, color=colors[variant], s=30, zorder=3)
            axis.plot([i-.25, i+.25], [row['rate']*100]*2, color=colors[variant], lw=2.5)
            axis.text(i, max(values)+4, f"{row['rate']*100:.2f}%", ha='center', color=colors[variant], fontsize=10)
        axis.set_title(task.title())
        axis.set_xticks(range(3), ['Contact\nbest 6k', 'Belief\nbest 10k', 'Future\nlast 10k'])
        axis.set_ylim(0, 108)
        axis.set_xlim(-.5, 2.5)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Success rate (%)')
    fig.suptitle('Five environment seeds x 300 episodes/task; different training iterations', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(output/f'success_by_seed.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, reference in zip(axes, ('belief', 'future')):
        for i, task in enumerate((*TASKS, 'macro')):
            comparison = data['comparisons'][reference]
            if task == 'macro':
                row = comparison['macro']['seed_difference']
                values = list(comparison['macro']['per_seed'].values())
            else:
                task_data = comparison['tasks'][task]
                row = task_data['seed_difference']
                values = [item['contact_minus_reference'] for item in task_data['per_seed'].values()
                          if item['contact_minus_reference'] is not None]
            if not values:
                continue
            axis.scatter(i+np.linspace(-.1, .1, len(values)), np.array(values)*100,
                         color='#506D79', s=32, alpha=.8, zorder=3)
            if row['t95'] is not None:
                axis.errorbar(i, row['mean']*100,
                    yerr=[[100*(row['mean']-row['t95'][0])], [100*(row['t95'][1]-row['mean'])]],
                    fmt='D', color='#9D3D45', capsize=6, markersize=5, lw=1.8,
                    label='Mean and approximate seed-level 95% t interval' if i == 0 else None)
        axis.axhline(0, color='#888888', lw=1)
        axis.set_title('Contact 6k minus '+LABELS[reference])
        axis.set_xticks(range(4), ['Throw', 'Roll', 'Bounce', '3-task mean'])
        axis.grid(axis='y', alpha=.2)
        axis.legend(loc='best', fontsize=8)
    axes[0].set_ylabel('Contact minus reference (percentage points)')
    fig.suptitle('Matching initial physical states; descriptive policy comparison', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(output/f'paired_differences.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=BASE/'formal')
    parser.add_argument('--reference-root', type=Path, default=REFERENCE/'formal')
    parser.add_argument('--output-root', type=Path, default=BASE/'report')
    parser.add_argument('--audit', type=Path, default=BASE/'source_audit.json')
    parser.add_argument('--reference-audit', type=Path, default=REFERENCE/'source_audit.json')
    args = parser.parse_args()
    contact_audit = json.loads(args.audit.read_text())
    reference_audit = json.loads(args.reference_audit.read_text())
    jobs = {}
    for variant in VARIANTS:
        folder = args.input_root if variant == 'contact' else args.reference_root
        audit_path = args.audit if variant == 'contact' else args.reference_audit
        audit = contact_audit if variant == 'contact' else reference_audit
        for task in TASKS:
            jobs[variant, task] = load_job(folder/f'{variant}_{task}', variant, task, audit_path, audit)
    data = aggregate(jobs, contact_audit, reference_audit)
    data['report_script_sha256'] = stats.sha(__file__)
    data['shared_statistics_script_sha256'] = stats.sha(stats.__file__)
    data['input_sha256'][str(args.audit)] = stats.sha(args.audit)
    data['input_sha256'][str(args.reference_audit)] = stats.sha(args.reference_audit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'summary.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    make_report(data, args.output_root)
    make_plots(data, args.output_root)
    print(json.dumps(dict(complete=True, contact_episodes=4500,
                          report=str(args.output_root/'REPORT.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()

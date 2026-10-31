"""Summarize the canonical, paired Belief/Future multi-seed evaluation.

Only reads formal/<variant>_<task>; pilot and discarded runs cannot enter the
analysis. Refuses unfinished runs by default. No simulator or GPU is required.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path('/home/isee/Catch')
BASE = ROOT / 'analysis_artifacts/distill_best_multiseed_20261029_v1'
VARIANTS = ('belief', 'future')
TASKS = ('throw', 'roll', 'bounce')
LABELS = {'belief': 'Belief best (10k)', 'future': 'Future last (10k)'}
METRICS = {'throw': ('success', 'throw_official'),
           'roll': ('roll_eval_success', 'region_retention_v5'),
           'bounce': ('success', 'bounce_catch_proximity_v2')}
SEEDS = (20261029, 20262029, 20263029, 20264029, 20265029)
AUXILIARY = ('success', 'roll_eval_success', 'roll_caught_once',
             'roll_stable_hold', 'roll_holding_at_end', 'roll_final_hold',
             'roll_legacy_success')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, n):
    """95% binomial Wilson interval; no claim of between-seed uncertainty."""
    if not n:
        return None
    z = 1.959963984540054
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def seed_statistics(values, rate=False):
    """Equal-weight seed summary; t interval is descriptive with only five seeds."""
    values = list(values)
    if not values:
        return {'n_seeds': 0, 'mean': None, 'sample_sd': None, 't95': None}
    mean = statistics.mean(values)
    result = dict(n_seeds=len(values), mean=mean, minimum=min(values),
                  maximum=max(values), sample_sd=None, t95=None)
    if len(values) > 1:
        # This protocol has five seeds. Other entries support partial reports.
        t_975 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053,
                 4: 2.7764451052}
        sd = statistics.stdev(values)
        half = t_975[len(values)-1]*sd/math.sqrt(len(values))
        interval = [mean-half, mean+half]
        if rate:
            interval = [max(0., interval[0]), min(1., interval[1])]
        result.update(sample_sd=sd, t95=interval)
    return result


def auxiliary(rows):
    result = {}
    for name in AUXILIARY:
        reported = [row['diagnostics'][name] for row in rows
                    if name in row.get('diagnostics', {})]
        if reported:
            successes = sum(bool(value) for value in reported)
            result[name] = dict(successes=successes, reported=len(reported),
                                rate=successes/len(reported))
    return result


def metric_version(row):
    diagnostics = row.get('diagnostics', {})
    return diagnostics.get('success_version',
                           diagnostics.get('roll_eval', {}).get('evaluation_version', 'missing'))


def summarize_rows(rows):
    n = len(rows)
    successes = sum(row['success'] for row in rows)
    return dict(episodes=n, successes=successes,
                rate=successes/n if n else None,
                conditional_wilson95=wilson(successes, n),
                mean_reward=statistics.mean(row['reward'] for row in rows) if n else None,
                mean_length=statistics.mean(row['length'] for row in rows) if n else None,
                termination_counts=dict(Counter(row['reason'] for row in rows)),
                metric_versions=dict(Counter(metric_version(row) for row in rows)),
                auxiliary=auxiliary(rows))


def load_job(path, variant, task, allow_incomplete):
    protocol_path = path/'protocol.json'
    if not protocol_path.exists():
        if allow_incomplete:
            return None
        raise RuntimeError('Missing protocol: '+str(protocol_path))
    protocol = json.loads(protocol_path.read_text())
    if protocol['variant'] != variant or protocol['task'] != task:
        raise RuntimeError('Variant/task protocol mismatch: '+str(path))
    if protocol['iteration'] != 10000 or protocol['policy_batch_size'] != 1:
        raise RuntimeError('Expected 10k weights and single-sample inference: '+str(path))
    if tuple(protocol['seeds']) != SEEDS or protocol['episodes_per_seed'] != 300:
        raise RuntimeError('Not the canonical 5 x 300 protocol: '+str(path))
    if not allow_incomplete and not (path/'COMPLETE').exists():
        raise RuntimeError('Evaluation is unfinished: '+str(path))
    streams, source_files = {}, {str(protocol_path): sha(protocol_path)}
    for seed in SEEDS:
        file = path/f'seed_{seed}.jsonl'
        if file.exists():
            rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            source_files[str(file)] = sha(file)
        else:
            rows = []
        if [row['episode_id'] for row in rows] != list(range(len(rows))):
            raise RuntimeError('Episode IDs are not a unique contiguous sequence: '+str(file))
        if len(rows) > 300 or (not allow_incomplete and len(rows) != 300):
            raise RuntimeError('Unexpected number of episodes: '+str(file))
        for row in rows:
            if row['seed'] != seed or type(row['success']) is not bool:
                raise RuntimeError('Invalid episode schema: '+str(file))
            if metric_version(row) != METRICS[task][1]:
                raise RuntimeError('Different success definition: '+str(file))
            if METRICS[task][0] in row['diagnostics'] and bool(row['diagnostics'][METRICS[task][0]]) != row['success']:
                raise RuntimeError('Success field and original terminal metric disagree: '+str(file))
            if not row['initial'].get('physical_hash') or not row['initial'].get('depth_sha256'):
                raise RuntimeError('Missing initial state hash: '+str(file))
        streams[seed] = rows
    return dict(protocol=protocol, streams=streams, source_files=source_files,
                complete=all(len(rows) == 300 for rows in streams.values()))


def paired_task(belief, future):
    per_seed, mismatches = {}, []
    contingency = Counter()
    exact_visual_contingency = Counter()
    for seed in SEEDS:
        left = {row['episode_id']: row for row in belief['streams'][seed]}
        right = {row['episode_id']: row for row in future['streams'][seed]}
        common = sorted(set(left) & set(right))
        counts, depth_counts, differences, depth_differences = Counter(), Counter(), [], []
        physical_mismatches = depth_mismatches = 0
        for episode in common:
            b, f = left[episode], right[episode]
            if b['initial']['physical_hash'] != f['initial']['physical_hash']:
                physical_mismatches += 1
                mismatches.append(dict(seed=seed, episode_id=episode, kind='physical'))
                continue
            key = ('both_success' if b['success'] and f['success'] else
                   'both_failure' if not b['success'] and not f['success'] else
                   'future_only' if f['success'] else 'belief_only')
            counts[key] += 1
            differences.append(int(f['success'])-int(b['success']))
            if b['initial']['depth_sha256'] == f['initial']['depth_sha256']:
                depth_counts[key] += 1
                depth_differences.append(differences[-1])
            else:
                depth_mismatches += 1
                mismatches.append(dict(seed=seed, episode_id=episode, kind='initial_depth'))
        contingency.update(counts)
        exact_visual_contingency.update(depth_counts)
        per_seed[str(seed)] = dict(common_episode_ids=len(common),
            unmatched_episode_ids=len(set(left)^set(right)),
            physical_matches=len(differences), physical_mismatches=physical_mismatches,
            initial_depth_matches=len(depth_differences),
            initial_depth_mismatches_with_matching_physics=depth_mismatches,
            future_minus_belief=statistics.mean(differences) if differences else None,
            same_depth_future_minus_belief=statistics.mean(depth_differences) if depth_differences else None,
            contingency=dict(counts), same_depth_contingency=dict(depth_counts))
    all_matches = sum(row['physical_matches'] for row in per_seed.values())
    depth_matches = sum(row['initial_depth_matches'] for row in per_seed.values())
    return dict(per_seed=per_seed,
        common_episode_ids=sum(row['common_episode_ids'] for row in per_seed.values()),
        physical_matches=all_matches,
        physical_mismatches=sum(row['physical_mismatches'] for row in per_seed.values()),
        initial_depth_matches=depth_matches,
        initial_depth_mismatches_with_matching_physics=sum(row['initial_depth_mismatches_with_matching_physics'] for row in per_seed.values()),
        matched_episode_weighted_difference=(contingency['future_only']-contingency['belief_only'])/all_matches if all_matches else None,
        same_depth_episode_weighted_difference=(exact_visual_contingency['future_only']-exact_visual_contingency['belief_only'])/depth_matches if depth_matches else None,
        seed_difference=seed_statistics(row['future_minus_belief'] for row in per_seed.values() if row['future_minus_belief'] is not None),
        same_depth_seed_difference=seed_statistics(row['same_depth_future_minus_belief'] for row in per_seed.values() if row['same_depth_future_minus_belief'] is not None),
        contingency=dict(contingency), same_depth_contingency=dict(exact_visual_contingency),
        mismatch_records=mismatches)


def aggregate(jobs, audit, incomplete):
    result = dict(complete=not incomplete, planned_episodes=9000,
                  seeds=list(SEEDS), episodes_per_seed_per_task=300,
                  comparison_direction='future_minus_belief', models={}, comparisons={},
                  input_sha256={}, notes=[
                      'Five evaluation/environment seeds, not five independently trained policies.',
                      'Wilson intervals describe episode-level binomial uncertainty conditional on these sampled environments.',
                      'The five-seed t intervals are approximate descriptive intervals with four degrees of freedom, not robust significance tests.',
                      'Roll main success is region_retention_v5 and is distinct from stable holding.',
                      'Per-episode reset reseeds the environment; episode duration does not itself invalidate the reset seed pairing.',
                      'Initial physical and rendered-depth hashes are checked separately.'])
    for variant in VARIANTS:
        tasks = {}
        for task in TASKS:
            job = jobs.get((variant, task))
            if job is None:
                continue
            streams = job['streams']
            rows = [row for seed in SEEDS for row in streams[seed]]
            data = summarize_rows(rows)
            data['per_seed'] = {str(seed): summarize_rows(streams[seed]) for seed in SEEDS}
            data['seed_rate'] = seed_statistics((data['per_seed'][str(seed)]['rate'] for seed in SEEDS
                                                 if data['per_seed'][str(seed)]['rate'] is not None), rate=True)
            data['environment_draws'] = {
                str(seed): dict(episodes=len(streams[seed]),
                    radii=sorted(set(row['initial']['radius'] for row in streams[seed])),
                    masses=sorted(set(row['initial']['object_mass'] for row in streams[seed]))) for seed in SEEDS}
            data['protocol'] = job['protocol']
            result['input_sha256'].update(job['source_files'])
            tasks[task] = data
        per_seed_macro = {}
        if len(tasks) == 3:
            for seed in SEEDS:
                rates = [tasks[task]['per_seed'][str(seed)]['rate'] for task in TASKS]
                if all(rate is not None for rate in rates):
                    per_seed_macro[str(seed)] = statistics.mean(rates)
        result['models'][variant] = dict(
            checkpoint=audit[variant]['checkpoint'], checkpoint_sha256=audit[variant]['checkpoint_sha256'],
            iteration=audit[variant]['iteration'], source_exceptions=audit[variant]['exceptions'],
            source_manifest_files=len(audit[variant]['files']),
            matching_source_manifest_files=sum(row['matching'] for row in audit[variant]['files']),
            additional_historical_assets=len(audit[variant].get('additional_required_assets', [])),
            tasks=tasks, per_seed_macro=per_seed_macro,
            macro=seed_statistics(per_seed_macro.values(), rate=True))
    for task in TASKS:
        b, f = jobs.get(('belief', task)), jobs.get(('future', task))
        if b and f:
            result['comparisons'][task] = paired_task(b, f)
    macro_differences = {}
    if len(result['comparisons']) == 3:
        for seed in SEEDS:
            values = [result['comparisons'][task]['per_seed'][str(seed)]['future_minus_belief'] for task in TASKS]
            if all(value is not None for value in values):
                macro_differences[str(seed)] = statistics.mean(values)
    result['macro_comparison'] = dict(per_seed=macro_differences,
        seed_difference=seed_statistics(macro_differences.values()))
    result['episodes'] = sum(task['episodes'] for model in result['models'].values() for task in model['tasks'].values())
    result['complete'] = result['episodes'] == 9000 and all(job['complete'] for job in jobs.values()) and len(jobs) == 6
    return result


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def pp(value):
    return '—' if value is None else f'{100*value:+.2f}'


def interval(value, difference=False):
    if value is None:
        return '—'
    return f'[{100*value[0]:+.2f}, {100*value[1]:+.2f}]' if difference else f'[{100*value[0]:.2f}%, {100*value[1]:.2f}%]'


def make_report(data, output):
    conclusions = []
    if data['complete']:
        differences = {task: data['comparisons'][task]['seed_difference']['mean'] for task in TASKS}
        bounce_wins = sum(row['future_minus_belief'] > 0 for row in data['comparisons']['bounce']['per_seed'].values())
        macro = data['macro_comparison']['seed_difference']
        macro_note = ('种子层面的平均差值区间仍跨0，不能据此断言综合优势稳定。'
                      if macro['t95'][0] <= 0 <= macro['t95'][1] else
                      '该区间仅描述这两份权重在5组环境种子下的差异，不能推断算法整体优劣。')
        holds = {variant: data['models'][variant]['tasks']['roll']['auxiliary']['roll_stable_hold'] for variant in VARIANTS}
        conclusions = [
            f"Bounce的Future−Belief差值为{pp(differences['bounce'])}个百分点，5组中有{bounce_wins}组Future较高；Throw和Roll差值分别为{pp(differences['throw'])}、{pp(differences['roll'])}个百分点。三任务平均差值为{pp(macro['mean'])}个百分点，{macro_note}", '',
            f"面向实机时尤其需要区分Roll的成功口径：主成功率是区域保持，严格的稳定抓持为Belief {holds['belief']['successes']}/{holds['belief']['reported']}、Future {holds['future']['successes']}/{holds['future']['reported']}。主指标不能直接代表可靠的稳定抓球能力。", '']
    text = [
        '# Belief 与 Future 多种子重复评估（2026-10-29）', '',
        f"状态：{'完整' if data['complete'] else '进行中，不能用于最终模型排名'}；已统计 {data['episodes']}/{data['planned_episodes']} 回合。", '',
        '本次比较 **Belief student_best（10000轮）** 与 **Future student_last（10000轮）**。Future student_best 是6000轮；本次没有将它混入一万轮对比。', '',
        '每模型、每任务使用5个新的评估种子，每种子300回合，共1500回合；两个模型、三个任务共9000回合。它们是5组环境随机种子，两个已训练好的策略各只有一份，不代表5次独立训练。', '',
        *conclusions,
        '## 成功率', '',
        '|任务|模型|成功/回合|成功率|回合级95% Wilson区间|5种子最小–最大|种子间标准差|',
        '|---|---|---:|---:|---|---|---:|']
    for task in TASKS:
        for variant in VARIANTS:
            row = data['models'][variant]['tasks'].get(task)
            if row is None:
                continue
            seed = row['seed_rate']
            text.append(f"|{task.title()}|{LABELS[variant]}|{row['successes']}/{row['episodes']}|{pct(row['rate'])}|{interval(row['conditional_wilson95'])}|{pct(seed.get('minimum'))}–{pct(seed.get('maximum'))}|{pct(seed['sample_sd'])}|")
    text += ['', '三个任务等权平均：', '', '|模型|平均成功率|5种子平均值的近似95% t区间|', '|---|---:|---|']
    for variant in VARIANTS:
        row = data['models'][variant]['macro']
        text.append(f"|{LABELS[variant]}|{pct(row['mean'])}|{interval(row['t95'])}|")
    text += ['', 'Wilson区间只近似反映固定这5组环境之后的回合抽样误差，不能把1500回合视作1500次独立的环境参数抽样。种子间差异和下方种子层面的区间更适合检查对环境变化的敏感度。只有5个种子，t区间依赖近似假设，应作描述性参考。', '',
             '## 同一初始物理状态的模型差异', '',
             '差值统一为 **Future − Belief**，单位是百分点；正值表示Future较高。配对只使用种子、回合编号相同且初始物理哈希一致的回合。', '',
             '|任务|物理配对/共同编号|初始深度逐位一致|配对差值（种子等权）|5种子差值的近似95% t区间|',
             '|---|---:|---:|---:|---|']
    for task, row in data['comparisons'].items():
        diff = row['seed_difference']
        text.append(f"|{task.title()}|{row['physical_matches']}/{row['common_episode_ids']}|{row['initial_depth_matches']}/{row['physical_matches']}|{pp(diff['mean'])}|{interval(diff['t95'], True)}|")
    macro = data['macro_comparison']['seed_difference']
    text.append(f"|三任务平均|—|—|{pp(macro['mean'])}|{interval(macro['t95'], True)}|")
    text += ['', '初始物理哈希和初始渲染深度哈希分别核对。物理一致并不保证图像浮点数组逐位一致；有图像差异时，不能把该回合称作完全相同的视觉输入。完整深度也相同的子集结果及所有不匹配编号保存在 summary.json。图像一致子集是敏感性检查，不替代全部正式回合。', '',
             '若差值区间跨0，当前5个环境种子无法支持稳定的优劣结论；即使区间不跨0，也不能据此推断不同训练随机种子下的算法整体优劣。', '',
             '## 每个评估种子的结果', '',
             '|任务|评估种子|Belief 成功率|Future 成功率|物理配对差值（百分点）|',
             '|---|---:|---:|---:|---:|']
    for task in TASKS:
        for seed in SEEDS:
            rows = [data['models'][variant]['tasks'].get(task, {}).get('per_seed', {}).get(str(seed), {}) for variant in VARIANTS]
            diff = data['comparisons'].get(task, {}).get('per_seed', {}).get(str(seed), {}).get('future_minus_belief')
            text.append(f"|{task.title()}|{seed}|{pct(rows[0].get('rate'))}|{pct(rows[1].get('rate'))}|{pp(diff)}|")
    text += ['', '## Roll口径与物体参数', '',
             '主成功率使用原模型配置的 roll_eval_success / region_retention_v5。它与“稳定抓持”是两个指标；区域保持成功不能直接写成稳定抓住球。', '',
             '|模型|指标|成功/已报告回合|比例|', '|---|---|---:|---:|']
    for variant in VARIANTS:
        for name, row in data['models'][variant]['tasks'].get('roll', {}).get('auxiliary', {}).items():
            text.append(f"|{LABELS[variant]}|{name}|{row['successes']}/{row['reported']}|{pct(row['rate'])}|")
    text += ['', 'Roll物体半径、质量按环境构造时的种子抽样；下面列出每个种子实际记录到的参数集合，便于确认同一环境内是否固定。', '',
             '|模型|种子|半径（米）|质量（千克）|', '|---|---:|---|---|']
    for variant in VARIANTS:
        for seed, row in data['models'][variant]['tasks'].get('roll', {}).get('environment_draws', {}).items():
            radii = ', '.join(f'{value:.6f}' for value in row['radii']) or '—'
            masses = ', '.join(f'{value:.6f}' for value in row['masses']) or '—'
            text.append(f"|{LABELS[variant]}|{seed}|{radii}|{masses}|")
    text += ['', '## 复现、源码与限制', '',
             '评估使用独立恢复的历史源码目录，沿用原网络、归一化器、动作适配、摄像头和各任务成功定义；没有重新训练或更改权重。环境可以并行运行，每次策略前向计算的batch固定为1，避免批量形状引起微小浮点差异后放大为不同闭环结果。短测和废弃批量试跑的数据没有计入本报告。', '',
             '原worker每回合显式重新设定种子，构造种子为 evaluation_seed + 1000000 + TASK_ID×10000，回合种子再加 episode_index×100003。因此，先前若将“终止时间不同消耗随机数”作为无法配对初始状态的直接原因，这一说法需要更正。本次根据实际初始物理哈希核验配对；旧导出没有这些哈希，不能反推其逐回合完全一致。', '']
    for variant in VARIANTS:
        model = data['models'][variant]
        text += [f"- {LABELS[variant]}：`{Path(model['checkpoint']).name}`，iteration={model['iteration']}；SHA256 `{model['checkpoint_sha256']}`。",
                 f"  原清单源码文件 {model['matching_source_manifest_files']}/{model['source_manifest_files']} 哈希一致，另补充 {model['additional_historical_assets']} 个经历史git版本逐字节核验的资源文件。"]
        for exception in model['source_exceptions']:
            text.append(f"  例外：`{exception['path']}` 未能恢复原文件；独立目录中放置了导入即报错的占位文件，评估同时断言它未被导入。该文件是未使用的训练启动器，不能将整个Future清单描述为全部原样恢复。")
    text += ['', '任务定义：Throw = success / throw_official；Roll = roll_eval_success / region_retention_v5；Bounce = success / bounce_catch_proximity_v2。当前本地运行库与原服务器可能不同，故本次是同一本地评估条件下的模型对比，不能保证复现旧300回合的逐回合轨迹或百分比。', '',
             '旧300回合不与新1500回合合并。此次也没有评估真实机器人、新YCB物体泛化或不同训练随机种子的方差。', '',
             '输入哈希、每种子统计、成败配对列联表和初始状态不匹配编号见 [summary.json](summary.json)。图中的点是每种子300回合的成功率，柱/粗线表示等权平均。', '',
             '![成功率与种子间变化](success_by_seed.png)', '',
             '![按种子配对的模型差值](paired_differences.png)', '']
    if 'depth_diagnostic' in data:
        diagnostic = data['depth_diagnostic']
        top = diagnostic['pooled_cameras']['top']
        wrist = diagnostic['pooled_cameras']['wrist']
        text += ['## 独立初始图像检查', '',
                 f"正式测试后另取Throw seed={diagnostic['seed']}、每模型{diagnostic['episodes']}回合保存原始策略输入，未加入9000回合统计。物理哈希一致{diagnostic['physical_pairs_equal']}/{diagnostic['episodes']}，深度逐位一致{diagnostic['depth_pairs_equal']}/{diagnostic['episodes']}。",
                 f"5幅top图像共{top['encoded']['values']}个像素，其中{top['encoded']['different']}个不同；编码最大绝对差{top['encoded']['max_abs']:.9g}，按相机范围换算为裁剪、量化后的深度最大差约{top['decoded_clipped_metres']['max_abs']*1000:.2f}毫米。wrist不同像素为{wrist['encoded']['different']}。这只是小样本诊断，不能替代全部不匹配图像的测量，也不能单凭差值小就断定对闭环成功没有影响。",
                 '完整诊断和逐回合NPZ见 [图像检查报告](../depth_diagnostic/REPORT.md)。', '']
    (output/'REPORT.md').write_text('\n'.join(text))


def make_plots(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    colors = {'belief': '#336DA8', 'future': '#D17936'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for axis, task in zip(axes, TASKS):
        for i, variant in enumerate(VARIANTS):
            row = data['models'][variant]['tasks'].get(task)
            if row is None or row['rate'] is None:
                continue
            rates = [100*row['per_seed'][str(seed)]['rate'] for seed in SEEDS
                     if row['per_seed'][str(seed)]['rate'] is not None]
            x = i+np.linspace(-.1, .1, len(rates))
            axis.bar(i, 100*row['rate'], width=.5, color=colors[variant], alpha=.2)
            axis.scatter(x, rates, s=35, color=colors[variant], zorder=3)
            axis.plot([i-.26, i+.26], [100*row['rate']]*2, color=colors[variant], lw=3)
            axis.text(i, max(rates)+4, f"{100*row['rate']:.2f}%", ha='center', color=colors[variant], fontsize=10)
        axis.set_title(task.title())
        axis.set_xticks([0, 1], ['Belief best 10k', 'Future last 10k'])
        axis.set_ylim(0, 105)
        axis.set_xlim(-.5, 1.5)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Success rate (%)')
    fig.suptitle('Five evaluation seeds; 300 episodes per seed and task', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(output/f'success_by_seed.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    labels = []
    for i, task in enumerate((*TASKS, 'macro')):
        if task == 'macro':
            row = data['macro_comparison']['seed_difference']
            values = list(data['macro_comparison']['per_seed'].values())
            label = '3-task mean'
        else:
            comparison = data['comparisons'].get(task)
            if comparison is None:
                labels.append(task.title())
                continue
            row = comparison['seed_difference']
            values = [r['future_minus_belief'] for r in comparison['per_seed'].values()
                      if r['future_minus_belief'] is not None]
            label = task.title()
        labels.append(label)
        if not values:
            continue
        axis.scatter(np.full(len(values), i)+np.linspace(-.1, .1, len(values)),
                     np.array(values)*100, s=38, color='#506D79', alpha=.8, zorder=3)
        if row['t95'] is not None:
            axis.errorbar(i, row['mean']*100,
                          yerr=[[100*(row['mean']-row['t95'][0])], [100*(row['t95'][1]-row['mean'])]],
                          fmt='D', color='#9D3D45', capsize=7, markersize=6, lw=2,
                          label='Mean and approximate seed-level 95% t interval' if i == 0 else None)
    axis.axhline(0, color='#888888', lw=1)
    axis.set_xticks(range(4), labels)
    axis.set_ylabel('Future minus Belief (percentage points)')
    axis.set_title('Difference on matching initial physical states')
    axis.grid(axis='y', alpha=.2)
    axis.legend(loc='best', fontsize=9)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(output/f'paired_differences.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=BASE/'formal')
    parser.add_argument('--output-root', type=Path, default=BASE/'report')
    parser.add_argument('--allow-incomplete', action='store_true', help='Generate clearly marked provisional summaries')
    args = parser.parse_args()
    audit = json.loads((BASE/'source_audit.json').read_text())
    jobs = {}
    for variant in VARIANTS:
        for task in TASKS:
            job = load_job(args.input_root/f'{variant}_{task}', variant, task, args.allow_incomplete)
            if job is not None:
                jobs[variant, task] = job
    data = aggregate(jobs, audit, args.allow_incomplete)
    if not args.allow_incomplete and not data['complete']:
        raise RuntimeError('Refusing to label incomplete evaluation as final')
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = BASE/'depth_diagnostic/comparison.json'
    if diagnostic_path.exists():
        data['depth_diagnostic'] = json.loads(diagnostic_path.read_text())
        data['input_sha256'][str(diagnostic_path)] = sha(diagnostic_path)
    data['report_script_sha256'] = sha(__file__)
    data['source_audit_sha256'] = sha(BASE/'source_audit.json')
    (args.output_root/'summary.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    make_report(data, args.output_root)
    make_plots(data, args.output_root)
    print(json.dumps(dict(complete=data['complete'], episodes=data['episodes'],
                          report=str(args.output_root/'REPORT.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Audit raw checkpoint evaluations and plot unsmoothed success-rate curves."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT.parent
MODELS = ('belief', 'future', 'spatial', 'camera_gate', 'contact')
TASKS = ('throw', 'roll', 'bounce')
STEPS = (2000, 4000, 6000, 8000, 10000)
COLORS = dict(belief='#276FBF', future='#E69F00', spatial='#7A5195', camera_gate='#D1495B', contact='#009E73')


def difference(a, b):
    """Newcombe interval from Wilson intervals, exploratory unpaired comparison."""
    p, q = a['success_rate'], b['success_rate']
    lp, up = a['wilson_95']
    lq, uq = b['wilson_95']
    d = p-q
    return dict(estimate=d, lower=d-math.sqrt((p-lp)**2+(uq-q)**2),
                upper=d+math.sqrt((up-p)**2+(q-lq)**2))


def main():
    points = {}
    contract = {}
    n_records = 0
    hashes = {}
    for model in MODELS:
        points[model] = {}
        for path in sorted((SOURCE/model).glob('eval_*.json')):
            iteration = int(path.stem.split('_')[1])
            assert iteration in STEPS
            data = json.loads(path.read_text())
            assert set(data) == set(TASKS)
            tasks = {}
            for task in TASKS:
                item = data[task]
                records = item['records']
                assert item['complete'] and item['episodes'] == item['requested_episodes'] == len(records) == 300
                assert sum(bool(r['success']) for r in records) == item['successes']
                assert abs(item['success_rate']-item['successes']/300) < 1e-12
                assert dict(Counter(r['reason'] for r in records)) == item['termination_counts']
                metadata = {k: item[k] for k in ('stage', 'seed', 'camera', 'success_metric',
                            'success_definition', 'evaluation_revision', 'policy_role')}
                if task in contract:
                    assert contract[task] == metadata
                contract[task] = metadata
                tasks[task] = {k: item[k] for k in ('successes', 'episodes', 'success_rate', 'wilson_95')}
                if task == 'roll':
                    tasks[task]['stable_hold_count'] = sum(bool(r['diagnostics'].get('roll_stable_hold')) for r in records)
                n_records += len(records)
            rates = [tasks[t]['success_rate'] for t in TASKS]
            points[model][iteration] = dict(iteration=iteration, tasks=tasks,
                mean=sum(rates)/3, minimum=min(rates), total_successes=sum(tasks[t]['successes'] for t in TASKS))
            hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert set(points[model]) == set(STEPS[:3] if model == 'contact' else STEPS)
    assert n_records == 20700
    comparisons = {it: {t: difference(points['belief'][it]['tasks'][t],
                                     points['camera_gate'][it]['tasks'][t]) for t in TASKS} for it in STEPS}
    changes = {}
    for model in MODELS:
        keys = sorted(points[model])
        changes[model] = [dict(start=a, end=b,
            mean_change=points[model][b]['mean']-points[model][a]['mean'],
            tasks={t: difference(points[model][b]['tasks'][t], points[model][a]['tasks'][t]) for t in TASKS})
            for a,b in zip(keys, keys[1:])]
    best = {m: max(points[m].values(), key=lambda p: p['minimum'])['iteration'] for m in MODELS}
    (OUTPUT/'summary.json').write_text(json.dumps(dict(points=points, contracts=contract,
        records_validated=n_records, source_sha256=hashes,
        belief_minus_camera_gate=comparisons, changes=changes,
        best_minimum_among_exported_points=best,
        best_checkpoint_caveat='Conditional on all these points participating in one minimum-task comparison without resume best-score resets; actual student_best requires checkpoint/audit.',
        interval_assumptions='Approximately independent Bernoulli episodes; unpaired exploratory intervals; no training-seed uncertainty or multiplicity correction.'), indent=2, ensure_ascii=False))
    write_report(points, comparisons, changes, best)
    plot_all(points)
    plot_focus(points)
    print('Validated:', len(hashes), 'files;', n_records, 'records')
    print('Conditional min-task optimum:', best)


def pct(x):
    return f'{100*x:.2f}%'


def write_report(points, comparisons, changes, best):
    lines = ['# 2000～10000轮：五组视觉辅助蒸馏曲线分析', '', '日期：2026-10-29。', '',
        '已核对23份JSON、20700个逐回合记录。原始文件未修改。四组有5个评估点，Contact仅有2000/4000/6000。',
        '所有文件的可见条件相同：Catch视觉学生、seed=123、每任务300回合、Top/Wrist双深度160×160，',
        'success metric/definition、evaluation_revision、camera完整字典均一致。五个末点与之前eval_export中的内容相同。', '',
        '## 关键结论：最终点的优势不是持续优势', '',
        'Camera Gate在前四个Throw评估点都高于Belief，仅到10000轮反转。Bounce领先关系也发生明显反转。',
        '因此，先前“Belief可能更适合Throw、门控可能更适合Bounce”的解释仍只能作为机制假设；',
        '这些曲线不支持把最终轮优势直接推广成架构的固定特长。', '',
        '| 轮次 | Belief Throw | Gate Throw | Belief Bounce | Gate Bounce |',
        '|---|---:|---:|---:|---:|']
    for it in STEPS:
        vals = [points[m][it]['tasks'][t]['success_rate'] for t in ('throw', 'bounce') for m in ('belief', 'camera_gate')]
        lines.append(f"| {it} | {' | '.join(map(pct, vals))} |")
    lines += ['', '## 三任务均值', '', '| 模型 | 2000 | 4000 | 6000 | 8000 | 10000 |',
              '|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        cells = [pct(points[m][it]['mean']) if it in points[m] else '—' for it in STEPS]
        lines.append(f"| {m} | {' | '.join(cells)} |")
    lines += ['',
        '- Belief均值随这些离散点逐步上升，但内部任务不单调；8k→10k的Throw上升伴随Bounce下降。',
        '- Camera Gate均值也在6k→8k回落，8k→10k的Bounce大增伴随Throw/Roll下降。',
        '- Future在6k→8k整体回落10.33个百分点，随后恢复；不能据此认为训练轮数越多越好。',
        '- Spatial的Throw从4k的59.00%降到6k的43.00%，后续又恢复，说明非单调现象并非只在Belief/Gate出现。', '',
        '## 300回合能解释多少波动', '',
        '单模型300回合、成功率约50%～80%时，95%评估抽样区间的半宽通常约4～6个百分点；',
        '两点差异还需考虑两份估计的不确定性，不能直接拿某个单点半宽当差异阈值。',
        '以下从原JSON Wilson区间计算Newcombe差异区间，正值代表Belief优于Gate。', '',
        '| 轮次 | Throw差异（95%区间，百分点） | Bounce差异（95%区间，百分点） |',
        '|---|---:|---:|']
    for it in STEPS:
        cells = [f"{100*comparisons[it][t]['estimate']:+.2f} [{100*comparisons[it][t]['lower']:+.2f}, {100*comparisons[it][t]['upper']:+.2f}]"
                 for t in ('throw', 'bounce')]
        lines.append(f"| {it} | {' | '.join(cells)} |")
    lines += ['', '8k→10k的主要变化：', '',
        '| 模型/任务 | 变化（百分点） | 约95%变化区间 |', '|---|---:|---:|']
    for m in ('belief', 'camera_gate', 'future'):
        row = next(x for x in changes[m] if x['start'] == 8000)
        for t in (('throw', 'bounce') if m != 'future' else ('bounce',)):
            x = row['tasks'][t]
            lines.append(f"| {m} / {t} | {100*x['estimate']:+.2f} | [{100*x['lower']:+.2f}, {100*x['upper']:+.2f}] |")
    lines += ['',
        '这些较大跃变难以只由300回合评估抽样误差解释，但不等于已经确定负迁移、过拟合或门控饱和的原因。',
        '区间是假设回合近似独立的探索性非配对比较，没有包含训练seed差异、源码改变和多重比较影响。',
        '原worker每回合显式重设seed，终止时间不同本身不会导致后续reset随机序列漂移。旧导出缺少初始状态记录，仍需核验源码和环境构造参数，不能仅凭相同seed认定严格配对；新的多种子评估已记录并核验初始状态，见[新报告](../../analysis_artifacts/distill_best_multiseed_20261029_v1/report/REPORT.md)。',
        '同一训练运行的5个checkpoint也不是5个独立训练seed，不能将其简单池化为1500个独立重复来证明架构优势。', '',
        '## 同轮次比较：6000轮', '',
        '| 模型 | Throw | Roll | Bounce | 均值 | 最低任务 |', '|---|---:|---:|---:|---:|---:|']
    for m in sorted(MODELS, key=lambda m: points[m][6000]['mean'], reverse=True):
        p = points[m][6000]
        vals = [p['tasks'][t]['success_rate'] for t in TASKS] + [p['mean'], p['minimum']]
        lines.append(f"| {m} | {' | '.join(map(pct, vals))} |")
    lines += ['',
        'Contact在共同6000轮点的均值和最差任务均最高；与Future均值仅差0.56个百分点，不能据此宣称稳定优胜。',
        '其Roll从4k的42.67%升至6k的67.67%，但还缺8k/10k，是否能保持均衡未知。', '',
        '## 最终模型不一定是best模型', '',
        '当前源码在skip_teacher_baseline=true时按三个任务中最低成功率选择student_best，而不是平均值。',
        '如果这些点都参加同一比较且best_score从未在恢复时重置，则导出点中的最优点如下。', '',
        '| 模型 | 按最低任务最优的轮次 | 最低任务成功率 | 同点均值 |', '|---|---:|---:|---:|']
    for m in MODELS:
        it = best[m]
        lines.append(f"| {m} | {it} | {pct(points[m][it]['minimum'])} | {pct(points[m][it]['mean'])} |")
    lines += ['',
        'Future的6k与10k最低任务只差1/300个成功，属于非常小的差异。',
        'Camera Gate的8k最低59.33%，10k为55.67%，但10k平均值更高，体现选择目标不同。',
        '**重要限制：**用户曾多次带--allow-source-changes恢复。该flag本身不必然重置best，',
        '但若实际接受了源码变化，当前validate_resume会重置best_score；改变评估回合数/跳过教师选项也可能重置。',
        '因此上表是条件性推算，不能替代读取student_best.pth的iteration及resume_audit.json。', '',
        '## 原因：能确认的现象与待验证假设', '',
        '能确认：后期任务成功率发生反向变化；多个结构存在非单调曲线。',
        '候选原因包括共享参数中的任务梯度取舍、优化阶段变化、在线采样分布改变及恢复时接受的源码变化。',
        '当前本地默认每任务回放容量8192帧、每轮8次更新；在线学生状态分布会随策略变化，',
        '但这些本地默认值不能替代服务器resolved_config，不能据此直接诊断某次退步。',
        'Belief/Gate都有GRU。前者额外预测视觉特征并按置信度融合，后者为双相机softmax加权；',
        '差别可能影响不同任务，但没有逐任务门控权重、置信度、prior误差日志，不能从曲线确定是哪一路相机或历史机制导致。', '',
        '## Roll质量提醒', '',
        '主Roll指标仍为region_retention_v5。各点roll_stable_hold仅0或1/300，五组最终点均0/300。',
        '区域保持成功不等价于低速稳定抓握成功，后续仍应结合视频和稳定保持诊断。', '',
        '## 建议的下一步', '',
        '1. 如仍保留8k/10k权重，用多个独立评估seed重测Belief/Gate，确认反转是否复现。',
        '   若只保留best/last，先固定这些现有权重重测；仅有评估JSON无法恢复该轮模型权重。',
        '   重点模型/点可增加到每任务1000回合，300→1000时抽样标准误约降为原来的0.55倍。',
        '2. 比较last、按最低任务选择的best，以及预先约定的均值选择规则，避免看完结果再挑单任务峰值。',
        '3. 补齐Contact至10k，保持6000共同点的公平比较；多个训练seed优先验证Belief/Future/Contact。',
        '4. 提供各运行resolved_config、resume_audit和metrics以排查退步是否跨越恢复/源码变更。',
        '5. 若要解释机制，再按任务和反弹前后记录门控权重、Belief置信度/预测误差和失败视频。', '',
        '## 图与数据', '',
        '![五模型曲线](all_learning_curves.png)', '', '![Belief与门控](belief_camera_gate_curves.png)', '',
        '误差棒为单任务95% Wilson区间；均值图不绘制混合指标的二项分布区间。直线只连接离散评估点，不做平滑。',
        'Contact不补8k/10k。源文件哈希和完整分项数据见summary.json。', '']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines))


def plot_all(points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, task in zip(axes.flat, TASKS+('mean',)):
        for m in MODELS:
            x = sorted(points[m])
            y = np.array([100*(points[m][it]['mean'] if task == 'mean' else points[m][it]['tasks'][task]['success_rate']) for it in x])
            ax.plot(np.array(x)/1000, y, '-o', color=COLORS[m], label=m.replace('_', ' ').title(), linewidth=1.8, markersize=4)
            if task != 'mean':
                lo = np.array([100*points[m][it]['tasks'][task]['wilson_95'][0] for it in x])
                hi = np.array([100*points[m][it]['tasks'][task]['wilson_95'][1] for it in x])
                ax.errorbar(np.array(x)/1000, y, yerr=[y-lo, hi-y], fmt='none', color=COLORS[m],
                            elinewidth=.7, capsize=2, alpha=.55)
        style(ax, 'Three-task mean' if task == 'mean' else task.title())
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False)
    fig.suptitle('Auxiliary distillation: checkpoint learning curves', fontsize=16, fontweight='bold', y=.985)
    fig.text(.5, .018, '300 episodes/task, seed 123. Task error bars: 95% Wilson CI. Contact ends at 6k; no smoothing or extrapolation.',
             ha='center', fontsize=9, color='#4c5963')
    fig.tight_layout(rect=[0,.04,1,.90])
    fig.savefig(OUTPUT/'all_learning_curves.png', dpi=180)
    fig.savefig(OUTPUT/'all_learning_curves.svg')
    plt.close(fig)


def style(ax, title):
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks([2,4,6,8,10])
    ax.set_xlim(1.65,10.35)
    ax.set_ylim(0,100)
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_xlabel('Training iteration (thousands)')
    ax.set_ylabel('Success rate (%)')
    ax.grid(alpha=.22)
    ax.spines[['top','right']].set_visible(False)


def plot_focus(points):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for ax, task in zip(axes, ('throw', 'bounce')):
        for m, marker in [('belief','o'), ('camera_gate','s')]:
            x = np.array(STEPS)/1000
            y = np.array([100*points[m][it]['tasks'][task]['success_rate'] for it in STEPS])
            lo = np.array([100*points[m][it]['tasks'][task]['wilson_95'][0] for it in STEPS])
            hi = np.array([100*points[m][it]['tasks'][task]['wilson_95'][1] for it in STEPS])
            ax.errorbar(x, y, yerr=[y-lo,hi-y], color=COLORS[m], marker=marker,
                        label=m.replace('_',' ').title(), linewidth=2, capsize=3, markersize=5)
            other = 'camera_gate' if m == 'belief' else 'belief'
            for index, (xx, yy, it) in enumerate(zip(x,y,STEPS)):
                higher = yy > 100*points[other][it]['tasks'][task]['success_rate']
                label_y = hi[index]+2 if higher else lo[index]-2
                ax.text(xx,label_y,f'{yy:.1f}',ha='center',
                        va='bottom' if higher else 'top',fontsize=9,color=COLORS[m])
        style(ax, task.title())
        ax.axvspan(8,10, color='#f4e5be', alpha=.30)
        ax.legend(loc='lower right', frameon=False)
    fig.suptitle('Belief vs Camera Gate: the final advantage is not persistent', fontsize=14, fontweight='bold', y=.99)
    fig.text(.5,.018,'Points: observed evaluations; whiskers: 95% Wilson CI. Shaded: 8k to 10k, where the ranking reverses.',
             ha='center',fontsize=9,color='#4c5963')
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(OUTPUT/'belief_camera_gate_curves.png',dpi=180)
    fig.savefig(OUTPUT/'belief_camera_gate_curves.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Read-only diagnosis of Contact's 8k-to-10k regression; no model evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'catch_it_copy/distillation_aux_contact/runs/catch_10000_seed123'
OUT = ROOT/'analysis_artifacts/contact_training_diagnosis_20261031'
TASKS = ('throw', 'roll', 'bounce')
LOSSES = ('loss', 'imitation_loss', 'position_loss', 'velocity_loss', 'module_loss')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(run):
    config = json.loads((run/'resolved_config.json').read_text())
    audit = json.loads((run/'resume_audit.json').read_text())
    raw = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    jumps = [dict(line=i+2, previous=a['iteration'], resumed_at=b['iteration'])
             for i, (a,b) in enumerate(zip(raw,raw[1:])) if b['iteration'] != a['iteration']+1]
    # Repeated early iterations are from rewound runs. Retain final occurrence.
    by_iteration = {row['iteration']: row for row in raw}
    ordered = [by_iteration[i] for i in sorted(by_iteration)]
    if [row['iteration'] for row in ordered] != list(range(1,10001)):
        raise RuntimeError('Expected complete iteration range 1..10000')
    if not all(math.isfinite(row[key]) for row in raw for key in LOSSES):
        raise RuntimeError('Non-finite logged loss')
    evaluations = {}
    fingerprints = {}
    visible_fields = ('success_metric','success_definition','evaluation_revision','seed','stage','camera','episodes')
    reference = None
    for iteration in (2000,4000,6000,8000,10000):
        path = run/('eval_%06d.json' % iteration)
        data = json.loads(path.read_text())
        entries = {}
        visible = {task:{key:data[task].get(key) for key in visible_fields} for task in TASKS}
        if reference is None:
            reference = visible
        if visible != reference:
            raise RuntimeError('Visible evaluation parameters differ')
        for task in TASKS:
            value = data[task]
            rows = value['records']
            if len(rows) != 300 or sum(row['success'] for row in rows) != value['successes']:
                raise RuntimeError('Evaluation counts disagree')
            entry = {key:value[key] for key in ('successes','episodes','success_rate','wilson_95','termination_counts','mean_reward')}
            if task == 'roll':
                entry['retention_counts'] = {key:sum(bool(row['diagnostics'].get(key)) for row in rows)
                    for key in ('roll_caught_once','roll_holding_at_end','roll_stable_hold')}
                unmet = Counter()
                for row in rows:
                    unmet.update(term for term in row['diagnostics'].get('roll_eval_unmet','').split(',') if term)
                entry['unmet_counts'] = dict(unmet)
            entries[task] = entry
        minimum = min(value['success_rate'] for value in entries.values())
        if abs(by_iteration[iteration]['min_task_success']-minimum) > 1e-12:
            raise RuntimeError('Training metrics do not match evaluation score')
        evaluations[str(iteration)] = dict(tasks=entries,
            mean_success_rate=statistics.mean(value['success_rate'] for value in entries.values()),
            minimum_success_rate=minimum)
        fingerprints[str(path)] = sha(path)
    windows = {}
    for start, end in ((5001,6000),(7001,8000),(8001,9000),(9001,10000),(7801,8000),(9801,10000)):
        selected = [row for row in ordered if start <= row['iteration'] <= end]
        windows['{}-{}'.format(start,end)] = {key:dict(mean=statistics.mean(row[key] for row in selected),
            sd=statistics.stdev(row[key] for row in selected)) for key in LOSSES}
    for name in ('metrics.jsonl','resolved_config.json','resume_audit.json','teacher_check.json'):
        fingerprints[str(run/name)] = sha(run/name)
    return config, audit, raw, ordered, jumps, evaluations, windows, fingerprints, reference


def inspect_students(run):
    import torch
    students = {}
    saved = {}
    for label in ('best','last'):
        path = run/('student_'+label+'.pth')
        value = torch.load(path, map_location='cpu', weights_only=True)
        saved[label] = value
        students[label] = dict(path=str(path), sha256=sha(path), iteration=value['iteration'],
                              schema=value['schema'], all_weights_finite=all(torch.isfinite(v).all().item() for v in value['student'].values()))
    return dict(checkpoints=students,
                config_equal=saved['best']['config']==saved['last']['config'],
                source_manifest_equal=saved['best']['metadata']['source_asset_sha256']==saved['last']['metadata']['source_asset_sha256'],
                teacher_hashes_equal=saved['best']['metadata']['teacher_hashes']==saved['last']['metadata']['teacher_hashes'])


def make_plot(rows, evaluations, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3,figsize=(16,4.6))
    iterations = [int(key) for key in evaluations]
    for task, color in zip(TASKS,('#3274A1','#DD8452','#55A868')):
        axes[0].plot(iterations,[100*evaluations[str(i)]['tasks'][task]['success_rate'] for i in iterations],
                     marker='o',label=task.title(),color=color)
    axes[0].plot(iterations,[100*evaluations[str(i)]['mean_success_rate'] for i in iterations],
                 '--',marker='s',color='#555555',label='3-task mean')
    axes[0].set(title='Evaluation: 300 episodes per task',xlabel='Iteration',ylabel='Success rate (%)',ylim=(0,100))
    axes[0].legend(fontsize=8,loc='lower right')
    x=np.array([row['iteration'] for row in rows])
    for panel, names, title in [(axes[1],('loss','imitation_loss'),'Total / imitation: rolling 200-iteration mean'),
                                (axes[2],('module_loss',),'Contact loss: rolling 200-iteration mean')]:
        for name in names:
            y=np.array([row[name] for row in rows])
            smoothed=np.convolve(y,np.ones(200)/200,mode='valid')
            visible=x[199:] >= 5000
            panel.plot(x[199:][visible],smoothed[visible],label=name)
        panel.set(title=title,xlabel='Iteration',ylabel='Logged loss',xlim=(5000,10000))
        panel.axvline(8000,color='#888888',ls=':',lw=1)
        panel.legend(fontsize=8)
    for panel in axes:
        panel.grid(alpha=.2)
    fig.suptitle('Contact: training losses improve while closed-loop success regresses',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ('png','svg'):
        fig.savefig(output/('training_and_success.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def make_report(data, output):
    evaluations=data['evaluations'];old=evaluations['8000'];new=evaluations['10000']
    window_old=data['loss_windows']['7801-8000'];window_new=data['loss_windows']['9801-10000']
    lines=['# Contact 后期训练退化分析（2026-10-31）','',
        '可确认的现象是：训练代理损失继续缓慢下降，但闭环成功率下降。没有日志证据显示损失数值发散；现有数据尚不能把退化归因于某一个模块或确定为传统意义的过拟合。','',
        '## 全部评估点','', '|轮次|Throw|Roll|Bounce|三任务平均|最低任务成功率|','|---:|---:|---:|---:|---:|---:|']
    for key, entry in evaluations.items():
        rates=[100*entry['tasks'][task]['success_rate'] for task in TASKS]
        lines.append('|{}|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|'.format(key,*rates,100*entry['mean_success_rate'],100*entry['minimum_success_rate']))
    lines += ['', '6000轮的最弱任务成功率最高，因此student_best保存在6000轮；8000轮的三任务平均最高；10000轮是last，不能代表峰值表现。','',
        '## 训练损失没有随成功率一起恶化','',
        '下面对比评估前200轮的平均值，避免用单轮随机波动判断趋势。训练数据来自不断变化的在线回放，这不是固定验证集上的误差。','',
        '|指标|7801–8000平均|9801–10000平均|','|---|---:|---:|']
    for name in LOSSES:
        lines.append('|{}|{:.8f}|{:.8f}|'.format(name,window_old[name]['mean'],window_new[name]['mean']))
    lines += ['', '10000个唯一轮次的所有已记录损失均有限；总损失、模仿损失和Contact模块损失的窗口均值都略有下降。不能用这些指标宣称闭环能力持续改善，也没有证据把这次下降称为loss爆炸。速度辅助权重为0，日志中的velocity_loss未参与加权训练目标。','',
        '## 后期更新强度和回放窗口','',
        '- 学习率配置为3e-4，训练实现没有学习率调度器。',
        '- 8000到10000轮又执行2000×8＝16000次optimizer.step；每任务计划采样2000×8×64＝1024000个环境控制步。',
        '- 每任务回放容量8192帧，每轮计划新增8×64＝512帧，容量约为16轮采样量。实际淘汰按完整回合进行，这只是量级估算。',
        '- 因而训练一直拟合近期策略产生的轨迹，8000轮附近的旧轨迹不会长期保留。策略漂移、旧行为被覆盖是有实现依据的候选机制，尚未通过对照实验确认。',
        '- Contact预测通过缩放/平移直接调制动作头特征；接触分类损失下降不保证保持球的动作更好。日志没有各任务动作误差、接触正负样本比例、预测准确率或梯度分解，无法证明模块本身是根因。','',
        '## Roll和Bounce的具体退化','',
        '|Roll诊断|8000轮|10000轮|','|---|---:|---:|']
    for key in ('roll_caught_once','roll_holding_at_end','roll_stable_hold'):
        lines.append('|{}|{}|{}|'.format(key,old['tasks']['roll']['retention_counts'][key],new['tasks']['roll']['retention_counts'][key]))
    lines += ['|below_palm|{}|{}|'.format(old['tasks']['roll']['unmet_counts'].get('below_palm',0),new['tasks']['roll']['unmet_counts'].get('below_palm',0)),'',
        '曾进入保持区域的次数未下降，但结束时保持住球的次数减少，球落到手掌下方的回合增加。这里是region_retention_v5区域保持，不能等同于严格稳定抓持。',
        'Bounce的ball_left失败从{}次增加到{}次。'.format(old['tasks']['bounce']['termination_counts'].get('ball_left',0),new['tasks']['bounce']['termination_counts'].get('ball_left',0)),'',
        '## 续训与文件一致性','',
        'metrics共有10497行，唯一轮次为1–10000。多出的497行是早期重复轮次；本分析按每个轮次最后一次记录构造曲线，没有改原日志。可识别的回退如下：','',
        '|日志行|上一轮|回退后的第一轮|','|---:|---:|---:|']
    for jump in data['resume_jumps']:
        lines.append('|{}|{}|{}|'.format(jump['line'],jump['previous'],jump['resumed_at']))
    lines += ['',
        '8000–10000轮编号连续，没有类似回退记录；这不能排除恰好从最新保存点继续的重启。resume_audit记录接受了{}个源码路径变化，old_config与new_config完全相同。该审计未记录续训起点轮次和时间，不能据此定位后期退化。'.format(data['resume_audit']['changed_source_count']),
        '现存best（6000）与last（10000）的配置、教师哈希、源码清单哈希均相同，权重全部有限。检查点中的源码清单是在训练入口建立的，不是逐轮实时追踪磁盘修改。',
        '五份评估JSON中可见的种子、相机、任务判据及评估版本一致；8000轮没有对应的已保存权重或独立源码审计，因此无法仅凭这些文件完全证明所有运行条件相同。','',
        'resume_last.pth在首次检查时仍在复制，文件大小持续增长，因而当时的读取失败不能当作训练损坏的证据。','']
    inspection=data.get('resume_inspection')
    if inspection:
        lines += ['复制完成后已使用torch.load（CPU、weights_only=True）成功读取完整resume：iteration={}；Adam参数组学习率={}；累计step={}；网络权重、配置和元数据均与student_last相同。best_score={}，与6000轮最弱任务成功率一致。'.format(
            inspection['iteration'],inspection['optimizer_groups'][0]['lr'],inspection['optimizer_step_values'],inspection['best_score']),'',
            '|任务|回放回合数|回放帧数|物体—手部接触帧比例|','|---|---:|---:|---:|']
        for task,value in inspection['replay'].items():
            lines.append('|{}|{}|{}|{:.2f}%|'.format(task.title(),value['episodes'],value['frames'],100*value['contact_frame_fraction']))
        lines += ['', 'Bounce末态回放中约48%的帧有物体—手部接触，三个接触预测目标的正样本比例约52%、47%、43%。因此，末态数据并非几乎没有接触正样本；但这些是标签统计，不等同于预测准确率。该回放来自训练采样环境，与seed=123的单环境评估也不是同一组轨迹。完整末态核验见[resume_inspection.json](resume_inspection.json)。','']
    else:
        lines += ['本次报告未读取其优化器/回放末态。','']
    lines += ['## 现阶段的结论与后续检查','',
        '优先保留现有student_best（6000轮）；8000轮评估也值得保留，但此目录仅有eval_008000.json，没有8000轮权重，不能从评估JSON恢复模型。',
        '目前更准确的诊断是“近期数据上的代理损失与闭环性能脱节”，而不是已经证明过拟合、灾难性遗忘或Contact模块有bug。后续实验优先检查后期学习率衰减、近期回放对旧成功轨迹的覆盖，以及Contact预测调制前后的动作差；一次只改变一个因素。',
        '如要严格定位，需固定回放验证集并分任务记录动作误差、接触分类的正负样本比例与准确率，或对同一检查点关闭Contact调制做独立诊断。上述实验本次均未运行。','',
        '![训练损失与闭环成功率](training_and_success.png)','',
        '完整数值、输入文件SHA256和检查点核验见[summary.json](summary.json)。']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    config,audit,raw,ordered,jumps,evals,windows,fingerprints,visible=load_data(args.run)
    students=inspect_students(args.run)
    resume=args.run/'resume_last.pth'
    data=dict(raw_metric_rows=len(raw),unique_iterations=len(ordered),duplicate_metric_rows=len(raw)-len(ordered),
              resume_jumps=jumps,resume_audit=dict(config_unchanged=audit['old_config']==audit['new_config'],
              source_changes_accepted=audit['source_changes_accepted'],changed_source_count=len(audit['changed_sources']),
              changed_sources=audit['changed_sources']),
              evaluations=evals,loss_windows=windows,visible_eval_metadata=visible,students=students,
              config=config,all_logged_losses_finite=True,input_sha256=fingerprints,
              resume_file=dict(path=str(resume),observed_bytes=resume.stat().st_size,
              earlier_growing_file_observations_bytes=[738197504,831782912,838598656,845938688],
              inspection='not loaded: file was growing during inspection; do not infer corruption'),
              scope='read-only log analysis; no training, GPU policy evaluation or ablation',
              analysis_script_sha256=sha(__file__))
    inspection_path=args.output/'resume_inspection.json'
    if inspection_path.exists():
        inspection=json.loads(inspection_path.read_text())
        if inspection['file_bytes']!=resume.stat().st_size or not inspection['student_equal_to_last']:
            raise RuntimeError('Resume inspection does not match the current run')
        data['resume_inspection']=inspection
        data['resume_file']['inspection']='Successfully loaded after transfer completed; optimizer/replay inspected on CPU'
        data['input_sha256'][str(resume)]=inspection['sha256']
        data['input_sha256'][str(inspection_path)]=sha(inspection_path)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    make_report(data,args.output)
    make_plot(ordered,evals,args.output)
    print(json.dumps(dict(report=str(args.output/'REPORT.md'),metrics=len(raw),unique_iterations=len(ordered)),ensure_ascii=False))


if __name__ == '__main__':
    main()

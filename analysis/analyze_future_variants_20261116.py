"""Read-only analysis of exported Future experiments; no simulation."""
import json
from pathlib import Path
import hashlib
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_artifacts/future_variants_analysis_20261116'
FOLDERS={'stable':ROOT/'catch_it_copy/distillation_aux_future_stable/runs/catch_10000_seed123',
         'uncertain':ROOT/'catch_it_copy/distillation_aux_future_uncertain/catch_10000_seed123'}
TASKS=('throw','roll','bounce')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={}
    sources={}
    for name,p in FOLDERS.items():
        configs=json.loads((p/'resolved_config.json').read_text())
        evaluations=[]
        for f in sorted(p.glob('eval_*.json')):
            data=json.loads(f.read_text());tasks={}
            for t in TASKS:
                r=data[t];assert r['episodes']==300
                assert abs(r['successes']/r['episodes']-r['success_rate'])<1e-12
                tasks[t]={k:r[k] for k in ('successes','episodes','success_rate')}
                tasks[t]['seed']=r.get('seed')
            rates=[tasks[t]['success_rate'] for t in TASKS]
            evaluations.append(dict(iteration=int(f.stem.split('_')[1]),tasks=tasks,mean=sum(rates)/3,minimum=min(rates)))
        weights={}
        for f in ('student_best.pth','student_last.pth'):
            s=torch.load(p/f,map_location='cpu',weights_only=True)
            assert all(torch.isfinite(v).all() for v in s['student'].values())
            weights[f]=dict(iteration=s['iteration'],schema=s['schema'],sha256=hashlib.sha256((p/f).read_bytes()).hexdigest())
            sources[name]=s['metadata']['source_asset_sha256']
        rows={};raw=0
        for line in (p/'metrics.jsonl').read_text().splitlines():
            r=json.loads(line);rows[r['iteration']]=r;raw+=1
        window=[]
        for end in (5000,6000,7000,8000,9000,10000):
            batch=[r for i,r in rows.items() if end-199<=i<=end]
            window.append(dict(end=end,count=len(batch),**{k:sum(r[k] for r in batch)/len(batch) for k in ('loss','imitation_loss','module_loss','learning_rate')}))
        summary[name]=dict(folder=str(p),config=configs,evaluations=evaluations,checkpoints=weights,
                           highest_mean=max(evaluations,key=lambda e:e['mean']),loss_windows=window,
                           log_rows=raw,unique_iterations=len(rows))
    shared=set(sources['stable'])&set(sources['uncertain'])
    summary['source_comparison']=dict(shared_files=len(shared),changed_shared_files=[k for k in sorted(shared) if sources['stable'][k]!=sources['uncertain'][k]])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,4,figsize=(15,3.7),sharey=True)
    for ax,t in zip(axes,(*TASKS,'mean')):
        for name,color in [('stable','#1976d2'),('uncertain','#e67e22')]:
            rows=summary[name]['evaluations']
            ys=[r['mean'] if t=='mean' else r['tasks'][t]['success_rate'] for r in rows]
            ax.plot([r['iteration']/1000 for r in rows],[100*y for y in ys],'-o',color=color,label=name)
        ax.set_title(t.capitalize());ax.set_xticks([5,6,7,8,9,10]);ax.set_xlabel('Training iteration (1000s)');ax.grid(alpha=.25);ax.set_ylim(40,90)
    axes[0].set_ylabel('Success rate (%)');axes[-1].legend()
    fig.suptitle('Future variants: 300 episodes/task/checkpoint, evaluation seed 20261110')
    fig.tight_layout();fig.savefig(OUT/'success_curves.png',dpi=180);fig.savefig(OUT/'success_curves.svg');plt.close(fig)
    lines=['# Future variants analysis — 2026-11-16','','每个检查点评估每任务300回合，固定评估种子20261110。此报告读取已有结果，没有启动模拟。','',
           '|版本|轮次|Throw|Roll|Bounce|平均|最低任务|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('stable','uncertain'):
        for r in summary[name]['evaluations']:
            vals=[r['tasks'][t]['success_rate'] for t in TASKS]+[r['mean'],r['minimum']]
            lines.append('|'+name+'|'+str(r['iteration'])+'|'+'|'.join(f'{100*v:.2f}%' for v in vals)+'|')
    lines+=['','## 模型选择','',
            '- Stable: student_best=9000轮，平均75.00%，最低69.67%；last=10000轮，平均72.22%，最低64.67%。',
            '- Uncertain: student_best和last均为10000轮，平均68.22%，最低61.00%。8000轮平均68.56%略高，但最低任务只有57%。',
            '- Best按三个任务最低成功率选择，而非平均成功率。',
            '', '## 解释与限制','',
            '- 稳定版9000→10000：Throw下降5.00个百分点、Roll下降4.00、Bounce上升0.67。不存在所有任务同步改善。',
            '- 不确定性版8000→9000：Bounce下降20.67个百分点，10000恢复到76.33%，是波动而非单调退化。',
            '- 两组公共源码hash一致；配置主要不同为架构及学习率策略。每组只有一个训练种子，不能将差异单独归因于学习率或不确定性。',
            '- 不确定性版负NLL可为负，总loss不能与稳定版SmoothL1直接比较。',
            '- 最后一轮的replay训练批诊断：增强版预测RMSE更低并不表示闭环控制更好；不同策略采集的状态分布也不同。',
            '- 原Future多种子1500回合结果：Throw73.20%、Roll65.07%、Bounce80.53%、平均72.93%。与当前300回合不同协议，不能据此宣布稳定版显著超过原版。',
            '- 300回合在这些成功率附近单任务95%抽样误差量级约±4–6个百分点；固定验证集被重复用于选模型，后续应使用新种子独立复测。',
            '- 同一种子不等于已经核实逐回合初始状态完全一致；本次旧JSON无初始状态hash，不计算严格配对显著性。',
            '- Roll仍采用接触后区域保留成功标准，并非稳定抓握。',
            '', '下一步优先独立复测Stable的9000与10000轮及原Future10000轮，保持相同环境、种子、并行数量、推理batch和成功判据。',
            '', '![curves](success_curves.png)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report:',OUT/'REPORT.md')
    print('Shared source differences:',summary['source_comparison']['changed_shared_files'])


if __name__=='__main__':main()

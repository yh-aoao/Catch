"""Publish the completed four-task table and paired Bounce scene comparison."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis_artifacts/throw_bounce_model_comparison_20261031_v1'
PREVIOUS = ROOT/'analysis_artifacts/three_task_three_ball_success_20261031_v1'
BALLS = ('original', 'tennis_ball', 'baseball')


def read(folder):
    assert (folder/'COMPLETE').is_file(), str(folder)
    rows = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 100 and [r['seed'] for r in rows] == list(range(100))
    s = json.loads((folder/'summary.json').read_text())
    p = json.loads((folder/'provenance.json').read_text())
    assert s['successes'] == sum(r['success'] for r in rows)
    assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest() == p['checkpoint_sha256']
    return s, p, rows


table = []
folders = {}
for task, model in [('throw','497.09'),('roll','252.54'),('bounce','464.31'),('basket','38.38')]:
    for ball in BALLS:
        folder = (OUT/('throw497_'+ball) if task == 'throw' else
                  PREVIOUS/'roll_metric_correction'/('roll_'+ball) if task == 'roll' else
                  PREVIOUS/(task+'_'+ball))
        s, p, rows = read(folder)
        folders[task+'_'+ball] = folder
        table.append(dict(task=task, model=model, ball=ball, episodes=100,
            successes=s['successes'], success_rate=s['success_rate'],
            hand_contact_episodes=s['hand_contact_episodes'], contact_0p3s=s['contact_0p3s'],
            criterion=p['criterion']['name'], checkpoint=p['checkpoint'],
            source=str(folder), ci95_low=s['wilson_95_interval'][0],ci95_high=s['wilson_95_interval'][1]))

controls = []
for model, scene, ball, folder in [
    ('653.87','legacy','original',OUT/'bounce653_legacy_original'),
    ('464.31','legacy','original',OUT/'bounce464_legacy_original'),
    *[('653.87','physical_v2',b,OUT/('bounce653_physical_'+b)) for b in BALLS],
    *[('464.31','physical_v2',b,folders['bounce_'+b]) for b in BALLS]]:
    s,p,rows = read(folder)
    controls.append(dict(model=model,scene=scene,ball=ball,successes=s['successes'],episodes=100,
        hand_contact_episodes=s['hand_contact_episodes'],contact_0p3s=s['contact_0p3s'],
        source=str(folder),criterion=p['criterion']['name'],checkpoint_sha256=p['checkpoint_sha256']))
paired = {}
for scene,old,new in [('legacy',OUT/'bounce653_legacy_original',OUT/'bounce464_legacy_original'),
                     ('physical_v2',OUT/'bounce653_physical_original',folders['bounce_original'])]:
    _,_,a = read(old);_,_,b = read(new)
    paired[scene] = dict(episodes=100, identical_initial_states=all(x['initial']==y['initial'] for x,y in zip(a,b)))
    assert paired[scene]['identical_initial_states']

for filename, records in [('four_task_success.csv',table),('bounce_model_scene_comparison.csv',controls)]:
    with (OUT/filename).open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
(OUT/'summary_all.json').write_text(json.dumps(dict(four_tasks=table,bounce_controls=controls,paired_initial_states=paired),indent=2))
(OUT/'paired_initial_states.json').write_text(json.dumps(paired,indent=2))

lookup = {(r['task'],r['ball']):r for r in table}
c = {(r['model'],r['scene'],r['ball']):r for r in controls}
lines=['# Throw三球补测与Bounce新旧模型对照（2026-11-01）','',
    '每个组合100局，seed0–99，真实权重及归一化、原PPO确定性推理。Throw与新增Bounce对照共800局；Roll、当前Bounce、Basket复用前次已完成且有来源记录的100局结果。没有训练或修改环境物理/奖励/成功条件。','',
    '| 任务与模型 | 原始球 | 网球 | 棒球 |','| --- | ---: | ---: | ---: |']
for task,model in [('throw','497.09'),('roll','252.54'),('bounce','464.31'),('basket','38.38')]:
    lines.append('| '+task.title()+' '+model+' | '+' | '.join(str(lookup[(task,b)]['successes'])+'%' for b in BALLS)+' |')
lines+=['','## 为什么旧Bounce看起来好很多','',
    '两模型使用相同的bounce_catch_proximity_v2成功指标；同一场景中逐局初态完全相同。legacy与physical_v2之间同时改变了发射分布和物理参数，因此交叉表识别场景适配差异，尚不能将影响单独归因于某一个参数。','',
    '| Bounce模型 | legacy原始球 | 当前physical_v2原始球 |','| --- | ---: | ---: |']
for model in ('653.87','464.31'):
    lines.append('| '+model+' | '+str(c[(model,'legacy','original')]['successes'])+'% | '+str(c[(model,'physical_v2','original')]['successes'])+'% |')
lines+=['','当前physical_v2三球上的新旧模型：','',
    '| 模型 | 原始球 | 网球 | 棒球 |','| --- | ---: | ---: | ---: |']
for model in ('653.87','464.31'):
    lines.append('| '+model+' | '+' | '.join(str(c[(model,'physical_v2',b)]['successes'])+'%' for b in BALLS)+' |')
lines+=['',
    '旧653.87在旧场景表现好，不能直接证明它在新物理场景仍应有同样效果。对照表应优先比较同一列。模型文件名中的653.87/464.31是训练奖励，不是成功率；发射分布、动力学和回合长度改变后，奖励数值也不能当作统一质量排名。','',
    '当前物理场景的变化包括：原始球惯量由旧7.33516e-5改为一致的3.2e-5 kg·m²；自由关节armature/damping归零；物理步长由2ms改为0.125ms（控制/策略周期仍2ms/40ms）；球地面使用独立接触对，球手临界阻尼；原始球初始距离由旧较远范围改为1.6–2.0m、水平速度由约1m/s改为1.3–1.7m/s。网球/棒球还有各自材料与棒球专用发射。','',
    '上述修改影响反弹高度、落点、接触时间和手接触后的响应。交叉测试支持“模型对训练场景的适配”是主要因素，而不是只因球的外观不同。','',
    '本对照没有单独训练同一physical_v2环境下的单球模型，因此不能据此量化“三球训练本身”对原始球成功率的影响。要识别这一项，应固定物理、发射、奖励、网络、训练预算和种子，对比单原始球与三球训练。','',
    '## Roll未见过YCB仍能成功的含义','',
    '252.54未使用网球/棒球重新训练，所以三球结果是旧模型直接换球的零样本迁移。策略输入为位置、速度、机器人状态和手关节；没有类别标签或RGB图像。不同物体若仍落在策略可处理的状态范围内，可以共享接近/承接行为，因此“未训练某种球”并不必然失败。',
    '但当前Roll主指标是region_retention_v5，关注终局球仍在手掌区域；不能把它等同于低速稳定闭合握持。原始球/网球/棒球的roll_stable_hold辅助指标在前次测试分别是1/100、1/100、0/100。','',
    '## 读表时的边界','',
    '- Throw使用当前项目教师497.09，按throw_official（到达2.5s截断，包括同时失败/截断的历史语义）统计。它不是与Roll相同的区域持球指标，也不是严格抓稳。网球/棒球使用现有object_test解析球碰撞；质量、尺寸、惯量不同，但仍沿用Throw历史自由关节阻尼和接触设置，未额外套用Roll/Bounce的材料版本。',
    '- Roll使用已更正的roll_eval_success，与正式PPO测试一致；不再把旧info.success作为主结果。',
    '- Bounce主指标为到达2.5s、未失败、进入grasping且末端距离<0.25m。contact_0p3s是独立辅助诊断，不是严格稳定握持成功率。',
    '- Basket网球使用[1.0,0.9,0.9]初始手姿，其余为[0.8,0.7,0.7]；使用真实自然托球与入篮判据。','',
    '## 下一步建议（依据本次对照的工程判断）','',
    '1. 保留Roll252.54为不可覆盖的换球基线。若重训，先做完整Catch模型小步微调对照，不必仅因没有YCB训练就从头训练。使用原始球占多数的混合分布，同时监控区域保持和稳定诊断。',
    '2. Bounce以当前合理物理为目标：先在可达发射条件下把单原始球的接触后控制练好，再逐步扩展材质和发射范围；原始球回归评估与三球评估同时保留。不能用旧场景接近满分作为新场景已学好的证据，也不能用当前38%证明混合训练本身失败。',
    '3. 补充接触后掉球、末态持球、相对速度和实际手指关节行为等诊断；保留现有主判据，避免用宽松成功率代替实机抓稳能力。已有controlled_hold_v1可作为下一版Bounce控制/奖励候选，但需要独立训练验证，不能声称它已提高成功率。','',
    '## 数据与复现','',
    '- four_task_success.csv：四任务三球主表及来源；bounce_model_scene_comparison.csv：Bounce交叉对照。',
    '- 每个新组合保存逐局episodes.jsonl、物理/模型/配置provenance.json、source_hashes.json和summary.json。',
    '- paired_initial_states.json确认Bounce两模型在同场景下100个种子的初态完全一致。',
    '- run_throw_bounce_comparison_20261101.py可复用已完成结果并拒绝静默覆盖未完成结果。',
    '- failed_legacy_api_attempts仅保留旧API兼容适配时的启动失败；没有回合数据，不计入正式100局。interrupted_legacy_runs保留中断的未完成重复试跑，也不计入正式结果。正式结果均有COMPLETE。',
    '- 本次评估仅改独立分析脚本，未启动训练。','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
(OUT/'COMPLETE').write_text('800 new episodes plus linked prior evaluation verified\n')
print('\n'.join(lines[:12]))
print('\n'.join(lines[lines.index('| Bounce模型 | legacy原始球 | 当前physical_v2原始球 |'):][:4]))

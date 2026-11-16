from pathlib import Path
import hashlib,json,csv,math
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/isee/Catch/catch_it_copy');BASE=ROOT.parent/'analysis_artifacts/episode_dr_catch_20261116/results'
assert (OUT/'COMPLETE').exists() and (OUT/'NOMINAL_ROBOT_COMPLETE').exists()
def load(folder):
 s=json.loads((folder/'summary.json').read_text());r=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
 assert s['completed'] and len(r)==20 and sum(x['success'] for x in r)==s['successes']
 assert len({(x['slot'],x['episode']) for x in r})==20 and all(sum(x['slot']==i for x in r)==5 for i in range(4))
 assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==s['checkpoint_sha256']
 return s,{(x['slot'],x['episode']):x for x in r}
summary=[];cache={};totals=[];ranges={}
for model in ('roll_best_236_00','bounce_best_1_57'):
 motion='roll' if model.startswith('roll') else 'bounce';names=list(json.loads((BASE/'catalog_used.json').read_text())[motion])
 for scale in (1.,.5,0.):
  successes=stable=0
  for obj in names:
   folder=BASE/model/obj if scale==1 else OUT/(model+'_range'+str(scale))/obj
   s,rows=load(folder);cache[model,scale,obj]=rows;successes+=s['successes'];stable+=s['stable_hold_episodes'] or 0
   if scale==1:assert len({r['physics']['randomization']['size_scale'] for r in rows.values()})==20
   else:
    ref=cache[model,1.,obj]
    for key,r in rows.items():
     a=ref[key];assert r['episode_seed']==a['episode_seed']
     d=r['physics']['randomization'];base=a['physics']['randomization']
     assert d['version']=='physical_dr_episode_v1' and d['sampling']=='episode_reset'
     for field in ('size_scale','mass_scale','friction_scale','rolling_scale','damping_scale'):np.testing.assert_allclose(d[field],1+(base[field]-1)*scale,atol=1e-12)
     np.testing.assert_allclose(np.asarray(r['initial']['qpos'])[[0,1,3,4,5,6]],np.asarray(a['initial']['qpos'])[[0,1,3,4,5,6]],atol=1e-12)
     np.testing.assert_allclose(r['initial']['qvel'][:2],a['initial']['qvel'][:2],atol=1e-12)
   f=[r for r in rows.values() if not r['success']]
   row=dict(model=model,motion=motion,object=obj,scale=scale,episodes=20,successes=s['successes'],success_rate=s['success_rate'],stable_hold_episodes=s['stable_hold_episodes'],failure_reasons={why:sum(r['reason']==why for r in f) for why in sorted({r['reason'] for r in f})})
   if motion=='roll':row.update(failed_with_contact=sum((r['roll_evaluation'] or {}).get('first_contact_time') is not None for r in f),failed_departed=sum(bool((r['roll_evaluation'] or {}).get('departed')) for r in f))
   else:row.update(failed_with_contact=sum(r['bounce_diagnostics'].get('bounce_grasp_had_contact',False) for r in f),failed_hold_0p05=sum(r['bounce_diagnostics'].get('bounce_grasp_max_hold_seconds',0)>=.05 for r in f))
   summary.append(row)
  n=len(names)*20;totals.append(dict(model=model,scale=scale,episodes=n,successes=successes,success_rate=successes/n,stable_hold_rate=stable/n if motion=='roll' else None))
# Additional continuous-contact diagnostics and specialist comparison: same initial physics/states.
extra=[]
for group,objects in [('bounce_best_1_57_range1.0',['original','baseball']),('bounce_three_ball_13_68',['original','tennis_ball','baseball'])]:
 for obj in objects:
  s,rows=load(OUT/group/obj);ref=cache['bounce_best_1_57',1.,obj]
  for key,r in rows.items():
   a=ref[key];assert r['episode_seed']==a['episode_seed'];assert r['physics']['randomization']==a['physics']['randomization'];assert r['physics']==a['physics']
   np.testing.assert_array_equal(r['initial']['qpos'],a['initial']['qpos']);np.testing.assert_array_equal(r['initial']['qvel'],a['initial']['qvel'])
   if group.startswith('bounce_best'):assert r['success']==a['success']
  failed=[r for r in rows.values() if not r['success']];probes=[r['bounce_diagnostics']['bounce_grasp_probe'] for r in failed];ct=sum(p['contact_ticks'] for p in probes)
  extra.append(dict(model=group,object=obj,successes=s['successes'],episodes=20,failed_never_two_groups=sum(p['max_groups']<2 for p in probes),failed_never_valid_hold=sum(p['gates'].get('valid',0)==0 for p in probes),failure_contact_gate_fractions={k:sum(p['contact_gates'].get(k,0) for p in probes)/max(1,ct) for k in ('capture','groups','span','speed','off_ground','joint','valid')}))
robot=[]
for model,obj in [('roll_best_236_00','apple'),('roll_best_236_00','lemon'),('bounce_best_1_57','original'),('bounce_best_1_57','baseball')]:
 s,rows=load(OUT/(model+'_nominal_robot_range0')/obj);ref=cache[model,0.,obj]
 for key,r in rows.items():
  a=ref[key];assert r['episode_seed']==a['episode_seed'];assert all(r['physics']['randomization'][k]==1. for k in ('size_scale','mass_scale','friction_scale','rolling_scale','damping_scale'))
  np.testing.assert_array_equal(r['initial']['qpos'],a['initial']['qpos']);np.testing.assert_array_equal(r['initial']['qvel'],a['initial']['qvel'])
 robot.append(dict(model=model,object=obj,successes=s['successes'],episodes=20,reference_object_fixed_successes=sum(r['success'] for r in ref.values())))
for spec in (json.loads((OUT/'run_spec.json').read_text()),):
 for name,h in spec['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
baseline=json.loads((BASE/'run_spec.json').read_text())
for name,h in baseline['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
(OUT/'validated_summary.json').write_text(json.dumps(dict(totals=totals,per_object=summary,extra_diagnostics=extra,nominal_robot=robot),indent=2))
(OUT/'VALIDATION.json').write_text(json.dumps(dict(new_cases=35,new_episodes=700,reused_baseline_cases=13,reused_baseline_episodes=260,total_evaluated_comparison_episodes=960,exact_quotas=True,checkpoint_hashes_verified=True,baseline_production_sources_identical=True,range_scaling_and_launch_seeds_verified=True,specialist_identical_physics_and_initial_state=True,instrumented_full_range_successes_reproduce_baseline=True,notes=['Fixed means nominal TARGET physics only; robot gains, hand friction, launch and observation/control noise remain in the main range experiment.','The separate nominal_robot test fixes PID multipliers and disables observation/control noise; it does not disable every possible environment randomization.','Per-object samples are small (20); these inference tests do not establish benefits of retraining under smaller ranges.','Explicit initial-state seeds are matched; later noise is not strictly paired because historical synchronous envs share a global NumPy RNG.']),indent=2))
lines=['# 三任务泛化对齐与Catch瓶颈诊断（2026-11-16）','','新增35组700回合，复用同一源码、同一模型、同一种子的原范围13组260回合，共960个对照回合。每目标每条件20回合，4个同步环境各5回合。全范围和半范围均每回合采样，固定参数组只固定目标物理参数，仍保留随机发射和原机器人随机化。正式代码、奖励和默认参数未修改。','','## 范围消融总体结果','','|任务|范围倍率|成功次数|成功率|Roll稳定持球率|','|---|---:|---:|---:|---:|']
for r in totals:
 st=f"{r['stable_hold_rate']:.1%}" if r['stable_hold_rate'] is not None else '—';lines.append(f"|{r['model']}|{r['scale']}|{r['successes']}/{r['episodes']}|{r['success_rate']:.1%}|{st}|")
lines+=['','## 各目标','','|模型|目标|当前范围|半范围|固定目标参数|','|---|---|---:|---:|---:|']
for model in ('roll_best_236_00','bounce_best_1_57'):
 names=[r['object'] for r in summary if r['model']==model and r['scale']==1]
 for obj in names:
  values=[next(r for r in summary if r['model']==model and r['object']==obj and r['scale']==scale)['success_rate'] for scale in (1.,.5,0.)]
  lines.append(f"|{model}|{obj}|{values[0]:.1%}|{values[1]:.1%}|{values[2]:.1%}|")
lines+=['','半范围相对偏差全部减半：尺寸±2.5%、质量±5%、滑动摩擦±7.5%、Roll滚动摩擦长度±10%、Bounce接触阻尼比±5%。当前完整范围分别是±5%、±10%、±15%、±20%、±10%。阻尼比不是COR。','缩范围对现有模型的收益较小且不一致，固定目标参数没有明显恢复到高成功率。不能把低成功率主要归于目标参数范围。半范围可作为待验证的课程训练起点，最终测试仍应在统一完整范围下比较。','','## 旧三球Catch在完全相同当前环境中的对照','','|目标|当前六球1.57|旧三球13.68|','|---|---:|---:|']
for obj in ('original','tennis_ball','baseball'):
 a=next(r for r in summary if r['model']=='bounce_best_1_57' and r['scale']==1 and r['object']==obj);b=next(r for r in extra if r['model']=='bounce_three_ball_13_68' and r['object']==obj)
 lines.append(f"|{obj}|{a['successes']}/20|{b['successes']}/20|")
lines+=['','逐项核验初始位置、速度、物理参数相同，成功判据都是当前controlled_hold_v1。环境条件可被已有策略抓住，当前六球策略学习效果有明显差距。这个对照不能单独证明是灾难性遗忘，也不能分开训练初始化、有效样本数、训练配方和学习率等因素。','','## 接触之后的失败','','原范围Roll失败84回合，其中80回合曾接触手、84回合曾离桌；Bounce失败57回合，其中55回合曾接触手，仅3回合最长有效持球达到0.05秒。曾接触不等于正确入掌，不要求这些接触一定发生在离桌后。','原始球失败12回合，有8回合完全没有形成有效持球；棒球失败18回合，有14回合完全没有形成有效持球。棒球失败时的手接触采样中，达到两个接触组约22.4%、满足低相对速度约32.2%、同时满足完整持球条件约4.6%。主要检查多指包覆、入掌位置、相对速度匹配以及接触后控制，不是简单把0.3秒成功门槛放宽。','','## 机器人随机化的附加诊断','','在目标物理固定基础上，另将PID倍率固定为1，观测和控制噪声关闭；其余条件不变，尤其不声称取消了所有环境随机化。','','|任务/目标|固定目标、原机器人随机化|再固定PID且关闭观测/控制噪声|','|---|---:|---:|']
for r in robot:lines.append(f"|{r['model']}/{r['object']}|{r['reference_object_fixed_successes']}/20|{r['successes']}/20|")
lines+=['','仍没有恢复到高成功率，故机器人PID/噪声也不足以单独解释当前差距。','','## Roll坐标核验','','同一手模在Roll和Bounce的FK一致。掌面网格PCA薄轴约[0.0265,-0.2675,0.9632]，接近局部Z；手指正向弯曲使指尖沿局部Z移动约5.28cm。Roll捕获点、grasp_terms和scoop姿态项仍把局部Y作为掌面法向，Bounce用Z。该坐标语义不一致值得优先修正并做新版本训练对照，但没有据此声称全部失败由这一个问题造成，也没有把修正后收益当作已验证结果。','','## 建议优先级','','1. Roll先版本化核对并统一捕获点、掌面法向、入掌区域和姿态奖励，按迎球/承托阶段设计；保留当前基线后重新训练验证。','2. Bounce优先用强三球完整Catch初始化六球训练，而不是仅从Track重建Catch；物体仍均衡分配，旧三球可用教师动作蒸馏约束保留能力。检验较小联合学习率及Track分支学习率倍率，用独立验证成功率选模型。','3. 将现有包覆、低相对速度和有效持球进度奖励做分阶段消融，先查接触后的控制，不先放宽成功条件。','4. 半范围只作为课程训练候选，逐步恢复当前范围；不通过缩小最终测试范围夸大提升。','5. Throw按共享对象catalog、每回合采样、质量/惯量/几何一致更新接入，修复实际质量固定和惯量不同步，保留legacy路线。完整原参数范围和统一设计见ALIGNMENT_DESIGN.md。','','所有结果均为已有模型的推理诊断，未做正式训练。每目标20回合，变化5个百分点对应一个回合，不足以证明小幅改善具有统计显著性。']
(OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(totals=totals,nominal_robot=robot),indent=2))

from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT/'catch_it_copy';OUT=ROOT/'analysis_artifacts/object_benchmark_20261103'
motion=json.loads((OUT/'motion_validation.json').read_text());drops=json.loads((OUT/'drop_validation.json').read_text())
assert len(motion['rows'])==70 and all(r['passed'] for r in motion['rows'])
assert all(r['passed'] for r in motion['half_timestep_seed0']) and all(r['passed'] for r in drops.values())
training=json.loads((OUT/'training_smoke.json').read_text());assert len(training)==4 and all(r['exit']==0 for r in training)
viewer=json.loads((OUT/'viewer_smoke.json').read_text());assert len(viewer)==3 and all(r['exit']==0 for r in viewer)
regression=json.loads((OUT/'baseline_regression.json').read_text());assert len(regression)==6 and all(r['exit']==0 and all(r['checks'].values()) for r in regression)
tests=(OUT/'tests.log').read_text();assert re.search(r'Ran 45 tests',tests) and '\nOK\n' in tests
profile=PROJECT/'ycb_objects/calibration_data/object_benchmark_v1.json';profile_sha=hashlib.sha256(profile.read_bytes()).hexdigest()
for r in viewer:
 m=json.loads((OUT/('replay_final_'+r['object'])/'manifest.json').read_text())
 assert m['material_sha256']==profile_sha
 assert len(m['episodes'])==2 and all(e['terminal_state_exact'] for e in m['episodes'])
 report=dict(object=r['object'],episodes=2,terminal_frames_exact=True,successes=sum(e['success'] for e in m['episodes']))
 r['recording_check']=report
sources=['train_DCMM.py','configs/config.yaml','configs/benchmark_bounce_viewer_v1.yaml','gym_dcmm/__init__.py','gym_dcmm/envs/object_benchmark.py','gym_dcmm/utils/roll_evaluation.py','ycb_objects/benchmark_geometry.py','ycb_objects/bounce_viewer_replay.py','scripts/view_object_benchmark.sh','scripts/train_object_benchmark.sh','tests/test_object_benchmark.py']
manifest={'status':'complete','formal_training_started':False,'catalog_sha256':profile_sha,'passive_motion_trials':70,'half_timestep_trials':7,'unit_tests':45,'training_smoke':training,'viewer_smoke':viewer,'baseline_regression':regression,'source_sha256':{n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in sources},'asset_provenance':{n:json.loads((PROJECT/'assets/ycb'/(n+'.provenance.json')).read_text()) for n in ['orange','apple','lemon','tomato_soup_can','racquetball','golf_ball','softball']}}
(OUT/'VALIDATION.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
lines=['# 新增目标运动验证','', '七种目标各10条被动轨迹，共70条均通过检查；这是运动与数值验证，不是模型抓取成功率。水果为塑料YCB模型的凸包，罐为侧放解析圆柱，球为各自材质/惯量的解析球。','', '|目标|轨迹通过|自然离桌时间 / 首次反弹净高度|','|---|---:|---|']
for name in ['orange','apple','lemon','tomato_soup_can','racquetball','golf_ball','softball']:
 rows=[r for r in motion['rows'] if r['object']==name]
 if rows[0]['motion']=='roll':
  values=[r['departure']['time'] for r in rows];text=f'{min(values):.3f}–{max(values):.3f}秒离桌'
 else:
  values=[r['apex_clearance_m'] for r in rows];text=f'{min(values):.3f}–{max(values):.3f}米'
 lines.append(f'|{name}|10/10|{text}|')
lines+=['','45项测试通过；4条64步短PPO训练链路通过；每新球2局真实策略预计算及逐帧恢复通过；旧三球两任务6个检查回合与原数据initial/final/reward/success完全相同。','', '所有材质/摩擦是有来源标注的仿真先验，不是实物测量；高尔夫球COR0.8尤其是工程假设。水平反弹窗口是发射设计目标，不是机械臂IK可达性证明。更长保持及新增目标抓取成功率未做正式评估。','', '水果默认更近释放，不能把它们和旧三球不同初始条件下的成功率直接当作纯物体差异。正式benchmark应固定并报告每组发射协议、物体尺寸质量、可达率及稳定持物标准。','', '接口、viewer及训练命令见项目OBJECT_BENCHMARK_V1_2026-11-03.md。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'COMPLETE').write_text('Motion, training plumbing, unit, baseline and viewer checks passed.\n')
print('\n'.join(lines[:13]))

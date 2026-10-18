# 蒸馏实验输出审查（2026-10-18）

来源：C:/Users/29491/Downloads。只分析下载快照，不代表服务器当前进度。排除 smoke/check，所有成功率取记录中的 success_rate；平均为三个任务等权平均。最佳指平均成功率最高的同一次评估，不拼接不同轮次的单任务峰值，也不等于程序按最差教师差距保存的 best 模型。

| 实验 | 最新训练轮次 | 最佳平均对应评估 | Throw | Roll | Bounce | 最佳平均 | 最新评估平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| catch_dualcam160_5000 | 3926 | eval_003500 | 57.5% | 62.5% | 60.0% | 60.0% | 60.0% |
| catch_dualcam_5000_fixed | 3910 | eval_002500 | 57.5% | 58.0% | 69.0% | 61.5% | 60.5% |
| distillation_action_heads\runs\action_heads_catch_train_seed123 | 2000 | eval_002000 | 27.0% | 44.0% | 58.0% | 43.0% | 43.0% |
| distillation_camera_gate\runs\camera_gate_catch_train_seed123 | 2000 | eval_001800 | 57.0% | 42.0% | 60.0% | 53.0% | 51.0% |
| distillation_event\runs\event_catch_train_seed123_run2 | 2000 | eval_001800 | 32.0% | 36.0% | 53.0% | 40.3% | 35.7% |
| distillation_feature\runs\feature_catch_train_seed123 | 1000 | eval_001000 | 24.0% | 37.0% | 45.0% | 35.3% | 35.3% |
| distillation_feature\runs\feature_catch_train_seed123_run2 | 542 | eval_000400 | 16.0% | 14.0% | 40.0% | 23.3% | 23.3% |
| distillation_moe\runs\moe_catch_train_seed123 | 2000 | eval_001800 | 40.0% | 34.0% | 62.0% | 45.3% | 41.0% |
| distillation_visual_aux\runs\catch_position_5000_new | 3542 | eval_003500 | 63.5% | 56.5% | 70.5% | 63.5% | 63.5% |
| track_dualcam_5000 | 5000 | eval_005000 | 72.5% | 94.0% | 83.0% | 83.2% | 83.2% |
| trb_v2_5000_CATCH | 5000 | eval_003000 | 72.0% | 52.0% | 70.0% | 64.7% | 61.0% |
| trb_v2_5000_TRACK | 5000 | eval_002800 | 86.0% | 89.0% | 88.0% | 87.7% | 84.7% |

## 解读与限制
- Track：双相机5000轮达到83.17%，Roll 94%接近教师94.5%，Throw72.5%相对教师97%是主要短板。
- Catch双相机160与位置辅助监督在相同3500轮、8环境/任务、200回合/任务下平均60%与63.5%；辅助监督Throw/Bounce提升，但Roll从62.5%下降到56.5%。不能仅凭一次评估宣布显著收益。
- 相机门控2000轮平均51%，同期双相机160为55.83%。目前没有证据说明门控优于普通融合；门控末轮平均使用top34.1%、wrist65.9%，不能据此断言分阶段正确选择。
- 动作分头2000轮43%，趋势仍上升；MoE最好45.33%、末轮41%，Roll弱；事件采样最好40.33%、末轮35.67%，末轮Roll17%明显退化。
- 特征蒸馏原run1000轮35.33%；run2仅训练到542轮、最新评估400轮23.33%，不能当2000轮结果。
- 位置辅助监督是预测物体位置，不等同于模仿教师隐藏特征。各方法loss定义/权重不同，不横向比较loss大小。
- trb_v2为旧状态输入MLP配置，无vision配置，与视觉学生信息条件不同；Catch的Bounce判定版本也不同（bounce_throw_style vs bounce_catch_proximity_v2），只能作历史参考。
- 新设计未找到正式Track评估结果；baseline_event目录只有check/smoke，缺少严格匹配的普通蒸馏正式对照。
- 所有正式实验配置seed123，尚无多训练种子证据；多轮选择最高评估存在选择偏差。后续统一训练预算/并行环境/教师/判定版本，对选定模型用独立测试种子至少500回合/任务复评，再做多训练种子实验。
- 优先完成位置辅助监督与普通双相机160对照；保留分头方案继续验证。事件、MoE、特征方案先统一条件并检查任务级退化，不建议立即组合全部模块。

## catch_dualcam160_5000
- 路径：C:\Users\29491\Downloads\catch_dualcam160_5000
- seed=123；环境数/任务=8；每任务评估回合=200；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.575, 0.625, 0.6]；最新训练loss=0.9044298063963652
- 各评估平均：eval_000500=28.8%, eval_001000=48.3%, eval_001500=45.8%, eval_002000=55.8%, eval_002500=51.3%, eval_003000=56.5%, eval_003500=60.0%
- 教师三任务成功率：{'throw': 0.825, 'roll': 0.66, 'bounce': 0.89}

## catch_dualcam_5000_fixed
- 路径：C:\Users\29491\Downloads\catch_dualcam_5000_fixed
- seed=123；环境数/任务=8；每任务评估回合=200；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.59, 0.55, 0.675]；最新训练loss=0.921324023976922
- 各评估平均：eval_000500=29.2%, eval_001000=39.5%, eval_001500=46.8%, eval_002000=51.8%, eval_002500=61.5%, eval_003000=53.3%, eval_003500=60.5%
- 教师三任务成功率：{'throw': 0.825, 'roll': 0.66, 'bounce': 0.89}

## distillation_action_heads\runs\action_heads_catch_train_seed123
- 路径：C:\Users\29491\Downloads\distillation_action_heads\runs\action_heads_catch_train_seed123
- seed=123；环境数/任务=1；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.27, 0.44, 0.58]；最新训练loss=0.9045983264222741
- 各评估平均：eval_000200=1.3%, eval_000400=17.0%, eval_000600=23.0%, eval_000800=31.3%, eval_001000=31.3%, eval_001200=35.7%, eval_001400=36.7%, eval_001600=34.3%, eval_001800=39.0%, eval_002000=43.0%
- 教师三任务成功率：{'throw': 0.82, 'roll': 0.62, 'bounce': 0.87}

## distillation_camera_gate\runs\camera_gate_catch_train_seed123
- 路径：C:\Users\29491\Downloads\distillation_camera_gate\runs\camera_gate_catch_train_seed123
- seed=123；环境数/任务=8；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.58, 0.42, 0.53]；最新训练loss=0.9240086022764444
- 各评估平均：eval_000200=9.0%, eval_000400=10.7%, eval_000600=18.0%, eval_000800=34.7%, eval_001000=30.3%, eval_001200=51.3%, eval_001400=27.3%, eval_001600=41.7%, eval_001800=53.0%, eval_002000=51.0%
- 教师三任务成功率：{'throw': 0.9, 'roll': 0.7, 'bounce': 0.89}

## distillation_event\runs\event_catch_train_seed123_run2
- 路径：C:\Users\29491\Downloads\distillation_event\runs\event_catch_train_seed123_run2
- seed=123；环境数/任务=1；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.35, 0.17, 0.55]；最新训练loss=0.8751634261570871
- 各评估平均：eval_000200=2.3%, eval_000400=14.0%, eval_000600=17.0%, eval_000800=35.7%, eval_001000=24.3%, eval_001200=28.3%, eval_001400=35.7%, eval_001600=33.0%, eval_001800=40.3%, eval_002000=35.7%
- 教师三任务成功率：{'throw': 0.9, 'roll': 0.7, 'bounce': 0.89}

## distillation_feature\runs\feature_catch_train_seed123
- 路径：C:\Users\29491\Downloads\distillation_feature\runs\feature_catch_train_seed123
- seed=123；环境数/任务=1；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.24, 0.37, 0.45]；最新训练loss=0.9289801502600312
- 各评估平均：eval_000200=1.3%, eval_000400=23.0%, eval_000600=20.0%, eval_000800=29.3%, eval_001000=35.3%
- 教师三任务成功率：{'throw': 0.9, 'roll': 0.7, 'bounce': 0.89}

## distillation_feature\runs\feature_catch_train_seed123_run2
- 路径：C:\Users\29491\Downloads\distillation_feature\runs\feature_catch_train_seed123_run2
- seed=123；环境数/任务=1；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.16, 0.14, 0.4]；最新训练loss=0.9539895765483379
- 各评估平均：eval_000200=4.3%, eval_000400=23.3%
- 教师三任务成功率：{'throw': 0.9, 'roll': 0.7, 'bounce': 0.89}

## distillation_moe\runs\moe_catch_train_seed123
- 路径：C:\Users\29491\Downloads\distillation_moe\runs\moe_catch_train_seed123
- seed=123；环境数/任务=1；每任务评估回合=100；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.39, 0.27, 0.57]；最新训练loss=0.8903564536012709
- 各评估平均：eval_000200=2.7%, eval_000400=20.3%, eval_000600=28.3%, eval_000800=30.0%, eval_001000=24.7%, eval_001200=33.7%, eval_001400=41.3%, eval_001600=36.3%, eval_001800=45.3%, eval_002000=41.0%
- 教师三任务成功率：{'throw': 0.82, 'roll': 0.62, 'bounce': 0.87}

## distillation_visual_aux\runs\catch_position_5000_new
- 路径：C:\Users\29491\Downloads\distillation_visual_aux\runs\catch_position_5000_new
- seed=123；环境数/任务=8；每任务评估回合=200；判定版本=2026-10-12-bounce-catch-v2
- 最新评估三任务：[0.635, 0.565, 0.705]；最新训练loss=0.9046966638416052
- 各评估平均：eval_000500=21.5%, eval_001000=34.2%, eval_001500=50.2%, eval_002000=50.8%, eval_002500=56.7%, eval_003000=62.2%, eval_003500=63.5%
- 教师三任务成功率：{'throw': 0.825, 'roll': 0.66, 'bounce': 0.89}

## track_dualcam_5000
- 路径：C:\Users\29491\Downloads\track_dualcam_5000
- seed=123；环境数/任务=8；每任务评估回合=200；判定版本=2026-10-11
- 最新评估三任务：[0.725, 0.94, 0.83]；最新训练loss=1.0751644540578127
- 各评估平均：eval_000500=35.5%, eval_001000=62.7%, eval_001500=70.5%, eval_002000=77.8%, eval_002500=81.7%, eval_003000=81.5%, eval_003500=79.8%, eval_004000=81.8%, eval_004500=79.0%, eval_005000=83.2%
- 教师三任务成功率：{'throw': 0.97, 'roll': 0.945, 'bounce': 0.88}

## trb_v2_5000_CATCH
- 路径：C:\Users\29491\Downloads\trb_v2_5000_CATCH
- seed=123；环境数/任务=2；每任务评估回合=100；判定版本=2026-10-11
- 最新评估三任务：[0.65, 0.47, 0.71]；最新训练loss=0.05452525510918349
- 各评估平均：eval_000200=43.7%, eval_000400=47.0%, eval_000600=41.7%, eval_000800=47.3%, eval_001000=45.7%, eval_001200=50.7%, eval_001400=50.0%, eval_001600=40.0%, eval_001800=42.3%, eval_002000=51.0%, eval_002200=56.3%, eval_002400=57.3%, eval_002600=44.7%, eval_002800=58.7%, eval_003000=64.7%, eval_003200=58.0%, eval_003400=55.3%, eval_003600=56.3%, eval_003800=49.0%, eval_004000=56.0%, eval_004200=55.3%, eval_004400=54.3%, eval_004600=50.0%, eval_004800=59.3%, eval_005000=61.0%
- 教师三任务成功率：{'throw': 0.82, 'roll': 0.62, 'bounce': 0.87}

## trb_v2_5000_TRACK
- 路径：C:\Users\29491\Downloads\trb_v2_5000_TRACK
- seed=123；环境数/任务=2；每任务评估回合=100；判定版本=2026-10-11
- 最新评估三任务：[0.78, 0.87, 0.89]；最新训练loss=0.06348391715437174
- 各评估平均：eval_000200=78.0%, eval_000400=70.0%, eval_000600=76.7%, eval_000800=78.3%, eval_001000=79.7%, eval_001200=73.7%, eval_001400=79.0%, eval_001600=78.0%, eval_001800=86.7%, eval_002000=83.0%, eval_002200=86.3%, eval_002400=77.0%, eval_002600=85.7%, eval_002800=87.7%, eval_003000=84.3%, eval_003200=82.7%, eval_003400=80.3%, eval_003600=83.3%, eval_003800=84.0%, eval_004000=86.3%, eval_004200=82.3%, eval_004400=80.7%, eval_004600=85.0%, eval_004800=82.0%, eval_005000=84.7%
- 教师三任务成功率：{'throw': 0.96, 'roll': 0.95, 'bounce': 0.86}
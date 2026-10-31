# 三任务三球成功率（2026-10-31，更正Roll统计字段）

**更正：此前Roll的0/100读取了旧字段info.success，不能代表当前Viewer成功率。现已按原PPO使用的roll_eval_success重跑三球各100局。只修正评估字段，没有修改环境、物理、策略、奖励或成功阈值。旧结果保存在superseded_legacy_roll_report中，原roll_*目录的0%仅作为legacy指标对照。**

每个组合100局，逐局设置 NumPy/Python/Torch/Gym seed=0–99，共900局。使用真实模型和保存的归一化参数、原PPO确定性动作适配，关闭Viewer。初始化失败计入分母，不重抽、不丢弃。

| 任务与模型 | 原始球 | 网球 | 棒球 |
| --- | ---: | ---: | ---: |
| Roll Catch 252.54 | 78%（78/100） | 70%（70/100） | 54%（54/100） |
| Bounce Catch 464.31 | 38%（38/100） | 44%（44/100） | 40%（40/100） |
| Basket 38.38 | 97%（97/100） | 0%（0/100） | 0%（0/100） |

Roll三球的手接触回合数分别为99/100、98/100、99/100。手接触、当前区域持球指标、旧低速指标和稳定持球诊断分别统计，不可互相替代。
Basket网球与棒球的初始化失败数分别为0和0，初始化成功与后续投篮成功分开统计。

## 成功判据与测试条件

- Roll：使用当前physical_v1、初速0.8–1.3m/s、legacy奖励、关闭桌前等待硬约束。主指标是roll_eval_success（region_retention_v5），与正式PPO的terminal_metrics相同：真实手接触启动持球状态；球已离桌、无桌地接触且回合结束仍在掌心附近区域，失败终止覆盖成功。掌心区域为水平半径12cm，高度相对掌心[-2cm, 球半径+12cm]。主指标不要求连续低速或固定保持时长，也不要求每个瞬间都持续接触。
- Roll旧指标info.success/roll_legacy_success：相对手速≤0.05m/s且连续15个策略步，仍为三球0/100。roll_stable_hold是另一项独立诊断：真实接触、近手、无桌地接触、速度≤0.15m/s累计0.30s（符合条件的短间隙可容许）。不能用旧指标或此诊断替代主成功率。
- Bounce：physical_v2、legacy抓取、baseball_intercept_v1发射。成功为到达2.5s时限、未失败、处于grasping且末端距离<0.25m；该数值不等于严格稳定握持率。保留原0.125ms物理步长。
- Basket：38.38保存配置、固定底盘、自然接触托球、物理自由飞行入篮与安全跟随。原始球和棒球初始手姿[0.8,0.7,0.7]；网球按给用户的测试命令改为[1.0,0.9,0.9]。网球这一列含初始手姿适配，不是完全相同初始条件下的泛化。
- 三球固定各自尺寸/质量/惯量和接触参数。网球与棒球为解析碰撞球＋YCB外观。参数是项目中的近似先验。
- 此前Roll的82%/81%/84%来自27.44的Tracking模型，衡量拦截，不能作为252.54 Catch的当前区域持球成功率。

## 接触诊断（各100局）

手接触和0.3s连续接触是辅助诊断，不代替成功判据。Basket托球准备阶段不计入下表接触时长。

| 任务/球 | 至少手接触一次 | 连续手接触≥0.3s | 成功且有≥0.3s接触 | 准备失败 |
| --- | ---: | ---: | ---: | ---: |
| Roll Catch 252.54 / 原始球 | 99 | 0 | 0 | 0 |
| Roll Catch 252.54 / 网球 | 98 | 1 | 1 | 0 |
| Roll Catch 252.54 / 棒球 | 99 | 0 | 0 | 0 |
| Bounce Catch 464.31 / 原始球 | 90 | 45 | 35 | 0 |
| Bounce Catch 464.31 / 网球 | 86 | 39 | 37 | 0 |
| Bounce Catch 464.31 / 棒球 | 90 | 44 | 38 | 0 |
| Basket 38.38 / 原始球 | 100 | 0 | 0 | 0 |
| Basket 38.38 / 网球 | 100 | 0 | 0 | 0 |
| Basket 38.38 / 棒球 | 100 | 0 | 0 | 0 |

## 终止原因

- Roll Catch 252.54 / 原始球：`{"timeout": 85, "ball_on_floor": 11, "out_of_bounds": 4}`
- Roll Catch 252.54 / 网球：`{"ball_on_floor": 12, "timeout": 79, "out_of_bounds": 8, "ball_left": 1}`
- Roll Catch 252.54 / 棒球：`{"ball_on_floor": 24, "timeout": 68, "out_of_bounds": 5, "ball_left": 3}`
- Bounce Catch 464.31 / 原始球：`{"ball_left": 56, "catch_success": 38, "failure_unspecified": 6}`
- Bounce Catch 464.31 / 网球：`{"ball_left": 45, "catch_success": 44, "failure_unspecified": 11}`
- Bounce Catch 464.31 / 棒球：`{"ball_left": 60, "catch_success": 40}`
- Basket 38.38 / 原始球：`{"basket_score": 97, "not_a_throw": 1, "ball_on_floor": 2}`
- Basket 38.38 / 网球：`{"ball_on_floor": 100}`
- Basket 38.38 / 棒球：`{"ball_on_floor": 100}`

## 可复核文件

- summary.csv：九个组合的成功数、成功率和95% Wilson区间。
- Roll更正数据在roll_metric_correction/roll_*；Bounce/Basket沿用原对应目录。目录内有episodes.jsonl逐局记录、provenance.json模型哈希/配置/物理参数、source_hashes.json代码哈希、summary.json汇总。
- run_status.json：本次启动命令、退出状态与耗时。smoke_*为独立入口检查，不计入900局。
- 评估脚本：analysis/evaluate_three_task_balls_20261031.py。汇总脚本：analysis/report_three_task_balls_20261031.py。

有限样本的0/100不代表真实成功概率严格为零；95% Wilson区间上界约3.7%。三个任务的成功定义不同，不宜直接横向排名。

# Roll 成功率统计字段更正

错误：原评估脚本直接读取info.success，把旧低速保持判据的0/100作为Roll主成功率。当前Roll PPO的terminal_metrics优先使用roll_eval_success，二者不是同一个指标。

现行实现：gym_dcmm/utils/roll_evaluation.py中的region_retention_v5。真正手接触启动区域持球，球已离桌且没有桌/地接触，回合结束仍位于掌心附近的世界坐标区域；失败终止覆盖主成功标志。主指标没有固定低速或最短保持时间要求。roll_stable_hold和roll_legacy_success是独立的对照指标。

更正只改独立评估脚本的指标选择与诊断记录。模型、物理、动作、奖励、终止和环境成功判据均未改动。使用原Roll PPO的terminal_metrics函数选择主指标，并对字段矛盾的两种情况做了回归检查：legacy=false/eval=true必须成功；legacy=true/eval=false必须失败。

三个目录均使用原252.54 checkpoint、相同physical_v1配置、固定球类和seed0–99各100局。trajectory_equivalence.json核对更正前后各局初态、末态、奖励、步数和结束原因，证明统计更正没有改变模型行为。

上一级REPORT.md、summary.csv和summary_all.json使用这里的更正结果。上一级原roll_*目录仅用于保留旧字段数据和轨迹对照，不作为正式Roll主成功率。

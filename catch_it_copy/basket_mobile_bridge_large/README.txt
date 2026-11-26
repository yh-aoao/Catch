上一版桥接模型的0.6m物理篮筐训练对照

训练入口 python -m basket_mobile_bridge_large.run。
与原 basket_mobile_bridge 保持相同网络、动作、持球->抛球切换、奖励权重、停车释放判据和课程。
只把篮筐名义直径设置为0.6m（圆环中心线半径0.3m）。24个实体圆环小球的位置同步移动，
圆环粗细保持原半径0.01m；判定仍要求球完整通过，使用0.3减球半径的球心允许误差。
不改变篮筐目标中心/高度/倾斜角，不修改全局DcmmCfg或旧任务配置。
新checkpoint schema为basket_mobile_bridge_large_v1，防止和旧模型尺寸混淆。

从旧桥接模型teacher_last.pth通过--initialize-from导入，新建优化器和计步。
--resume只用于本新包自己的resume_last.pth；旧包resume会恢复旧配置，不能用来修改框的尺寸。
旧代码与权重未改动，候选权重另存backups/bridge_candidate_20261126。
服务器需同步新包，原basket_mobile_bridge/basket_mobile_curriculum/basket_autonomous/gym_dcmm依赖保持现有版本。

物理篮筐扩大后会改变球与圆环的碰撞，因此不能直接等同此前虚拟0.6m区域78%的命中率。
原训练仍要求低速稳定释放，原统计中的stopped_success_rate只有14%；本对照不放宽该规则。
评估仍是安全进筐率，不能承诺达到80%。

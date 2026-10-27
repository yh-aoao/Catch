# YCB诊断数据索引

首选说明：`catch_it_copy/YCB_GENERALIZATION_DIAGNOSIS_2026-10-27.md`。

## 当前正式诊断记录

- `ycb_reset_state_evidence_20261027.json`：直接加载修复前源码副本与当前源码的重置状态对照。
- `ycb_generalization_20261027/`：接入状态错误尚未修复的120回合，用于复现旧故障。
- `ycb_generalization_reset_fixed_20261027/`：重置修复后的120回合原生物理测试。
- `ycb_generalization_physics_matched_20261027/`：40回合匹配原球物理的排错对照，不是真实YCB泛化成绩。
- `ycb_baseball_low_speed_reset_fixed_20261027/`：额外10回合低速0.5m/s/5秒棒球测试。
- `bounce_stability_report_20261027.json`：无机器人/无自旋单球数值敏感性测试，未做实物标定。
- `bounce_stability_fine_steps_20261027.json`：更细步长对照。
- `ycb_generalization_diagnosis_20261027.png`：主结果图；源码在analysis/plot_ycb_generalization_20261027.py。
- `ycb_generalization_diagnosis_summary_20261027.json`：批次汇总、版本与验证说明。

## 中间探索文件，不作为最终物理结论

- `bounce_timestep_sensitivity_20261027.json`：按接触进出瞬间估算速度比，受接触采样相位影响，
  且部分对照使用统一0.058kg质量，不是三个实际配置的一一比较。
- `bounce_apex_stability_20261027.json`：接触微小抖动会触发“首次离地峰值”，不作为完整反弹高度。
- `bounce_peak_height_stability_20261027.json`：修正上述峰值识别后的探索版，正式复现以bounce_stability_report为准。

2026-10-26的旧YCB完整机器人策略实验也受到重置姿态错误影响，不能直接用于公平泛化结论；
独立球—桌物理实验不使用该重置工厂，需要与策略结果分开看待。保留旧数据便于追溯，没有覆盖或删除。

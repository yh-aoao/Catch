# 2026-11-09 参数泛化多目标 Catch 简测

Roll 212.69 / Bounce -14.97。physical_dr_v1 开启。每目标4套物理参数×5完整回合=20回合，seed41000；确定性策略，单进程同步4环境。仅统计每环境最先完成的5回合。尺寸±5%、质量±10%、滑动摩擦±15%；Roll滚动阻力±20%，Bounce阻尼比±10%。已核对每回合物理元数据与采样一致。

模型目录只有权重，未提供服务器 resolved_config。本次使用此前 all_targets_catch_setup_20261104 的 Roll/Bounce resolved 配置，覆盖目标、模型、测试设备与随机化选项。Bounce检查点记录 controlled_hold_v1、freeze_track_steps=0、track_lr_scale=1。不是使用Track碰手指标。

|目标|Roll区域留存|Roll稳定持球|Bounce连续持球|
|---|---:|---:|---:|
|original|8/20|0/20|4/20|
|tennis_ball|9/20|3/20|8/20|
|baseball|9/20|1/20|1/20|
|orange|15/20|4/20|—|
|apple|7/20|2/20|—|
|lemon|12/20|3/20|—|
|tomato_soup_can|2/20|0/20|—|
|racquetball|—|—|7/20|
|golf_ball|—|—|9/20|
|softball|—|—|10/20|

Roll区域留存：结束时球留在手部区域，已离桌、无桌地支撑、失败覆盖；稳定持球是回合中曾相对速度≤0.15m/s持续0.3秒等条件，不保证持到最后。Bounce是controlled_hold_v1，连续0.3秒满足多区域接触与低相对速度等条件。

Roll区域留存62/140=44.3%，曾稳定持球13/140=9.3%；Bounce39/120=32.5%。每目标仅4套物理样本，结果不代表全参数范围，也不能与固定物理/其他目标数量的结果直接归因比较。

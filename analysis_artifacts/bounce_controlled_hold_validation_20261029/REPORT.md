# Bounce controlled_hold_v1 验证（2026-10-29）

实现文档：[训练、测试与回退](../../catch_it_copy/BOUNCE_CONTROLLED_HOLD_V1_2026-10-29.md)。

- 新版环境与PPO、旧物理与材质的27项相关测试全部通过。包含奖励/接触/保持、实际极值动作、三球reset、球物理不变、Track严格初始化与冻结/解冻、Adam恢复后同批样本更新一致。
- 扩展运行的3项历史结构检查中有1失败、1错误；使用修改前备份源文件复跑得到相同结果。具体见 final_suite 和 baseline_historical_checks 日志，未把这些既有失败算作通过。
- 两环境异步短训练已完成；冻结checkpoint在128步保存。真实预训练Track权重及归一化逐位不变，恢复到256步后Track已按计划解冻更新，归一化仍保持原值。见 training_verification.json。
- 新测试脚本加载短训练checkpoint、固定网球、无viewer推理64步后正常退出。短训练模型仅用于管线验证，不是正式训练结果。
- 同一旧Catch模型三球各两局、同种子对照：新版目标越限全部为0，2ms采样下实际最大关节越限原始球/网球约.0144rad，棒球约.0285rad；旧版分别2.018/5.389/1.425rad。见control_probe.json；不用于估计新策略成功率。
- 掌面几何和保持条件另有静态接触正例与step合同测试。静态接触正例不是动态握稳证明。
- 尚未进行正式2500万步训练，没有新版正式成功率；手部力矩/软限位参数仍是工程仿真先验。

源文件指纹见source_manifest.json。生产训练参数预览见production_config_preview.yaml。旧版完整备份：catch_it_copy/backups/bounce_before_controlled_hold_20261028_195129/manifest.json。

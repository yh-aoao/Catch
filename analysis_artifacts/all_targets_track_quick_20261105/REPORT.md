# 2026-11-05 多目标 Track 简测

Roll 26.25 / Bounce 20.19；每目标20个完整回合，seed123，确定性策略，单环境同步，参数随机化关闭。使用之前七目标/六球训练的 resolved_config.yaml，覆盖本地路径、checkpoint、测试模式和单目标设置。没有提供新模型的服务器原始训练配置，故此处结论限定于上述评估设置。

|任务|目标|成功数|成功率|上次简测|
|---|---|---:|---:|---:|
|roll|original|19/20|95%|80%|
|roll|tennis_ball|19/20|95%|90%|
|roll|baseball|19/20|95%|85%|
|roll|orange|20/20|100%|90%|
|roll|apple|18/20|90%|95%|
|roll|lemon|18/20|90%|90%|
|roll|tomato_soup_can|20/20|100%|60%|
|bounce|original|19/20|95%|100%|
|bounce|tennis_ball|18/20|90%|85%|
|bounce|baseball|19/20|95%|95%|
|bounce|racquetball|19/20|95%|100%|
|bounce|golf_ball|20/20|100%|100%|
|bounce|softball|20/20|100%|100%|

Roll：沿用 PPO terminal_metrics 的 roll_eval_success，离桌后曾被手拦截；后续失败可能仍保留成功。Bounce：碰手且当步无失败。均不是Catch稳定持球率。

每项20回合，单次成败对应5个百分点，不能据此宣称统计显著提升。测试使用同一初始总seed，但自动reset的后续随机流受回合步数影响，不是逐回合完全配对试验。

roll: 133/140 = 95.0%

bounce: 115/120 = 95.8%

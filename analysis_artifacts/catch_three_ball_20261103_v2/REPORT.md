# 11-3-2 Catch模型三球评估

2026-11-03：Roll269.97、Bounce9.40，各球100局，种子0–99，共600局。原PPO确定性推理、严格加载actor/RMS，失败全部计入分母。与11-2、11-3 best/last逐球100/100初始状态一致。
目录没有服务器原始resolved_config.yaml；沿用前两轮重建配置，若服务器改过环境或动作参数需补原配置复核。Bounce checkpoint确认controlled_hold_v1、439万步附近、联合训练不冻结。
Roll主成功=最终区域保持且排除失败，稳定持球另列（曾达到相对速度<=0.15m/s持续0.3秒）。Bounce主成功=连续0.3秒真实多区域受控保持，成功立即终止；两任务口径不同。

|任务|原始球|网球|棒球|
|---|---:|---:|---:|
|roll|76%|85%|83%|
|Roll曾稳定持球|0%|0%|0%|
|bounce|72%|81%|81%|

## 诊断

- roll original：触手99/100；失败但曾触手23；终止原因{'timeout': 82, 'out_of_bounds': 8, 'ball_on_floor': 8, 'base_collision': 1, 'ball_left': 1}。
  主要成功且曾稳定持球0局；失败末态区域诊断{'outside_lateral_region': 3, 'below_palm': 18, 'other': 3}。
- roll tennis_ball：触手98/100；失败但曾触手13；终止原因{'timeout': 90, 'ball_on_floor': 7, 'out_of_bounds': 1, 'ball_left': 2}。
  主要成功且曾稳定持球0局；失败末态区域诊断{'below_palm': 13, 'other': 2}。
- roll baseball：触手100/100；失败但曾触手17；终止原因{'timeout': 85, 'ball_on_floor': 10, 'out_of_bounds': 5}。
  主要成功且曾稳定持球0局；失败末态区域诊断{'below_palm': 14, 'other': 2, 'outside_lateral_region': 1}。
- bounce original：触手100/100；失败但曾触手28；终止原因{'controlled_hold': 72, 'base_collision': 1, 'timeout': 18, 'out_of_bounds': 9}。
  失败中曾满足短暂有效保持10局。
- bounce tennis_ball：触手99/100；失败但曾触手18；终止原因{'controlled_hold': 81, 'out_of_bounds': 6, 'timeout': 13}。
  失败中曾满足短暂有效保持6局。
- bounce baseball：触手100/100；失败但曾触手19；终止原因{'controlled_hold': 81, 'out_of_bounds': 4, 'timeout': 15}。
  失败中曾满足短暂有效保持9局。

## 历史对照

|任务|球|参照模型|旧成功率|本次成功率|
|---|---|---|---:|---:|
|roll|original|11-2|69%|76%|
|roll|original|11-3 best|78%|76%|
|roll|original|11-3 last|77%|76%|
|roll|tennis_ball|11-2|73%|85%|
|roll|tennis_ball|11-3 best|78%|85%|
|roll|tennis_ball|11-3 last|74%|85%|
|roll|baseball|11-2|85%|83%|
|roll|baseball|11-3 best|87%|83%|
|roll|baseball|11-3 last|79%|83%|
|bounce|original|11-2|61%|72%|
|bounce|original|11-3 best|55%|72%|
|bounce|original|11-3 last|65%|72%|
|bounce|tennis_ball|11-2|62%|81%|
|bounce|tennis_ball|11-3 best|67%|81%|
|bounce|tennis_ball|11-3 last|80%|81%|
|bounce|baseball|11-2|74%|81%|
|bounce|baseball|11-3 best|76%|81%|
|bounce|baseball|11-3 last|79%|81%|

比例有抽样误差，95% Wilson区间见summary_all.json。相同种子既用于比较也用于选模型，不是独立保留验收集。reward文件名不代表抓取成功率；未修改训练奖励或控制代码。各子目录包含模型/配置/源码哈希、真实物理参数和逐回合记录。

## 结论

本轮Bounce9.40受控保持72/81/81%，相对上一轮推荐last65/80/79%呈现改善，原始球增幅较大，网球和棒球小差异不宜过度解读。Roll269.97区域保持76/85/83%，对比262.02的78/78/87%为球类间权衡，三球稳定持球仍0/100；保留11-02稳定持球较好的250.65备份。未覆盖任何历史模型或训练代码。

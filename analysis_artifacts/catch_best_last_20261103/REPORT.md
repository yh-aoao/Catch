# 2026-11-03 Roll / Bounce best与last对照

每模型/球100回合、种子0–99，共1200回合。原PPO确定性推理，严格加载actor与RMS，所有失败计入分母。best/last对应100/100初始状态完全相同。
训练服务器未提供原始resolved_config.yaml，沿用11-02已验证的重建配置；若服务器覆盖过速度、发射、手部动作尺度或奖励选项，需原配置复核。Bounce checkpoint确认controlled_hold_v1且track_lr_scale=1、freeze_track_steps=0、freeze_track_normalization=false。
Roll主要成功=region_retention_v5（结束时区域保持且排除失败）；曾稳定持球另列（0.3秒、相对速度<=0.15m/s，不要求结束时仍稳定）。Bounce主要成功=连续0.3秒真实多区域接触受控保持。两任务不能直接横向比较。

|任务|模型|原始球|网球|棒球|三球合计|
|---|---|---:|---:|---:|---:|
|roll|best|78%|78%|87%|243/300|
|roll|last|77%|74%|79%|230/300|
|bounce|best|55%|67%|76%|198/300|
|bounce|last|65%|80%|79%|224/300|

## 稳定性及失败诊断

- roll_best original：触手100/100；失败且曾触手22；终止原因{'timeout': 82, 'ball_on_floor': 14, 'ball_left': 2, 'out_of_bounds': 2}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'below_palm': 19, 'outside_lateral_region': 3}。
- roll_best tennis_ball：触手99/100；失败且曾触手21；终止原因{'timeout': 83, 'ball_on_floor': 12, 'ball_left': 2, 'out_of_bounds': 3}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'below_palm': 21, 'other': 1}。
- roll_best baseball：触手99/100；失败且曾触手12；终止原因{'timeout': 90, 'ball_on_floor': 6, 'out_of_bounds': 3, 'ball_left': 1}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'below_palm': 11, 'outside_lateral_region': 1, 'other': 1}。
- roll_last original：触手98/100；失败且曾触手21；终止原因{'ball_on_floor': 11, 'timeout': 85, 'out_of_bounds': 2, 'ball_left': 2}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'below_palm': 18, 'outside_lateral_region': 4, 'other': 1}。
- roll_last tennis_ball：触手93/100；失败且曾触手19；终止原因{'timeout': 85, 'ball_on_floor': 9, 'out_of_bounds': 4, 'ball_left': 2}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'below_palm': 19, 'other': 3, 'outside_lateral_region': 4}。
- roll_last baseball：触手98/100；失败且曾触手19；终止原因{'timeout': 85, 'ball_left': 2, 'out_of_bounds': 7, 'ball_on_floor': 6}。
  曾稳定持球0%，其中主要成功且曾稳定0%；失败末态区域{'outside_lateral_region': 1, 'below_palm': 19, 'other': 1}。
- bounce_best original：触手100/100；失败且曾触手45；终止原因{'controlled_hold': 55, 'out_of_bounds': 14, 'timeout': 30, 'base_collision': 1}。
  失败中曾满足短暂有效保持18局。
- bounce_best tennis_ball：触手100/100；失败且曾触手33；终止原因{'controlled_hold': 67, 'out_of_bounds': 11, 'base_collision': 2, 'timeout': 20}。
  失败中曾满足短暂有效保持8局。
- bounce_best baseball：触手100/100；失败且曾触手24；终止原因{'controlled_hold': 76, 'timeout': 16, 'out_of_bounds': 8}。
  失败中曾满足短暂有效保持7局。
- bounce_last original：触手100/100；失败且曾触手35；终止原因{'controlled_hold': 65, 'out_of_bounds': 12, 'timeout': 23}。
  失败中曾满足短暂有效保持8局。
- bounce_last tennis_ball：触手100/100；失败且曾触手20；终止原因{'controlled_hold': 80, 'timeout': 16, 'out_of_bounds': 4}。
  失败中曾满足短暂有效保持10局。
- bounce_last baseball：触手99/100；失败且曾触手20；终止原因{'controlled_hold': 79, 'out_of_bounds': 4, 'timeout': 17}。
  失败中曾满足短暂有效保持11局。

## 模型选择

best_reward仅按训练奖励选择，不能保证真实抓取更好；last只是最后保存。按三球独立成功率、稳定性和最弱球种共同选择。小差异存在抽样误差，95% Wilson区间见summary_all.json。本次用于模型选择的种子集并非另留的最终独立验收集。
训练期间success为探索策略及滑动窗口，不能与这里确定性独立回合直接比较。Roll累计日志也不等于当前模型表现。
完整配置、物理参数、模型/源码哈希、逐局数据均在各模型/球子目录。未修改训练奖励和控制。

## Viewer

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/view_roll_bounce_20261103.sh roll original best
bash scripts/view_roll_bounce_20261103.sh roll original last
bash scripts/view_roll_bounce_20261103.sh bounce original best
bash scripts/view_roll_bounce_20261103.sh bounce original last
```
original可替换为tennis_ball/baseball。Bounce先预计算10局，再1倍速回放，不是在线交互控制。

## Roll稳定性补查

best原始球种子0–9完整物理采样；增加只读记录后，逐回合reward/final/success与主测试完全一致。

{
  "episodes": 10,
  "totals": {
    "ticks": 12461,
    "contact": 2698,
    "clear_near": 3833,
    "contact_clear_near": 2664,
    "slow_contact_clear_near": 545,
    "fast_contact_clear_near": 2119,
    "gap_clear_near": 1169,
    "stable_break_speed": 72
  },
  "trajectory_equal": true,
  "max_stable_duration": 0.04600000000000003
}

在离桌离地且位于手区、真实触手的2664个采样中，2119个相对速度大于0.15m/s（约79.5%），10局中观察到72次低速稳定区间因相对速度超限而中断；最长稳定区间0.046秒。该小样本支持接触后速度控制是问题，不证明所有球或所有回合均为同一原因。

核对Roll环境DcmmVecEnv_roll_645edc4.py、physical_v1、配置DcmmCfg_roll_645edc4.py、评价roll_evaluation.py与11-02已测版本哈希一致，当前稳定性差异不是这些代码变化造成。

## 本轮选择与历史基线

推荐查看本轮Roll best262.02与Bounce last。Roll best区域保持78/78/87，高于last77/74/79，但两个模型三球stable_hold均0/100，昨日250.65为21/23/27；应保留旧版，不称本轮整体抓稳能力改善。Bounce last65/80/79，高于本轮best55/67/76；昨日-0.73为61/62/74（相同受控保持标准）。

只改变模型权重的配对对照，不隔离训练次数/策略分支变化等具体成因。选模型后若要最终验收，可使用未参与选择的种子另测。

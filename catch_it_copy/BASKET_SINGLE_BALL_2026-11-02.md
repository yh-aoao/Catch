# 固定底座三种单球独立实验：ballistic_v2

三个实验分别只使用原始球、网球或棒球，各自从零训练独立模型。没有启动训练；没有对真实模型执行优化器更新。

单球实验用于区分奖励/探索难度和混合训练难度。此前的三球模型已经接收半径与质量，不能由低成功率直接推出它只能学一个折中动作，更不能归因于从外貌识别质量失败（策略没有图像输入）。

## 三条训练命令

同步代码，激活dcmm环境后，分别启动所需实验：

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_single.sh original
```

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_single.sh tennis_ball
```

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_single.sh baseball
```

同一GPU建议一次运行一个实验。每条命令默认32个异步环境、GPU0、学习率3e-4、1万亿步上限，可手动停止。可追加`num_envs=16`或`device_id=1`等Hydra参数。

输出目录分别为：

- outputs/BasketOriginalBallisticScratch/
- outputs/BasketTennisBallisticScratch/
- outputs/BasketBaseballBallisticScratch/

三个配置分别是basket_single_original、basket_single_tennis、basket_single_baseball，都继承basket_single_delivery。所有权重路径均为空，网络随机初始化，归一化从头建立，无旧模型参考损失。原41.30及12.48都不参与初始化。

原始球已有41.30结果可以保留。新的原始球任务是相同新奖励下的对照实验，并非要求覆盖旧成果。

## 新奖励

仅新配置启用ballistic_v2，旧的basket_three_ball及固定/移动配置仍使用原奖励，不改变旧实验。

| 项目 | 旧 | ballistic_v2 |
|---|---|---|
| 有效抛出一次性奖励 | +5 | +2 |
| 飞行推进总上限 | +5 | +3 |
| 释放后的弹道瞄准 | 最高+10 | 保留 |
| 真实投进 | +20 | 保留，不放宽成功条件 |
| 机械臂前送 | 最高+6 | 原始上限+2，再乘当前/释放弹道质量 |
| 前送期间掌内保持 | 最高+2 | 原始上限+1，再乘弹道质量 |
| 前送释放 | 最高+4 | 原始上限+2，再乘释放弹道质量 |
| 持球加速的弹道进展 | 无 | 单回合最多约+8，只奖励新最好质量 |
| 未投进的终止回合 | 有效抛出后通常不扣失败分 | 额外-8一次 |

已有无效抛出-5、危险碰撞-40、掌内滑移和手指协同/平滑惩罚保留。因此未有效抛出且失败的回合仍可能同时承担原失败成本和新增未投进成本。

### 弹道质量

根据球当前实际位置、速度、重力和倾斜框平面，预测向下穿过框平面的误差e。质量为：

`q = clip(朝框的水平速度/0.3, 0, 1) * exp(-(e/0.35)^2)`。

无未来有效交点时q=0。这里不预设三种球必须使用同一出手速度或角度，较高位置的合理向下释放也可获得高质量。

持球时以2ms物理步测量，仅在安全手接触、未确认自由抛出时计量。进展奖励为`8 * max(q - 本回合历史最好q, 0)`，初始稳定持球状态作为起点；重复摆动、等待、重复接触不会反复支付同一峰值。实际释放质量在首次确认抛出时根据最后接触状态保存。

额外未投进成本8与进展预算8相当，降低仅拿准备信用却不完成任务的收益。真实投进判定、跟随安全检查、篮框大小、球物理参数均不放宽。

## 环境与手型

底座固定，框保持原投篮位置。保留37维输入/18维机械臂与手指动作；单球实验的半径和质量恒定，不需要判断物体外貌或类别。球类ID仅作日志元数据。

原始球/棒球长指初始姿势[0.8,0.7,0.7]，网球[1.0,0.9,0.9]。原物理准备和碰撞检查保留，不固定球、不添加助力。

## 查看日志与测试

日志包含各球SR、当前单球Task SR、PPO KL，以及Actor updates（本轮实际actor更新次数/尝试次数）。超过KL阈值后暂停本轮后续actor更新，下一轮重采样后重新判断。最佳成功率按当前单球统计选择，不等待其他两球。

例如网球：

```bash
python3 train_DCMM.py --config-name basket_single_tennis \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/网球训练输出/nn/throw_basket_catch_best_success.pth \
  basket_mixed_source='' viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=100 output_name=BasketTennisBallistic_Test
```

原始球/棒球分别使用其对应配置与权重。单球模型不允许在其他球配置下静默测试；旧12.48模型也不允许静默套新奖励配置。旧模型使用原basket_three_ball配置复现。

## 验证

125项相关测试通过，覆盖：

- 能到达框口的出手速度组合优于短抛，不强制所有释放必须向上。
- 静止/反向运动无引导、单回合进展有界、重复摆动不重复得分。
- carry随弹道质量缩放，滑移/安全惩罚保留。
- 失败只结算一次，真正成功不扣未进球成本。
- 实际环境返回奖励与分项/累计分项一致，重置不泄漏上一回合状态。
- 旧奖励权重与新配置不混用、单球best_success选择正确。

三个配置分别使用两个spawn环境验证了正确球类、37/18维存储、固定底座、ballistic_v2奖励与内嵌保存加载；初始化时禁止torch.load，确认未读取旧权重。每个配置还通过真实CLI test=True入口执行了一个完整随机初始化模型回合，验证加载/奖励日志和结束流程；这些不是训练成功率。

检查中未调用真实训练循环，合成更新测试的optimizer.step为空操作。记录：`/home/isee/Catch/analysis_artifacts/basket_single_delivery_20261102/`。

## 服务器同步文件

- configs/config.yaml
- 新增configs/basket_ballistic.yaml（单球/新三球共享奖励）
- configs/basket_single_delivery.yaml（继承共享奖励）
- 新增configs/basket_single_original.yaml
- 新增configs/basket_single_tennis.yaml
- 新增configs/basket_single_baseball.yaml
- train_DCMM.py
- gym_dcmm/envs/basket_mixed.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py
- 新增gym_dcmm/utils/basket_delivery.py
- 新增scripts/train_basket_single.sh
- 新增tests/test_basket_delivery.py

还需保留此前三球版本的模型工具、注册与其他依赖。当前权重加载兼容NumPy标量元数据。

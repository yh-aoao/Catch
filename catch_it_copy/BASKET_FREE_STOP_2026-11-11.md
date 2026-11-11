# 三球距离适应模型加入移动：自主选择区域内的停车位置

初始化权重：/home/isee/Catch/11-11/throw_basket_catch_best_reward_37.86.pth
配置：basket_mobile_free_stop
输出：outputs/BasketThreeBallFreeStop/日期/时间/nn/

## 行为

原模型的框前后偏移范围为±10厘米。本版将该能力转换为相对框的投球区域：在原标称投球距离前后±8厘米内，机器人可以自主停车；保留2厘米余量。这里只改变前后停车距离，横向仍需对准框（默认4厘米容差），不是任意二维或任意朝向抛球。

底座策略通过速度动作选择行驶方向、减速时机和最终位置。区域内没有朝向中心停车点的奖励；关闭旧的固定点刹车锁存和自动接近减速，因此可以穿过区域，选择更里面的位置。区域内速度低于0.04m/s并保持0.30秒、球仍受手支撑后，环境自动解锁抛球。没有新增单独的“开始抛”动作。解锁后底座继续制动；释放仍要求位于区域内，允许底座受机械臂反作用产生的速度不超过原配置0.15m/s。

导航进度、距离与速度奖励按到“最近区域边界”的误差计算；进入区域后这部分误差归零。原标称目标位置仍用于观测，使继承策略的坐标语义保持一致。不存在专门奖励唯一投球距离的项。这个版本可学习可行区域内的位置选择，未承诺学出全局最优距离。

每回合等概率选择三种球，框位置随机且回合内静止。70%移动回合，30%固定位置抛球复习。继续使用ballistic_v2、原球物理与持球手型。底座策略和价值网络新建；完整继承37维抛球策略及归一化，抛球策略继续训练。导航输入53维，动作20维。学习率：底座1e-4、抛球1e-5，沿用分支KL限制及复习约束。

## 训练

在dcmm环境中：
```bash
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_mobile_free_stop \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint=/home/isee/Catch/11-11/throw_basket_catch_best_reward_37.86.pth \
  checkpoint_catching='' basket_mixed_source='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketThreeBallFreeStop
```

服务器需替换工作目录、权重路径和GPU编号。上限1e12，可手动停止。没有自动启动训练。若扩大投球区域，必须先确认源固定模型已在更宽范围训练；源范围校验会拒绝超范围初始化。

## Viewer

使用本版训练产生的新移动权重：
```bash
python3 train_DCMM.py --config-name basket_mobile_free_stop \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/新移动权重.pth \
  basket_mobile.fixed_throw_probability=0.0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=50
```

info中的basket_throw_distance_offset记录相对标称投球距离的偏移，basket_free_stop标识新逻辑。旧basket_mobile_three_ball仍是固定停车点逻辑，其权重不能用新配置直接测试。

## 验证

39项相关测试通过，包括多个停车位置都有效、区域内部无中心引导、可穿过区域、释放必须在区域内、速度门槛与旧移动任务回归。37.86实际权重动作等价、三球重置、固定复习/移动完整回合推理、保存/加载及异步入口检查通过。验证只做推理或模拟优化器调用，没有训练更新；新任务成功率需要训练后评估。

## 服务器同步

- configs/config.yaml
- configs/basket_mobile_free_stop.yaml
- train_DCMM.py
- gym_dcmm/__init__.py
- gym_dcmm/envs/basket_free_stop.py
- gym_dcmm/utils/basket_mobile.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py
- tests/test_basket_free_stop.py（可选）

依赖此前已同步的三球移动、范围固定抛及ballistic_v2代码。

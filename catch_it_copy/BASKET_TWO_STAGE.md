# Basket两阶段训练

入口使用`task=Catching_TwoStage`与Basket专用两阶段适配器。
适配器共享现有PPO回放/更新实现，但两阶段有独立Actor、优化器、输入归一化和动作空间。
旧的其他运动模式路由保持原样。

第二阶段停稳延迟、共享时限和无效动作更新的修正版使用`basket_throw_first`，
见[BASKET_THROW_FIRST.md](BASKET_THROW_FIRST.md)。原命令保留用于已有模型复现。

## 第一阶段：持球移动和停车

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_stage1 \
  test=False num_envs=32 vector_mode=async device_id=0 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=BasketStage1_Parking
```

51维状态输入，策略只输出底盘2维；臂手通过原关节控制器保持物理托球姿态。
框每回合随机，回合内静止。停车点参考框距、减速限制与迁移配置一致。
持球、到位且速度合格，连续停稳0.30秒后以`parking_score`成功结束，成功不扣“未投掷”罚分。
落球、危险碰撞和超时失败。这个阶段`mean_success`是停车成功率，不是入框率。
`safe_basket_success`仍为false，避免将停车和入框混淆。
不加载已有投掷模型，不在停车阶段学习臂手动作。

先评估停车成功率再进入第二阶段：

```bash
python3 train_DCMM.py --config-name basket_stage1 \
  test=True num_envs=1 vector_mode=sync device_id=0 viewer=true basket_log=true \
  checkpoint_catching=/绝对路径/第一阶段模型.pth \
  train.ppo.test_num_episodes=200 output_name=BasketStage1_Test
```

两阶段均将权重保存在实际输出目录的`nn/`内。默认累计至少50个完整回合后，
按成功统计改善保存`throw_basket_catch_best_success.pth`；也保留best_reward和last。
停车成功率尚低时，不能仅凭出现best_success文件就开始第二阶段。

## 第二阶段：冻结停车策略，训练臂手投篮

```bash
python3 train_DCMM.py --config-name basket_stage2 \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_parking_checkpoint=/绝对路径/第一阶段模型.pth \
  basket_warm_start=/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=BasketStage2_Throw
```

第一阶段的底盘Actor及其输入归一化被冻结，使用确定性动作；不加入第二阶段优化器。
到位后请求底盘零速度，通过原控制器制动，没有固定底盘位置或直接修改速度。
第二阶段策略只输出臂6＋手12共18维，PPO概率、熵和梯度更新只针对这些动作。
到位前这些动作由环境屏蔽、保持托球；到位后开放，学习投掷。
旧固定基座Actor按35维到51维输入显式映射，臂手输出保持18维，不发生输出行偏移。
对应归一化参考处理和价值网络重新初始化规则见`BASKET_MOBILE_TRANSFER.md`。
第二阶段臂手Actor参与训练，以适应移动后的持球状态。

如需第二阶段臂手也从零开始，可覆盖`basket_warm_start=''`。
`basket_parking_checkpoint`用于加载第一阶段策略，`basket_warm_start`用于初始化旧投掷Actor，
两者用途不同；训练时不传`checkpoint_catching`。

第二阶段`mean_success`才是安全入框率；不会把持球停车作为最终成功。
第一阶段和第二阶段的目标范围、停车容差、限速等移动配置须一致；仅`parking_only`不同。
代码拒绝不匹配的第一阶段模型及原单策略20动作模型。

## 第二阶段测试

```bash
python3 train_DCMM.py --config-name basket_stage2 \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  checkpoint_catching=/绝对路径/第二阶段模型.pth \
  train.ppo.test_num_episodes=200 output_name=BasketStage2_Test
```

第二阶段权重包含冻结的停车策略及其归一化统计，测试不再依赖第一阶段文件或旧投掷初始化文件。

## 验证边界

80项相关测试通过，包括阶段动作布局、第一阶段成功奖励、旧Actor映射、冻结策略及其统计。
真实MuJoCo中两个阶段均完成CPU异步短训练和保存；逐张量核实第二阶段PPO更新后，
底盘Actor和其输入统计与第一阶段文件完全一致。
第二阶段保存后的模型可独立加载并运行有限回合评估。
这些检查验证接口和更新逻辑，不能替代完整训练的收敛与安全入框评估。

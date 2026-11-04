# 三球弹道模型的移动基座第二阶段

源模型：/home/isee/Catch/11-4/throw_basket_catch_best_reward_33.25.pth
新配置：basket_mobile_three_ball
输出：outputs/BasketThreeBallMobileJoint/日期/时间/nn/

每回合三球等概率抽样，半径与质量输入保留。继承固定模型完整37维机械臂/手指策略及归一化；底座策略和价值网络新建。导航原51维状态追加半径和质量，成为53维。内部归一化输出为53维导航输入、37维抛球输入、原始阶段标记，共91维；最终动作仍为底座2维、机械臂6维、手指12维。

70%回合框位置随机，x在[-0.20,0.20]，y在[1.15,1.55]，回合内静止；持球导航、停车稳定、抛球，抛球阶段底座制动。30%回合在固定位置复习原抛球任务，依然随机抽球。所有当前策略分支可训练，固定的参考策略仅用于复习约束。两类回合的抛球损失按现有稳定方案平衡。

保留ballistic_v2、各球物理参数和手型。移动阶段不支付弹道进度奖励；增加导航、停车、制动等项；未入框使用ballistic_v2的-8终止扣分，避免重复移动任务失败扣分。底座学习率1e-4，抛球1e-5，分别限制KL，延续已验证的移动训练机制。

## 启动

在dcmm环境下：
```bash
cd /home/isee/Catch/catch_it_copy
bash scripts/train_basket_three_ball_mobile.sh
```

等价完整命令：
```bash
python3 train_DCMM.py --config-name basket_mobile_three_ball \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint=/home/isee/Catch/11-4/throw_basket_catch_best_reward_33.25.pth \
  checkpoint_catching='' basket_warm_start='' basket_mixed_source='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketThreeBallMobileJoint
```

服务器上将basket_throw_checkpoint改为服务器实际权重路径，并按需指定device_id。脚本接受相同Hydra覆盖参数。没有自动启动训练。

## 监控与测试

Mobile SR是移动回合，Replay SR是固定复习回合，Basket SR是总体；新增original Mobile SR、tennis_ball Mobile SR、baseball Mobile SR分别统计各球的移动回合。不要将总体成功率当作纯移动成功率。

新模型viewer：
```bash
python3 train_DCMM.py --config-name basket_mobile_three_ball \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/移动训练权重.pth \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  basket_mobile.fixed_throw_probability=0.0 \
  train.ppo.test_num_episodes=100
```

当前输入仍为仿真球半径、质量真值。

## 验证与同步文件

44项相关测试通过，包含模拟损失反向传播检查（optimizer.step被替换，未更新权重）。实际33.25源模型动作等价性、异步环境入口、固定/移动完整回合推理、三球重置和保存/加载均做检查；新任务移动成功率需训练后评估。

服务器需同步以下文件（及已有单球移动/三球弹道依赖）：
- configs/basket_mobile_three_ball.yaml
- scripts/train_basket_three_ball_mobile.sh
- train_DCMM.py
- gym_dcmm/envs/basket_mixed.py
- gym_dcmm/algs/ppo_dcmm/basket_mobile_joint.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py

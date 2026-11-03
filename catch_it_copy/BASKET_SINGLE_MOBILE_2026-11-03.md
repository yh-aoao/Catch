# 网球、棒球移动基座第二阶段

两个独立任务，分别从 11-3 中对应的 last 抛球权重初始化；底座策略和价值网络从零开始，机械臂与手指策略继续训练。原来的单球固定底座配置仍可使用。

## 行为与训练

- 网球、棒球各自保持其固定底座训练的半径、质量、惯量、摩擦、持球手型及 ballistic_v2 抛球奖励。
- 70% 回合随机框位置：世界坐标 x ∈ [-0.20, 0.20]，y ∈ [1.15, 1.55]，回合内静止。携球移动，停车并稳定后解锁抛球；抛球时底座制动。
- 30% 回合使用原固定位置复习抛球，防止遗忘。复习与移动抛球的策略损失平衡；原策略只作为复习约束目标，当前机械臂/手指依然接受 PPO 更新。
- 底座学习率 1e-4，抛球学习率 1e-5；分支 KL 上限分别为 0.02、0.01；保留原稳定版本的分支停止更新机制。
- 添加导航进度、停车、制动与控制奖励；ballistic_v2 的终止未命中扣分为 -8，移除移动奖励额外的 -15 task_failure，避免重复惩罚。碰撞与无效抛球惩罚保留。
- 移动期间不支付弹道进度奖励，避免用底座运球速度刷抛球奖励。
- 不增加 RGB 或球类识别输入。单球训练中的半径、质量是常量，37 维网络的这两个常量输入被精确合并到第一层偏置，抛球分支保留35维接口，动作输出不变。这只适用于指定单球，不支持将混合三球策略这样转换。

## 训练命令

在 dcmm 环境下：
```bash
cd /home/isee/Catch/catch_it_copy
bash scripts/train_basket_single_mobile.sh tennis_ball
bash scripts/train_basket_single_mobile.sh baseball
```

两个命令分别启动独立训练。默认32个异步环境、GPU0、上限1e12步，按需手动停止。两项同时运行须按机器资源分配 GPU，例如命令末尾添加 device_id=1。

换机器或更换初始权重时：
```bash
bash scripts/train_basket_single_mobile.sh tennis_ball \
  basket_throw_checkpoint=/完整路径/throw_basket_catch_last.pth
```

默认输出分别为 outputs/BasketTennisMobileJoint 与 outputs/BasketBaseballMobileJoint。

## Viewer 测试

使用第二阶段新生成的移动权重，不是原固定底座权重：
```bash
python3 train_DCMM.py --config-name basket_mobile_tennis \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/nn/throw_basket_catch_best_success.pth \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=100
```

棒球改用 --config-name basket_mobile_baseball 和对应移动权重。默认测试仍含30%固定复习回合，查看 Mobile SR 与 Replay SR；只测移动回合可加 basket_mobile.fixed_throw_probability=0.0。

## 验证与同步

40项相关单元测试通过；两个球种均通过异步入口检查和固定复习/移动完整回合推理。验证包括源策略等价性、球种/奖励协议拒绝检查、51维移动接口、参数可训练性、推理、保存/加载以及异步命令入口。未执行训练或真实优化器更新。新任务实际移动成功率须训练后评估。

服务器需同步：
- train_DCMM.py
- gym_dcmm/envs/basket_mixed.py
- gym_dcmm/algs/ppo_dcmm/basket_mixed_policy.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py
- configs/basket_mobile_tennis.yaml
- configs/basket_mobile_baseball.yaml
- scripts/train_basket_single_mobile.sh

这些文件依赖已有完整的 basket_mobile41 与单球 ballistic_v2 代码。

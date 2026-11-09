# 固定底座三球投球距离适应

这是自主停车之前的第一步：底座保持固定，框的世界Y坐标在原固定位置基础上随机偏移，回合内静止。框X、高度和倾角不变。三种球等概率抽样，半径和质量仍作为策略输入。

配置 basket_three_ball_range 从三球固定抛33.25模型继承完整37维策略、归一化和价值网络，重新建立优化器，以1e-5学习率继续PPO训练。策略参数均可训练，保留原归一化坐标系。奖励、物理、持球和释放协议保持ballistic_v2不变，不额外规定抛球力度或动作。

默认70%回合在原框Y坐标±0.10米内均匀抽样，30%回合保持原位置。注意这是框的位置偏移范围，不是球的飞行距离，也不是机器人底座中心到框的距离。不会自动扩大范围，以免在尚未学会时增加难度。当前移动配置及指定停车点逻辑保持原样。

## 训练

在dcmm环境中：
```bash
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_three_ball_range \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source=/home/isee/Catch/11-4/throw_basket_catch_best_reward_33.25.pth \
  checkpoint_catching='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 \
  output_name=BasketThreeBallRange10cm
```

服务器替换basket_mixed_source为该服务器的固定底座三球弹道权重完整路径（如33.74），不可使用移动权重37.56。默认不是从零训练，不会自动启动。

## Viewer及距离分层测试

```bash
python3 train_DCMM.py --config-name basket_three_ball_range \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/新距离适应模型.pth \
  basket_mixed_source='' \
  basket_range.original_probability=0.0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=100
```

分别添加以下参数做固定偏移测试，避免总体成功率掩盖边缘位置失败：
- basket_range.y_offset_range='[-0.10,-0.10]'
- basket_range.y_offset_range='[0.0,0.0]'
- basket_range.y_offset_range='[0.10,0.10]'

每个偏移分别检查三种球；可用basket_mixed_ball指定球种评估已训练mixed模型。三种球及各距离稳定后，用新距离适应权重继续训练并设置更宽范围，例如basket_range.y_offset_range='[-0.15,0.15]'。这不是自动课程。

## 验证和同步

23项相关测试通过，包括范围随机、原位置复习、目标回合内静止、无基准漂移及无效参数拒绝。实际33.25权重严格继承、三球重置、零底座动作、推理、保存/加载和异步命令入口均通过。只运行推理和模拟反向传播检查，没有真实优化器更新。新范围成功率须训练后测量。

同步：
- configs/config.yaml
- configs/basket_three_ball_range.yaml
- train_DCMM.py
- gym_dcmm/envs/basket_mixed.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py
- tests/test_basket_range.py（可选，用于服务器验证）

本版没有实现移动策略自主选择停车点或释放阶段；下一步须在验证可投距离范围后，修改停车点约束及阶段选择。

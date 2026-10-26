# 固定底盘近目标入框 basket_aim_v1

从零开始，不加载旧 checkpoint。旧 launch_simple_v1 奖励和目标保持不变。
动作18维（arm6+hand12），输入沿用原Basket35维，包含目标相对位置。
近目标中心世界坐标[0,1.05,0.55]，半径沿用0.20m，倾角沿用25度。
这是待真实MuJoCo验证的起点，非已验证最优目标位置。
配置位于 gym_dcmm/utils/basket_aim.py 的 make_config，以环境私有配置隔离旧任务。

奖励：有效释放+5；前向自由飞行距离累计最多+5（0.5m封顶）；
释放状态的下降穿越预测误差e给予10*exp(-(e/0.25)^2)，一次；无有效未来交点为0；
真实有效入框+20，一次；未释放失败-5；机器人危险碰撞-40一次；
机械臂归一化动作变化惩罚-0.01*mean(delta_action^2)。
实际入框采用球中心轨迹从入口正侧到负侧穿越，穿越半径<=框半径-球半径。
要求有效释放，无再接触/其他碰撞，至少0.1秒空中飞行、0.2米水平位移。
命中后观察0.3秒，落地不结束观察，机器人碰撞仍终止并扣分。
这是穿过入口任务，不是实体容器内稳定停留任务。
info.basket_hit表示曾有效穿过；safe_basket_success与success表示完成安全观察后的入框。
原 PPO mean_success 因而可用于该版本的安全入框统计。

训练（catch_it_copy目录）：
```bash
python3 train_DCMM.py test=False task=Catching_OneStage num_envs=32 vector_mode=async \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_reward_version=basket_aim_v1 \
  basket_curriculum_stage=3 basket_log=false basket_self_collision_debug=false \
  device_id=0 output_name=BasketAimV1_FromScratch
```
不传checkpoint_tracking/checkpoint_catching。首次服务器验证可用vector_mode=sync并开启basket_log。
测试同一套任务参数，加test=True num_envs=1 viewer=true basket_viewer_speed=0.25
basket_log=true checkpoint_catching=/path/to/model.pth。

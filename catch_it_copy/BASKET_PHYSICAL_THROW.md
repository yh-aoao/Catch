# Basket：由臂手接触完成投掷

后续新增接触加速、有限支撑、一次性有效脱手奖励和控制诊断，详见
[Roll / Basket 最新塑形](ROLL_BASKET_REACH_THROW_UPDATE.md)。下文基础机制仍保留。

替换此前“准备 0.3 秒后注入速度”的实现。用户要求模型自己抛球，因此主环境不再提供
定时弹射、额外初速度、投掷阶段的反重力托举或反复写入球位置。

## 当前流程

- Tracking 及 Catching 的停车阶段：保留脚本托球和固定臂手姿态，提供持球初态，停车模型可复用。
- 连续停稳后进入 preparing：关闭托举，此后不再写球的 qpos/qvel，也不会因底座偏移重新吸附球。
  机械臂和手指动作由策略控制。球受重力和真实碰撞接触作用，必须通过臂手运动获得投掷速度。
- 检测到真实掌/指接触后，球脱离接触且距 link6 超过两倍球半径，标记 flight。
  这个标记只用于日志和奖励切换，不施加力，也不设置速度。
- 从 preparing 开始就检测落地、越界和穿框；从未接触手便掉落同样失败。
  得分要求曾有手部接触、已经脱手、当前无手部接触，并真实穿过有效圆孔。

策略通过原有 6 维机械臂和 12 维手指动作学习加速和松手，没有额外的“释放按钮”。
底座两维策略在 TwoStage 默认冻结，停车后给零速度指令。观测/动作维度仍为 35/20。

## 奖励

停车奖励不变。准备阶段以**仿真中的实际球速度**预测弹道，
`r = -2*predicted_error - 0.02*||arm_action||² - 0.01*||hand_action||² - 0.05`。
相比上一版降低臂手控制惩罚，避免有效挥臂被过度惩罚；不直接奖励速度大小。
飞行阶段保留距离进展奖励 `5*(d_previous-d)-0.05`；入框 +100，失败/超时 -20。
预测弹道只作为塑形，不代替穿框成功判定。回合仍为 4 秒，无固定准备时长。

## 训练与验证

需要重新训练 Catch；旧模型依赖外加速度，不能用其表现评价新机制。

```bash
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=throw_basket checkpoint_tracking=/path/to/basket_track.pth output_name=BasketPhysicalThrow
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=throw_basket checkpoint_catching=/path/to/new_basket_catch.pth object_eval=false viewer=true basket_log=true
```

先短跑观察是否能从停车托球初态自然获得接触支撑，再开始长训练。如果球在准备阶段
立刻落下，需要检查掌心初始球位、碰撞几何和握持姿态，不能恢复脚本弹射来掩盖问题。
这个版本仍简化了停车运输和持球初始化，不宣称已学会全程自主抓取、携带和投掷。

CPU 回归测试验证停车后不再设置球位姿/速度、时间不触发弹射、脱手检测不改变球速，
以及奖励、穿框与冻结底座的相关逻辑。本机 MuJoCo DLL 无法加载，尚未验证真实物理
训练的收敛或投篮成功率。

Linux 需同步主环境、`gym_dcmm/utils/basket_catching.py`、`roll_rewards.py`、
Python 配置及此前的 TwoStage PPO/模型改动。旧的释放时长、速度补偿和速度封顶配置已删除。

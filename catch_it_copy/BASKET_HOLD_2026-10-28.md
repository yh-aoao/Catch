# 下一步：稳定持球后投掷

先训练固定底盘速度请求、固定近目标下的延时投掷，再接回已有停车模块。
`basket_hold`是独立任务配置：只更新臂6＋手12的18维Actor，51维状态输入，
不加载停车Actor；已有停车模型保留给后续完整移动任务。
默认以原39.44模型初始化Actor，价值网络重新初始化，输入归一化保持冻结。

## 物理起点和动作开放

- 仅在reset放球时沿世界Z抬高到与手碰撞几何至少1mm间隙，随后通过重力落入杯状手。
  搜索范围最多8cm，无法清除穿入时报错；不在仿真过程中传送球或施加托球力。
- 当前模型诊断使用约2.4cm抬升，最小间隙约1.3mm，初始球接触不再穿入。
- 底盘始终请求零速度；物理动力学保留，未焊接底盘或直接改写实际速度。
- 等待期间臂手控制器保持初始托球关节目标，策略动作被屏蔽且不参与Actor更新。
- 默认每回合等待0.3～0.6秒，再检查球仍接触手、底盘满足停稳条件、球世界速度<0.15m/s。
  条件满足才开放臂手，投掷另有2.5秒预算；不满足时继续等待或按原失败条件结束。
- 始终要求真实有效释放、最小自由飞行、真实入口穿越和命中后0.3秒安全观察。

## 奖励

沿用释放＋5、飞行距离累计最多＋5、一次性瞄准最多＋10、实际命中＋20，
以及失败、危险碰撞、平滑和时间代价。
新增仅在开放后且球接触手时的弹道准备进度：只奖励历史最好预测质量的改善，累计最多＋4。
首次持球状态作为基线，不给初始姿态奖励；反复摆动不能重复得到同一质量的奖励。
未释放失败的总罚分仍大于这项准备奖励，因此不能靠持续持球获利。
本任务没有停车成功＋3，也不会将等待结束作为回合成功。

## 先做有限预算训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_hold \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_warm_start=/home/isee/Catch/10-25/throw_basket_catch_best_reward_39.44.pth \
  train.ppo.max_agent_steps=1000000 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  output_name=BasketHold_03_06
```

这是从旧Actor开始适应稳定持球状态的新训练，不加载当前失败的移动模型，也不运行长期训练脚本。
初期负奖励不等于故障；主要看`HeldThrow SR`和有效释放、入框、碰撞结果。
先用100万环境步作为一次检查预算，不承诺该预算内收敛。
如果该预算结束后成功率仍接近0，应检查握球/准备动作和奖励反馈，避免原样扩大长训练。

## 测试

```bash
python3 train_DCMM.py --config-name basket_hold \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  checkpoint_catching=/绝对路径/本次训练生成的模型.pth \
  train.ppo.test_num_episodes=200 output_name=BasketHold_Test
```

此时`mean_success`是真实安全入框率，不是等待或停车成功率。
模型保存为`held_arm6_hand12`布局，避免误用成原35维模型或底盘Actor。
测试允许覆盖等待区间，例如：

```bash
python3 train_DCMM.py --config-name basket_hold \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  viewer=false 'basket_mobile.hold_time_range=[0.3,2.0]' \
  checkpoint_catching=/绝对路径/本次训练生成的模型.pth \
  train.ppo.test_num_episodes=200 output_name=BasketHold_LongWait_Test
```

建议先让0.3～0.6秒这档200回合安全入框率达到约90%，且没有明显碰撞问题，
再扩大等待时间并训练覆盖0.3～2秒；这是工程验收目标，不是已验证的训练结果。
延时区间内还应分档检查，避免均值掩盖较长等待失败。

## 通过后再接回移动

`basket_after_hold`用于后续第二步：新持球投掷Actor初始化、已有停车Actor冻结，
固定目标回放也等待后再投；重置使用相同的无穿入放球方式。
显式的51维持球Actor复用不会再次对停车特征做0.2m归一化偏移。

```bash
python3 train_DCMM.py --config-name basket_after_hold \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_warm_start=/绝对路径/通过延时测试的新持球投掷模型.pth \
  basket_parking_checkpoint=/home/isee/Catch/catch_it_copy/outputs/BasketStage1_Parking/2026-10-27/19:23:21/nn/throw_basket_catch_best_reward_3.50.pth \
  viewer=false basket_log=false output_name=BasketAfterHold
```

此命令等固定任务通过后再运行。当前没有可宣称已通过的持球投掷权重。

## 验证边界

- 92项相关测试通过，新增延时开放、球速度门槛、零底盘动作、准备奖励上限、51维持球Actor复用测试。
- MuJoCo持球诊断验证初始无穿入、等待期间没有球驱动力、臂手动作被屏蔽、底盘零速度请求。
- CPU异步短训练、保存、有限回合测试和后续Actor加载接口已跑通。
- 短训练测试入框仍为0，没有证明模型已学会稳定持球投掷；未启动长期GPU训练。

诊断脚本：`MUJOCO_GL=egl python3 basket_hold_probe.py --episodes 15 --hold-min 0.3 --hold-max 2.0`。
它检查持球和动作开放条件，不评估学到的投掷策略。

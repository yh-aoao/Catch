# Basket 41.30 → 第二阶段联合强化学习

当前已应用[联合训练稳定性修正](BASKET_JOINT_STABILITY_2026-11-01.md)：固定学习率、分策略KL保护、复习损失平衡和原策略软约束。启动前请同步全部修改。

当前 `configs/basket_mobile41.yaml` 已改为 `basket_training_stage: mobile_joint`，不再冻结机械臂/手指 actor。旧的 `navigate_throw` 冻结实现保留供历史实验使用，默认脚本不再选择它。

没有启动训练，没有执行优化器更新。

## 第一阶段与第二阶段

第一阶段是现有固定底座模型 `/home/isee/Catch/11-1/throw_basket_catch_best_reward_41.30.pth`，不需要重训。

第二阶段创建底座策略，继承41.30的机械臂/手指 actor及其探索标准差，底座和机械臂/手指均参加PPO更新。独立critic从头学习整个移动抛球回合的价值。

| 部分 | 初始化 | 第二阶段更新 | 初始学习率 |
|---|---|---|---|
| 底座XY策略 | 随机 | 是，导航阶段 | 1e-4 |
| 机械臂6+手指12策略 | 41.30 | 是，停车后抛球阶段及复习回合 | 1e-5 |
| 整个回合的critic | 随机 | 是，所有阶段 | 1e-4 |

训练的20维动作是base2+arm6+hand12。导航时执行底座动作，手臂/手指保持已有托球目标；停车后执行机械臂/手指策略，底座制动。两个策略在同一次第二阶段PPO训练中持续更新，并非把抛球actor冻结。

未执行的动作通道不计入PPO概率、熵或KL。只有一种阶段的minibatch会清除另一个actor的梯度（设为None），避免Adam动量更新没有执行动作的分支。critic继续学习所有阶段。

机械臂/手指归一化统计沿用第一阶段，不随移动样本重算；冻结的是坐标变换的统计量，actor权重和探索标准差可训练。底座归一化从头统计。默认固定学习率，机械臂为底座的0.1；可选KL调度分别控制两组且不超过初始上限。

每回合有30%概率在原固定框位置复习抛球，70%概率随机框位置训练移动后抛球。复习回合也执行物理持球准备，不辅助固定球。终端日志分别报告Mobile SR和Replay SR，最佳成功率模型按Mobile SR选取。

## 移动环境及奖励

框位置每回合随机、回合内静止。默认范围X=[-0.20,0.20]m，Y=[1.15,1.55]m。停车点保持原固定抛球的框到arm_base间距，由实际初始位置标定。

底座速度≤0.30m/s，加速度≤0.35m/s²。XY各轴误差<0.04m、底座速度<0.04m/s，连续稳定0.30s后进入抛球。导航最多10s，到位后另给2.5s抛球和飞行。最后接触时释放资格为各轴位置误差<0.08m、底座速度<0.15m/s，以容纳机械臂前送反作用，不替代开始抛球前的严格停车条件。

保留原有效抛出、飞行推进、弹道瞄准、真实投进、碰撞、机械臂平滑、前送/持球和手指协同奖励，叠加导航进展、位置/速度误差、停车、制动、底座控制/平滑、时间/失败成本以及导航中的掌内滑移成本。前送奖励从停车解锁后的机械臂相对底座运动计量，底座移动不赚机械臂前送奖励。

配置中的历史名称`preserve_fixed_throw: true`表示继续使用41.30的物理准备和奖励约定；是否冻结actor由训练器决定，当前`mobile_joint`不会冻结actor。

## 训练

激活dcmm环境，先同步最新代码到实际训练服务器，再执行：

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_mobile41.sh \
  ../11-1/throw_basket_catch_best_reward_41.30.pth
```

等价命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_mobile41 \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint="$(realpath ../11-1/throw_basket_catch_best_reward_41.30.pth)" \
  checkpoint_catching='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketMobile41JointStable
```

32个异步环境、GPU0、1万亿步上限，手动停止。来源参数是`basket_throw_checkpoint`，不要把原始41.30填入`checkpoint_catching`。新权重保存在`outputs/BasketMobile41JointStable/日期/时间/nn/`，包括last、best_reward和best_success。新权重内含两个actor和两套归一化，无需原始41.30文件即可测试。

新联合权重与之前冻结版本的checkpoint结构不同，不能混用。

## 测试纯移动任务

```bash
python3 train_DCMM.py --config-name basket_mobile41 \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/移动训练输出/nn/throw_basket_catch_best_success.pth \
  basket_throw_checkpoint='' basket_mobile.fixed_throw_probability=0.0 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=100 output_name=BasketMobile41Joint_Test
```

观察parking_success、qualified_release、safe_basket_success，以及训练日志的Mobile SR/Replay SR。移动版本增加了奖励项和成本，其总奖励不直接与固定版本41.30比较。

## 本次验证

- 94项Basket测试通过：包括源actor输出/标准差一致、两套统计、联合动作与存储维度、阶段概率屏蔽、两个actor的非零PPO梯度、单阶段清理梯度、KL尺度、保存重载，以及已有Basket回归测试。
- 用真实41.30构造联合训练器，确认所有actor参数在优化器中、没有重复注册、学习率比例10:1。对合成样本只执行反向传播，未执行optimizer.step。
- 新联合checkpoint的CLI推理加载通过，测试时原始41.30路径置空，固定复习10回合安全投进10/10；说明继承与加载未破坏初始抛球，不代表第二阶段训练后的移动成功率。
- 编译、shell语法、git diff --check通过。

本次没有启动训练。诊断文件在`/home/isee/Catch/analysis_artifacts/basket_mobile_joint_20261101/`。

本次需同步的文件：

- `configs/basket_mobile41.yaml`
- `configs/config.yaml`
- `train_DCMM.py`
- `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py`
- 新增`gym_dcmm/algs/ppo_dcmm/basket_mobile_joint.py`
- 新增`gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py`
- `scripts/train_basket_mobile41.sh`
- 新增`tests/test_basket_mobile_joint.py`

如果服务器此前尚未同步移动版本，还需同步前一版修改的`gym_dcmm/envs/DcmmVecEnv.py`、`gym_dcmm/utils/basket_mobile.py`、`gym_dcmm/utils/basket_throw_policy.py`等移动支持文件。

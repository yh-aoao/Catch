# Basket 41.30 → 移动底座版本

新配置：`configs/basket_mobile41.yaml`。没有启动训练，也没有执行任何优化器更新。

## 策略结构

- 复用 `/home/isee/Catch/11-1/throw_basket_catch_best_reward_41.30.pth` 的机械臂/手指 actor、35 维观测归一化，冻结其参数。使用确定性动作。
- 新增独立的底座 actor/critic：51 维观测、2 维 XY 速度动作，从随机初始化训练；归一化重新统计。
- 这是在已学会抛球技能之上学习移动，不是从零重学抛球，也不是让 41.30 的机械臂参数在第一版中继续更新。
- 两个策略一并保存到新 checkpoint；测试时不再读取原始 41.30 文件。固定底座模型不能直接作为这个模式的 `checkpoint_catching`。
- PPO 的底座 actor 仅在导航阶段计算动作损失；停车后的回报仍通过优势估计训练之前的导航动作，critic 学习整个回合。

## 环境衔接

重置时以新手型进行无外力辅助的物理持球准备。此后不设置球位置/速度、不绑定球。移动期间使用原托球关节目标；到位并稳定后切换到冻结抛球 actor。只重置前送奖励的计量起点，不重置机器人或球状态。

框位置每回合随机，回合内静止。第一版范围 X=[-0.20,0.20]m、Y=[1.15,1.55]m。停车目标保持原固定抛球的框到 arm_base 间距，由实际初始 arm_base 位置标定。

底座速度≤0.30m/s，加速度≤0.35m/s²。XY 位置各轴误差<0.04m、速度<0.04m/s，连续稳定0.30s后抛球。到位停车带速度包络；离开到位容差即恢复位置修正，避免旧的停车滞回带造成无法解锁。

抛球阶段底座指令为零；加速度限制逐步归零。最后接触时速度资格放宽到0.15m/s，位置各轴容差0.08m，用于容纳机械臂前送反作用。它不替代抛球开始前的严格停车条件。导航最多10s，到位后另给2.5s抛球/飞行窗口。

## 奖励

保留原有效抛出、飞行推进、弹道瞄准、真实投进、碰撞、机械臂平滑，以及前送/持球、手指协同项。前送奖励只在解锁后计量，底座移动不赚机械臂前送奖励。碰撞成本沿用40。

叠加现有移动任务的有符号距离进展、位置/速度误差、停车一次性奖励、制动、底座控制/动作平滑、时间和失败成本。导航时增加球相对初始稳定掌内位置的滑移成本：`-2*(1-exp(-sum((超出容差的局部偏差/sigma)^2)))*dt`，不奖励站着无限等待。

## 训练命令

激活 dcmm 环境后：

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_mobile41.sh \
  ../11-1/throw_basket_catch_best_reward_41.30.pth
```

脚本用32个异步环境、GPU0、步数上限1000000000000，可手动停止。输出为 `outputs/BasketMobile41/日期/时间/nn/`，包含 last、best_reward 和 best_success 模型。不要把原始41.30填入 checkpoint_catching；本版本的来源参数是 basket_throw_checkpoint。

等价命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_mobile41 \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_throw_checkpoint="$(realpath ../11-1/throw_basket_catch_best_reward_41.30.pth)" \
  checkpoint_catching='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketMobile41
```

## 测试命令

```bash
python3 train_DCMM.py --config-name basket_mobile41 \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/移动训练输出/nn/throw_basket_catch_best_success.pth \
  basket_throw_checkpoint='' viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=100 output_name=BasketMobile41_Test
```

优先看 parking_success、qualified_release、safe_basket_success 三个指标。41.30原始训练奖励和叠加移动成本后的奖励不可直接比较。

## 验证（仅推理和构造，不训练）

- 87项 Basket 回归/新测试通过，覆盖原actor输出一致性、归一化冻结、导航时屏蔽抛球、动作通道分离、停车解锁/制动、停车滞回边界、滑移奖励单次结算、权重导出重载。
- 已用实际41.30构造新PPO，确认被冻结actor不在优化器参数中；新权重可内嵌加载原actor，随后执行一次向量环境动作。没有调用 train/train_epoch/optimizer.step。
- 在原投篮位置通过新结构推理10回合：10/10安全投进。
- 在默认随机目标范围，由诊断用比例底座控制器导航，再由冻结41.30抛球，seed0..9：10/10安全投进。**这验证环境/技能链路，不是训练后的底座策略成功率；实际底座actor还未训练。**
- CLI test=True 使用临时的未训练底座 checkpoint，验证配置、嵌入actor读取和回合结束均正常；该随机底座回合超时，符合未训练状态。临时权重不属于可用训练成果。
- Python编译、shell语法和git diff --check通过。

记录：`/home/isee/Catch/analysis_artifacts/basket_mobile41_20261101/`。

服务器运行前需同步本次代码，新增与修改文件包括：configs/basket_mobile41.yaml、configs/config.yaml、train_DCMM.py、gym_dcmm/envs/DcmmVecEnv.py、gym_dcmm/utils/basket_mobile.py、gym_dcmm/utils/basket_throw_policy.py、gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py、gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_two_stage.py、scripts/train_basket_mobile41.sh；测试文件 tests/test_basket_mobile41.py。

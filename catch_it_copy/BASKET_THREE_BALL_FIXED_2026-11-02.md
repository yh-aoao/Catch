# 固定底座三球混合抛球（已知半径/质量）

已提供代码、配置和脚本；没有启动训练，也没有对真实模型执行优化器更新。

## 任务与观测

每回合按相同概率选择原始球、网球或棒球；回合内不切换球。底座固定，框固定在原投篮位置。使用原抛球、机械臂前送、持球和手指协同奖励，保留原有成功与安全判定。

原35维观测后追加球半径（米）、质量（千克），共37维；策略动作仍为机械臂6+手指12共18维。进入网络前，新增两维转换为`(radius-0.04)/0.01`与`log(mass/0.05)`。不使用图像、颜色或球类ID作为策略输入。球类ID只用于采样记录、损失平衡和分类成功率。

这是一版“已知物理参数条件下”的三球策略：仿真直接提供半径和质量，不代表模型学会从外观识别质量。实际部署需要测量/估计这些参数；未知动力学自适应仍需观测历史或力觉等支持。

## 球与持球准备

- 原始球沿用现有41.x协议的实际球模型参数（包括原惯量、摩擦、阻尼），从其他球切回时恢复缓存值。
- 网球：半径约0.03328m，质量0.058kg。
- 棒球：半径约0.03618m，质量0.138kg。
- 其他两球的惯量与接触摩擦来自现有`balls.roll_table_priors_v1.json`，是仿真先验，不是实物标定结果。使用解析球形碰撞体和颜色区分，未加载YCB视觉网格。
- 原始球和棒球初始长指屈曲为[0.8,0.7,0.7]，网球为[1.0,0.9,0.9]；拇指均为[0.4,0.4,0.4]。
- 初始手型由环境按已知球类选择，本版不训练从任意手型抓稳球。物理准备仍靠重力和真实接触，不绑定球、不施加助力；稳定后才开始策略回合。

三种球各20个种子的物理准备均通过；这不是投篮成功率。

## 初始化及PPO稳定性

`basket_mixed_source`支持：

1. 原固定底座41.x手指协同模型：35维、arm6_hand12。
2. 当前移动联合模型：`joint_base2_arm18_phase_v1`，只提取其抛球actor、标准差与35维抛球归一化，丢弃底座与旧critic。

新actor第一层保留原35列，新加物理属性两列以0初始化，最初输出与源抛球策略一致；所有actor参数均可学习。critic为新37维网络，重新学习价值。原35维归一化统计保留不更新，新物理参数使用固定的显式尺度。

学习率固定3e-5；KL按18动作维求和，单轮PPO任一球的平均KL超过0.01时，暂停本轮后续actor更新，critic继续学习，下一次采样重新判断。整轮更新始终保留采样时的旧分布。三球各自标准化优势，按整份rollout对三球等权计算损失，不因各球回合长短产生样本数量偏置。原始球另有0.05权重的源策略KL软约束，减少旧技能遗忘。

终端显示original SR、tennis_ball SR、baseball SR与Macro SR。最佳成功率权重以三球等权平均成功率选择，且每种球至少有20个近期完成回合才开始选择；前期可能还没有best_success文件。整体Basket SR和Current Reward仍是混合回合统计，优先看分类SR和Macro SR。

## 训练命令

激活dcmm环境并同步代码后：

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_three_ball.sh \
  ../11-1/throw_basket_catch_best_reward_41.30.pth
```

也可直接使用当前移动联合模型：

```bash
bash scripts/train_basket_three_ball.sh \
  /完整路径/当前移动模型/nn/throw_basket_catch_best_success.pth
```

服务器上请使用实际存在的权重路径。脚本支持追加Hydra参数，例如`num_envs=16`。默认32环境、GPU0、异步、固定3e-5学习率，上限1万亿步，可手动停止。

输出：`outputs/BasketThreeBallFixed/日期/时间/nn/`，含last、best_reward、best_success。

等价命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_three_ball \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source=/原固定模型或移动联合模型的完整路径.pth \
  checkpoint_catching='' viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketThreeBallFixed
```

## 测试命令

三球混合测试：

```bash
python3 train_DCMM.py --config-name basket_three_ball \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/三球训练输出/nn/throw_basket_catch_best_success.pth \
  basket_mixed_source='' basket_mixed_ball=mixed \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=300 output_name=BasketThreeBallFixed_Test
```

公平比较时分别固定每一种球各测试100回合，将`basket_mixed_ball`设为`original`、`tennis_ball`、`baseball`。不要使用旧test_object入口。测试时无需源模型文件，新checkpoint内含完整新策略与物理协议记录。

新模型为37维，不能直接使用旧35维或移动51维的测试配置。后续三球移动训练需要额外把物理参数接入移动观测与迁移逻辑。

## 验证

- 原有与新增测试共115项通过（含新模型迁移、条件输入可学习、原归一化保留、三球损失平衡、物理切换恢复、随机种子复现、终端自动重置分类统计，以及KL与旧分布保护）。合成更新测试中optimizer.step被替换为空操作，没有真实更新模型参数。
- 真实41.30初始化、新checkpoint导出重载、两个spawn异步环境、37维观测/18维动作及球类元数据写入rollout检查通过。
- 三种球各20次物理准备通过。
- 未训练的新结构CLI推理：原始球5/5投进；网球0/3、棒球0/3。只验证初始化和测试链路；两种新球需要训练，这些不是训练后的成绩。
- Python编译、shell语法及本次代码范围diff检查通过。

记录：`/home/isee/Catch/analysis_artifacts/basket_three_ball_20261102/`。

## 需同步的文件

- 新增configs/basket_three_ball.yaml
- configs/config.yaml
- gym_dcmm/__init__.py
- 新增gym_dcmm/envs/basket_mixed.py
- 新增gym_dcmm/algs/ppo_dcmm/basket_mixed_policy.py
- 新增gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py
- train_DCMM.py
- 新增scripts/train_basket_three_ball.sh
- 新增tests/test_basket_mixed.py、tests/test_basket_mixed_update.py

依赖已有的basket_joint_stability.py、Basket持球/手指奖励工具和球物理先验JSON；服务器应先同步此前稳定版代码。

# 固定底座三球从零训练

当前默认版本已纠正为从零训练：所有actor/critic随机初始化，不加载41.30或移动模型，不使用旧模型参考损失，归一化统计从头建立。没有启动训练。

## 默认训练命令

先同步最新代码到服务器，激活dcmm环境：

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_three_ball.sh
```

不需要传模型路径。默认32异步环境、GPU0、固定学习率3e-4，步数上限1万亿，可手动停止。输出为`outputs/BasketThreeBallScratch/日期/时间/nn/`。

等价显式命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 train_DCMM.py --config-name basket_three_ball \
  test=False num_envs=32 vector_mode=async device_id=0 \
  basket_mixed_source='' checkpoint_catching='' checkpoint_tracking='' basket_warm_start='' \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=1000000000000 output_name=BasketThreeBallScratch
```

启动时应看到：

```text
[basket-three-ball] SCRATCH: random actor/critic; fresh normalization; no old-policy loss
```

## 环境与网络

每回合开始，原始球、网球、棒球各以1/3概率被选中；回合内不切换。底座固定，框固定在原投篮位置。等概率采样不保证有限回合中三球数量恰好相等。

保留现有持球姿势、机械臂前送、持球位置、手指协同和投篮奖励，这些是环境/奖励设计，不是模型权重。三种球均先在物理仿真中稳定持球，再开始策略回合，不绑定球、不施加助力。

策略为37维观测、18维动作（机械臂6+手指12）。原35维状态后增加半径和质量。新增物理输入缩放为`(radius-0.04)/0.01`、`log(mass/0.05)`，其余35维归一化从新数据统计；每轮PPO内部保持统计不变，完成更新后再更新统计。没有图像或球类ID策略输入。

这是“已知半径/质量条件下”的多球训练，仿真直接提供这些参数，不代表能从外貌推断质量。

## 球与持球姿势

- 原始球沿用原固定抛球环境的实际参数。
- 网球约半径0.03328m、质量0.058kg；棒球约半径0.03618m、质量0.138kg。
- 两种新球的惯量/摩擦来自现有球物理先验JSON，尚非实物标定。使用球形碰撞体，颜色只用于Viewer区分。
- 原始球和棒球长指初始屈曲[0.8,0.7,0.7]；网球[1.0,0.9,0.9]；拇指[0.4,0.4,0.4]。初始手型由环境按已知球类选择，不训练从任意手型抓稳球。

## PPO与指标

所有机械臂/手指参数从随机值开始更新，critic也从头学习。三球分别标准化优势，按整份rollout等权计算损失，避免不同回合长短造成某种球主导训练。KL限制只约束本轮PPO相对本轮采样策略的更新，不涉及任何旧模型。

日志显示各球SR及Macro SR。最佳成功率权重以三球等权平均成功率选择；每球至少20个近期完成回合后才开始选取。原始球本次也从零学习，因此启动时不会保留41.30的高成功率。

## 测试

```bash
python3 train_DCMM.py --config-name basket_three_ball \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=/完整路径/三球训练输出/nn/throw_basket_catch_best_success.pth \
  basket_mixed_source='' basket_mixed_ball=mixed \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=300 output_name=BasketThreeBallScratch_Test
```

分类测试分别设置`basket_mixed_ball=original`、`tennis_ball`、`baseball`，各测试100回合。不要使用旧test_object入口，也不要使用原35维或移动51维测试配置。后续三球移动需继续接入物理参数和迁移逻辑。

## 验证与同步

116项相关测试通过。新增了“无参考模型、全新统计”更新测试；合成更新中optimizer.step替换为空操作。实际构造检查禁止初始化调用torch.load，核对无参考网络、无旧权重、全新统计、37维/18动作、两个spawn环境、元数据存储和保存重载。没有执行真实训练或优化器更新。

需同步：configs/basket_three_ball.yaml、train_DCMM.py、scripts/train_basket_three_ball.sh、gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mixed.py；以及前一版三球环境、模型工具、注册与配置支持文件。原迁移功能仅保留为显式可选入口；默认命令不传路径、不进行迁移。

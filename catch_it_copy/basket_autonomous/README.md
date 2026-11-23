# Basket 自主教师：随机持球起点、完整20维控制

这是独立的新教师训练任务，不覆盖旧 Basket 教师或已有蒸馏模型。

## 与旧版本的区别

- 每回合随机采样机械臂、手指、球在手内的位置及初始底盘XY位置。默认使用通用托球邻域，加上4份此前真实 Throw 接球后的姿态记录，并施加连续关节扰动；不把其中任何一个作为目标姿态。
- 使用关节限位、碰撞和短时间重力/接触检查拒绝无效初始化；失败重采样，超过次数明确报错，没有固定姿态兜底。球的位置调整仅存在于 reset，保证初始无穿模；随后只在重力和真实接触下运动。
- 初始化的物理准备发生在回合计时前。它仅生成有效随机持球状态；策略开始后不再维持预设姿态、不粘球、不重新放置球、不施加额外球体推力。
- 一个统一Actor每步输出全部20维：底盘2、机械臂关节速度6、手指增量12。没有底盘/机械臂分阶段网络、阶段动作掩码、自动持球、阶段制动、指定停车点或停车解锁机制。
- 底层PID、关节限位、速度/加速度限制仍保留，它们是执行动作的电机控制，不是任务辅助。
- 使用固定网球物理：质量0.058kg、半径0.0332757853m、惯性因子0.55、自由关节阻尼/armature为0；篮筐边缘启用物理碰撞。

## 观测与成功标准

这是有特权状态的教师，不是视觉学生。观测65维：机器人状态36、小球相对位置/速度6、篮筐相对位置3、上一动作20。小球和目标坐标在机械臂基座坐标系中。没有阶段、稳定进度、剩余时间、停车参考或目标持球姿态输入。

模型自己选择停车位置和挥臂/松手时机。成功必须是真实脱手后穿过篮筐有效开口，不能手拿球穿框：至少离手0.1秒、飞行0.2m、释放点离篮筐平面距离至少0.3m。因为任务要求停稳后投球，释放前底盘平动速度需低于0.05m/s、角速度低于0.15rad/s并持续0.12秒。此项只检查动作结果，不锁定任何执行器，不给模型发送阶段提示。

奖励包含一次性的真实投中、弹道质量进步、释放质量，以及时间、动作平滑和失败代价；没有追踪预设机械臂/手指姿态或固定停车点的奖励。持球小奖励小于时间代价，弹道进步和释放奖励有界，避免一直持球刷分。

这些回合从“随机有效持球状态”开始，仍然不是完整接球→投框训练。后续应继续扩展实际接球姿态库，覆盖真实接球后的速度和姿态分布。

## 从头训练

```bash
conda activate dcmm
cd /home/yuhao2/Catch/catch_it_copy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/catch_mpl
python -m basket_autonomous.run train \
  --output basket_autonomous/runs/tennis_random_hold_seed123 \
  --device cuda:2 \
  --num-envs 32 \
  --rollout-steps 64 \
  --total-steps 25000000 \
  --eval-every 1000000 \
  --eval-episodes 100 \
  --save-every 100000
```

`num-envs`是总环境数，不是每任务数量。训练仅使用状态，worker不创建深度渲染器/EGL上下文，因此不需要`--egl-device-id`。GPU只运行PPO网络。可在CPU和内存允许时增加到64；总采集预算由`total-steps`固定，增加并行不会扩大预算，要求总步数能被环境数整除。25,000,000可被32或64整除；若换128等数量需选择可整除的总步数。

`rollout-steps`是每个环境采集多少步再更新；增大环境数时，可按比例减小它，保持每轮数据量，例如32×64与64×32均为2048步。

这里的`eval-every`和`save-every`单位均为累计环境交互步数，不是蒸馏迭代。跨过阈值时在该轮结束后执行，所以文件名记录实际步数。每次评估保存`teacher_XXXXXXXXX.pth`和`eval_XXXXXXXXX.json`，另外有`teacher_best.pth`、`teacher_last.pth`、`resume_last.pth`及`metrics.jsonl`。100回合评估默认最多8个CPU仿真进程。

旧的40.98等阶段式教师不能作为本版resume。建议先独立训练和验证本教师，达到可靠成功率后再接入蒸馏；现有`distillation_visual_basket.FrozenTeacher`不支持本65维新协议，不能只修改SHA256就混用。

## 续训与观察

```bash
python -m basket_autonomous.run train \
  --resume basket_autonomous/runs/tennis_random_hold_seed123/resume_last.pth \
  --output basket_autonomous/runs/tennis_random_hold_seed123 \
  --device cuda:2

python -m basket_autonomous.run eval \
  --checkpoint basket_autonomous/runs/tennis_random_hold_seed123/teacher_best.pth \
  --output basket_autonomous/evaluations/viewer_01 \
  --device cpu --eval-episodes 10 --viewer
```

续训恢复模型、优化器及随机数状态；物理worker从新回合开始，不能视为仿真状态逐位无缝恢复。初始化姿态库内容已嵌入checkpoint，不依赖服务器原文件路径。eval不使用自动持球/阶段制动。

## 检查

```bash
python -m unittest basket_autonomous.test_autonomous
```

检查覆盖随机持球起点差异、真实手部接触、全动作可执行、没有旧停车参考调用、没有策略回合内球位置重置、球体控制力为零，以及策略概率/梯度。小规模训练检查只证明采集、PPO更新、保存、评估、续训可运行，不能证明已经学会自主投球。

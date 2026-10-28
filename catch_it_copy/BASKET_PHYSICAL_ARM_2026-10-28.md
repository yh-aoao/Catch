# 无初始重叠的前送抛球训练

CPU 试跑于 2026-10-28 16:26:18 启动，输出目录为
`outputs/BasketPhysicalArm/2026-10-28/16:26:18/`。随后确认 GPU 可在沙箱外正常访问，已停止 CPU 试跑并切换为正式 GPU 训练，输出至 `outputs/BasketPhysicalArm_GPU/2026-10-28/16:29:38/`。CPU 日志另存为 `train_cpu.log`。

日志：`../analysis_artifacts/basket_physical_20261028/train.log`

## 实现

配置 `basket_physical_arm` 继承固定 35 维观测、18 维动作的训练入口，新手型保持 `[0.8,0.7,0.7]`。从用户已训练的 39.70 权重加载网络和输入统计，输入统计冻结；优化器和训练计数重新开始。原模型文件保留。

1. 仅重置放球时向上寻找不重叠位置，球与手部碰撞几何间隙至少 1 mm。本次测得上移 24 mm 后间隙 1.303 mm。
2. 策略回合开始前，控制器保持机械臂和手指姿态，让球在重力与接触下自然稳定。期间不绑定球、不助推、不重设球速或球位置。
3. 至少准备 0.4 秒，并要求连续 0.1 秒有手部接触、球速低于 0.08 m/s、掌部坐标相对滑动速度低于 0.03 m/s、局部 Y 位于 0.02～0.075 m 支撑范围。1.5 秒仍不满足或发生非手部球接触／机械臂自碰撞则报错，不悄悄瞬移纠正。
4. 从实际稳定后的状态开始策略控制与回合计时，机械臂和手指都由策略控制，保持原来的 35 维输入，避免增加一个不可观测的等待阶段。
5. 离手前肩肘前送仍有最多 2 分进度奖励、最多 2 分释放奖励。新增合格释放质量：`min(clip(前向位移/0.06), clip(肩肘前向速度/0.30))`，各裁剪到 0～1。数据取最后手部接触时刻，不取过去最大值，也不取离手后动作。
6. 实际穿框奖励为 `20 × (0.25 + 0.75 × 质量)`，保留渐进学习信号；只有质量达到 1 且安全随动检查通过，终止原因才是 `basket_score` 并计为成功。其余实际穿框回合记为 `basket_unqualified`。因此“穿框”和“合格安全进框”明确分开。

不规定固定前送轨迹、固定开手时刻或最小出手时间；让策略学习协调。6 cm 和 0.3 m/s 是新训练任务的初始合格标准，不代表已验证的硬件能力或最优动作。

日志的 `basket_control` 包含：放球间隙、准备用时与稳定速度、策略时间、离手相对策略开始时间、最后接触时前送位移和肩肘速度、释放质量。`QualifiedThrow SR` 是合格安全进框率。新模型保存协议标记，防止误用旧初始化配置测试。

## 验证

- 10 个随机种子全部自然稳定，准备时间 0.472～0.518 秒；开始策略时球速 0.0059～0.0097 m/s，掌内滑动速度 0.0043～0.0085 m/s。
- 双环境 CPU 512 步短训练完成采样、奖励计算、更新、保存。
- 64 项相关测试通过，包括真实飞行分支对合格／不合格穿框的区别及安全随动检查。
- 正式异步训练首批约 2,000 步正常完成并保存。初始奖励约 9.72，合格成功率 0%；这是准备状态与成功标准改变后的新任务，不能与原 39.70 直接比较。

原始验证数据与短训练日志位于 `../analysis_artifacts/basket_physical_20261028/`。此报告记录启动与实现验证，不宣称 200 万步已完成或动作效果已提高。

## 本次实际启动命令

沙箱内检测显示 CUDA 不可用；进一步在沙箱外检查确认 RTX 5070 Laptop GPU 和 PyTorch CUDA 均正常。正式训练已改为 GPU、32 个异步环境，CPU 试跑已停止。

```bash
cd /home/isee/Catch/catch_it_copy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/basket-mpl \
/home/isee/miniconda3/envs/dcmm/bin/python -u train_DCMM.py \
  --config-name basket_physical_arm \
  test=False num_envs=32 vector_mode=async device_id=0 \
  viewer=false basket_log=false basket_self_collision_debug=false \
  train.ppo.max_agent_steps=2000000 output_name=BasketPhysicalArm_GPU
```

训练正在运行时不要重复启动上述命令。可在自己的终端查看：

```bash
tail -f /home/isee/Catch/analysis_artifacts/basket_physical_20261028/train.log
```

完成后使用同一配置测试新权重：

```bash
python3 train_DCMM.py --config-name basket_physical_arm \
  test=True num_envs=1 vector_mode=sync device_id=0 \
  checkpoint_catching=新权重完整路径 \
  viewer=true basket_viewer_speed=1.0 basket_log=true \
  train.ppo.test_num_episodes=30 output_name=BasketPhysicalArm_Test
```

测试时若 GPU 同样不可用，将 `device_id=0` 改为 `device_id=-1`。优先对比 `best_success`、`best_reward` 和 `last` 权重的合格安全进框率，而不是仅以最高总奖励选择动作。

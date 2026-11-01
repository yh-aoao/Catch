# Roll252.54奖励来源与Bounce三球训练版本

## Roll252.54的来源边界

实际检查`/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth`：仅有model、running_mean_std_track、running_mean_std_hand、value_mean_std。没有原训练配置、奖励源码哈希或Git版本。现有resolved_config中引用252.54的记录均为后来的测试/微调，不能用于证明其原训练奖励。

历史文档在2026-10-02已将252.54列为既有Catch模型。2026-10-09的ROLL_CATCH_GRIP_V1明确以252.54为微调起点，所以该次新增的捕获窗口、手指分支接触、保持进度和滑落惩罚不是原模型最初训练时的奖励。后加的adaptive_wait_v1也不是其原始奖励。

可恢复的历史候选基线为Git d2e7712（2026-09-29恢复旧Roll版本）及之前相关提交。该基线包括预测接球点/距离与靠近、高度及手桌约束、掌心朝上塑形、手指准备/包球与手速/动作变化正则、抓取阶段精度和真实接触低相对速稳定项，再减控制/碰撞/限位惩罚。它是可检查的候选源码，**尚无证据把该提交唯一绑定到252.54原训练**，不能伪称已恢复精确原训练配方。

## 版本对应

| 对象 | 奖励版本 | 说明 |
| --- | --- | --- |
| 已测Bounce464.31 | legacy | 三球physical_v2训练，当前成功率38%/44%/40% |
| 当前配置默认值 | legacy | 不显式切换时仍使用旧版 |
| 新训练脚本train_bounce_three_ball_hold_v1.sh | controlled_hold_v1 | 之前实现的手部控制/抓稳候选，尚未完成正式长训练和成功率验证 |

physical_v2是球动力学版本；controlled_hold_v1是Catch控制、奖励和成功判据版本，二者不是同一个开关。

## 推荐本轮Catch训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_bounce_three_ball_hold_v1.sh device_id=0 num_envs=16
```

脚本已核对能找到现有Track19.31：
`/home/isee/Catch/bounce_three_ball_tracking_v1_seed123/2026-10-28/00:42:24/nn/bounce_track_best_reward_19.31.pth`。

保留Tracking→Catching_TwoStage流程；此命令执行第二阶段，加载Track、重新学习手指分支，不加载464.31或653.87 Catch权重。16环境、2500万环境步、seed123。原始球/网球/棒球每回合采样概率60%/20%/20%，这是期望比例。physical_v2、各球材质及棒球专用发射沿用已验证设置。

前50万环境步冻结Track，之后Track学习率为其他分支的0.1倍；Track归一化固定，手指动作尺度0.08。输出到`outputs/bounce_three_ball_catching_controlled_hold_v1_seed123/日期/时间/`，保留resolved_config、源码/初始化模型哈希、逐局奖励日志和包含优化器/计步/RNG的last权重。

如果需要连Track也重新训练，先运行：

```bash
bash scripts/train_bounce_three_ball_v1.sh \
  task=Tracking device_id=0 num_envs=16 \
  output_name=bounce_three_ball_tracking_retry_seed123
```

结束后将新Track权重显式传给第二阶段：

```bash
bash scripts/train_bounce_three_ball_hold_v1.sh \
  checkpoint_tracking=/新Track训练目录/nn/实际Track模型.pth \
  device_id=0 num_envs=16
```

## 新版奖励

原Track奖励保持：3D距离/靠近、触球、控制/碰撞/限位。下面是第二阶段controlled_hold_v1：

有界状态分数组成Phi：位置2、近距离精度4、近球掌面朝向1、相对速度匹配1、接触前准备开度0.3、真实接触2、包球/接触分布2、有效保持进度8。这些是势能分量权重，不是逐步直接相加发放的奖励。

实际奖励为gamma*Phi_next-Phi_prev，再加成功25、失败终端-25，以及每步-0.01、手动作幅度/变化、关节越限、底盘/臂控制与碰撞惩罚。gamma跟PPO一致，终端Phi归零。真实手接触、多个手部区域与低相对速保持形成收益；没有“越用力闭合越高分”的无限线性奖励。

新版成功条件为连续0.3s真实受控保持：至少两个手部区域接触、有一定接触分布、球在捕获区域、相对手速≤0.35m/s、球底离地>0.02m、无腕/前臂/地面外部支撑、关节越限≤0.08rad，策略步末仍成立。该成功率需另记，不能与旧proximity成功率直接比较。

同时使用有界手部目标、抗积分饱和/测量微分PID和输出限幅。因此这版是控制与奖励联合候选，不是只改奖励。控制参数为仿真先验，正式效果仍需训练验证。

## 若要复现464.31的旧奖励

```bash
bash scripts/train_bounce_three_ball_v1.sh \
  task=Catching_TwoStage bounce_grasp_version=legacy \
  checkpoint_tracking=/home/isee/Catch/bounce_three_ball_tracking_v1_seed123/2026-10-28/00:42:24/nn/bounce_track_best_reward_19.31.pth \
  device_id=0 num_envs=16 \
  output_name=bounce_three_ball_catching_legacy_retry_seed123
```

此命令同样三球60/20/20。旧Catch奖励在接近阶段为3D距离+靠近增量+姿态+触球减控制/碰撞/限位；抓取阶段再加10*exp(-50*d²)精度和从首次接触起累计的时间稳定项。该稳定项不逐个物理tick验证受控持球。

旧手指闭合/协同块读obs['hand']['joints']，实际obs['hand']是12维ndarray，访问抛异常后这几项被清零。新版直接使用正确手关节数组，已绕开该旧分支问题。不能把源码中写着手指奖励等同于它在旧训练中实际生效。

本次已通过两套Catch命令的Hydra配置解析与脚本语法检查，并实际检查旧环境手观测数据结构；没有启动正式训练。

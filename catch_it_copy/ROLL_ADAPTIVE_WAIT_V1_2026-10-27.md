# Roll：自适应桌前等待奖励 v1

日期：2026-10-27。用户要求在当前Roll基础上加入可学习的桌前等待，
等待位置/高度跟随球的方向和轨迹，同时保留可退回的旧版。

## 旧版已备份

目录：`backups/roll_before_adaptive_wait_20261027_001703/`。
基线Git commit：`90d9d1474fb89b84885d08af50d2055d163cc40f`。
包含164个实际文件副本及SHA256清单，总字节数4,669,653。
已在临时目录执行恢复并逐文件核验，没有对正在使用的项目执行恢复。
本次仅改变了三个已有集成文件；快照中的其余源码、配置、模型和XML均校验一致。
原有`进度.txt`用户修改未改动。

旧模型252.54保留，SHA256为
`40020e1b13a29ad49de9e9b6ad2e6316678f418bdd8712be898b7a257af3d0ba`。
快照内另含240.13 Catch和33.28 Track模型，不覆盖原文件。

## 修改前的真实训练路线

| 阶段 | 环境 | 配置 | PPO |
|---|---|---|---|
| Track | `DcmmVecEnv.py::RollTrackingEnv`（07c176f） | `DcmmCfg_roll_track_07c176f.py` | `ppo_dcmm_645edc4/ppo_dcmm_track.py` |
| Catch | `DcmmVecEnv_roll_645edc4.py::DcmmVecEnv` | `DcmmCfg_roll_645edc4.py` | `ppo_dcmm_645edc4/ppo_dcmm_catch_two_stage.py` 或 one_stage |

路由经过`roll_compat.py`；不能把其他未被此入口选择的Roll类当作当前教师环境。
Track观测18维、动作8维；Catch观测30维、动作20维，动作组为底盘2/机械臂6/手12。
Catch动作缩放为[1.5,0.025,0.15]；观测归一化和网络仍沿用当前PPO实现。

默认训练物体仍为原解析球，而非YCB混合训练。
当前可动底盘Roll初始速度0.8～1.3m/s，方向±30°，初始x在±0.25m、y在2.1～2.6m。
桌面高0.70m，前缘y=0.90m，默认回合2.5秒，策略间隔为20个物理步。
质量、半径、惯量、摩擦、阻尼、armature、PID和动作延迟均保留原源码。

旧等待奖励通过`roll_waiting.target/shaping`计算：

- 根据当前水平速度预测到达桌前边缘及落到固定0.50m接取高度的位置。
- XY接近项权重1，尺度0.45m；靠近增量权重5。
- 接取高度0.50m±0.06m，超出带宽后最多扣1分。
- 全手保守前缘进入桌前3cm裕量时开始扣分，15cm范围内最多扣2分。
- 其余掌心朝上、手指、接触、保持、控制、底盘碰撞和关节约束奖励按原阶段计算。
- 原成功、失败及阶段切换保持，Catch使用现有roll_eval_success/保持区域口径。

PPO默认：gamma=0.99、GAE tau=0.95、学习率5e-4、KL调度阈值0.016，
horizon=64、minibatch=512、mini_epochs=6、reward_scale=0.1、最多25,000,000 agent steps。
默认32环境时batch=2048。原PPO按累计回报选择best模型；不同奖励版本的best_reward不可直接比较。
`checkpoint_catching`恢复模型和输入归一化用于继续学习，但原restore_train并不恢复完整optimizer/RNG等状态，
所以它是权重初始化/微调，不是严格断点续训。本次没有改变这些训练逻辑。

## 新版实现及开关

默认仍为`roll_reward_version=legacy`，旧命令继续走旧路线。
显式`roll_reward_version=adaptive_wait_v1`选择新注册的`RollAdaptiveWorld-v0`。
实现位于`gym_dcmm/envs/roll_adaptive.py`及`gym_dcmm/utils/roll_adaptive_wait.py`。

新版调用同一个原环境执行动作，随后仅调整返回的奖励：

`new_reward = old_reward - sum(old_roll_wait_terms) + sum(adaptive_wait_terms)`。

替换的只有原XY、靠近、高度和桌前边界四个位置塑形项，避免新目标与旧固定0.50m项冲突。
其他原奖励仍保留。没有更改动作、物体运动、质量/惯量、摩擦、碰撞、观测、终止或成功标准。
既有测试辅助`roll_wait_guard`必须关闭，不通过硬限制动作冒充策略学会等待。

## 新目标如何变化

球未完全离桌时，以当前世界坐标位置/速度估计出桌x，再选择出桌后的自由飞行接取点。

- 前向速度`u=max(0,-vy)`；期望桌外距离为`0.10 + 0.08*u`米。
- 用距离/u选择下落时间，再根据可接取高度范围0.38～0.62m限制下落时间。
- 目标x/y由出桌位置加水平速度×下落时间得到，目标z由重力下落公式得到。
- 高度限制通过限制下落时间实现，保持目标在预测轨迹上，不独立把z截断到另一条轨迹。
- 来球偏左或偏右时目标x对应变化；较慢来球通常对应较低的接取点，较快来球对应较高的接取点。
- 预测暂采用滚动阶段水平速度不变的近似，不声称已精确预测未来摩擦减速。
- 球无前向运动时保留桌外有限目标，不把手引导到桌内静止球下方。
- 完整球体越过桌前边缘且无桌面接触后，本回合转为追踪实际球位置，停止等待奖励和桌前等待惩罚。

使用物理掌面几何中心沿掌心法线偏移一个球半径作为接取点；全手前缘包含掌面及手指碰撞几何的保守包围范围。
宽高度带是软奖励，不固定手在某个高度。所有权重是待验证的初值，不是已验证最优参数。

## 新分项

| 分项 | 默认定义 |
|---|---|
| XY接近 | `1/(1+(XY误差/0.30)^2)`，全手接近桌边裕量内时衰减 |
| 靠近增量 | 同一当前目标下，前后接取点距离的改善×5；等待时用XY，离桌后用3D |
| 高度 | 等待时允许目标高度±0.06m，超出后最多扣1.5；离桌后提供较弱3D跟踪高度项 |
| 全手边界 | 球未离桌时，全手前缘侵入桌前3cm裕量开始扣分，最多扣4 |
| 到位停稳 | 最多+0.5；须接近目标、在高度带内、手速度低且有桌前净空 |
| 提前接触 | 等待阶段球同时接触桌面和手时扣2 |

靠近奖励的两个距离都使用同一个当前目标，因此目标随球运动漂移本身不产生靠近分数。
等待/追踪阶段切换时清除增量历史；reset清除目标历史、离桌状态和分项累计。
这些软奖励不会保证零越界，也未改成实机安全控制器。

## 训练命令

先在原球及原速度分布下验证奖励效果，以下用252.54旧模型初始化，原模型保留：

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 \
  object_motion=roll roll_reward_version=adaptive_wait_v1 \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth \
  output_name=RollAdaptiveWaitV1 device_id=0
```

从头训练Catch时去掉checkpoint_catching；Track使用task=Tracking及新output_name。
本次没有启动正式长训练。当前YCB仍只支持测试；训练命令不要设置test_object=ycb:...。
`roll_speed`仍保留为现有物体测试入口的选项，没有暗中扩大旧训练速度分布。
因此此版奖励本身不等于已覆盖0.5m/s；若后续扩大低速分布或多物体训练，应另立实验并同步记录。

输出沿用按日期时间创建的独立目录，另保存：

- `resolved_config.yaml`：奖励版本、参数、seed、PPO及实际命令配置。
- `roll_reward_provenance.json`：旧版快照manifest哈希、运行源码哈希、初始化模型路径及哈希。
- `roll_reward/episodes_*.jsonl`：每个worker独立记录回合终止原因、成功标志、新旧奖励及分项累计。
- 原PPO模型与TensorBoard输出。

参数可通过例如`roll_adaptive_wait.hand_weight=5`覆盖；实验名应明确区分权重版本。
最佳模型仍由原PPO按新总回报选取，正式比较应使用同一成功标准、相同测试seed和足够回合数。

## 测试新模型

替换真实的新模型路径，不要把短训练smoke模型当作学会的新教师：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll roll_reward_version=adaptive_wait_v1 \
  checkpoint_catching=/新训练目录/nn/roll_catch_last.pth viewer=true
```

YCB棒球测试可以追加`test_object=ycb:baseball object_eval=false`；
需要低速对照时再追加`roll_speed=0.5 test_env_time=5`。
`roll_wait_guard=false`必须保持；奖励版本改变不意味着动作被硬限制。
权重单独复制后需一并保留所在运行目录的配置与provenance文件。

## 回退

1. 奖励回退：使用旧模型，并设置`roll_reward_version=legacy`（或省略该参数）。
2. 训练回退：保留新输出，另用旧模型/旧参数和新输出名运行legacy。
3. 源码完整恢复：先检查，再显式应用快照：

```bash
python3 backups/roll_before_adaptive_wait_20261027_001703/restore.py
python3 backups/roll_before_adaptive_wait_20261027_001703/restore.py --apply
```

后者只覆盖三个已有集成文件，覆盖前自动保存当前文件，不删除新输出或模型。
旧算法和环境文件本次从未修改，无需通过git reset回退其他任务。
如果未来对这三个文件又做了其他功能修改，优先用legacy开关；完整文件恢复也会回退那些后续修改。

## 已完成验证及边界

- 7项新增测试通过：方向/速度自适应、弹道高度、等待与停稳激励、全手越界/提前接触惩罚、
  目标漂移不获利、完整离桌/重置、YCB路径、Gym vector factory接口。
- Track和Catch的同seed、同动作真实MuJoCo对照：qpos、qvel、物体观测逐元素完全一致，终止一致；
  返回奖励严格等于上述替换公式。
- Catch真实两进程PPO短训练完成4次更新，Track完成3次更新，检查保存模型参数均有限。
- Catch smoke保存了回合分项JSONL、配置、源码/初始化模型哈希和模型。
- 快照164项完整性通过；恢复脚本在临时目录实际复制并逐文件校验通过，当前项目未执行恢复。
- `test_main_historical_envs.py`有既有失败：BounceEnv历史AST不一致及Bounce Tracking路由stub检查报错。
  修改前的snapshot运行同一测试同样失败，确认不是本次引入；没有修改Bounce或其历史测试。
- 上述不证明收敛、成功率提高或实机可迁移，仍需正式训练与独立评估。

短训练目录：

`outputs/RollAdaptiveWaitV1Smoke_20261027/2026-10-27/00:25:06/`

`outputs/RollAdaptiveTrackSmoke_20261027/2026-10-27/00:26:51/`

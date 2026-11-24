# Basket 三阶段：真实接球姿态来源，持球→投篮→移动投篮

这是独立版本，入口`python -m basket_three_stage.run`；修复后的两阶段`basket_autonomous.run`保持可用。网球物理、关节速度输出和真实物理接触保持一致，没有回合内球体重放、粘球、固定手型或自动底盘制动。

## 随机起点

附带姿态库`configs/caught_poses.json`共20条：Throw 4、Roll 10、Bounce 6。Throw来自真实视觉接球后的稳定交接日志，Roll/Bounce来自既有接球教师的实际推理接触轨迹。记录包含来源任务、策略文件/哈希、种子、时间、关节角、球在手掌坐标中的位置，以及Basket迁移验证信息。失败或不适合迁移的记录在`caught_poses.report.json`中列出，没有伪造缺失任务。

默认每回合先选择采样类型：60%使用记录不额外加噪声、35%使用记录邻域扰动、5%使用通用托球邻域。前两类先等概率选择Throw/Roll/Bounce，再从对应任务的记录中抽样；不会因某个任务记录更多就更常采到该任务。近邻扰动：机械臂每关节±0.035rad、手指±0.04rad、球局部XYZ±[3,3,2]mm。

采样任务与类型在重试时不变；姿态不合格只在该任务内重采样，达到上限明确报错，不偷偷换成通用姿势。训练开始时不需要读取接球模型文件；姿态库嵌入checkpoint，服务器只需同步代码和姿态库。

**95%表示从真实记录及其邻域生成，不表示逐位复现完整接球末态。**初态仍需经过关节限位、reset内清穿模与短时重力/接触落稳。准备过程中球可临时抬升最多5cm，最终策略起点的球相对位置必须距原记录小于2.5cm、机械臂最大偏差小于0.15rad、手指最大偏差小于0.60rad且RMS偏差小于0.25rad；不合格则重采样。旧接球策略有部分手指明显超出限位，不能原封不动作为有效初态。此筛选是保留来源邻域的折中，详情见逐条验证记录。

当前只迁移姿态和球相对位置，初始速度置零；不声称这是完整连续接球→投篮。20条记录仍是小规模启动库，未来应采集更多接球后状态及残余速度再扩展。

## 三阶段

| 配置 | 训练回合 | 目标 |
| --- | --- | --- |
| stage1_hold.json | 全部固定底盘、持球任务 | 真实接触、低相对滑速、球处于宽松手部范围，并连续稳定0.8–1.6秒；最长4秒。不指定标准关节角或掌心某一点，不奖励投篮。 |
| stage2_throw.json | 80%固定底盘投篮、20%持球复习 | 加载第一阶段，全部参数继续学调整、挥臂、真实释放和入框。 |
| stage3_mobile.json | 70%移动投篮、20%固定投篮、10%持球 | 加载第二阶段，继续训练全部参数，自己选择位置、减速并投篮。 |

持球复习和固定投篮回合的底盘只执行零速度指令；移动投篮回合20维动作从开始就全部开放，没有阶段停车解锁。零速度使用已修复的目标积分控制，不强制回到某个预设姿态。

观测66维：上一版65维状态+1维任务指令（0持球、1投篮），动作仍为底盘2+机械臂6+手指12。任务指令不做连续状态标准化；指令输入列初始化为0，保证第一阶段切换第二阶段时不会仅因新的指令值破坏初始动作。指令用于区分目标，不锁定执行器。固定回合底盘两维不计入PPO概率和熵。

第一阶段奖励：稳定持球最高+0.5/秒、滑移最高-0.5/秒、时间-0.02/秒、原动作幅度/平滑代价、连续稳定成功+20、失败-15；没有弹道或释放奖励。第二、三阶段投篮回合沿用V2奖励和两处修复。持球成功和入框成功分开记录。`hold_sr`、`fixed_throw_sr`、`mobile_sr`统计各类近期回合；自动评估分别只测当前阶段主任务，不混入复习回合。

## 训练

```bash
conda activate dcmm
cd ~/Catch/catch_it_copy
python -m basket_three_stage.run train \
  --config basket_three_stage/configs/stage1_hold.json \
  --output basket_three_stage/runs/tennis_stage1_hold_seed123 \
  --device cuda:0 --num-envs 32 --rollout-steps 64 \
  --total-steps 1000000000000 \
  --eval-every 500000 --eval-episodes 100 --save-every 100000
```

第一阶段持球率稳定、滑移量降低后启动第二阶段，而非固定步数自动切换：

```bash
python -m basket_three_stage.run train \
  --config basket_three_stage/configs/stage2_throw.json \
  --initialize-from basket_three_stage/runs/tennis_stage1_hold_seed123/teacher_best.pth \
  --output basket_three_stage/runs/tennis_stage2_throw_seed123 \
  --device cuda:0 --num-envs 32 --rollout-steps 64 \
  --total-steps 1000000000000 \
  --eval-every 500000 --eval-episodes 100 --save-every 100000
```

第二阶段真实投中率稳定后启动第三阶段：

```bash
python -m basket_three_stage.run train \
  --config basket_three_stage/configs/stage3_mobile.json \
  --initialize-from basket_three_stage/runs/tennis_stage2_throw_seed123/teacher_best.pth \
  --output basket_three_stage/runs/tennis_stage3_mobile_seed123 \
  --device cuda:0 --num-envs 32 --rollout-steps 64 \
  --total-steps 1000000000000 \
  --eval-every 500000 --eval-episodes 100 --save-every 100000
```

同阶段续训使用`--resume .../resume_last.pth`，切换阶段使用`--initialize-from`并重置优化器、步数和best统计。只允许新三阶段权重按1→2→3导入；旧65维两阶段或旧阶段式权重不能直接使用。

Viewer评估任意阶段（输出目录需未存在）：

```bash
python -m basket_three_stage.run eval \
  --checkpoint basket_three_stage/runs/tennis_stage1_hold_seed123/teacher_best.pth \
  --output basket_three_stage/evaluations/stage1_viewer_01 \
  --device cpu --eval-episodes 20 --viewer
```

## 扩充真实状态库

`collect_poses.py`只执行既有接球模型推理，不更新权重。先按本机/服务器实际路径修改`configs/catch_sources.json`及对应SHA256，再运行：

```bash
python -m basket_three_stage.collect_poses \
  --config basket_three_stage/configs/catch_sources.json \
  --output basket_three_stage/collections/new_catch_states \
  --per-task 30 --max-episodes 300
python -m basket_three_stage.prepare_bank \
  --inputs basket_three_stage/collections/new_catch_states/caught_poses.json \
  --output basket_three_stage/configs/caught_poses_expanded.json
```

采集要求真实手部接触、局部距离<18cm、相对速度<0.5m/s且持续至少一个策略步，最多每回合3条、间隔至少0.16秒。它们是接触持球状态，不冒称整回合稳定接球成功；之后仍需Basket物理筛选。未达数量会报错并保存已采集的分任务原始记录；不会合成假数据。Bounce较难采集，可能需要更多回合或更好的源策略。

用`--pose-bank .../caught_poses_expanded.json`替换第一阶段新实验的状态库，后续两阶段使用同一状态库。不要在resume或已保存模型的eval中更换初态分布。

## 同步与验证

同步`basket_three_stage/`完整目录、`basket_autonomous/environment.py`、`basket_autonomous/carry_reward.py`和已修复的`gym_dcmm/envs/DcmmVecEnv.py`。三阶段仅为父环境增加采样钩子，默认两阶段行为不变；修改前父环境快照在`backups/before_three_stage_20261124/parent_source.tar.gz`。

检查覆盖三阶段导入顺序、任务指令、动作梯度、真实分组重置、无球体辅助、持球奖励不混入投篮分数及旧两阶段回归。另有真实CLI→双进程采集→自动reset→未训练权重保存/导入检查，未执行优化器更新或正式训练。不能据此推断训练成功率。

# 成功判定调整（2026-10-10）

修改前备份：backups/success_criteria_20261010_220925（按相对路径保存）。未改用户进度.txt。

## Throw：官方判定

参考 hang0610/Catch_It main 的 DcmmVecEnv.py 与 PPO。
Track 仅 step_touch（手掌接触）触发成功截断，不再因时间截断。
Catch 距离 link6 < 0.25 m 进入 grasping，之后 >= 0.25 m 失败；去掉后加的低速连续5步提前成功。
Catch 超过 env_time 截断成功。info.success 精确对应 truncated，包括同时 terminated/truncated 的官方边界行为。
现有 PPO 从 terminal info 读取，因此无需恢复旧 PPO 全部实现。
这恢复的是结束/成功判定，不代表整个 Throw 环境已与上游完全相同。会改变未来训练的回合长度和回报。

## Roll

Track 未修改。
Catch 评估版本 retention_v4_final：主成功读取 held_now，不再读取 caught_once；手部接触启动、位于原有掌心区域且离桌离地、保持至少0.5秒。无接触超过0.08秒清零并要求重新接触。失败终止优先（旧 catch_success 除外）。
roll_caught_once 保留历史曾经持有的辅助指标。低速只影响稳定性指标。
奖励、控制、终止和时限不变。需要重新评估，无须为统计变化重新训练。

## Bounce：确认后的实际入口

train_DCMM.py -> Tracking: bounce_compat.make_bounce_july22 -> DcmmVecEnv_bounce_july22.py；PPO ppo_dcmm_july22.track。
Catching: bounce_compat.make_bounce_5fe75d5f -> 主文件 BounceEnv；配置与 Catch PPO 为 5fe75d5f。

本次仅修复统计：Track success=step_touch且无失败；Catch PPO读取环境info.success（含向量自动reset的final_info），不再拿truncated当成功。
Catch 保留回溯基线要求：grasping阶段手掌接触且观测球速<=0.05连续5步，同时XY距离<=0.03、平均MCP屈曲>0.3、球相对arm_base高度>0.05。因此这仍是严格旧判定，不是Roll的0.5秒最终持有指标。
未新增首次反弹门槛，未改奖励、物理、动作、终止、模型保存规则。后续若要统一科学评估，应另立评估指标，不混入本次基线修复。
启动输出已明确success统计覆盖，不能再声称PPO完全原版。

## 验证

test_roll_evaluation.py 13项、test_bounce_tracking_metrics.py 4项、test_success_criteria_revision.py 3项通过；修改Python语法检查通过。
未运行MuJoCo真实轨迹或重新训练；需在服务器用已有checkpoint复测。原有成功率与新口径不可直接混比。

## 后续按用户要求更新 Bounce

Track（July22实际入口）：手掌或手指接触均触发 step_touch；无接触超时不算成功，同步失败仍优先。
Catch（BounceEnv实际入口）：改用当前Throw的官方抓取阶段逻辑，link6距离<0.25m进入grasping，随后>=0.25m则ball_left；超过env_time时info.success=truncated，与Throw一致（包括同步失败/截断边界）。原低速、MCP、XY、高度与failed_control分支不再用于Bounce Catch。
保留Bounce自身的落地反弹、越界及底座碰撞等仿真条件，非将整套Throw物理复制过来。Roll未修改。
该修改影响Bounce的结束时刻/阶段切换，因此可能改变新训练回报；已有模型可直接测试。
新增手指接触、Catch阶段切换和超时统计测试，success_criteria_revision共5项通过；未运行MuJoCo实测。

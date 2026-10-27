# Roll / Bounce 重训前检查（2026-10-27）

## Roll 等待奖励核查

已接入 `train_DCMM.py`、Gym 工厂、Track/Catch 环境和奖励日志。
必须显式设置 `object_motion=roll roll_reward_version=adaptive_wait_v1 roll_wait_guard=false`。
默认仍为 `legacy`，以保证旧模型复现；仅写 `object_motion=roll` 不会启用等待奖励。

实现位于 `gym_dcmm/envs/roll_adaptive.py` 和 `gym_dcmm/utils/roll_adaptive_wait.py`。
它用自适应项替换旧的四项等待位置奖励，保留其他抓取、碰撞、控制和终止逻辑。
球在桌面时预测运动方向、离桌位置和下落拦截点，奖励手在桌前对齐并稳定等待，
惩罚整只手侵入桌面范围及在桌上提前碰球。等待高度是 0.38–0.62 m 范围内的轨迹目标，
不是固定停在 0.5 m。球完全越过前沿且不再接触桌面后，解除等待惩罚并跟踪实际球位。
这是奖励塑形，没有强行限幅动作；预测暂时假设水平匀速，尚未标定滚动减速模型。

7 项测试通过，包括方向/速度自适应、静止球、离桌切换、目标漂移不产生虚假进度奖励，
以及相同行动下 legacy / adaptive 的物理状态、观测和终止完全一致。
这些检查证明实现接入和激励方向，不代表尚未训练的新策略已经学会等待。

原始奖励和训练备份：`backups/roll_before_adaptive_wait_20261027_001703/`。
更完整说明：`ROLL_ADAPTIVE_WAIT_V1_2026-10-27.md`。

## Bounce 最终候选：stable_v1

启用：`object_motion=bounce bounce_physics=stable_v1 object_eval=false`。
仍默认 `bounce_physics=legacy`。训练和测试均可选择 stable_v1，支持 Track/Catch、原始球，
以及现有 YCB 测试入口。尚未添加训练时 YCB / 多物理参数采样。

最终配置：RK4，物理时间步 0.000125 s；每次原有控制更新执行 16 个物理子步。
底层控制更新仍为 0.002 s；默认 steps_per_policy=20，策略更新仍为 0.04 s / 25 Hz。
不扩大动作延迟缓冲，不提高 PID 更新频率，也不改变奖励函数。
球和地面 solref 保持原有 [0.008, 0.1]，solimp 保持 [0.95, 0.998, 0.002, 0.5, 2]。
每次 reset 后重新应用配置，避免被旧环境或 YCB 重置覆盖；日志记录版本。

0.1 是接触阻尼比，旧配置中“restitution=0.8”不是实测恢复系数。
采用较细物理步长先解决旧接触模型的数值问题，不给不同真实球虚构反弹系数。
候选 .00025 s / solref [.02,.1] 也做过测试，但接触压入更大，未选用。
implicit/implicitfast 在所测低阻尼接触下没有优于细步长 RK4，未采用。

### 数值验证

MuJoCo 3.6.0；独立被动球，居中均匀球体，无机器人、水平速度或自旋。
三组球质量/半径，释放中心高度 0.898–0.902 m，各 17 点：
最终步长下三组反弹峰值分别约 0.508–0.527、0.506–0.525、0.507–0.526 m。
仍存在毫米至厘米级离散误差，不能称为精确收敛或真实材料标定。

另以固定 seed=20261027 随机抽取 60 组：质量 0.04–0.20 kg、半径 0.025–0.05 m、
释放高度 0.6–1.2 m、初始向下速度 0–1 m/s。
与 0.0000625 s 参考步长比较，最大峰值差约 0.0090 m，最大接触压入约 0.00334 m，
未发现超过初始能量允许高度的情况。
本轮数值验收门槛为峰值差 <0.015 m、压入 <0.005 m、无异常增能；60 组通过。
该范围是数值测试范围，并没有自动成为训练随机化范围。

实际 Catch 场景另测原始球、网球、棒球各 10 个种子，共 30 回合，保留历史发射与自旋，
施加零策略动作。全部出现地面接触且状态有限；第一次落地后、首次手接触之前的高度峰值：
原始球约 0.511 m，网球约 0.502–0.525 m，棒球约 0.507–0.527 m。
这是轨迹检查，不是抓取成功率。完整记录为 `bounce_stable_v1_scene_validation_20261027.json`。

相关回归测试共 31 项通过；包括新配置的实际 Track/Catch × 三种球 × 重复重置、
40 ms 策略周期保持、奖励透传、细步长收敛，以及 Roll / YCB 原有回归。
双进程异步向量环境也已完成 reset 和 step，输出 25 Hz、stable_v1。

复现（在 dcmm 环境、项目目录运行；输出使用新文件名）：

```bash
python -m ycb_objects.bounce_pretrain_validation --output /tmp/bounce_random_check.json
python -m ycb_objects.bounce_stability --timesteps .00025 .000125 .0000625 \
  --time-constant .008 --samples 17 --output /tmp/bounce_phase_check.json
```

正式证据位于 `/home/isee/Catch/analysis_artifacts/`：
`bounce_stable_v1_final_random_validation_20261027.json` 和
`bounce_stable_v1_final_convergence_20261027.json`。
较早包含 stable_v1 的非 final 报告对应探索候选，应以这里的最终参数和 final 报告为准。

### 使用限制与回退

这次验证解决数值稳定性；尚未辨识网球/棒球各自的真实摩擦、恢复、质心和惯量。
自由关节阻尼、armature 及手/地面接触混合规则仍继承历史环境，不能等同实机参数。
独立落球不涵盖带自旋的全部三维碰撞，后续还需实测与多参数任务验证。
默认 16 倍物理子步会增加计算量，实际训练吞吐需测量。
旧模型用 stable_v1 的成绩不能替代重新训练的评估。

回退只需 `bounce_physics=legacy`；本次编辑前文件和 SHA256 已保存到
`backups/before_bounce_stable_20261027_142721/`。
未启动长期训练，未修改旧 checkpoint，也未改 Bounce 奖励或 PPO。

## 下一阶段

在以上基础上加入训练时按 episode 采样球的物理参数：Roll 启用 adaptive_wait_v1，
Bounce 启用 stable_v1、奖励沿用原版。先做短训练与吞吐检查，再启动正式多球重训。

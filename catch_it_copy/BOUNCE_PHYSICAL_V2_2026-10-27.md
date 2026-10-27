# Bounce physical_v2：合理物理近似与近距离拦截发射

最终目标是数值稳定、质量/尺寸/惯量一致，并把首跳放在机器人前方较近区域。
不再复刻 legacy 原始球的异常高跳。此前草稿的 1.75–1.85 m 释放高度已撤销。

## 启用与回退

显式添加 `object_motion=bounce bounce_physics=physical_v2 object_eval=false`。
默认仍是 legacy；stable_v1 不变。回退选择 `bounce_physics=legacy` 或 `stable_v1`。
本轮编辑前备份：`backups/before_bounce_physical_v2_20261027_145429/`。
Roll 继续选择 `roll_reward_version=adaptive_wait_v1`，本轮没有进一步修改 Roll。

## 最终物理配置

- RK4，dt=0.000125 s；每次原有控制更新执行 16 个子步。
  控制周期仍为 2 ms，默认策略周期仍为 40 ms。
- 原始球 50 g、半径 4 cm，惯量 I=0.4*m*r²。
  网球 58 g、半径约 3.3276 cm，以薄球壳 I=(2/3)*m*r² 近似；
  棒球 138 g、半径约 3.6176 cm，以实心球 I=0.4*m*r² 近似。
  三个主轴各使用上述值，质心放在球心。
- 球自由关节 armature=0、damping=0；取消未经测量的额外惯性和六自由度统一阻尼。
- 显式球/地面 contact pair：solref=[0.008,0.11]、condim=4、
  滑动摩擦 0.3、扭转摩擦 0.005。不是通过两侧几何体的摩擦最大值间接控制。
- 球与手的接触用临界阻尼接触参数，与弹跳地面分开；保留手部既有摩擦。
  地面/机器人滑动摩擦保持地面的既有值。
- 不修改奖励函数、动作维度、PPO 或旧 checkpoint；释放后不补速、不添加向上力。

以上是物理自洽的仿真先验，不是实测材料参数。网球有限壁厚、棒球层状结构及真实
球/地面恢复、摩擦仍需要辨识。两个真实球当前共用接触先验，不代表它们在现实中
具有相同恢复系数；damping_ratio=0.11 也不等于 COR=0.11。

## 发射配置

`bounce_v2` 默认值与模块 DEFAULTS 一致：
释放中心高度 0.9–1.1 m；初始世界 y=1.6–2.0 m；x=-0.25–0.25 m；
水平速度 1.3–1.7 m/s，朝 -y，偏航 ±15°；初始 vz=-0.2 m/s；
自旋为滚动匹配分量的 0.5–1.5 倍，另有竖直轴随机自旋 ±3 rad/s。

通过靠近释放位置和调整朝向速度，使首跳靠近机器人，而不是增加弹性去模仿旧版异常。
这些范围提供可供训练的拦截窗口；尚未对每条轨迹完成包含底盘加速度、关节限制、
碰撞和策略反应的可达性证明，也没有保证旧策略或重训策略一定成功。
用户修改参数范围后应重新验证；代码允许的范围不等于全部经过验证。

## 验证

Track/Catch × 原始球/网球/棒球，重复 reset/step，通过姿态保持、质量/惯量关系、
显式接触对、40 ms 策略步长、状态有限检查；混合采样可复现且覆盖三种球。
新增测试还确认 step 透传旧奖励及终止标志。
23 项相关核心回归测试通过。

独立复制实际环境的球体、惯量、接触对及初始运动（无机器人），三种球各 10 个种子，
共 30 条主轨迹与 30 条半步长参考轨迹：
首跳峰值高度 0.493–0.589 m，世界 y=0.376–0.967 m；组平均峰值时间约 0.75 s。
首跳高度与半步长参考的最大差为 0.00928 m，未发现首跳超过释放高度的情况。
这是数值与空间位置验证，不是抓取成功率测试。
数据：`/home/isee/Catch/analysis_artifacts/bounce_physical_v2_near_launch_validation_20261027.json`。

主入口还完成 CPU、双异步环境、混合球参数的 2 次短 PPO 更新；这只验证训练链路，
没有开展长期重训，也不能据此判断收敛或泛化性能。
短训记录：`outputs/bounce_physical_v2_smoke/2026-10-27/15:27:31/`。
每次新训练保存 resolved_config.yaml 与带源码 SHA256 的 bounce_physics_provenance.json。

## 测试命令

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=bounce bounce_physics=physical_v2 \
  test_object=ycb:tennis_ball ycb_collision=sphere object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/bounce_catch_best_reward_653.87.pth
```

棒球改为 `test_object=ycb:baseball`；原始球用 `test_object=default` 并去掉显式 ycb_collision。
旧 checkpoint 仅供观察对照，不能把它的成绩当成新配置重训后的效果。

## 多球训练入口

新入口允许 `bounce_ball_mode=mixed`：每回合等概率选择原始球/网球/棒球的物理先验，
质量再扰动 ±10%、半径 ±5%，惯量随 m*r² 变化。保持同一个解析球视觉模型，
适用于当前状态教师；这不是 YCB 视觉外观混合训练。
`bounce_ball_mode=fixed` 是默认值；YCB 外观测试使用 fixed，mixed 使用 test_object=default。

从头训练示例（需自行按硬件选择 num_envs；本轮未启动正式训练）：

```bash
python train_DCMM.py test=False task=Catching_OneStage \
  object_motion=bounce bounce_physics=physical_v2 bounce_ball_mode=mixed \
  object_eval=false viewer=false num_envs=16 output_name=bounce_physical_v2_mixed
```

默认材料接触先验固定；如需额外材料随机化，可显式设置
`'bounce_v2.damping_ratio=[0.09,0.13]'`，但该扩展范围应先做独立轨迹验证再正式训练。

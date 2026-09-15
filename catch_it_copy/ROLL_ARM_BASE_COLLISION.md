# Roll link5 与底盘碰撞修复（2026-09-15）

用户日志为 `ranger_base` 对 `geom#60/body#15`，穿透约 0.6–1.5mm。
当前主 XML 按 body/geom 顺序核对：body15=link5，geom60=link5 碰撞 mesh。
这是机械臂自碰撞，不能通过放宽手桌距离、忽略底盘碰撞来解决。
运行时日志现在同时打印 body 名称，可核对服务器模型是否与本地一致。

## 修改

- 仅 Roll：IK 从实际六个臂关节角开始，避免内部未执行目标累积。
- 候选 IK 成功后，在独立 MjData 中按每关节最大约 0.03rad 的间隔检查关节插值路径。
  用 `mj_geomDistance` 检查 link2–link6 的有效碰撞 geom 到 ranger_base 的距离。
  默认余量 `roll_arm_base_margin=0.005m`。已经位于余量内时允许不进一步恶化的退让动作。
- 拒绝危险目标或 IK 失败时保持当前实测关节目标，并回滚辅助 IK 模型状态。
  使用已有约束罚分，实际碰撞仍终止；没有关掉碰撞或改变成功口径。
- Roll reset 的关节控制目标改为与实际 Roll 初始关节角一致，避免未来修改
  `roll_arm_joints` 后控制器又拉回通用初态。
- `[roll-check]` 增加 `guard_blocked`，碰撞日志增加实际 body 名称。

## 验证和局限

CPU 测试执行真实 guard 方法，覆盖危险目标拒绝、退让允许、实时状态不被试算改写。
已有奖励/策略回归一并运行。本机 MuJoCo DLL 无法加载，未完成真实网格距离及训练验证。
这不是动力学安全保证：只检查 ranger_base 与 link2–link6，使用当前手部姿态，
不能保证不同步关节运动、PID 超调、底盘倾斜或未检查部件的碰撞被拦截。
较旧 MuJoCo 的凸网格正距离查询也有精度限制，需以运行时接触日志验证。
API 参考：https://mujoco.readthedocs.io/en/3.2.6/APIreference/APIfunctions.html#mj-geomdistance

同步 `gym_dcmm/envs/DcmmVecEnv.py` 和 `configs/env/DcmmCfg.py` 到 Linux，先用原命令
设置 `num_envs=1 roll_log=true` 短跑。确认实际碰撞是否减少，以及是否频繁 guard_blocked。
如果一直被拒绝，应继续调整初始臂姿/可达拦截目标或检查控制器，不能直接取消检查。
旧策略可用于复测故障，新训练使用新输出目录，避免覆盖旧模型。

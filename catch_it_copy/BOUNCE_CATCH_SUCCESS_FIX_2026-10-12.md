# Bounce Catch 成功判定修复

实际标准入口 `DcmmVecEnv.__new__` 将 bounce/tan/弹 的 Catching 分发至 `_make_main_bounce`，再创建主文件 `DcmmVecEnv.py::BounceEnv`（5fe75d5f路线）。原 train_DCMM 的 Catch PPO采用 `ppo_dcmm_5fe75d5f`，其统计函数读取终止info中的success；有标签与视觉蒸馏也读取同一字段。Tracking单独分发至 `bounce_compat.make_bounce_july22`，不走本次修复分支。

旧判定 `success=truncated` 把没有进入抓取阶段但坚持到超时的回合也计为成功。已改为 `bounce_catch_proximity_v2`：

- 达到既有回合时间上限；
- 已进入既有 `grasping` 阶段；
- 结束时小球与link6距离有限且小于原阈值0.25米；
- 没有失败终止，失败与超时同时发生也不能算成功。

保留原有阶段切换、动力学、动作、奖励和结束时机；不新增持续时间、接触或速度阈值。这是对旧近距离保持口径的修复，不是“接触并稳定夹持”的新标准，距离接近本身不证明严格物理抓握。

失败超时会报告 `timeout_not_caught`，成功报告 `catch_success`；`failure_diagnostic`包含阶段、距离、阈值、未满足条件和旧超时标记。Track、throw、roll均未改判定。

原单模型测试、有标签Catch蒸馏、视觉Catch蒸馏统一生效。两个Catch配置的success_definition已更新；Track配置不变。旧结果不可直接与新成功率比较，需重测教师。蒸馏源码/资产哈希校验可能拒绝旧checkpoint续训/评估，不应绕过校验混用成功标准；新训练使用新输出目录。

建议先运行：

```bash
python -m distillation_visual.run teacher-eval --stage catch --tasks bounce --device cuda:0 --gl-backend egl --episodes 100 --output distillation_visual/runs/bounce_teacher_success_v2
```

检查输出success_version应为bounce_catch_proximity_v2。Windows本地MuJoCo DLL环境无法完成真实仿真验证，回归测试通过提取实际源码分支验证路由与成功判定，并核对PPO终止info统计。

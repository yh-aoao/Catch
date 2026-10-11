# 蒸馏接口更新（2026-10-11）

## 教师清单

| 阶段 | throw | roll | bounce |
|---|---|---|---|
| Track，18维状态/8维动作 | best_model_track/throw_track_best_reward_58.56.pth | best_model_track/roll_track_best_reward_33.45.pth | best_model_track/bounce_track_best_58.18.pth |
| Catch，30维状态/20维动作 | best_model/throw_catch_best_reward_497.09.pth | best_model/roll_catch_best_reward_252.54.pth | best_model/bounce_catch_best_reward_653.87.pth |

本地 Throw Catch 完整权重已补齐，20维动作、观测与动作转换以及推理一致性检查通过，推理误差为0。六个文件均记录实际SHA256；加载时严格核对维度、权重及归一化统计，不通过文件名中的奖励值推断成功率。

## 新指标与原测试保持一致

| 阶段/任务 | 主字段 | 判定 |
|---|---|---|
| Track throw | success | 原环境 throw_official：step_touch触发成功截断，保留同步失败/截断的官方边界语义 |
| Track roll | roll_eval_success | 原有历史有效拦截指标，不要求回合末仍接触 |
| Track bounce | success | 手掌或手指有效接触，且没有失败终止；超时不算成功 |
| Catch throw | success | 当前官方式阶段与结束判定，环境发布的截断成功；不再用旧低速连续5步规则解释 |
| Catch roll | roll_eval_success | region_retention_v5：实际接触且在区域、离桌离地后启动保持；结束时仍有效保持；失败终止优先。无0.5秒门槛、无0.08秒失联门槛 |
| Catch bounce | success | 当前Bounce环境的Throw式阶段/超时成功，与原Catch测试读同一字段 |

Roll区域以掌部geom中心世界坐标为参考，水平距离≤12cm，高度范围为中心下2cm至中心上球半径+12cm；离开区域失效，再进入需新接触。旧成功率不能直接与新版混比。legacy_truncated_rate、track_contact_success、原始success及failure_diagnostic保留作辅助诊断，不替代主字段。评估文件新增policy_role、success_definition、evaluation_revision，说明教师/学生及判定版本；轮次仍见eval_XXXXXX.json文件名。

## 训练设计

保持两个独立实验：Track学生29→256→256→128→8；Catch学生53→256→256→128→20。输入分别为原状态、上一动作及任务标签；每任务教师使用自己的冻结归一化统计。共享MLP在线DAgger，默认学生执行(beta=0)，教师仅标注，每任务均衡采样，按动作组等权MSE更新。Track不控制手指；Catch输出完整底座/臂/手动作。

不新增GRU，不把Track学生自动拼接到Catch学生，不在这次同步中改变优化器或环境。默认训练1000轮，每200轮评估一次，每次每任务100回合，教师基线每任务100回合；最佳模型仍按最差任务相对同环境教师基线的差距选择。新版先重新测教师，再训练学生；零教师成功率默认阻止正式训练。

## 服务器命令（在catch_it_copy目录）

```bash
python -m distillation_track.run check --tasks throw roll bounce
python -m distillation.run check --tasks throw roll bounce
python -m distillation_track.run teacher-eval --device cuda:0 --episodes 100 --output distillation_track/runs/v20261011_teacher
python -m distillation.run teacher-eval --device cuda:0 --episodes 100 --output distillation/runs/v20261011_teacher
python -m distillation_track.run train --device cuda:0 --iterations 1000 --num-envs 2 --output distillation_track/runs/v20261011_1000
python -m distillation.run train --device cuda:0 --iterations 1000 --num-envs 2 --output distillation/runs/v20261011_1000
```

两个训练命令按需分别执行。模型、判定、源码已变化，使用新目录，不复用旧baseline或resume；旧学生eval读取内嵌旧配置，不会自动转换成新版评价。无显示服务器仍保留Bounce Track render_mode=null。

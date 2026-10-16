# 双相机无标签蒸馏：并行物体状态辅助预测

独立实验目录，基于 distillation_visual 的 160×160 双相机版本。
Track、Catch 分别训练，默认仅位置辅助预测。旧目录未修改。

## 信息流

Top/Wrist 深度图 → 两路独立 CNN → 与机器人状态、上一步动作拼接 → GRU
- 动作头：输出 8/20 维动作，模仿当前任务教师。
- 并行辅助头：256→128→6，前三维位置、后三维相对位置变化率。
预测结果不拼入动作头；辅助损失更新共享 CNN/GRU。eval 的 forward 不计算辅助头。
任务标签、物体真值和教师特征不进入动作网络。

## 监督定义

采图同一时刻调用环境 _get_relative_object_pos3d()，使用未添加观测噪声的相对位置。
坐标沿用环境：相对 arm_base 原点，水平轴随底盘 yaw，z 为相对高度。
速度目标是这个坐标随时间的有限差分，因此包含机器人运动的影响，不是世界绝对速度。
新回合清空差分历史，第一帧只监督位置。标签在动作执行前存入回放，与图像同步。
位置全部监督（包括遮挡），暂未实现可见性检测；长时间遮挡下标签可能不可观测。
仿真标签没有注入噪声，但差分是平均速度，不应描述为瞬时解析速度。

总损失 = imitation_loss + position_weight * position_loss + velocity_weight * velocity_loss。
辅助损失使用 Smooth-L1；位置除以 1m，速度除以 5m/s 后计算。
按完整回合长度归一化并等权任务采样；padding 与首帧速度被屏蔽。
权重 0.1 是待验证初值，不是已验证最优参数。各损失单独写入 metrics.jsonl。

## 配置

- configs/{track,catch}.json：默认仅位置（位置权重0.1，速度0）。
- configs/{track,catch}_position.json：同默认。
- configs/{track,catch}_position_velocity.json：位置、速度权重均0.1。
- configs/{track,catch}_control.json：辅助权重均0，作为同架构对照。
所有配置保持原教师、成功标准、双相机范围与160分辨率。不要同时改变分辨率做辅助消融。

## 运行

从 catch_it_copy 目录执行。先检查5轮（小样本零成功允许仅用于排查流程）：

```bash
python -m distillation_visual_aux.run train --stage catch --device cuda:0 --gl-backend egl --iterations 5 --num-envs 2 --baseline-episodes 3 --episodes 3 --eval-every 5 --allow-zero-teacher-success --output distillation_visual_aux/runs/catch_position_smoke
python -m distillation_visual_aux.run train --stage track --device cuda:0 --gl-backend egl --iterations 5 --num-envs 2 --baseline-episodes 3 --episodes 3 --eval-every 5 --allow-zero-teacher-success --output distillation_visual_aux/runs/track_position_smoke
```

正式训练（每任务8环境，共24；每500轮每任务200回合评估）：

```bash
python -m distillation_visual_aux.run train --stage catch --device cuda:0 --gl-backend egl --iterations 5000 --num-envs 8 --baseline-episodes 200 --episodes 200 --eval-every 500 --output distillation_visual_aux/runs/catch_position_5000
python -m distillation_visual_aux.run train --stage track --device cuda:0 --gl-backend egl --iterations 5000 --num-envs 8 --baseline-episodes 200 --episodes 200 --eval-every 500 --output distillation_visual_aux/runs/track_position_5000
```

位置+速度实验：改用 --config distillation_visual_aux/configs/catch_position_velocity.json
或 track_position_velocity.json，并指定不同输出目录。零辅助对照同理使用 control 配置。

```bash
python -m distillation_visual_aux.run eval --checkpoint distillation_visual_aux/runs/catch_position_5000/student_best.pth --tasks throw roll bounce --device cuda:0 --gl-backend glfw --viewer --episodes 200 --output distillation_visual_aux/runs/catch_position_view
python -m distillation_visual_aux.run train --resume distillation_visual_aux/runs/catch_position_5000/resume_last.pth --iterations 6000 --device cuda:0 --output distillation_visual_aux/runs/catch_position_5000
```

新 schema 与原112/160基线不兼容，默认从头训练。resume读取checkpoint配置，不能靠--config切换实验。
保存 student_best.pth、student_last.pth、resume_last.pth，保持原来源严格校验。
最佳模型规则沿用最差教师成功率差距，不等于平均成功率最大。

## 验证范围

测试覆盖辅助梯度到两路CNN/GRU、标签不影响动作推理、padding与首帧屏蔽、
分块损失、环境reset标签配对、Track/Catch模拟训练/评估/续训。
运行 python -m unittest discover -s distillation_visual_aux/tests -v。
这些是CPU网络和替身环境验证，真实MuJoCo相机、服务器显存和学习收益需要实测。

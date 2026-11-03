# Contact 从头重训至 9000 轮

在 catch_it_copy 目录，激活 dcmm 后执行：

```bash
python -m distillation_aux_contact.run train \
  --output distillation_aux_contact/runs/catch_9000_seed123_retrain \
  --device cuda:5 \
  --gl-backend egl \
  --egl-device-id 5 \
  --seed 123 \
  --num-envs 8 \
  --iterations 9000 \
  --eval-every 2000 \
  --episodes 300 \
  --skip-teacher-baseline
```

不加 --resume：从头训练，不读取旧 Contact 权重。
保持旧训练的学习率 0.0003、每任务 8 环境、rollout 64、每轮更新 8 次、损失权重等设置。
2000、4000、6000、8000 轮及最终 9000 轮评估，每个任务 300 回合。
每次评估前独立保存 student_002000.pth 等权重，确保保留 student_008000.pth 和 student_009000.pth。
独立快照仅包含模型及元数据；完整优化器/replay/RNG 续训使用 resume_last.pth。
跳过教师闭环测试；仍加载冻结教师生成蒸馏标签。

只新增评估时的权重快照保存，不改变网络、损失或优化器。
旧 10000 轮运行和门控/固定相加融合运行使用各自原有目录，不覆盖。
重训的 8000 轮结果不保证复现旧日志 76.67%：环境源码、运行条件等可能不同，需独立验证。
服务器运行前需同步更新后的 distillation_aux_contact/run.py。

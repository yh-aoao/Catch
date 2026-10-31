# Future + Contact gated fusion

独立新实验，从头训练；不覆盖 Future、Contact 的代码和权重，不可直接 resume 单模块模型。

共享双相机 CNN、GRU；两个分支都读取原始 GRU 特征 h：

- Future：预测未来 1/3/5 步的世界坐标位移，经 Linear(9,256) 得到 ΔF。
- Contact：预测三个独立接触事件，经有界仿射调制得到 ΔC。
- 两个独立 sigmoid 门控读取 h：z = h + gF ΔF + gC ΔC，再由动作头输出。
- 门控初值均为 0.5，Contact 调制保持原版零初始化。不使用任务 ID 或物体真值作为推理输入。

损失 = 原版 imitation + 0.1 position + 0.1 Future + 0.1 Contact。
保留原版 Future 接触前有效掩码、Contact 截尾窗口掩码、循环状态重计算和 replay。
两个辅助损失独立于门控计算，避免关闭分支后丢失该分支的监督。

默认 seed=123、Adam lr=0.0003，每任务 8 个环境（共 24）、rollout=64、每轮更新 8 次。
10000 轮；5000/6000/7000/8000/9000/10000 各评估 Throw/Roll/Bounce 每任务 300 回合。
跳过教师闭环基线测试；蒸馏仍需要加载教师、核验权重和生成动作标签。

训练（在 catch_it_copy 目录）：

```bash
python -m distillation_aux_future_contact.run train \
  --output distillation_aux_future_contact/runs/catch_10000_seed123 \
  --device cuda:1 --gl-backend egl --egl-device-id 1 \
  --seed 123 --num-envs 8 --iterations 10000 \
  --eval-start 5000 --eval-every 1000 --episodes 300 \
  --skip-teacher-baseline
```

每次评估前保存 student_005000.pth 等独立模型；student_last.pth 每轮保存；
resume_last.pth 每 200 轮保存完整模型、优化器、replay 和 RNG。
student_best.pth 仍按三个任务成功率的最小值选择，非平均值。

中断续训：同一命令加 `--resume distillation_aux_future_contact/runs/catch_10000_seed123/resume_last.pth`。
源码有变动时，核查后可加 `--allow-source-changes`，变动会写入 resume_audit.json。

CPU 检查：`python -m unittest distillation_aux_future_contact.test_fusion`。
已验证合成数据下梯度、序列推理、padding、保存/加载/续训及评估时间表；未启动真实物理训练或 GPU 效果评估。

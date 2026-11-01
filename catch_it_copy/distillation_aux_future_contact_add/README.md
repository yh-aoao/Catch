# Future + Contact 固定相加融合

独立实验 `distillation_aux_future_contact_add`，从头训练。
共享双相机 CNN + GRU 后，Future 与 Contact 分支并行读取原始 GRU 特征 h：

```text
ΔFuture = Linear(forecast(h))
scale, shift = Linear(sigmoid(phase(h)))
ΔContact = 0.1 h * tanh(scale) + 0.1 tanh(shift)
z = h + ΔFuture + ΔContact
动作 = tanh(动作头(z))
```

没有门控网络、任务选择或任务 ID 输入；两个分支的融合系数固定为 1。
保留原 Contact 的有界调制和零初始化；其内部的 0.1 属于模块定义。
这与门控版本初始权重各 0.5 的贡献幅度不同，不是把其门控固定在 0.5。

总损失 = imitation + 0.1 position + 0.1 Future + 0.1 Contact。
沿用原版 replay、未来位移接触前有效掩码、Contact 截尾窗口掩码。
运行时代码、配置、schema、输出目录独立，不依赖门控版本。

训练设置：seed 123，Adam lr 0.0003，每任务 8 个环境（合计 24），
每轮每环境采样 64 步，每轮更新 8 次，训练至 10000 轮。
5000、6000、7000、8000、9000、10000 轮评估，每个任务 300 回合。
跳过教师闭环评估；教师仍需加载并生成蒸馏动作标签。

在 catch_it_copy 目录运行：

```bash
python -m distillation_aux_future_contact_add.run train \
  --output distillation_aux_future_contact_add/runs/catch_10000_seed123 \
  --device cuda:1 \
  --gl-backend egl \
  --egl-device-id 1 \
  --seed 123 \
  --num-envs 8 \
  --iterations 10000 \
  --eval-start 5000 \
  --eval-every 1000 \
  --episodes 300 \
  --skip-teacher-baseline
```

每次评估保存独立 student_005000.pth 等模型；student_last.pth 每轮保存，
resume_last.pth 每 200 轮保存模型、优化器、replay 和 RNG。
student_best.pth 按三个任务成功率的最小值选取。

中断续训：在相同命令上增加
`--resume distillation_aux_future_contact_add/runs/catch_10000_seed123/resume_last.pth`。
不兼容门控或单模块模型的 resume；源码变更经核查后可加 `--allow-source-changes`。

验证：`python -m unittest distillation_aux_future_contact_add.test_fusion`。
CPU 合成测试覆盖固定和确实传入动作头、两分支梯度、序列推理、掩码、保存及续训、评估时间表。
尚未执行真实 MuJoCo/GPU 训练，不代表已验证成功率。

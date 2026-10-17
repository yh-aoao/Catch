# 独立模型实验实施记录

## 当前完成范围

已实现以下五个独立代码目录，每个目录包含自己的网络、收集器、训练/评估/续训入口、Track/Catch配置、说明和测试。原有蒸馏目录、机器人任务奖励和成功判定未修改。

| 实验 | 目录 | 本轮状态 |
|---|---|---|
| B1 事件窗口分层动作蒸馏 | distillation_event | 已实现；真实环境待验证 |
| B2 教师中间特征蒸馏 | distillation_feature | 已实现；真实环境待验证 |
| B3 双相机门控 | distillation_camera_gate | 已实现；真实环境待验证 |
| B4 共享骨干＋分支动作头 | distillation_action_heads | 已实现；真实环境待验证 |
| B5 特征级MoE C | distillation_moe | 已实现；真实环境待验证 |

只实现方案C，没有实现选择完整policy的方案A。五个版本独立训练，没有将它们组合。

每个目录的README记录具体实现与局限。五个版本均从当前`distillation_visual_aux`复制训练基础设施，关闭物体位置/速度辅助损失，用于单因素比较。现有原始目录内未提交的修改被保留；实验副本包含当时环境加载修复。

匹配对照B0使用`distillation_event/configs/{track,catch}_baseline.json`关闭事件分层，保留相同CNN/GRU和完整回合动作监督。这个B0是本次实验的匹配对照，不应把历史训练的不同网络/辅助损失结果直接当作同一对照。

## 已验证与尚未验证

- 五个版本的Track/Catch均跑过合成环境的收集、优化、checkpoint保存加载、续训和整段/逐步GRU一致性测试。
- 事件采样检查了损失位置预算、窗口偏重、padding排除、无事件回退和子步短接触记录。
- 特征蒸馏检查了hook清理、教师无梯度、投影头不影响推理动作；已读取六个真实Track/Catch教师的动作与特征。
- 五版本及匹配B0的Catch教师接口检查通过；Track教师接口也已验证。接口一致不代表闭环性能验证。
- 本机PyTorch是2.5.1+cpu，CUDA不可用；MuJoCo动态库加载报WinError1114。因此没有完成真实环境短训、正式训练、成功率比较或效果提升验证。
- 合成测试中的success=1来自测试替身，不是机器人实验结果，不能引用为成功率。

## 服务器执行顺序

进入`catch_it_copy`项目目录，使用原来能运行MuJoCo训练的环境。以下命令依次执行各实验，前一个失败就停止，不会自动越过错误。

```bash
# 1. 真实教师接口检查，不需要GPU或渲染
python -m model_experiments.run check --stage track
python -m model_experiments.run check --stage catch

# 2. 短程真实环境验证：每版本2轮迭代，不用于报告结果
python -m model_experiments.run smoke --stage track --device cuda:0 --gl-backend egl
python -m model_experiments.run smoke --stage catch --device cuda:0 --gl-backend egl

# 3. 正式独立训练：包括匹配B0，然后依次B1~B5
python -m model_experiments.run train --stage track --device cuda:0 --gl-backend egl --iterations 1000
python -m model_experiments.run train --stage catch --device cuda:0 --gl-backend egl --iterations 1000
```

`egl`适用于可用的Linux NVIDIA无头环境，其他系统按实际后端选择。默认每任务一个环境，三任务共三个；可用`--num-envs`调整。正式训练不绕过教师零成功率检查。`1000`迭代是继承的起始预算，不保证收敛。

只跑某个版本可加`--variants feature`或`--variants moe`等。预览命令加`--dry-run`。输出保存在各自`runs/<variant>_<stage>_<mode>_seed123`。同目录的正式训练禁止覆盖，换种子或用对应模块的`--resume`恢复；源代码/资产/配置变化时会拒绝原样续训。

Catch实验直接模仿完整20维Catch教师，并不依赖先训练好的本实验Track学生。两个阶段可以分别比较。

例如评估MoE Catch：

```bash
python -m distillation_moe.run eval \
  --checkpoint distillation_moe/runs/moe_catch_train_seed123/student_best.pth \
  --episodes 300 --device cuda:0 --gl-backend egl \
  --output distillation_moe/runs/moe_catch_eval_seed123
```

门控版metrics记录camera_usage，MoE版记录expert_usage和router_balance。平均使用均匀不等于学到有效专门化，还需分析不同任务/接触阶段的路由。分支头、MoE改变参数量，正式论文还需要容量匹配与多种子比较。

## 尚未完成的原计划

1. 斜坡滑落接取：还未实现/验证机器人环境与专家训练。
2. 主动抛球：原有basket仍需闭环诊断与训练改进，本轮未宣称解决。
3. 多物体形态：还未接入物理资产、碰撞与跨形态评估。
4. 上述五种模型的真实环境训练和独立效果评估：等待可运行MuJoCo/CUDA的服务器。

后续应依据真实短训日志继续推进，保留已有Throw/Roll/Bounce逻辑。模型版本未经独立验证前不组合，也不把待办项目标成完成。

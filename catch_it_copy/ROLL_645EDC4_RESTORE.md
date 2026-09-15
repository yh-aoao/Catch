# Roll 恢复到 645edc4（9 月 11 日上午）

目标提交：`645edc4e21631683254fba51cc66b035b409518d`。
当前项目直接按任务分流，无需切换 Git 分支或创建另一个工作目录：

- Roll：645edc4 的环境、物理配置、机器人封装和 PPO。
- Bounce：保持此前恢复的 61709fc。
- Basket、Throw：继续使用当前代码。

## 使用

上传修改及所有新增文件到 GitHub，服务器 git pull 后重启训练：

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll output_name=Roll645Track
```

启动应显示：

```text
[roll-baseline] commit=645edc4e21631683254fba51cc66b035b409518d environment/config/PPO=original
```

第二阶段使用本次 Tracking 的模型：

```bash
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=roll output_name=Roll645Catch checkpoint_tracking=/绝对路径/roll_track模型.pth
```

## 恢复内容

旧文件原文保存在以下位置，仅替换配置和机器人封装的 import 名称：

- `gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py`
- `configs/env/DcmmCfg_roll_645edc4.py`
- `gym_dcmm/agents/MujocoDcmm_roll_645edc4.py`
- `gym_dcmm/algs/ppo_dcmm_645edc4/`

默认构造入口（Gym、直接构造、环境脚本）均路由到该旧环境。
初始化、奖励、阶段切换、成功/失败条件、旧 PPO 统计和保存逻辑一起恢复。
后来的桌下等待奖励、整手空间惩罚、IK 候选检查及 execution_guard_v1 不再用于 Roll。
真实底盘碰撞仍按旧环境终止，没有通过忽略碰撞制造成功。
roll_log=true 仅打印旧版本身份，不提供后来新增的 roll-check/control-check 日志。
以后调整本次 Roll 实验参数应修改私有配置 DcmmCfg_roll_645edc4.py。

## 验证和限制

ROLL_645EDC4_MANIFEST.json 记录恢复源码及哈希；换行归一化后验证。
已逐文件与目标提交比较 AST，仅允许声明的 import 替换。
共用的原机器人资源、IK/PID、基础工具和训练 YAML 与目标提交无差异。
回归测试验证三种任务入口及对应 PPO 的选择、旧源码完整性和参数传递。

这只确认代码恢复，不保证该版本就是当时好模型使用的代码，也不保证复现相同分数。
本机未完成实际 MuJoCo 训练验证。建议先短跑，再决定是否进行完整训练。

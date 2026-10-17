# B2 教师特征蒸馏

状态：代码和CPU合成训练流程检查；真实MuJoCo训练/效果未验证。

冻结原教师，通过临时forward hook取得actor_mlp（Track）或actor_mlp_t/actor_mlp_c（Catch）输出。每分支L2归一化、拼接后归一化，并补零至512维。每个教师有独立的学生GRU特征投影头；避免不同教师表示空间被强行混合。动作损失之外增加0.1倍特征平方距离。任务索引只选择训练期投影头，不进入策略forward。部署不计算这些投影头，无需教师特征。

## 隔离与接口

独立的core.py、run.py、environments.py、configs/track.json、configs/catch.json、tests/和runs/。基于2026-10-17当前distillation_visual_aux副本；原目录及已有任务环境/奖励/成功标准未修改。
所有版本关闭物体位置/速度辅助损失，单独比较各自改动。学生输入仍为真实top/wrist深度、本体状态、上一步实际动作和GRU历史；不输入任务标签或物体真值。
动作标签仍来自同一组冻结教师，训练采用学生访问状态上的教师监督。各自checkpoint schema隔离，不能直接续训旧基线权重。

## 从项目目录运行

```bash
python -m distillation_feature.run check --stage track --device cpu --output distillation_feature/runs/check_track
python -m distillation_feature.run train --stage track --device cuda:0 --num-envs 1 --gl-backend egl --output distillation_feature/runs/track_seed123
python -m distillation_feature.run eval --checkpoint distillation_feature/runs/track_seed123/student_best.pth --episodes 300 --device cuda:0 --gl-backend egl --output distillation_feature/runs/track_eval
```

Catch实验把训练命令的`--stage track`改为`--stage catch`，使用新的输出目录。Catch学生直接模仿完整20维Catch教师，不先加载本实验Track学生。教师checkpoint路径与哈希在各自配置中。
无头Linux NVIDIA服务器可用egl；本机/其他环境须选择实际支持的后端。num-envs为每任务数量，默认三任务共3个worker。
已有输出目录禁止覆盖。恢复用`--resume <运行目录>/resume_last.pth`，并保持源代码/资产/配置一致。测试必须用对应目录的入口。

## 验证与比较

```bash
python -m unittest distillation_feature.tests.test_pipeline
```

此测试使用合成环境，验证收集、反向传播、保存、加载、续训和GRU历史一致性，不是机器人成功率。
统一队列见`python -m model_experiments.run --help`。先check，再smoke，再正式训练；正式训练不要使用allow-zero-teacher-success绕过教师失败。
保留同一教师、评估标准、种子、采样预算；先独立消融，不组合改动。成功率/置信区间由现有evaluate产生，未完成真实训练前不能声称改善。

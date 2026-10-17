# B1 事件时间窗口分层动作蒸馏

状态：代码和CPU合成训练流程检查；真实MuJoCo训练/效果未验证。

网络与对照一致。采集物理子步的离开台面接触、地面接触、首次手接触、连续手接触0.12s事件。每回合抽32个损失位置，默认一半全回合均匀抽样，一半在可用事件类别中均匀抽样、再在事件前后3个策略步窗口中抽样。重复抽样以计数权重实现。完整回合仍从零隐状态顺序通过GRU，TBPTT长度不变。台面接触分离不等同于已验证离开桌边；当前版本未实现斜坡/投掷释放事件。没有事件时退回均匀采样。events.enabled=false为匹配B0。

## 隔离与接口

独立的core.py、run.py、environments.py、configs/track.json、configs/catch.json、tests/和runs/。基于2026-10-17当前distillation_visual_aux副本；原目录及已有任务环境/奖励/成功标准未修改。
所有版本关闭物体位置/速度辅助损失，单独比较各自改动。学生输入仍为真实top/wrist深度、本体状态、上一步实际动作和GRU历史；不输入任务标签或物体真值。
动作标签仍来自同一组冻结教师，训练采用学生访问状态上的教师监督。各自checkpoint schema隔离，不能直接续训旧基线权重。

## 从项目目录运行

```bash
python -m distillation_event.run check --stage track --device cpu --output distillation_event/runs/check_track
python -m distillation_event.run train --stage track --device cuda:0 --num-envs 1 --gl-backend egl --output distillation_event/runs/track_seed123
python -m distillation_event.run eval --checkpoint distillation_event/runs/track_seed123/student_best.pth --episodes 300 --device cuda:0 --gl-backend egl --output distillation_event/runs/track_eval
```

Catch实验把训练命令的`--stage track`改为`--stage catch`，使用新的输出目录。Catch学生直接模仿完整20维Catch教师，不先加载本实验Track学生。教师checkpoint路径与哈希在各自配置中。
无头Linux NVIDIA服务器可用egl；本机/其他环境须选择实际支持的后端。num-envs为每任务数量，默认三任务共3个worker。
已有输出目录禁止覆盖。恢复用`--resume <运行目录>/resume_last.pth`，并保持源代码/资产/配置一致。测试必须用对应目录的入口。

## 验证与比较

```bash
python -m unittest distillation_event.tests.test_pipeline
```

此测试使用合成环境，验证收集、反向传播、保存、加载、续训和GRU历史一致性，不是机器人成功率。
统一队列见`python -m model_experiments.run --help`。先check，再smoke，再正式训练；正式训练不要使用allow-zero-teacher-success绕过教师失败。
保留同一教师、评估标准、种子、采样预算；先独立消融，不组合改动。成功率/置信区间由现有evaluate产生，未完成真实训练前不能声称改善。

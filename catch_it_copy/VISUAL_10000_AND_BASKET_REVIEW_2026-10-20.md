# 10000 轮实验与 Basket 视频反馈

以下命令在服务器 catch_it_copy 目录执行，先同步本次代码。GPU 均写为 cuda:0，按实际分配修改。
Catch 三个运动模式每个各评估 300 回合，每 2000 轮评估，总目标为 10000 轮。
前三组恢复完整训练状态，保留原分辨率、种子、环境数和优化器；不额外再训练 10000 轮。
后两组从原 160 基线配置开始新训练。续训模拟器回合重新开始，不是逐步完全复现。

## 先检查服务器原件

Downloads 中下面三份 resume_last.pth 本次均报 failed finding central directory，不能用于续训。
请使用服务器完整原件，不要用损坏的下载副本覆盖。以下检查应全部打印 iteration=5000：

```bash
python - <<'PY'
import torch
paths = [
 'distillation_visual_aux/runs/catch_position_5000_new/resume_last.pth',
 'distillation_visual/runs/catch_dualcam160_5000/resume_last.pth',
 'distillation_visual/runs/catch_dualcam_5000_fixed_10_14/resume_last.pth',
]
for path in paths:
    s = torch.load(path, map_location='cpu', weights_only=True)
    print(path, 'iteration=', s['iteration'], 'image=', s['config']['vision'],
          'seed=', s['config']['seed'], 'envs=', s['config']['num_envs_per_task'])
    for key in ('student', 'optimizer', 'replay', 'rng'):
        assert key in s, key
PY
```

## 命令

1. 辅助位置预测，160，seed123，从已完成的5000轮继续：

```bash
python -m distillation_visual_aux.run train --stage catch \
  --resume distillation_visual_aux/runs/catch_position_5000_new/resume_last.pth \
  --output distillation_visual_aux/runs/catch_position160_seed123_to10000 \
  --device cuda:0 --gl-backend egl --iterations 10000 \
  --skip-teacher-baseline --eval-every 2000 --episodes 300 --allow-source-changes
```

2. 基础模型160，seed123续训：

```bash
python -m distillation_visual.run train --stage catch \
  --resume distillation_visual/runs/catch_dualcam160_5000/resume_last.pth \
  --output distillation_visual/runs/catch160_seed123_to10000 \
  --device cuda:0 --gl-backend egl --iterations 10000 \
  --skip-teacher-baseline --eval-every 2000 --episodes 300 --allow-source-changes
```

3. 基础模型112，seed123续训：

```bash
python -m distillation_visual.run train --stage catch \
  --resume distillation_visual/runs/catch_dualcam_5000_fixed_10_14/resume_last.pth \
  --output distillation_visual/runs/catch112_seed123_to10000 \
  --device cuda:0 --gl-backend egl --iterations 10000 \
  --skip-teacher-baseline --eval-every 2000 --episodes 300 --allow-source-changes
```

4. 基础模型224，seed123从头：

```bash
python -m distillation_visual.run train --stage catch \
  --config distillation_visual/runs/catch_dualcam160_5000/resolved_config.json \
  --image-size 224 --seed 123 --num-envs 8 \
  --output distillation_visual/runs/catch224_seed123_10000 \
  --device cuda:0 --gl-backend egl --iterations 10000 \
  --skip-teacher-baseline --eval-every 2000 --episodes 300
```

5. 基础模型160，seed42从头：

```bash
python -m distillation_visual.run train --stage catch \
  --config distillation_visual/runs/catch_dualcam160_5000/resolved_config.json \
  --image-size 160 --seed 42 --num-envs 8 \
  --output distillation_visual/runs/catch160_seed42_10000 \
  --device cuda:0 --gl-backend egl --iterations 10000 \
  --skip-teacher-baseline --eval-every 2000 --episodes 300
```

续训为6000/8000/10000评估；从头为2000/4000/6000/8000/10000评估。300为每个运动模式，合计900回合。
跳过教师测试不跳过教师动作标签计算。num-envs=8为每个运动模式的环境数。
--allow-source-changes 显式接受更新后的源码/资源差异，会在 resume_audit.json 保存差异清单。
它不能绕过模型配置、教师权重、种子和环境数的检查；若差异包含其他未审阅的环境/奖励更改，请不要把结果视为严格续训对照。
改变评分方式或评估样本量后会重新累计 best_score，旧输出目录不覆盖。
本地已通过两个CLI帮助检查、语法检查和4项续训约束测试；未实际进行GPU/MuJoCo续训。

## Basket 视频反馈

视频长约3.84秒，抽取序列可见反复小幅抬降/下摆，球在手附近离开，没有形成朝远处篮筐的明显飞行弧线。
球脱手附近的手指接触与力不能从此视角精确判断，不能直接断言是主动松手。
建议先验证低位自然稳定持球姿态，训练直接向前上方加速，不强制先下摆。
持球时分别给有限前向、上向速度进展奖励，再奖励二者协同；先学会真实释放和飞行，再强化入框。
当前launch_v4的几何平均奖励在前向或上向比例为0时整体为0，可能增加初期探索难度，但视频不能证明测试模型使用了这个奖励版本。
需要记录最后接触时刻、手掌姿态、手指动作、球相对手速度及释放速度，区分失手和主动释放。
本轮仅给Basket反馈，没有修改Basket控制或奖励。

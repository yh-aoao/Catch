# 多目标 Catch：每目标 200 回合评估

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python scripts/evaluate_multitarget_catch.py \
  --model-dir /home/isee/Catch/11-11 \
  --episodes 200 --instances 20 --jobs 2 \
  --output evaluation_results/1111_200
```

默认模型清单为 `scripts/eval_multitarget_models_1111.json`：

|名称|文件|测试物理|
|---|---|---|
|roll_fixed_261_94|roll_catch_best_reward_261.94.pth|固定|
|roll_dr_229_39|roll_catch_best_reward_229.39.pth|随机|
|roll_dr_relaunch_222_64|roll_catch_best_reward_222.64.pth|随机|
|bounce_fixed_3_39|bounce_catch_best_reward_3.39.pth|固定|
|bounce_dr_3_96|bounce_catch_best_reward_3.96.pth|随机|

每个Roll模型评估7个目标，每个Bounce模型6个目标，共33组、6600个完整回合。
本命令只测试，不训练、不修改模型；使用CPU，无需viewer或显示器。

## 物理与种子

使用当前完整目标目录，并在启动时复制到输出目录的`catalog_used.json`。
默认苹果速度1.5–1.7m/s，苹果和柠檬起点y1.9–2.1m。
DR每个目标使用20套物理参数，每套10回合，参数在该环境内固定；
尺寸±5%、质量±10%、滑动摩擦±15%，Roll滚动阻力±20%，Bounce阻尼比±10%。
固定组20个环境的物理参数相同，分别测试10个发射轨迹。

参数种子默认41000，各环境按rank采样；每回合发射种子由基础seed、rank、回合号
确定，并显式重置。这比旧简测的4套物理参数覆盖更广，发射种子也不同，
因此不要把与旧20回合结果的差异全部归因于策略变化。

`--instances`必须整除`--episodes`。例如`--instances 10`意味着10套参数各20回合。
`--jobs`是同时运行的“模型×目标”任务数，各任务内部使用同步环境；不是GPU编号。
内存不足先把jobs降到1，再考虑降低instances，注意后者会减少参数覆盖。

## 单个模型 / 目标子集

```bash
python scripts/evaluate_multitarget_catch.py \
  --model-dir /home/isee/Catch/11-11 \
  --only roll_dr_relaunch_222_64 \
  --episodes 200 --instances 20 --jobs 1 \
  --output evaluation_results/roll_22264_200
```

可添加`--objects apple lemon`只评估这两个目标。目标必须属于所选模型的任务。
用`--seed 52000`更换物理样本和发射轨迹，建议换一个输出目录。

## 改模型路径与参数模式

`--model-dir`替换整个模型目录。若文件名改变，复制并编辑模型清单，
通过`--models-file /path/models.json`传入。清单每项包含：

```json
[
  {"name":"my_roll","motion":"roll","physics":"dr","checkpoint":"/absolute/path/roll.pth"},
  {"name":"my_bounce","motion":"bounce","physics":"fixed","checkpoint":"bounce.pth"}
]
```

相对checkpoint路径相对于model-dir解析。physics可为fixed或dr；
测试“固定参数训练模型在随机物理下表现”时，将其physics设为dr，
使用单独name和输出目录。只支持当前物理版本与兼容的TwoStage Catch模型，
Bounce按controlled_hold_v1设置评估，不能把旧legacy模型结果混作同口径。

## 中断后继续

原命令保持参数不变，加`--resume`：

```bash
python scripts/evaluate_multitarget_catch.py \
  --model-dir /home/isee/Catch/11-11 \
  --episodes 200 --instances 20 --jobs 2 \
  --output evaluation_results/1111_200 --resume
```

跳过已完整完成的模型/目标；未完成的目标从头重测，不把残缺回合算入成功率。
模型、代码、目录或采样设置变化会拒绝续测，防止混合不同实验条件。
模型文件在测试期间不要继续覆盖写入。脚本会校验checkpoint摘要。

## 输出与口径

- `REPORT.md`：每目标成功率与模型总体。
- `summary.csv`、`summary.json`：已完成目标的汇总。
- `model_totals.json`：模型总体，包含已完成目标数。
- `模型名/目标/episodes.jsonl`：逐回合种子、初态、成功、失败原因、实际物理参数。
- `模型名/目标/summary.json`：参数样本、Wilson95%区间、耗时和摘要。
- `configs/`：通过正式训练入口解析的实际测试配置。
- `worker.log`：每目标的运行日志。
- `COMPLETE`：全部目标完成后才生成；没有该文件时汇总可能尚不完整。

Roll主指标为结束时区域留存；stable_hold是回合中曾达到稳定条件，不保证保持到最后。
Bounce主指标为controlled_hold_v1连续持球，不能与Roll区域留存直接比较。
不丢弃失败或停桌回合，不启用发射筛选。固定与随机两种测试分布不相同，
若分析随机化训练的因果影响，还应让两组模型都接受同一个测试分布。

正式评估默认20套参数仍是有限采样，不代表完整参数区间或实机成功率。

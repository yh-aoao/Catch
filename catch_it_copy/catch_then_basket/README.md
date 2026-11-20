# 连续接球 → 投框测试

使用四任务 Future 学生，先运行原 Throw 接球控制，接球稳定后自动切到 Basket。切换不调用 reset、不重放置球、不更换模型；逐项校验 qpos、qvel、ctrl、act、仿真时间、球质量惯量和几何接触参数完全一致。切换时清空 GRU 和上一动作，并将 Basket 搬运保持姿态设为接球后的实际姿态。

稳定接球默认要求连续 0.3 秒：球接触手部、球相对手部速度不超过 0.3 m/s、球距 link6 不超过 0.18 m。原 Throw 超时 success 不作为稳定接球成功。该检测使用仿真状态，是实验用任务调度器；实机需要用触觉/夹持反馈及视觉速度估计替代。

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python -m catch_then_basket.run \
  --checkpoint /home/isee/Catch/11-19/student_last.pth \
  --output /home/isee/Catch/analysis_artifacts/catch_then_basket_viewer_01 \
  --episodes 10 --device cuda:0 --viewer
```

去掉 `--viewer` 使用 EGL；可加 `--egl-device-id 0`。输出目录必须是新目录，避免覆盖记录。默认 CPU，可按机器修改 device。可调 `--hold-seconds`、`--relative-speed`、`--palm-distance`，修改阈值后结果不宜直接对比。

`summary.json` 包含稳定接球率、全流程投框率、接到后投框条件成功率及每回合失败阶段；`episodes.jsonl` 随每回合增量保存。`--check-handoff --episodes 1` 强制切换仅检查接口与物理状态，不代表真实接球成功，不能用于评估成功率。

这是零额外训练的串接基线。Basket 原本从预设托球姿态训练，实际接球姿态和球动力学可能不同；因此独立 Basket 高成功率不保证串接成功。保留原 Throw 球参数，篮筐从场景开始就存在。仅本测试环境关闭 Basket 手指姿态奖励初始化约束（该约束拒绝非预设杯状手势），控制尺度和动作头不变；原训练与评估文件不修改。

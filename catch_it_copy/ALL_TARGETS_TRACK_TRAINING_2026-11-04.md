# Roll七目标 / Bounce六目标 Track训练

使用新增的全目标目录`ycb_objects/calibration_data/object_benchmark_all_targets_v1.json`，不会把原来的三球训练脚本或仅新增物体的benchmark目录改成其他分布。

|任务|目标|默认并行环境|
|---|---|---:|
|Roll|原始球、网球、棒球、橙子、苹果、柠檬、侧放番茄汤罐|28，每目标4个|
|Bounce|原始球、网球、棒球、Racquetball、高尔夫球、垒球|24，每目标4个|

默认从头训练Track，不加载Track或Catch权重。seed123、GPU0、viewer关闭，最大10亿环境交互步。这里只提供正式训练命令，未自动启动长训练。

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
```

Roll：

```bash
bash scripts/train_all_targets_track_v1.sh roll device_id=0
```

Bounce：

```bash
bash scripts/train_all_targets_track_v1.sh bounce device_id=0
```

输出目录分别为：

- `outputs/roll_seven_object_tracking_v1_seed123/日期/时间/nn/`
- `outputs/bounce_six_ball_tracking_v1_seed123/日期/时间/nn/`

两条命令是独立进程。如果在不同服务器运行，需同步最新代码、全目标catalog及已下载的YCB资产。原三球的材质文件也需一起保留。

## 分配和物理

每个环境在创建时固定目标身份；每次reset重新采样轨迹、控制扰动和观测噪声，不重新编译几何。全目标以等比例环境槽位分配，不代表每目标实际完成的回合数严格相等（回合长度不同）。

旧三球委托现有physical_v1/physical_v2工厂，加载已有材质文件。Roll仍用0.8–1.3m/s；Bounce棒球保留baseball_intercept_v1。这里为每环境固定单球配置，不施加旧mixed模式才有的质量/半径抖动；目标间仍有真实尺寸、质量、惯量和材质差异。没有把所有新目标替换成原始球。

新Roll保留凸包水果与侧放圆柱。水果释放y=2.0–2.2m；橙子1.1–1.4、苹果1.6–1.9、柠檬1.4–1.7m/s。圆柱罐保留其原配置。新Bounce保留各自解析球、阻尼及按名义弹性设计的初始发射。所有目标仍有原有物理近似限制，非实物精确标定。

Track沿用当前各任务的跟踪奖励、PPO和动作适配；此次没有修改奖励函数。未来训练Catch应显式加载这轮七/六目标Track，不要误用旧三球Track。

## 环境数与batch

默认horizon64：Roll batch1792/minibatch896，Bounce batch1536/minibatch768。脚本自动把minibatch设为`num_envs*horizon/2`，保持每rollout两份minibatch并保证整除；可显式覆盖合法minibatch。

资源有限时可减少到14/12环境，每目标2个：

```bash
bash scripts/train_all_targets_track_v1.sh roll device_id=0 num_envs=14
bash scripts/train_all_targets_track_v1.sh bounce device_id=0 num_envs=12
```

`train.ppo.max_agent_steps=...`、`seed=...`等仍可覆盖。若要完整覆盖全部目标，环境数至少为7/6；建议使用目标数的整数倍。改变seed时可同时覆盖output_name，默认名称中的seed123只是标签。

## 代码与验证

训练入口：`scripts/train_all_targets_track_v1.sh`，调用已有`train_object_benchmark.sh`；完整目录：`object_benchmark_all_targets_v1.json`；分配及旧球委托：`gym_dcmm/envs/object_benchmark.py`。

单测覆盖7/6目标清单、默认每目标4槽位、异步进程完整覆盖、每目标实际step，以及原三球与直接旧工厂的reset/动态逐项一致。短PPO训练覆盖每目标至少一个并行环境，用于确认实际训练链路，不代表模型已训练好。结果在`/home/isee/Catch/analysis_artifacts/all_targets_track_20261104/`。

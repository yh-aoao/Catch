# Bounce 原物理轨迹正常速度观看

## 启动

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/view_bounce_realtime.sh tennis_ball
```

球名可选 `original`、`tennis_ball`、`baseball`。默认加载
`outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_464.31.pth`。

程序先在无窗口模式下计算3局原始策略轨迹，再打开MuJoCo viewer，按仿真时间1倍速循环播放。开始前需要等待预计算；终端会逐局显示进度和缓存目录。

窗口操作：

- 鼠标：沿用MuJoCo的旋转、移动和缩放。
- 空格：暂停/继续。
- 左右方向键：切换回合。
- R：重播当前回合。
- Esc或关闭窗口：退出。

这是明确标注的**策略轨迹预计算回放**。它用于正常速度观看已计算的动作，不会根据观看时的鼠标施力重新推理；交互扰动实验仍需使用原来的在线测试入口。

## 常用参数

```bash
# 从seed10开始计算5局
bash scripts/view_bounce_realtime.sh baseball --episodes 5 --seed 10

# 指定其他physical_v2 Bounce Catch模型；自动读取同训练目录的保存配置
bash scripts/view_bounce_realtime.sh original \
  --checkpoint /绝对路径/训练目录/nn/bounce_catch_last.pth

# 播放缓存，跳过物理预计算；目录用终端打印的cache路径
bash scripts/view_bounce_realtime.sh --load /绝对路径/缓存目录

# 播一轮后退出
bash scripts/view_bounce_realtime.sh tennis_ball --once
```

默认缓存位于 `/home/isee/Catch/analysis_artifacts/bounce_replay_球名_时间戳/`。每组缓存包含约125MB的编译场景，多局共用一份；每局轨迹通常约1–3MB。再次观看可用 `--load`。

## 如何保证展示原轨迹

预计算严格加载模型和归一化，使用原PPO动作适配、原质量/材质/初始发射条件，保留0.125ms物理步长、16个物理子步、2ms控制周期和40ms策略周期。

每2ms保存真实完整积分状态及qpos/qvel；每局最后一帧与原仿真的终局逐位核对。回放从该次实际编译场景恢复记录状态，只调用 `mj_forward`，不重新积分，也不改变球的初速、弹性、摩擦、奖励或模型权重。60fps显示时选择对应仿真时间的真实记录帧，时间量化误差不超过一个2ms控制周期。

回放直接使用被动viewer，没有原在线路径中未使用的640×480离屏RGB渲染。

## 验证

已验证网球单局、原始球两局、棒球三局的多次reset记录与加载，所有记录帧恢复后qpos/qvel逐位一致。网球seed0的终局位置和时长与此前正式成功率评估对应回合一致。

本机实际窗口测试约59fps、0.99倍实时速度。原始物理计算仍可能慢于实时，但这部分开销在预计算时完成。

另外对0.25/0.5/1ms物理步长做了对照：0.5ms计算可达实时，但首反弹高度最大改变约6.1cm，6组闭环中3组成功结果改变。因此没有为了观看自动改变项目的物理设置。实验见：
`/home/isee/Catch/analysis_artifacts/bounce_viewer_speed_20261029/DT_PROBE_REPORT.md`。

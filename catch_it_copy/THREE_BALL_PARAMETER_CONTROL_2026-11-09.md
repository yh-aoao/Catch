# 三球参数泛化对照实验

入口：`scripts/train_three_ball_parameter_v1.sh roll|bounce track|catch [dr|fixed] [Hydra overrides]`。
默认 dr，独立输出目录，不改旧模型或旧训练入口。

## 唯一实验变量

两个模式使用同一份三球目录：`object_benchmark_three_ball_control_v1.json`。
原始球、网球、棒球各1/3，24环境，每球8环境；每个环境固定一种球。
dr 在环境创建时采样一套物理参数，后续回合保持固定，发射轨迹每回合随机。
fixed 不采样物理扰动，物理参数固定为基准。

- 尺寸/半径±5%、质量±10%、运动表面滑动摩擦±15%。
- Roll额外滚动阻力±20%；Bounce额外接触阻尼比±10%。
- 视觉、碰撞和惯量同步缩放；不改变自由关节阻尼、重力和奖励。
- Bounce的阻尼变化不是COR直接±10%，同此前physical_dr_v1已经验证的实现。
- 环境数、horizon64、minibatch768、seed123、最大步数10亿均相同。
- Catch：加载对应Track，Catch分支与critic重新初始化；Track/Catch联合训练。
- Roll沿用当前legacy抓取奖励；Bounce沿用controlled_hold_v1及当前手指奖励，
  Track不冻结，学习率倍率1，归一化统计不冻结。

这里 fixed 是新建的严格对照，不宣称能完整复现历史三球模型：历史环境、
采样比例和旧mixed路径的质量/半径扰动可能不同。

## 参数泛化组：先 Track 后 Catch

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/train_three_ball_parameter_v1.sh roll track dr device_id=0
bash scripts/train_three_ball_parameter_v1.sh bounce track dr device_id=0
```

输出目录：`outputs/roll_three_ball_tracking_parameter_dr_v1_seed123/` 与
`outputs/bounce_three_ball_tracking_parameter_dr_v1_seed123/`。

Track训练完成并评估后填写相应pth：

```bash
bash scripts/train_three_ball_parameter_v1.sh roll catch dr device_id=0 \
  checkpoint_tracking=/实际目录/roll_track_best_reward_XX.pth
bash scripts/train_three_ball_parameter_v1.sh bounce catch dr device_id=0 \
  checkpoint_tracking=/实际目录/bounce_track_best_reward_XX.pth
```

Catch输出：`outputs/{roll,bounce}_three_ball_catching_parameter_dr_v1_seed123/`。
Track默认从头训练，Catch必须显式指定checkpoint，防止误加载其他目标组合。

## 固定参数对照组

把上述命令中的 dr 换成 fixed。Catch应加载 fixed 组自己的Track。
输出目录中的 parameter_dr 相应变成 parameter_fixed，两组互不覆盖。

如降低并行环境，可用 `num_envs=12`，每球4环境；对照两组必须一致。
换seed时建议同时设置独立output_name，保持实验目录清晰。

## 如何判断是否受随机化影响

两组使用相同累计训练步数及同样的模型选取规则，然后分别测试：

|模型|基准物理测试|未见过参数样本测试|
|---|---|---|
|fixed训练|测|测|
|dr训练|测|测|

同列用相同物理样本和发射种子，不把“fixed模型/固定测试”直接与“dr模型/随机测试”
的差异全部归因于训练。奖励分数不一定对应最高成功率；保留best与last后独立测试。
Roll报告区域留存和曾稳定持球，Bounce报告controlled_hold_v1。
训练输出保存resolved_config与每个环境的物理样本provenance，传模型时一并保留。

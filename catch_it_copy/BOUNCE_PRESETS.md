# Bounce 参数组使用说明

参数位于 `configs/env/bounce_presets.py`。P0–P8 是待验证候选，尚未证明可接。
默认 `bounce_physics=legacy bounce_launch=legacy`，沿用 DcmmCfg 和原发球范围。
以下训练／测试命令均在 `catch_it_copy` 目录执行。

## 组名

| 物理组 | 相对 P0 的变化 |
|---|---|
| P0 | r=0.040 m，m=0.050 kg，摩擦=[0.50,0.05,0.015]，弹性配置=0.80 |
| P1 / P2 | 半径 0.039 / 0.041 m |
| P3 / P4 | 质量 0.045 / 0.055 kg |
| P5 / P6 | 滑动摩擦 0.45 / 0.55 |
| P7 / P8 | 弹性配置 0.78 / 0.82 |

共同参数：接触时间常数 0.008、阻尼比缩放 0.50、自由关节阻尼 0.0002、手摩擦 [2.0,0.5,0.1]。
P 组物理值是固定标量；修改 `PHYSICS_BASE` 改共同默认值，修改 `PHYSICS_GROUPS` 改单组。
摩擦配置写入球 geom，不等于实测接触对摩擦；弹性配置值不是已标定的真实 COR。

| 发球组 | 设置 |
|---|---|
| L0 | 位置 [0,2.4,0.9]，水平速度 1.0，方向 0°，vz=-0.2，自旋倍率 1，wz=0 |
| L1 / L2 | 在 L0 基础上，水平速度 0.95 / 1.05 m/s |
| L3 / L4 | 在 L0 基础上，方向 -10° / +10° |
| gentle | 在 L0 基础上，水平速度 [0.95,1.05]、方向 [-10°,10°] 随机 |
| legacy | 原 x=[-0.25,0.25]、y=[2.1,2.7]、方向 ±25°、自旋倍率 [0.5,1.5]、wz=±3；高度和速度读取 DcmmCfg |

方向相对世界 -Y，非精确瞄准机器人；速度单位 m/s，角速度 rad/s。
`LAUNCH_BASE` / `LAUNCH_GROUPS` 中所有字段都是 [下限,上限]；相等即固定。
固定 L0 发球不代表机器人初态、PID、动作延迟和观测噪声全部固定。

## 查看：无需 checkpoint 的键盘 viewer

```bash
python3 gym_dcmm/envs/DcmmVecEnv.py --viewer --object_motion bounce --bounce_physics P0 --bounce_launch L0 --bounce_log
```

沿用原键盘控制机制（初始暂停时按空格开始，按键以环境回调为准）。
换 P1–P8 或 L1–L4 逐组观察。此入口沿用原来的 steps_per_policy=1，
用于手动看轨迹；策略评估应使用下面 train_DCMM.py 入口的正常控制频率。

## 单组训练 tracking

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=bounce bounce_physics=P0 bounce_launch=L0
```

## 单组训练 catching

将示例 checkpoint 路径替换为你实际训练出的 tracking 模型。

```bash
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=bounce bounce_physics=P0 bounce_launch=L0 checkpoint_tracking=/path/to/track.pth
```

## 指定物理组混合训练

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=bounce bounce_physics='"P0,P1,P2"' bounce_launch=L0
```

Hydra 需要保留双引号，把逗号解释为字符串而非 sweep；上面的外层单引号适用于服务器 Bash。
也可直接在 `configs/config.yaml` 中设置 `bounce_physics: "P0,P1,P2"`，省去命令行引用。
每回合均匀选择一个组，回合中途不更改。物理组、发球组的多个 ID 独立抽样，
因此同时指定多个 P 和多个 L 会产生交叉组合；初期建议只混合 P、固定 L0。
小幅速度／方向随机化可用 `bounce_launch=gentle`；恢复原发球分布用 `legacy`。

## 指定组观看已训练模型

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 viewer=True object_motion=bounce bounce_physics=P1 bounce_launch=L0 bounce_log=True checkpoint_tracking=/path/to/track.pth
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 viewer=True object_motion=bounce bounce_physics=P1 bounce_launch=L0 bounce_log=True checkpoint_catching=/path/to/catch.pth
```

逐组测试时保持 checkpoint 相同，依次替换 P0、P1……；无头测试用 `viewer=False`。
默认不打印每回合配置；单环境观看时建议开启 `bounce_log=True`，32 环境训练时建议关闭。
日志记录组名、实际质量、半径、弹性配置、初始位置和六维速度；
reset/step 的 info 额外提供 `bounce_physics_group`、`bounce_launch_group`，不添加到策略观测。
当前 PPO 不会自动生成按组成功率汇总表，逐组运行仍使用原测试统计。

## 验证

```bash
python3 -m unittest discover -s tests -p test_bounce_presets.py -v
```

此检查不加载 MuJoCo，验证组选择、默认兼容、配置隔离和实际发球函数的速度／自旋。
完整仿真与可达性仍需在训练服务器运行。本地 Windows 的 MuJoCo DLL 初始化失败，未完成 viewer 或训练端到端验证。

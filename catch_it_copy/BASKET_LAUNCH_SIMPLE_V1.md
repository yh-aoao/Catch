# Basket：先学会抛出

`launch_simple_v1` 是独立奖励分支，原有 legacy、launch_v4、launch_v5 保留。
本阶段只训练向篮筐方向释放小球，不优化落点和入框精度。

| 奖励 | 定义 |
|---|---|
| 有效抛出 +5 | 曾接触手；最后接触时朝篮筐的水平速度至少 0.3 m/s；随后连续无接触至少 0.05 s。无向上速度要求。每回合一次。 |
| 飞行距离 0～10 | 从最后手接触位置起，沿当时朝篮筐的水平方向记录最大自由飞行距离；每增加 0.05 m 得 1 分，0.5 m 封顶。按增量发放，不奖励地面滚动、碰撞后位移或重复来回运动。 |
| 未抛出 −5 | 回合结束或超时仍未有效抛出，扣一次。有效抛出后落地不扣此项。 |

不叠加旧版持球、弹道质量、加速度、距离、控制、时间或入框奖励。
有效抛出后再次接触手或其他物体，将永久停止本回合飞行距离奖励。
原有物理、动作控制、入框判定和终止条件保留。`info.success` 仍表示严格入框；
新增 `launch_success`、`basket_success`、`launch_forward_distance`，并写入 Basket control 日志。
因此原有测试程序显示的 success 不能直接当成抛出成功率。
按累计奖励保存的 best_reward 模型在本版本主要代表抛出能力。

## 从头训练

在 catch_it_copy 下运行，建议使用新输出名，不加载 v5 权重：

```bash
python3 train_DCMM.py test=False task=Catching_OneStage num_envs=32 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_reward_version=launch_simple_v1 \
  basket_curriculum_stage=3 basket_log=false device_id=0 \
  output_name=BasketLaunchSimpleV1
```

评估时仍传入同样的 reward/control/fixed-base 参数，设置 `test=True`、
`num_envs=1`、`basket_log=true`，通过项目原有 checkpoint_catching 参数加载新模型。
先观察 launch_success、forward_distance 以及 viewer 中是否主动抛出，再考虑入框训练。

参数集中在 configs/env/DcmmCfg.py 的 basket_simple_*。
纯逻辑测试：`python -m unittest discover -s tests -p 'test_basket_simple.py'`。
这些测试不验证 MuJoCo 的实际动力学或训练收敛，需要在训练服务器进行短训练和可视化验证。


## 碰撞诊断与测试播放（更新）

新增独立的机械臂/手撞底盘惩罚 -20（basket_simple_collision_cost），有效抛出后仍扣，
每回合只扣一次；未抛出失败 -5 可与其叠加。球撞底盘不算机器人自碰撞，仍保留原终止。
此项仅影响 launch_simple_v1。检测范围为 link1 及其后代与既有 base_id 几何的接触，
尚非所有非相邻连杆之间的自碰撞检测。

固定底盘 Catching_OneStage 的 test=True viewer=true num_envs=1 测试，默认实时限速。
追加 basket_viewer_speed=0.25 可四分之一速播放；0 取消限速。只改变等待时间，不改变物理步长。
训练不受限速影响。basket_log=true 时，每回合输出 [basket-end]，包含仿真时长、结束原因、
抛出标记、机器人撞底盘标记和碰撞几何/连杆名称。旧模型也可以直接使用诊断与慢放；
新碰撞奖励需要继续训练或重新训练才会改变模型行为。


## 腕部回收自碰撞诊断

测试命令增加 `basket_self_collision_debug=true basket_viewer_speed=0.25`。
每个物理步检测 link1～link6 的非相邻连杆，按几何对/类别每 0.1 仿真秒最多输出一次。
`[basket-self] kind=contact` 为 MuJoCo 接触，附法向力、接触距离、关节角度；
`geometry_proximity` 为小于 5 mm 的几何距离探测，不受常规接触过滤决定是否执行。
负距离表示所用几何代理重叠；视觉网格回退和凸包代理可能与可见外观不同，不能单独证明实际碰撞。
仅有 proximity 而无 contact 时检查模型 exclude、掩码、焊接体过滤和几何近似。
此诊断不覆盖相邻连杆、手指之间或所有手臂-手部组合，不改变奖励、物理或终止。
默认关闭；应在单环境测试开启，不建议大规模训练时开启。

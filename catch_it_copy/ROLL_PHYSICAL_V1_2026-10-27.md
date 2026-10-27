# Roll physical_v1：独立球体动力学与桌面材质

显式启用 `roll_physics=physical_v1`；默认仍为 legacy，旧 checkpoint 和历史环境源文件不变。
本轮新增物理版本，不进一步改动自适应等待奖励或 PPO。无长期训练启动。
编辑前备份：`backups/before_roll_physical_20261027_192527/`。
回退只需去掉新物理开关及对应 roll_material_file/roll_ball_mode/roll_ball_weights 参数。

## 修正内容

- 解析球碰撞；YCB 网球/棒球保留各自视觉模型。
- 移除球自由关节 armature=0.001 和统一 damping=0.0001，均改为零。
  不再用附加惯性改变自旋响应，也不使用六自由度统一阻尼模拟滚动阻力。
- 重建球心质心和对称惯量 I=k*m*r²；reset 后重新计算模型常量并保留机器人状态。
- 仅对球/桌面定义独立 condim=6 接触对，启用滑动、扭转和滚动摩擦。
  不会被桌面另一侧的默认摩擦覆盖，也不会把桌面滚动阻力施加到手指接触上。
- 每次 reset 同步质量、尺寸、惯量、接触参数和满足 v=ωr 的初始滚动分量。
  球初始底部离桌约 0.2 mm，之后不持续补速/施力；仍自然减速、离桌、自由下落。
- 保持历史仿真步长 2 ms 和默认策略周期 40 ms；不破坏旧奖励中使用步长计算的时间项。
- 可继续选择 roll_reward_version=adaptive_wait_v1；该奖励只塑形，不强制限制动作。

## 材质先验

默认文件：`ycb_objects/calibration_data/balls.roll_table_priors_v1.json`。

|球|质量 kg|半径 m|惯量系数 k|滑动 μ|滚动参数 m|
|---|---:|---:|---:|---:|---:|
|原始参考球|0.050|0.04000|0.40|0.40|0.00015|
|网球|0.058|0.03328|0.55|0.50|0.00030|
|棒球|0.138|0.03618|0.40|0.30|0.00012|

扭转参数暂共用 0.0001 m；法向接触 solref=[0.02,1]，solimp=[0.9,0.95,0.001,0.5,2]。
原始球在此版本是固定质量/尺寸的参考球，区别于 legacy 的历史随机原始球。
mixed 模式当前逐回合在文件配置中选择，不额外扰动质量/尺寸；与 Bounce 的采样实现独立。

网球 k=0.55 采用有限壁厚球壳近似，棒球 k=0.4 为实心球近似，见 Rod Cross (2002)：
https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf
并非实测 YCB 转动惯量。滑动摩擦是文献启发的工程近似；特定球和桌面的值未标定。
滚动/扭转参数也是工程先验，未声称 YCB 或论文提供了这些数值。
MuJoCo 的滚动/扭转参数具有长度量纲，不能直接当作滑动摩擦或恢复系数：
https://mujoco.readthedocs.io/en/3.6.0/computation/#contact
不将上述桌面参数冒充球/手、球/地面的完整材料标定；手接触与落地接触仍沿用历史耗散近似。
软接触正则化下极低速可能缓慢爬行，没有增加人为的速度归零阈值。

## 验证证据

- 21 项核心回归通过：新物理 Track/Catch × 三球重复 reset/step、质心/惯量/阻尼、
  初始状态不被覆盖、40 ms 周期、材质采样、自适应等待奖励兼容，以及原有 Roll/YCB 回归。
- 30 条独立球/桌面轨迹，速度 0.8–1.3 m/s、位置/方向沿历史范围抽样，全部自然滚到桌边。
  网球离桌速度约 0.797–1.098 m/s，原始参考球约 0.844–1.133，棒球约 0.847–1.134。
  与 1 ms 步长对照，最大离桌速度差约 0.000216 m/s；未发现机械能异常增加。
- 同为 1 m/s，在长桌滚动 2 s 后：参考球约 0.950、网球约 0.888、棒球约 0.955 m/s。
  关闭滚动阻力的对照仍约 0.999 m/s，说明差异来自独立滚动接触，而非人为关节阻尼。
- 实际机器人场景零策略动作，Track/Catch × 三球共 6 回合，状态有限，均在约 1.54–1.59 s 离桌。
  没有用这些结果冒充旧策略抓取成功率。
- 主入口加载旧 Catch TwoStage checkpoint 完成短测试；双异步环境下，混合三球的 Track 与
  Catching_TwoStage 各完成 2 次短 PPO 更新，Catch 加载了旧 Track 模型。
  这些只验证接口与训练链路，不证明收敛或泛化。

证据位于 `/home/isee/Catch/analysis_artifacts/`：
`roll_physical_v1_validation_20261027.json`、`roll_physical_v1_scene_validation_20261027.json`。
复现独立对照：

```bash
python -m ycb_objects.roll_physics_validation --output /tmp/roll_physics_check.json
```

## Viewer

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll roll_physics=physical_v1 \
  roll_reward_version=adaptive_wait_v1 roll_wait_guard=false \
  test_object=ycb:tennis_ball ycb_collision=sphere object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth
```

棒球改 test_object=ycb:baseball；参考球用 test_object=default 并删除显式 ycb_collision。
初始速度可加 `roll_speed=0.8` 或 `'roll_speed=[0.8,1.3]'`，新物理版本也支持原始参考球。
低速观察应给足仿真时间，例如 `roll_speed=0.5 test_env_time=5.0`。
旧模型未按新动力学训练，是否成功抓住需要实测，不能仅由奖励开关推断。

## 后续重训入口

新物理可设置 roll_ball_mode=mixed，默认等概率采样三种球；如需原始球更多：
`'roll_ball_weights={original:0.6,tennis_ball:0.2,baseball:0.2}'`。
仍应保持 Track→Catching_TwoStage 流程，不自动改成 OneStage。
配置与材料文件快照、源码 SHA256 保存到 roll_physics_provenance.json；
开启自适应奖励时还保留 roll_reward_provenance.json 和逐回合奖励日志。

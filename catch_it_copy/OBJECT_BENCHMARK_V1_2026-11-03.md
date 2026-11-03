# 新增 YCB 运动目标：Roll / Bounce benchmark v1

新增目标通过 `object_benchmark=roll_v1|bounce_v1` 显式开启，默认 `off`。已有三球的物理、奖励和训练入口保留。未启动正式训练。

## 目标与碰撞

|任务|目标|YCB质量|碰撞设计|
|---|---|---:|---|
|Roll|orange 橙子|47g|详细扫描模型的凸包，非球体|
|Roll|apple 苹果|68g|详细扫描模型的凸包，非球体|
|Roll|lemon 柠檬|29g|详细扫描模型的凸包，非球体|
|Roll|tomato_soup_can 番茄汤罐|349g|半径约34.6mm、长度约102.7mm的解析圆柱，侧放|
|Bounce|racquetball 美式短柄墙球|41g|解析球，半径约27.84mm|
|Bounce|golf_ball 高尔夫球|46g|解析球，半径约21.26mm|
|Bounce|softball 垒球|191g|解析球，半径约47.60mm|

水果是YCB塑料仿真水果，不代表相同形状的鲜果。水果使用高分辨率视觉网格的凸包近似，保留不规则外形；不使用上游过粗的五块低面数碰撞片来制造人为棱角。凹陷及柔性没有建模。

不采用上游统一的`0.001`惯量：水果按均匀密度凸包计算质心、惯量并按YCB质量缩放；视觉和碰撞一起移到估计质心。罐体按实心等效圆柱、球按指定结构系数计算惯量。均为结构近似，非惯量实测。

## Roll初始化

- 摩擦与滚阻是真实接触求解器参数，不加持续向前力。
- 罐子长轴水平且垂直于前进方向，按圆柱半径初始化滚动。
- 水果按初始姿态的真实最低支撑点求解 `v + omega × r = 0`；非对称支撑允许初始质心少量升降，不套球体`v/r`。
- MuJoCo自由关节角速度按局部坐标转换，已用实际世界角速度验证初始无滑动。
- 台面顶面约z=0.7m，前缘y=0.8m；根据形状最低点放置，初始间隙0.3mm。
- 释放位置更新：水果和罐子均为y=2.0–2.2m（桌边y=0.8m），初始方向±10°。水果使用各自初始速度：橙子1.1–1.4、苹果1.6–1.9、柠檬1.4–1.7m/s；罐子仍为0.8–1.3m/s。增加释放距离给机器人准备时间，并用合理初速减少途中停滚；释放后不再补力。`roll_speed=null`表示使用目标默认值，显式`roll_speed`优先。
- 允许自然摆动、偏转、减速。并不保证所有随机姿态、所有自定义速度都能滚到桌边。
- 无碰撞的透明`object`球仅是旧代码的尺寸/ID标记；水果和罐子真正的碰撞形状不是球。桌边/桌面完全离开的判断改为使用实际姿态的几何范围，仅新benchmark调用该可选接口。

奖励与策略观测/动作维度未变。旧奖励的预测目标、捕获点和手区尺度仍含中心/包围尺度近似；物理接入不等于非球体抓取已经解决。未来正式非球体benchmark需进一步确定姿态观测及统一的稳定持物指标。

## Bounce初始化与材质

|球|名义恢复系数|默认首次反弹净高度目标|
|---|---:|---:|
|Racquetball|0.84|0.65–0.85m|
|高尔夫球|0.80|0.65–0.85m|
|垒球|0.44|0.50–0.65m|

Racquetball的值来自规定反弹高度比的近似；高尔夫球0.80是工程先验，不是USGA规定的地面COR；垒球0.44为标准试验/文献启发的先验，也不是这颗YCB实物在该地面的实测参数。摩擦、接触时间常数与结构惯量同样是仿真近似，来源与状态在catalog中记录。

采用被动落球数值拟合，将MuJoCo阻尼参数映射到名义COR；不是直接把solref阻尼当恢复系数。保持2ms控制、40ms策略，物理步0.125ms。垒球地面接触时间常数0.004s，其他0.008s；手部接触继续用临界阻尼。

发射根据 `h_apex ≈ e²*(h_release + vz²/(2g))` 计算初始向下速度，再估计水平速度，使反弹进入前方约定窗口。低弹性的垒球需要更强的初始向下速度，绝不通过提高其弹性或落地后加力来保证到达。水平碰撞损失用显式先验估计，实际到达位置以被动轨迹检查为准，并非IK可达性证明。

Bounce Catch沿用controlled_hold_v1与四个手指奖励覆盖；没有因为目标变大而放宽成功条件。现有模型未针对这三种新增球训练，不能保证抓取成功。

## Viewer

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
```

Roll在线viewer，默认加载11-3-2的269.97：

```bash
bash scripts/view_object_benchmark.sh roll orange
bash scripts/view_object_benchmark.sh roll apple
bash scripts/view_object_benchmark.sh roll lemon
bash scripts/view_object_benchmark.sh roll tomato_soup_can
```

可指定模型和速度，例如：

```bash
bash scripts/view_object_benchmark.sh roll apple /home/isee/Catch/11-2/roll_catch_best_reward_250.65.pth 'roll_speed=[0.8,1.0]'
```

更改释放范围（保持在支持的台面范围内）：

```bash
bash scripts/view_object_benchmark.sh roll apple '+benchmark_options.distance_m=[1.4,1.6]'
```

Bounce正常速度回放，默认加载11-3-2的9.40：

```bash
bash scripts/view_object_benchmark.sh bounce racquetball
bash scripts/view_object_benchmark.sh bounce golf_ball
bash scripts/view_object_benchmark.sh bounce softball
```

Bounce先计算5局真实策略轨迹，再1倍速回放；不能通过鼠标扰动测试策略。`--episodes 10 --speed 1`可覆盖。可在物体名后传入其他Catch模型路径。

## 新目标的训练入口

建议先把新增目标当作未见物体评测组，避免与已训练三球混淆。

以下是单独的新目标训练入口，不自动包含原始球/网球/棒球：Roll为四个新增目标，Bounce为三个新增目标。原三球继续用已有equal-v2脚本。

```bash
bash scripts/train_object_benchmark.sh roll device_id=0 num_envs=16
bash scripts/train_object_benchmark.sh bounce device_id=0 num_envs=24
```

默认从头训练Track。每个并行环境在创建时固定一种几何，不在reset时重新编译模型；用分层分配保证环境覆盖不同目标，避免fork后全选同一目标。16个Roll环境各4个目标槽位；24个Bounce环境各8个目标槽位。每回合随机发射状态，几何身份不变，所以实际回合数不保证等量。可用roll_ball_weights/bounce_ball_weights指定新增目标子集及槽位比例。

之后Catch（必须显式提供Track）：

```bash
bash scripts/train_object_benchmark.sh roll task=Catching_TwoStage checkpoint_tracking=/实际新Track.pth
bash scripts/train_object_benchmark.sh bounce task=Catching_TwoStage checkpoint_tracking=/实际新Track.pth
```

Bounce按当前模型实际方式联合训练：不冻结Track，倍率1.0，允许更新归一化；四个手指奖励及控制约束沿用。训练上限可覆盖`train.ppo.max_agent_steps=1000000000`。这里是可运行入口，不是已完成正式训练的模型。

## 验证及回退

初版近释放验证在`/home/isee/Catch/analysis_artifacts/object_benchmark_20261103/`；后移释放的更新验证和变更前catalog在`/home/isee/Catch/analysis_artifacts/fruit_launch_revision_20261103/`。旧目录对应历史参数，不应当作当前水果释放结果。

初版检查：

- 三新球从0.5/1/1.5m净高度落下，验证名义COR、半时间步误差和能量上限。不是实物标定。
- 七种物体各10条被动轨迹（70条）验证自然离桌/反弹窗口、有限状态及能量；另做每目标种子0的半时间步检查。没有机器人或持续助力。
- 45项单元/运行回归覆盖几何、惯量、角速度坐标、圆柱侧放、接触ID映射、fork多环境目标分配，以及旧Roll评价与Bounce受控保持。
- Roll/Bounce各Track和Catch完成64环境步短PPO更新，只验证链路；不证明策略学得好。
- 每种新球完成两局当前策略精确回放、逐帧恢复检查。
- 原三球两任务共六个种子0检查回合，initial/final/reward/success与11-3-2主评估完全一致。

关闭`object_benchmark`即可使用旧入口，默认本来就是off。原始最佳版本备份仍在`backups/roll_bounce_best_20261102_201641/`；本次入口修改前文件在`backups/before_object_benchmark_20261103/`。不覆盖历史模型，不自动上传远端。

主要代码：`gym_dcmm/envs/object_benchmark.py`、`ycb_objects/benchmark_geometry.py`；参数：`ycb_objects/calibration_data/object_benchmark_v1.json`；被动验证脚本：项目上级`analysis/validate_object_benchmark_20261103.py`；数值COR拟合脚本：`analysis/fit_object_benchmark_bounce_20261103.py`。

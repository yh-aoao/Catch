# 网球、棒球实物标定准备

日期：2026-10-27。目标是让仿真匹配后续实机使用的具体球、桌面和地面。
当前没有收到实物测量，因此尚未完成真实标定，没有修改训练物理或覆盖原参数。

## 已核对的当前仿真值

实际创建两种球的Roll/Bounce Catch环境并reset，记录在项目根目录
`analysis_artifacts/ycb_calibration_reference_20261027.json`。
这些是仿真参考值，不是实测值。

| 参数 | 网球 | 棒球 |
|---|---:|---:|
| 资源质量 | 58 g | 138 g |
| 当前解析球直径（网格近似） | 66.552 mm | 72.352 mm |
| 主惯量约值 kg·m² | 1.93～2.08×10^-5 | 5.48～6.00×10^-5 |
| Roll旋转armature | 三轴各0.001 kg·m² | 三轴各0.001 kg·m² |
| Bounce旋转armature | 三轴各0.0001 kg·m² | 三轴各0.0001 kg·m² |

两球当前Roll均继承friction=[1.2,0.1,0.1]、condim=4、solref=[0.02,1]；
Bounce均继承friction=[0.5,0.05,0.015]、condim=4、solref=[0.008,0.1]。
资源质量不能代替手中具体球的称重，网格推算惯量也不是实测材料惯量。

标定前需要处理的模型假设：

- 自由关节的armature是额外惯性；Roll当前值明显大于球的上述物体主惯量。
  不能先保留任意额外惯性，再用摩擦补偿一条滚动曲线，随后声称参数具有真实物理意义。
- condim=4包含滑动和扭转摩擦，没有独立滚动摩擦维度；condim=6才启用滚动摩擦。
- solref第二项是接触模型阻尼比，不能直接用实测恢复系数替代。
- 球—桌、球—地面、球—手指是不同接触组合，不能只拟合桌面就把同一接触参数套到手指。

以上含义参见[MuJoCo接触参数](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
和[关节armature](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-armature)。
实际施加的接触参数还取决于接触双方的参数混合或显式pair设置。

## 第一步：尺寸与质量

每个实物球分配specimen_id，后续所有测量使用同一球，并记录是否磨损、潮湿等条件。

1. 电子秤称重，记录g；多次称量检查稳定性。
2. 测三个近似垂直方向的直径，记录mm。网球不要明显压扁后读数。
3. 记录桌面、地面材料及表面覆盖层。不同材料用不同surface_id。

解析球半径的初步估计使用三个直径均值/2。工具会把mm转换为m、g转换为kg。
若三轴直径相差明显，先检查测量与球形近似，不强行认定严格球体。
这些数据不能唯一确定转动惯量；网球的空心结构、棒球的多层结构需要另列惯量假设或测量。

## 第二步：滚动测量

在后续机器人实际使用的桌面进行，先检查桌面水平。
在画面中放置长度标尺，固定相机，尽量避免视差。可用手机高帧率录像。

- 每球至少测几个不同初速度，建议覆盖0.4、0.7、1.0m/s附近；这是采样建议，不要求手工精确打出这些速度。
- 同一条件重复约5次，并做相反方向的对照，检查桌面坡度或方向性影响。
- 手离开球后，从稳定运动片段记录至少4个、建议6～10个时间—位置点。
- 位置是沿近似直线运动方向的前进距离，单位m；记录真实采集时间，单位s。
- 慢动作视频应使用拍摄帧率计算时间，不要误用慢放后的播放帧率。
- 停止后的平坦片段不放入滚动拟合；明显滑动、跳动、反向或弯曲轨迹分开分析。
- 如能看清球面标记，可对照v与ωr检查明显滑动；只看质心轨迹不能确认纯滚动。

工具拟合`distance = offset + v0*t + 0.5*a*t²`，输出初速、加速度、位置拟合误差和
常减速度模型外推停止距离。它只是经验摘要，不能直接把减速度/g写成MuJoCo滑动摩擦系数。
实际衰减可能不是常减速度；拟合误差大时需要分段或改模型，不能硬套。

## 第三步：自由落球反弹

对每一种实际接触表面单独测量，可先采用球底离表面0.30、0.50、0.80m的释放高度，
每个高度重复约5次。球自然释放，不主动向下推，也不主动加自旋。

记录释放时球底离表面的高度h0、第一次反弹最高点球底离表面的高度h1。
两者使用同一高度定义，因此不需要再减球半径。
工具计算`e_effective = sqrt(h1/h0)`，作为忽略空气阻力等影响后的有效反弹目标。
这是物体—表面组合的经验量，不是MuJoCo solref第二项。
先使用首次反弹，之后再检查连续反弹是否也能匹配。

## 记录文件与命令

已创建空白模板，没有填入虚构测量值：

- `ycb_objects/calibration_data/tennis_ball.measurements.json`
- `ycb_objects/calibration_data/baseball.measurements.json`

可直接回复测量数字，也可以在JSON中填写数据。不同重复实验在trials列表中新增条目，
trial_id保持唯一、surface_id对应表面名称；首轮可留一部分重复实验作为后续独立验证集。

分析测量记录：

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python -m ycb_objects.calibration analyze \
  --input ycb_objects/calibration_data/baseball.measurements.json \
  --output ../analysis_artifacts/baseball_measured_targets_v1.json
```

空白数据的status为awaiting_measurements；有基础尺寸但无轨迹为basic_dimensions_only；
已有滚动/反弹目标才为empirical_targets_available。这些状态均不等于MuJoCo参数已标定。
输出记录输入文件SHA256；为避免覆盖测量结果，输出路径必须是尚不存在的文件。

需要另一份空模板或重新读取当前环境参考值：

```bash
python -m ycb_objects.calibration template --object baseball --output /新路径/baseball.json
python -m ycb_objects.calibration inspect --output /新路径/simulation_reference.json
```

## 收到测量数据后的拟合顺序

1. 固定实测质量、尺寸，明确惯量假设；在单独标定场景中审查/移除自由球不应有的额外惯性。
2. 分别拟合球—桌滚动和球—地面反弹，避免同时放开所有参数造成多解。
3. 将模拟运动与实测时间序列对齐，比较减速曲线、停止距离和不同高度的反弹。
4. 使用没有参与拟合的重复实验或初速度验证，再测试球—手指接触。
5. 导出带测量来源和假设的新物理配置；明确选择新配置才应用到测试/训练，保留旧教师配置。

当前尚未执行以上实测参数拟合，也没有将任何估计值自动写回训练环境。
离线工具的4项合成数据测试通过，仅验证单位换算、拟合与异常数据处理，不代表真实标定完成。

# 真实球材质：独立参数入口与标定流程

当前没有实测记录，真实网球/棒球材质尚未标定。YCB 外观、资产质量、理想球壳/实心球
惯量近似都不能提供球与具体地面间的真实反弹、摩擦或滚动阻力。
本轮新增的是独立材质加载、法向落球拟合、追溯和合成数据验证流程。
不增加猜测的网球/棒球默认材料差异，不改变 physical_v2 无材质文件时的行为或奖励。

## 已实现

- `bounce_material_file=/absolute/path/materials.json`，仅适用于 physical_v2。
- 同一个文件对应一个实际表面（surface_id），每种球有独立的接触阻尼、时间常数、
  滑动/扭转摩擦，可选实测质量与半径。每次 reset 根据球身份重新加载进显式接触对。
- mixed 模式只采样文件中有配置的球；fixed 模式必须有当前球的配置。
  mixed 仍扰动质量/尺寸，惯量同步变化。材质文件的接触值当前按球固定，不自动做材料区间随机化。
- 文件区分 `bounce_status` 和 `friction_status`，避免把只拟合反弹称为完整材质标定。
- resolved_config 和 provenance 保存材质文件快照、路径及 SHA256、配置和源码 SHA256。
- `ycb_objects.materials fit-drop` 从多次静止落球的高度拟合 MuJoCo 阻尼比，
  检查拟合误差与更细步长的差异；该阻尼比不等于恢复系数。
- 相关 11 项测试通过。合成数据恢复已知接触参数，用不同合成材料验证了 Track/Catch
  的逐回合切换；这不是实际材料测量。

## 最小测量

先选实机实验实际使用的一块地面，给它稳定的标识，例如 laboratory_tile_A。
同一球、同一表面、同一温度/充气状态尽量保持一致；记录品牌/型号/磨损/球编号。

每种球建议：

1. 称质量、测三个互相垂直方向的直径。缺失时拟合工具暂用资产值，不能声称尺寸已实测。
2. 手机慢动作录像，画面内有长度参考，轻轻释放，不向下推、不主动加自旋。
3. 从球底离地约 0.5 m、1.0 m 各释放三次，记录实际释放高度及第一次反弹高度。
   所有高度均指球的最低点离表面的距离，不混用球心高度。
4. 额外保留另一个高度（例如 0.75 m）的录像作独立验证，不拿全部数据拟合后再当验证。

空模板（数值与表面信息全部为空，没有虚构测量）：

- `ycb_objects/calibration_data/tennis_ball.material_measurements.json`
- `ycb_objects/calibration_data/baseball.material_measurements.json`

填写 specimen_id、surfaces.floor、质量/直径及 bounce_trials。六次落球无需完美恰好在同一高度，
填写各自的实际值；至少三个完整记录，且释放高度范围至少覆盖 0.1 m。
当前工具只拟合该表面的法向反弹，不应混入不同表面的数据。

## 拟合与使用（测量填完后执行）

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python -m ycb_objects.materials fit-drop \
  --input ycb_objects/calibration_data/tennis_ball.material_measurements.json \
  --output ycb_objects/calibration_data/tennis_ball.floor_fit.json
python -m ycb_objects.materials fit-drop \
  --input ycb_objects/calibration_data/baseball.material_measurements.json \
  --output ycb_objects/calibration_data/baseball.floor_fit.json
python -m ycb_objects.materials merge \
  --input ycb_objects/calibration_data/tennis_ball.floor_fit.json \
  --with-file ycb_objects/calibration_data/baseball.floor_fit.json \
  --output ycb_objects/calibration_data/balls.floor_fit.json
```

输出文件必须是新文件，避免覆盖测量与既有标定。
拟合通过要求输入点 RMSE<=0.02 m、半步长峰值差<=0.015 m、无被动异常增高；
这些只是拟合与数值检查，不是独立实机验证。
材料参数有不同高度/冲击速度相关性，单一阻尼不能拟合时应检查数据或换模型，不能强称成功。

先测试某个球，在原 physical_v2 测试命令中增加：

```bash
bounce_material_file=/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.floor_fit.json
```

混合训练使用 `test_object=default bounce_physics=physical_v2 bounce_ball_mode=mixed`，
并传入上述文件。文件只有网球和棒球时就仅采样这两种球，不悄悄补充未标定原始球。
原始球测试需要文件里有 original 配置，或明确不传材质文件继续使用仿真先验。
网球/棒球 YCB 测试仍使用原 `test_object=ycb:...` 和解析球碰撞。

## 尚未完成的真实材料工作

目前空模板不能生成真实参数，仓库内没有可冒充实测结果的默认材质库。
fit-drop 输出的 `bounce_status=fitted_drop` 只表明拟合了法向反弹；
`friction_status=simulation_prior`，默认 0.3/0.005 摩擦和接触柔度仍是近似。
下一步需要带水平速度的斜向碰撞录像，记录碰撞前后速度和自旋，才能约束滑动摩擦。
滚动阻力需要额外滚动轨迹及适当的接触模型，不能从法向反弹高度推出；
本轮没有修改 Roll，也没有将 condim=4 的配置称作独立滚动摩擦模型。
手掌材质也不是地面材质，当前手/球接触仍是原先的耗散近似，需另做抓取接触验证。

本轮编辑前备份：`backups/before_ball_materials_20261027_154021/`。
回退本功能只需不传 `bounce_material_file`；legacy/stable_v1 的路径未修改。

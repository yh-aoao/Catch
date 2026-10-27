# 网球/棒球文献近似参数

用途：在没有实测数据的阶段，为多球训练提供不同材质的合理仿真先验。
不代表 YCB 官方标定，也不代表用户实际球和地面的测量。
YCB 官方论文（https://arxiv.org/pdf/1502.03143）及模型下载页
（https://ycb-benchmarks.s3.amazonaws.com/index.html）未找到两种球可直接使用的
摩擦/恢复系数标定表。第三方仿真资产中的摩擦或接触默认值不能视为官方真实参数。

## 已生成的配置

`ycb_objects/calibration_data/balls.hard_floor_literature_v1.json`

|球|法向恢复系数目标 e|滑动摩擦 μ|I/(m*r²)|MuJoCo 接触阻尼比|
|---|---:|---:|---:|---:|
|网球|0.75|0.50|0.55|约 0.09776|
|棒球|0.50|0.30|0.40|约 0.25472|

质量、半径继续使用项目 YCB 资产值；armature、自由关节 damping 保持 physical_v2 的零值。
保留当前近距离发射条件、奖励函数、动作接口和控制周期。
使用 ball/floor 显式接触对，因此摩擦不会被地面另一侧默认的 1.0 覆盖。
扭转摩擦 0.005、时间常数 0.008、刚体接触模型均仍为仿真近似。

## 文献依据与适用条件

1. Rod Cross, *The bounce of a ball*, American Journal of Physics 67 (1999)。
   https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/BallBounce.pdf
   Table I 的低速刚性测试装置中，棒球入射/反弹速度为 1.25/0.61 m/s，
   e 约 0.49；网球低速样本约 0.81。文中亦引用网球混凝土标准反弹条件，e 约 0.745。
   本项目取网球 0.75、棒球 0.50 为跨实验的代表近似，不是同一地面的配对测量。
2. Rod Cross, *Grip-slip behavior of a bouncing ball*, AJP 70 (2002)。
   https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf
   文中硬表面网球法向恢复约 0.75，网球/球场滑动摩擦通常不低于约 0.5；
   光滑表面的一个试验测得约 0.27，说明摩擦高度依赖表面。
   本配置将 0.5 作为干燥硬地面的近似，不能泛指所有地面。
   该文还给出网球有限壁厚惯量近似 0.55*m*r²、棒球 0.4*m*r²。
3. 棒球同一目标地面的可靠滑动摩擦值未在这些资料中找到。
   0.30 明确是工程初值，不能声称是文献实测值；应在后续训练/评估做敏感性检查。
   https://physics.usyd.edu.au/~cross/baseball.html 也说明棒球法向恢复约 0.5 且随速度变化。

## 数值转换与验证

真实恢复系数 e 不能直接写成 MuJoCo solref 阻尼比。
生成脚本对 0.5/1.0/1.5 m 球底释放高度的静止落球，优化阻尼比，使模拟 e 接近上述目标，
并比较 0.000125 s 和更细步长 0.0000625 s。
两种球均通过 e 目标误差 <0.02、反弹峰值细化差 <0.015 m、无被动异常增高检查。
原始数据与误差保存在 JSON 的 numerical_validation 字段。
这只是把文献目标转成仿真接触参数，没有伪造实测记录。
相关 7 项回归测试通过，独立材质切换和惯量覆盖已检查。

重建到新文件：

```bash
python -m ycb_objects.literature_materials --output /tmp/balls.literature_rebuild.json
```

## 测试/训练启用

在已有 `bounce_physics=physical_v2` 测试命令后追加：

```bash
bounce_material_file=/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.hard_floor_literature_v1.json
```

测试仍用 `test_object=ycb:tennis_ball` 或 `ycb:baseball`。
混合训练使用 `test_object=default bounce_ball_mode=mixed` 并传入该文件，
会在这两种球之间采样，保留质量/半径扰动与关联惯量。
不传材质文件时维持原 physical_v2 共用接触先验，不会自动更换默认环境。
该文件未包含 original；固定测试原始球时应不传该文件。

棒球使用 e=0.5 后，从球底 1 m 静止释放只反弹约 0.25 m；网球约 0.56 m。
这些是 e²*h 推算的球底高度，不能为了抓取方便把两者强行调成同样高。
若棒球抓取窗口太低，应调整释放高度、距离、速度或拦截时机，保持材质先验可追溯。
本文没有声称新材质配置的旧策略成功率或实机迁移已经通过验证。

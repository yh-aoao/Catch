# YCB测试接入

> 2026-10-27关键修复：`mj_setConst`曾将YCB reset后的关节/物体位置改回qpos0。
> 现已保存和恢复完整积分状态，启动日志应含`reset=state_preserving_v1`。
> 旧YCB闭环失败不能全部归为泛化；最新对照和Bounce数值问题见
> [YCB兼容与泛化诊断](YCB_GENERALIZATION_DIAGNOSIS_2026-10-27.md)。

> 2026-10-26 Linux 本机更新：`dcmm` 环境现已完成三个 YCB 物体编译、完整 Roll 环境重置/步进及教师闭环对照。
> 下文 Windows DLL 不可用的描述为历史记录。最新结果见 [Roll 动态诊断](YCB_ROLL_DIAGNOSIS_2026-10-26.md)。

> 球类碰撞更新：现在默认 `ycb_collision=auto`，网球、棒球使用解析球碰撞并保留 YCB 视觉网格；
> 其他 YCB 物体继续使用网格。指定 `ycb_collision=mesh` 可复现旧网格碰撞。

来源 https://github.com/elpis-lab/YCB_Dataset 。三个已下载物体：tennis_ball（0.058kg）、baseball（0.138kg）、tomato_soup_can（0.349kg）。每个对象的上游commit和文件SHA256记录于assets/ycb/*.provenance.json，许可证在UPSTREAM_LICENSE。下载器不执行上游代码。

## 用法

从catch_it_copy运行，保持object_eval=false：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=throw test_object=ycb:tennis_ball object_eval=false viewer=true checkpoint_catching=/home/isee/Catch/best_model/throw_catch_best_reward_497.09.pth
```

Roll/Bounce沿用对应模型已验证的测试命令，添加test_object=ycb:baseball或test_object=ycb:tomato_soup_can。Track必须用Track checkpoint，不要给checkpoint_tracking传Catch模型。默认test_object=default保持原任务。

若服务器没有同步资源，或需要添加其他上游物体：

```bash
python -m ycb_objects.download --objects tennis_ball baseball tomato_soup_can
python -m ycb_objects.download --objects lemon
```

下载器读取同一次确定的上游commit，再下载XML所引用的网格、纹理和许可证。新增物体结构仍需单环境验证；不是所有YCB物体适合本手或本发射任务。

## 物理与接触

使用真实外观网格。网球、棒球默认使用解析球碰撞，其他对象使用CoACD碰撞部件。
额外保留不可见且不碰撞的object几何作为旧接口的尺寸标记；多部件的物体接触合并，手/底座接触中的物体部件ID映射回该标记。保持一个自由关节，不改变观测动作维数。

移除上游统一1e-3惯量，用各碰撞部件均分总质量后由MuJoCo推算惯量。这是近似，不是实测惯量；凸包重叠、均分质量都会影响精度。每次reset保留目标质量和惯量，物体接触参数跟随原任务，并同步到各碰撞部件。
解析球版本显式保留上述网格推算的质量、质心、惯量和主轴，便于先比较碰撞形状；没有把网球当作均匀实心球重新计算惯量。
解析球半径由视觉网格三轴包围盒直径的均值的一半近似：网球33.276mm、棒球36.176mm。
当前未用实物尺寸或滚动/反弹数据标定；实机迁移前仍需测量并调整。
原碰撞网格禁用碰撞且质量清零，新增独立解析球碰撞；旧object尺寸标记仍不碰撞。
阻尼、armature、摩擦和接触软硬参数沿用原任务，除非显式使用已有阻尼覆盖选项。

网格与纹理不意味着材料动力学准确：真实网球弹性、罐子质量分布、空心壳体惯量仍未标定。旧奖励/成功判定中的球半径读取采用网格关于局部原点的包围半径近似；尤其罐子的保持区域、桌面支撑和离地检测不应直接用于正式泛化论文指标。未宣称完成任意形状评估标准的重设计。

## 验证状态

已做下载资源存在性、质量总和、非碰撞标记、多部件接触映射、路径校验及语法检查。本机MuJoCo DLL无法加载，未完成真实模型编译/重置/运行验证。先num_envs=1 viewer=true，确认形状、接触和放置正常后再统计成功率。

仅接入train_DCMM.py现有物体测试入口，蒸馏学生独立CLI未扩展；未修改既有Throw/Roll/Bounce默认行为。

## 2026-10-26 接触参数修复

占位 object 几何现在同时继承原任务的 friction、solref、solimp、condim，继续保持不可见、零质量、不可碰撞。
此前重置将占位体的默认接触参数复制到网格，覆盖了 Throw/Roll 初始化时设置的参数。
现在重置同步保留原任务参数；Bounce 在基础环境重置时更新占位体的参数后，仍同步这些最新值。
本次接触修复当时没有改变网格形状、质量和惯量计算方式；后续球类解析球替换见本文顶部更新。

在能加载 MuJoCo 的服务器上，可不启动 viewer 检查网格编译、接触属性和惯量：

```bash
python -m ycb_objects.audit --objects tennis_ball baseball tomato_soup_can
```

输出包括碰撞部件数、质量、质心、主惯量及主轴姿态，并列出旧物体 XML 的惯量。
2026-10-27起audit默认auto与测试入口一致，球类审计解析球；加`--collision mesh`检查旧网格。
该命令验证独立物体编译，不替代完整任务 reset/step 和抓取效果测试。
当前本机仍有 MuJoCo DLL 初始化错误，因此不能声称已完成编译或动态验证。

Bounce 默认 Catch（5fe75d5f）和 Track（july22）配置有位置、方向、自旋随机化；
质量、半径、摩擦、反弹参数、关节阻尼及初始高度/速率的范围上下限相同，实际未开启这些参数的域随机化。

## Roll 网球到达桌边前超时

当前 `train_DCMM.py` 的 Roll 默认回合时限是 2.5 秒。网格网球减速后，部分回合仍在桌面运动就已超时。
复现此次旧网格诊断时，显式选择 `ycb_collision=mesh`，再延长时限：

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll test_object=ycb:tennis_ball object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth \
  test_env_time=5 ycb_collision=mesh
```

`test_env_time=null` 恢复原任务时限；覆盖值必须为有限正数且只允许 `test=True`。
可额外加 `ycb_joint_damping=0` 做阻尼消融：覆盖物体自由关节全部六个自由度的阻尼，
保留网格、质量、惯量、armature 和接触摩擦；默认 `null` 继承原任务。
此参数只允许 YCB 测试，不能用于偷偷改变历史训练配置。
重置信息记录实际阻尼与回合时限，完整入口也保存 resolved_config.yaml。
5 秒评估与旧 2.5 秒评估属于不同条件，统计必须标明时限。

新增纯物理诊断 `python -m ycb_objects.roll_probe --output /新目录`，
以及实际 Roll Catch 教师对照 `python -m ycb_objects.roll_policy_probe --checkpoint /模型路径 --output /新目录`。
前者的解析球选项使用旧包围半径，仅作诊断；正式入口现在按上述近似半径采用球类解析球。
viewer 每进程只播放一种对照：例如后者追加 `--viewer --variants long --episodes 3 --seed 1`。
教师诊断默认显式使用旧 `mesh` 碰撞以保持历史对照可复现；新版本可加 `--collision sphere`。

## 当前解析球播放

原测试命令无需添加参数，即自动切换两类球；也可以显式指定：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll test_object=ycb:baseball object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth \
  ycb_collision=sphere
```

换成 `test_object=ycb:tennis_ball` 即测试网球。控制台应出现 `shape=ycb_sphere`。
8项YCB测试通过；两种球×三种运动×Track/Catch共12种真实MuJoCo环境完成两次reset及一步step。
这些检查验证加载与接口，不代表抓取成功率。集成记录位于项目根目录
`analysis_artifacts/ycb_sphere_integration_20261026.json`。

## 命令行设置 Roll 初速度

新增可选 `roll_speed`，单位 m/s，仅限 `test=True object_motion=roll` 且
`test_object` 不为 default 的物体测试，支持 Track/Catch、YCB及已有基础形状预设。

- 固定初速：`roll_speed=0.5`，每回合水平速度大小均为0.5 m/s。
- 随机初速：`'roll_speed=[0.4,0.7]'`，每回合在范围内均匀采样。
- 省略或 `roll_speed=null`：沿用原任务速度范围；当前可动底盘Roll为0.8～1.3 m/s。

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll test_object=ycb:baseball object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth \
  roll_speed=0.5 test_env_time=5
```

设置的是释放时的水平线速度大小，不是持续恒速。随机方向和释放位置沿用原任务，
角速度同步按当前球半径更新为 `wx=-vy/r, wy=vx/r`，之后由物理自然演化。
允许速度为0；负数、非有限数、倒序范围会报错。
低速可能需要更长回合，因此示例显式使用5秒时限；时限和初速均记录在resolved_config.yaml。
重置信息新增 `roll_initial_speed`、`roll_initial_velocity`。

## 球未离桌时在桌前等待

可选 `roll_wait_guard=true`，默认false。只用于 `test=True object_motion=roll` 且
`test_object` 非default的评估，可用于Track/Catch。它是额外控制辅助，开启后的结果
必须标明guard，不能当作原教师的直接泛化成绩。

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=roll test_object=ycb:baseball object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth \
  roll_speed=0.5 test_env_time=5 roll_wait_guard=true
```

球尚未完全越过桌前边缘时：

- 用手掌和手指碰撞几何的保守包围范围，限制机械臂向桌内推进的目标命令。
- 底盘向桌前进时提前减速，手部接取点的目标世界高度为0.50m。
- 保留水平方向对齐动作，暂缓腕部旋转和手指闭合，保持等待姿态。
- 整个球越过桌前边缘且没有桌面接触后，本回合锁存解除约束，恢复原策略动作。

该机制投影控制命令，不修改机器人位置、物体轨迹、奖励、成功标准或终止条件。
由于控制延迟、IK跟踪误差和惯性，不能保证实际几何永远不越界，也不是实机碰撞防护。
reset会清除锁存状态；step info记录 `roll_wait_guard.active/intervention` 等诊断。
球停在桌上时会继续等待直到原回合终止，不会强制推球。

原因核对：原程序的 `roll_wait_target` 是奖励塑形，没有测试时动作限制。
改变碰撞形状、速度、惯量等会改变接球时序；尤其0.5m/s低于当前教师的0.8～1.3m/s初始化范围，
旧教师可能提前将手伸入桌下。等待辅助不能代替针对新物体/速度分布的重新训练。

初步验证：同seed=0～4、baseball解析球、0.5m/s、5秒时限的两组对照，
原策略在球仍在桌面时接取点最大y约1.09～1.19m（桌前边缘y=0.90m）。
guard启用期间手部保守前缘最大y约0.783～0.864m，五回合均在球离桌后恢复原策略。
两组均0/5抓取成功，不能声称此选项已提升抓取率。
数据位于 `analysis_artifacts/baseball_wait_guard_pilot_20261026.json`。
3项新测试覆盖提前下探命令、真实几何前缘、完整离桌解除、跨回合重置、Track与关闭路径。

# Catch 项目对话交接文档

整理日期：2026-10-26。用途：迁移到另一台电脑后继续项目。

本文是对当前长对话的结构化总结，不是逐字聊天记录。实验数字来自对话中用户提供的表格；没有重新读取所有历史附件。文件路径、权重、运行配置以迁移后的实际文件为准。不要把设计建议当作已经实现或验证的功能。

## 1. 项目目标与当前工作

项目基于 Catch It，分别训练不同运动模式的强化学习教师，再蒸馏为统一学生。

- Throw：接住从空中抛来的物体。
- Roll：接住沿桌面滚来的物体。
- Bounce：接住落地反弹后运动的物体。
- Basket / throw_basket：机器人主动抛球，最终目标为入篮；当前先训练有效抛出。
- 项目另有 Slide 模式；不要与 Basket 或三个蒸馏任务混淆。

Track 与 Catch 分别有独立教师和蒸馏流程。历史教师先训练 Track，再基于它训练 Catch。

最近的工作重点从视觉蒸馏转向：

1. 简化 Basket 奖励，使机器人先学会主动抛球。
2. 用 YCB 网球、棒球等替换原物体测试泛化。
3. 修复 YCB 接触参数覆盖问题。
4. 诊断 YCB 网球在 Roll 中低速时停在桌面的问题。

**当前尚未解决的具体问题：YCB tennis_ball 在 Roll 测试中，有时初速度较小时停住，无法滚到桌边。尚未证明主因是摩擦、阻尼、网格形状或惯量。**

## 2. 工作目录与迁移

- 当前 Windows 项目根目录：`F:\yjs\Catch`。
- Python 模块运行目录：`F:\yjs\Catch\catch_it_copy`。
- 历史服务器路径：`/home/yuhao2/Catch/catch_it_copy` 或 `/home/isee/Catch/catch_it_copy`。
- 历史资料和运行输出常被复制到 `C:\Users\29491\Downloads`。
- 对话附件原来位于 `C:\Users\29491\.codex\attachments`；这些绝对路径不会自动在新电脑上可用。

迁移时需要代码、权重、实验输出、YCB 资源和必要视频/附件。Git 忽略的 `.pth`、runs、Downloads 内容需要另行复制。本文不能替代这些文件，也不能恢复 Codex 原生会话。

## 3. 蒸馏方案演进

### 3.1 带任务标签的状态学生

- 三个教师独立冻结，训练程序根据当前环境任务选择教师。
- 学生接收机器人状态、小球位置/速度、任务标签和上一步实际执行动作等输入。
- 上一步动作指同一 episode 的上一策略步动作；新回合需要重置相关状态。
- 学生使用 MLP；具体维数、归一化范围和动作适配以对应模块代码为准。
- 在线采集状态并由教师提供动作标签，训练学生模仿教师动作。
- 不要把它描述成学生在部署时自动识别运动模式；带标签学生依赖标签接口。

目录：`distillation`（Catch）、`distillation_track`（Track）。

历史 Throw 权重存在 6/8 维 Track、18/20 维 Catch 的版本差异。曾因加载旧 6 维 Track 教师被适配检查拒绝。不得凭文件名认定维数，也不得盲目补零；应检查 checkpoint 和原控制器语义。

### 3.2 无标签双相机视觉学生

- 训练程序知道任务并选教师，学生不接收任务标签。
- 学生去掉直接的小球位置/速度输入，改用 Top 与 Wrist 相机深度图。
- 机器人自身状态与上一步动作仍作为输入。
- 基本结构：双路 CNN → 特征拼接 → GRU → MLP 动作头。
- Track 输出 8 维、Catch 输出 20 维，具体版本应核对 checkpoint/config。
- 使用深度图历史和 GRU 学习时序信息；真实小球状态可以供教师使用。
- 实验包含 112×112、160×160、224×224，不能仅凭目录名称确认实际分辨率，应读取 resolved_config。

目录：`distillation_visual`。

视觉学生必须通过视觉评估入口测试，不能直接拿原带标签单模型入口代替。`--tasks` 用于选择测试环境，不代表给学生输入标签。

示例（必须替换实际 checkpoint 和新输出目录）：

```bash
python3 -m distillation_visual.run eval \
  --checkpoint /实际路径/student_best.pth \
  --tasks throw roll bounce --device cuda:0 \
  --gl-backend glfw --viewer --episodes 10 \
  --output distillation_visual/runs/view_new
```

### 3.3 小球位置辅助预测

目录：`distillation_visual_aux`。

GRU 特征分出 MLP 辅助预测头，预测小球位置（3 维），使用仿真真值监督。辅助损失反向更新 CNN/GRU，促使共享特征保留与小球有关的信息。

讨论中的 B6 是“并行辅助预测”：预测位置不直接作为动作头输入，部署不需要仿真位置标签。速度预测是可选项，不能因为日志出现 velocity_loss 就认定它有非零训练权重。不同新增变体可能改变辅助分支连接方式，须核对具体代码。

### 3.4 模型变体

第一组 B 系列：

| 编号 | 模型 |
|---|---|
| B0 | 视觉无标签基准 |
| B1 | 事件分层动作蒸馏 |
| B2 | 特征蒸馏 |
| B3 | 双相机门控 |
| B4 | 分支动作头 |
| B5 | MoE |
| B6 | 小球位置辅助预测 |

B1～B5 的入口为 `model_experiments.run`，variants 分别为 `event`、`feature`、`camera_gate`、`action_heads`、`moe`。

后续新增五个独立实验目录：

| 目录 | 设计方向 |
|---|---|
| `distillation_aux_future` | 未来运动辅助预测 |
| `distillation_aux_belief` | 当前视觉与历史信息融合 |
| `distillation_aux_spatial` | 空间定位特征 |
| `distillation_aux_contact` | 接触相关辅助预测 |
| `distillation_aux_camera_gate` | 位置辅助预测＋双相机门控 |

实现细节查各目录 README 和 `visual_module_experiments`。对话没有给出这五个变体全部完成后的可靠结果，不要补造成功率。

## 4. 已讨论的实验结果

以下是用户最终提供的 Catch 5000 轮比较表；B0 已由 112 替换为 160 分辨率结果。

| 模型 | Throw | Roll | Bounce | 三任务均值 |
|---|---:|---:|---:|---:|
| B0 视觉无标签基准（160） | 63.5% | 71.0% | 64.5% | 66.3% |
| B1 事件分层动作蒸馏 | 51.0% | 52.5% | 47.0% | 50.2% |
| B2 特征蒸馏 | 48.5% | 57.5% | 67.5% | 57.8% |
| B3 双相机门控 | 62.0% | 55.0% | 71.0% | 62.7% |
| B4 分支动作头 | 60.5% | 57.0% | 66.5% | 61.3% |
| B5 MoE | 67.0% | 55.5% | 59.5% | 60.7% |
| B6 小球位置辅助预测 | 68.0% | 65.0% | 69.0% | 67.3% |

这些数字不能单独证明某模块稳定优于基准：应统一图像大小、并行数、采样量、seed、checkpoint 选择方式和评估回合。历史讨论中部分变体并行数不同。B6 对 B0 的均值优势约 1 个百分点，需要重复实验。

用户另提供过带标签学生 5000 轮最终结果：

| 阶段 | 任务 | 教师 | 学生 5000 轮 |
|---|---|---:|---:|
| Track | Throw | 96% | 78% |
| Track | Roll | 95% | 87% |
| Track | Bounce | 86% | 89% |
| Catch | Throw | 82% | 65% |
| Catch | Roll | 62% | 47% |
| Catch | Bounce | 87% | 71% |

历史表格还出现过其他学生数值，可能来自不同 checkpoint 或评估轮次，用户曾追问来源。正式汇报应重新追溯 JSON，不把 best 与 last、不同 seed 或不同成功标准混在一起。

历史 1000～10000 轮曲线和教师比较多保存在用户附件或 Downloads 输出目录中；本文没有恢复完整时间序列。训练轮次增加不保证成功率单调提高，动作模仿 loss 收敛也不等于成功率已达到上限。

## 5. 训练运行经验

- `num-envs` 在当前相关蒸馏入口中按任务计数：三个任务、每任务 8 个，合计 24 个环境 worker。
- 视觉训练同时消耗 CPU 仿真、GPU 网络计算、渲染和主机内存，不能只看 GPU 利用率。
- `--device cuda:N` 不保证 EGL 自动选择同一块物理 GPU。相关新增入口支持 `--egl-device-id N`，正确写法有空格或等号，不能写 `--egl-device-id:N`。
- CUDA 与 EGL 设备编号还需结合 CUDA_VISIBLE_DEVICES 和机器枚举核实。
- `EOFError` 可能只是子进程退出的结果。已有实际 SIGSEGV 发生在 MuJoCo Renderer 构造中；不能把所有 EOF 都判成 OOM。
- worker timeout 且 exitcode=None 表示当时没有返回结果，不能声称它已经崩溃。
- `gladLoadGL error` 发生在渲染上下文创建阶段，不是策略网络维数问题。
- 输出目录存在时，使用新目录或真正的 resume checkpoint；不要删除有用训练结果来绕过检查。
- 断点恢复通常用 `resume_last.pth`，策略评估使用匹配的 student checkpoint。恢复的 iterations 是上限还是追加量，应查该入口实现。
- 源码校验拒绝 resume 时，应审查变化再考虑 `--allow-source-changes`，不是无条件跳过。
- 五轮小测试成功只证明采集、训练、评估链路能跑，不代表策略已经学会抓取。

用户后期实验常用：10000 iterations、每 2000 轮评估、每任务 300 episodes。

## 6. Basket 简化奖励

详细说明：`BASKET_LAUNCH_SIMPLE_V1.md`。

已新增 `launch_simple_v1`，保留 legacy / launch_v4 / launch_v5。

| 项目 | 定义 |
|---|---|
| 有效抛出 +5 | 曾与手接触；最后接触时朝篮筐水平速度 ≥0.3 m/s；随后连续无接触 ≥0.05 s；每回合一次 |
| 自由飞行距离 0～10 | 沿释放时朝篮筐水平方向计算最大距离增量；0.5 m 封顶 |
| 未抛出失败 −5 | 回合终止或超时仍未确认有效抛出，扣一次 |

不要求向上速度，不叠加原有持球、弹道、加速度、控制、时间或入框奖励。确认抛出后再接触手/其他物体会停止飞行奖励；地面滚动不计分。

原有物理、碰撞终止和严格入框判定保留。`info.success` 仍表示入框；新增 `launch_success`、`basket_success`、`launch_forward_distance`。不能将入框 success 当成当前抛出成功率。

```bash
python3 train_DCMM.py \
  test=False task=Catching_OneStage num_envs=32 \
  object_motion=throw_basket basket_fixed_base_training=true \
  basket_arm_control=joint_velocity basket_reward_version=launch_simple_v1 \
  basket_curriculum_stage=3 basket_log=false device_id=0 \
  output_name=BasketLaunchSimpleV1
```

实现：`gym_dcmm/utils/basket_simple.py`，并接入 `gym_dcmm/envs/DcmmVecEnv.py`、`train_DCMM.py` 和 `configs/env/DcmmCfg.py`。

当时报告新增及相关奖励测试 33 项通过；完整 Basket 测试有其他兼容问题，没有完成服务器训练收敛验证。

`[early_term] reason=base_collision` 是回合碰撞提前终止日志，不是 Python 报错。仅凭该日志不能判定具体碰撞物体；需要接触 geom 名称。不要为了让回合变长就直接取消碰撞终止。

## 7. YCB 接入范围与命令

详细说明：`YCB_TESTING.md`。

已下载 tennis_ball、baseball、tomato_soup_can；资源位于 `assets/ycb`，下载与加载器位于 `ycb_objects`。

当前支持 `train_DCMM.py` 中单模型的 Throw/Roll/Bounce 测试，Track/Catch 各走对应环境。要求 `test=True`、`object_eval=false`。未开放该入口的 YCB 训练，独立蒸馏 CLI 未接入 YCB；Basket/Slide 也不由此入口支持。

在原来已验证的模型测试命令上添加：

```bash
test_object=ycb:tennis_ball object_eval=false
```

可换 `ycb:baseball` 或 `ycb:tomato_soup_can`。默认恢复为 `test_object=default`。模型路径和任务入口沿用对应任务，不混用 Track/Catch checkpoint。

```bash
python -m ycb_objects.download --objects tennis_ball baseball tomato_soup_can
python -m ycb_objects.audit --objects tennis_ball baseball tomato_soup_can
```

第二条是本次新增的无 viewer 编译审计：输出质量、碰撞部件数、包围半径、质心、主惯量、主轴姿态，并与旧 XML 惯量对照。它检查独立物体编译，不替代整个环境的 reset/step 和抓取验证。

### 7.1 物体参数对比

下列为当前默认单模型路线源码/资源结果，不代表历史 checkpoint 训练时一定使用相同配置。

| 参数 | Throw 原球 | Roll 原球 | Bounce 原球 | YCB 网球 | YCB 棒球 |
|---|---|---|---|---|---|
| 质量 | 35 g | 50～200 g | 50 g | 58 g | 138 g |
| 半径 | 35～45 mm | 30～50 mm | 40 mm | 包围半径 34.16 mm | 包围半径 37.81 mm |
| 外形尺寸 | 直径 70～90 mm | 直径 60～100 mm | 直径 80 mm | 66.50×66.66×66.49 mm | 72.02×72.50×72.53 mm |
| 碰撞形状 | 解析球 | 解析球 | 解析球 | 单凸网格 | 单凸网格 |

Throw 配置 object_mass 虽写 0.035～0.075，但当前采样调用下界两次，实际固定 35 g。

原球接触/关节参数：

| 参数 | Throw | Roll | Bounce |
|---|---|---|---|
| damping | 0.005～0.02 | 0.0001 | 0.00015 |
| 显式 armature | 未额外设置 | 0.001 | 0.0001 |
| friction | 0.95 0.3 0.1 | 1.2 0.1 0.1 | 0.5 0.05 0.015 |
| solref | 0.001 1 | 0.02 1 | 0.008 0.1 |
| solimp 前三项 | 0.998 0.998 0.001 | 0.9 0.95 0.01 | 0.95 0.998 0.002 |
| condim | 4 | 4 | 4 |

Bounce restitution=0.8 通过 `(1-0.8)*0.5` 映射为接触阻尼比 0.1，不等于已经测得实际恢复系数为 0.8。

### 7.2 接触参数覆盖修复

问题：compose() 原先为旧接口创建不可碰撞的 object 占位球，但没给它继承原球接触参数；reset 又从占位球复制参数到 YCB 网格，导致 Throw/Roll 的参数被默认值覆盖。

修复：`ycb_objects/loader.py` 中占位球也继承 friction、solref、solimp、condim；继续保持 mass=0、contype=0、conaffinity=0。Bounce 基础环境在 reset 中更新占位参数后，仍同步最新参数。

新增三任务×三个物体的 XML 参数继承回归检查。`tests/test_ycb_objects.py` 四个测试通过，语法检查通过。

### 7.3 网格与惯量

网球、棒球各有一个碰撞网格；视觉网格不碰撞，占位球不碰撞。原小球上述历史路线保留 XML 显式惯量 `7.33516e-05 kg·m²`（三个主分量）；YCB 删除上游统一惯量，按碰撞网格和分配质量重新推算。

这些值不是实测材料参数。未把 YCB 改回原球惯量，也未改碰撞形状。

当前 Windows Python 在 import mujoco 时出现 WinError 1114（DLL 初始化失败）。因此没有本机 MuJoCo 编译、完整重置或动态验证结果；不要声称已证实真实惯量数值或修复后成功率。

## 8. Bounce 是否做了泛化

当前 Catch 的 `configs/env/DcmmCfg_bounce_5fe75d5f.py` 与 Track 的 `configs/env/DcmmCfg_bounce_july22.py`：

- 初始 x 为 −0.25～0.25 m、y 为 2.1～2.7 m。
- 水平偏角 ±25°。
- 自旋滚动分量乘 0.5～1.5，绕 z 轴 −3～3 rad/s。
- 质量固定 0.05 kg，半径固定 0.04 m。
- 摩擦、反弹参数、接触时间常数和关节阻尼的上下限相同。
- 初始高度 0.9 m，水平速率 1 m/s，竖直速度 −0.2 m/s，均固定。

因此有初始状态随机化，但没有开启上述主要物体物理参数的域随机化。代码调用 uniform 不代表实际范围非零。机器人 PID/延迟等另有随机化机制，不应将其等同于多物体训练。

未修改 Bounce 泛化范围。未来如修改，必须区分新训练配置与历史教师评估配置。

## 9. 当前问题：Roll 网球低速停住

用户观察：替换 tennis_ball 后，部分低初速回合物体停在桌面，无法滚下去。

目前确认的条件：

- 桌面水平，物体靠初速度运动，没有持续前进驱动力。
- Roll 自由关节 damping=0.0001，同时阻碍平移与旋转。
- YCB 碰撞网格不是严格解析球面，可能引入摇摆、接触切换和局部平面停靠。
- 初始角速度按包围半径计算，与网格实际接触半径可能不一致。
- condim=4 通常包含滑动与扭转摩擦，滚动摩擦需要 condim=6；不能直接把 friction 第三项当成当前停球的已知原因。
- 约 0.034 m 半径、1 m/s 速度对应约 29 rad/s 自旋，数字较小的旋转阻尼也可能产生影响。

以上是候选原因，不是定量诊断结论。尚未执行以下对照实验：

| 实验 | 条件 | 目的 |
|---|---|---|
| A | 当前修复后的参数 | 建立速度、停止位置基准 |
| B | 仅将物体自由关节阻尼设为 0 | 检验阻尼耗能 |
| C | 保持网球质量、惯量，将碰撞形状换为解析球 | 检验网格接触影响 |

实验必须用相同位置、线速度、角速度、姿态和 seed，并记录 v、ω、质心、惯量、接触数量/geom、滑移及到达桌边情况。C 是诊断，不是 YCB 真网格成绩。

建议优先 B，再 C。不要直接整体降低摩擦使物体滑到桌边，然后声称滚动问题解决；不要靠强制持续推力掩盖问题。

## 10. 下一台电脑继续工作的提示词

> 请先阅读 CONVERSATION_HANDOFF_2026-10-26.md、YCB_TESTING.md，并核对当前源码。当前重点是 YCB tennis_ball 在 Roll 中低初速时停住。接触参数覆盖已修复，四项静态测试通过，但旧电脑 MuJoCo DLL 无法加载，没有动态验证。先运行 ycb_objects.audit，再设计相同初始条件下的阻尼与解析球碰撞对照，记录速度、惯量和接触情况，判断根因。保持历史教师和默认任务参数可复现；不要未经实验就把原因认定为摩擦，也不要将建议对照实验当作已实现功能。

需要继续更早的蒸馏工作时，先找实际 runs/config/checkpoint/eval JSON。不要仅依赖本文重建缺失的历史曲线或权重名字。

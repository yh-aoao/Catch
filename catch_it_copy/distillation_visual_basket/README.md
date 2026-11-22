# 四任务多目标蒸馏：视觉 Basket、自主停车与投球

学生输入为 Top/Wrist 深度图、机器人自身状态36维、上一动作20维、Catch/Basket任务指令1维。没有篮筐真实坐标、停车参考、阶段、进度、接触提示、剩余时间、物体类别、质量或半径输入。内部数据结构仍用10维context保存模式兼容序列工具，其后9维必须为零，网络仅拼接第一维；非零会直接报错。

学生沿用 Future Stable 接球结构，共享视觉和 GRU；Catch/Basket分别使用末端增量/关节速度动作头。新增 `256→128→3` 篮筐位置预测头，以机械臂基座坐标系中的篮筐中心相对位置为训练标签，Smooth L1 损失系数0.1。真值和预测均不额外拼入动作输入；辅助损失通过共享GRU训练视觉表示。推理不需要该标签。

## 教师与目标

- Throw 515.51：sphere、box、cylinder、capsule、ellipsoid，使用原尺寸/质量范围；每个worker固定类别，尺寸/质量按原环境创建时采样。
- Roll 403.15：original、tennis_ball、baseball、orange、apple、tomato_soup_can、coke_can。
- Bounce 14.04：original、tennis_ball、baseball、racquetball、golf_ball；使用checkpoint标识的controlled_hold_v1。
- Basket 40.98：单网球、free-stop范围与物理参数从checkpoint恢复；学生仅使用移动场景，不混入固定位置rehearsal。

Roll/Bounce使用creation_v3目录及physical_dr_v1，在创建环境时采样物理扰动。Catch旧checkpoint缺少完整环境清单，这些配置结合本地训练/评估配置重建，不能声称完整恢复了服务器所有训练选项。SHA256固定校验四份教师文件。

## 自主控制与训练标签

`AutonomousBasketEnv`允许底盘、机械臂、手指始终执行学生动作。没有阶段动作掩码、自动持臂/持手、阶段触发底盘制动、参考位置制动或接近制动；保留通用速度/加速度/关节限位及初始物理持球准备。旧环境默认行为保持不变，只有本子类启用新的控制分支。

教师查询仍可使用真实坐标和内部阶段，但它们只用于生成标签。移动阶段把教师环境保持的机械臂姿态转为有界关节反馈速度，把手部保持转为增量；投球阶段补齐底盘零速度标签。这是控制器行为的有界表达，在目标不可一步到达时会饱和，并非声称与原强制保持每个时间步严格等价。

初始采集以80%概率执行教师标签，随后在5000轮内降至10%，其余执行学生动作；评估100%执行学生。归一化校准阶段使用教师。四任务各占25%训练权重，目标在任务内均匀抽取。完整Basket序列若含两个阶段，模仿损失在两阶段各占一半，避免移动数据淹没投球数据；阶段仅是损失元数据，不进入模型。

环境仍保留内部停车/接触/释放状态用于成功判定、教师查询和训练统计，不会以该状态锁定学生执行器。本版验证自主控制下模仿学习是否能复现行为，不是已经完成可靠的连续接球→投框训练；必要时仍需学生轨迹纠正或RL微调。

## 训练

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/catch_mpl
python -m distillation_visual_basket.run train \
  --output distillation_visual_basket/runs/catch_10000_seed123 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 \
  --num-envs 32 --rollout-steps 16 \
  --iterations 10000 --eval-start 1000 --eval-every 1000 \
  --episodes 300 --skip-teacher-baseline
```

`--num-envs`指每任务环境数：32即总128个仿真进程，48即总192个。建议先16或32，监控CPU、内存、显存和每轮秒数再增加。独立进程各自创建MuJoCo和EGL上下文；CPU充足不保证GPU渲染或内存也足够。线程环境变量需在Python启动前设置。

四任务每步先分别批量推理并派发所有worker，再统一接收结果，使四任务仿真同时运行。默认每次优化固定抽取每任务2个完整序列，总8个；不会随目标数增加而扩大优化批次。目标更多意味着充分覆盖通常需要更多经验，不意味着每步推理必须成倍变慢。

32环境×4任务×16步=2048条环境交互/轮。增加环境数但不降低rollout长度，会增加每轮数据量和总训练预算，不能只按迭代秒数比较加速。这里的32/16与旧6/64的1536条/轮接近但不相同。旧版和新版迭代编号不代表相同数据量。

自动评估每任务合计300回合，各目标尽量均分，默认每个目标4进程；Basket仅统计移动场景。1000、2000、…、10000轮保留`student_001000.pth`等模型，另保存best、last和可续训resume_last。best按最低任务的目标宏平均成功率选择；不保证每个任务同时最优。

服务器若四个教师都放在同一目录，在上述命令增加：

```bash
--teacher-root /home/yuhao2/Catch/catch_it_copy/best_model_new
```

该目录需包含四个原文件名、内容不变；无需改SHA256。没有统一目录时编辑`configs/catch.json`各任务checkpoint路径。

续训：同一训练命令增加`--resume distillation_visual_basket/runs/catch_10000_seed123/resume_last.pth`。代码确实变更并核对后才增加`--allow-source-changes`。旧版学生和resume不兼容本schema，应从头训练。

## 检查

```bash
python -m distillation_visual_basket.run check --output /tmp/visual_basket_check --device cpu
python -m distillation_visual_basket.run env-check --output /tmp/visual_basket_envcheck --device cpu --gl-backend egl --check-steps 2
python -m distillation_visual_basket.smoke
python -m unittest distillation_visual_basket.test_student distillation_visual_basket.test_controls
```

check恢复教师并核对推理映射，不运行教师成功率基线；env-check覆盖18种条件的真实渲染与动作接口。smoke使用四个真实worker采集短序列并执行一次反向传播，短序列人为截断仅作工程验证，不是成功率评估。

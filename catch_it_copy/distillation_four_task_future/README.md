# 四任务 Future Stable 视觉蒸馏

四教师冻结，学生从头初始化；Throw、Roll、Bounce、Basket 从第一轮同时采样、每次一起优化。
这是独立新实验，不直接 resume 旧 Future、Stable、Contact 或融合模型。

## 教师、物体与动作

| 任务 | 教师（相对 Catch 根目录）|物体|动作缩放 base/arm/hand |
|---|---|---|---|
|Throw|best_model/throw_catch_best_reward_497.09.pth|original|1.5 / 0.025 / 0.15|
|Roll|11-5/roll_catch_best_reward_270.37.pth|original、tennis_ball、baseball|1.5 / 0.025 / 0.15|
|Bounce|11-5/bounce_catch_best_reward_13.68.pth|original、tennis_ball、baseball|1.5 / 0.025 / 0.08|
|Basket|11-4/throw_basket_catch_best_reward_47.35.pth|original|0.3 / 0.025 / 0.15|

配置对四个教师逐一锁定SHA256；路径相对 catch_it_copy 解析，因此服务器也须保持上述文件布局。
Throw普通环境；Roll physical_v1；Bounce physical_v2 + controlled_hold_v1；Basket mobile51_v1。
三球每个环境固定一种球，不做新的物理参数域随机化。保留原任务的初始位置/速度/旋转等分布。
Roll/Bounce环境参数来自已保存的20261105重建评估配置；Basket mobile/carry/finger选项从checkpoint提取。
旧教师未包含完整原始源码清单，因此不宣称完全恢复服务器历史环境。

## 学生推理接口

```python
action, hidden, logits = student(depth, proprio, previous, context, hidden)
```

- depth: `[B,T,2,160,160]`，固定归一化的Top/Wrist深度。
- proprio: `[B,T,36]`：底盘速度2、末端位置3、姿态四元数4、末端线速度3、手指状态12、机械臂关节位置6、关节速度6。
- previous: `[B,T,20]`，上一实际发给对应控制接口的归一化命令，回合重置时归零。
- context: `[B,T,10]`，布局如下。

|列|含义|
|---|---|
|0|外部任务指令：catch=0、basket=1；它也作为目标有效开关|
|1:4|篮筐中心相对arm_base的位置，遵循当前环境的世界轴对齐坐标差，并非任意改成车体旋转坐标|
|4:6|停车目标的车体平面相对位置（同教师）|
|6|外部阶段控制器：导航0、投球1|
|7|停车稳定进度|
|8|持球接触状态|
|9|剩余时间比例|

Catch模式下，列1:10全部关闭；模型编码前后均屏蔽缺失目标，不会因目标MLP偏置而泄露目标信息。
不输入Throw/Roll/Bounce运动类别、球种ID、质量、半径或物体真值。训练器知道任务/球种仅用于环境、教师、采样和统计。

共享CNN、GRU。目标5维→64维编码；任务及阶段状态参与循环输入。
Catch头使用 `h + Linear(Future(h))`；Basket头直接使用h。
两个头均20维，但机械臂控制语义不同：Catch末端增量，Basket关节速度，部署必须使用对应动作适配器。
Basket按外部阶段屏蔽：导航仅base2、投球仅arm6+hand12；环境保留教师对应的持球、刹车等控制规则。
任务切换应在回合边界进行，并重置hidden和previous；不支持未经验证的回合中途切头。

**部署边界**：本版不是自动判断投篮意图的系统。运行程序必须提供catch/basket指令。
当前Basket目标与阶段状态由仿真环境提供；实机需要用篮筐标定/定位、机器人反馈、接触估计及计时复现，不能当成纯视觉自动阶段识别。

## 数据与损失

- 每个任务独立环境组；默认6环境/任务，四任务共24。
- Roll、Bounce各3球，每球2个环境槽位。可使用8环境，但槽位数不完全均匀；Replay优化采样仍等权。
- 学生执行动作，各任务教师对同一个观测状态给标签。
- `Replay[任务/物体]` 独立FIFO；总容量8192帧/任务，三球各约2730帧。
- 每个优化批，每个物体抽1个完整回合，共8回合；按任务/物体权重形成四任务各1/4，不按总帧数加权。
- 每个回合按长度归一化；Basket模仿损失只计算该阶段实际启用的动作组。
- 前三任务：模仿 + 0.1位置SmoothL1 + 0.1未来位移SmoothL1。
- Basket：动作模仿；不使用当前位置和Future辅助损失。
- Future预测1/3/5策略步的世界球心位移，手部首次接触前有效，不跨终止回合。
- CNN/GRU/两个头从第一轮一起训练，无先三任务后Basket的预训练阶段。

## 训练

服务器同步本目录及依赖公共代码（尤其当前gym_dcmm、distillation_visual_aux、future_prediction_experiments、model_experiments和ycb_objects）。

```bash
cd ~/Catch/catch_it_copy
conda activate dcmm

python -m distillation_four_task_future.run train \
  --output distillation_four_task_future/runs/catch_basket_10000_seed123 \
  --device cuda:0 \
  --gl-backend egl \
  --egl-device-id 0 \
  --seed 123 \
  --num-envs 6 \
  --iterations 10000 \
  --eval-start 5000 \
  --eval-every 1000 \
  --episodes 300 \
  --skip-teacher-baseline
```

学习率前5000轮3e-4，之后余弦衰减至10000轮3e-5。
rollout=64、每轮Adam更新8次、BPTT=16，梯度范数上限1。
启动时读取已有教师权重并核验推理/观测/物理动作一致性；不训练教师，不做教师完整成功率基线评估。
先校准机器人状态归一化，并等待每个任务/物体至少一个完成回合，然后开始计数训练轮次。

## 模型保存与续训

- student_last.pth：每轮保存学生。
- student_005000.pth等：5000/6000/7000/8000/9000/10000评估前独立保存。
- student_best.pth：四任务各自物体宏平均成功率的最小值最高的检查点；并非直接选四任务平均。
- resume_last.pth：每200轮保存模型、Adam、Replay、RNG。大Replay可能占数GB磁盘。

续训：上述命令追加

```bash
--resume distillation_four_task_future/runs/catch_basket_10000_seed123/resume_last.pth
```

模型/优化器/Replay/RNG恢复，物理环境回合重新开始。LR按恢复的iteration及固定5000–10000调度重算。
源码变更经核查后可追加 --allow-source-changes；教师hash或模型配置变更不能用此开关绕过。

## 评估

默认5000轮起每1000轮评估，每次每任务总300回合，四任务共1200：
Throw单球300；Roll/Bounce各每球100；Basket单球300。固定验证种子20261117。
记录每球结果和任务宏平均，最终选择模型后应另外用未使用的新种子测试。

成功字段分别为 Throw success、Roll roll_eval_success、Bounce controlled_hold_v1 的success、Basket safe_basket_success。
Roll主成功率仍为接触后区域保留，不等同稳定握持。
Basket保留checkpoint的30%固定投球rehearsal采样分布，属于训练匹配验证；分别记录phase_groups.mobile和phase_groups.rehearsal，不能把混合结果直接解释成纯移动投框成功率。

评估入口（串行，1环境/当前物体，--num-envs不控制评估并行数）：

```bash
python -m distillation_four_task_future.run eval \
  --checkpoint distillation_four_task_future/runs/catch_basket_10000_seed123/student_best.pth \
  --output distillation_four_task_future/evaluations/best_seed20261117 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 --episodes 300 --eval-seed 20261201
```

## 检查

```bash
python -m unittest distillation_four_task_future.test_four_task
python -m distillation_four_task_future.run check --device cpu \
  --output /tmp/four_task_check_unique
python -m distillation_four_task_future.run env-check --device cpu \
  --gl-backend egl --egl-device-id 0 --check-steps 2 \
  --output /tmp/four_task_env_check_unique
```

check不创建物理环境；env-check逐一创建8个任务/物体环境，渲染真实深度，核对学生/教师接口并执行少量教师动作，不做梯度更新或成功率评估。

## 本地验证记录（2026-11-16）

- 四个真实教师的观测、推理和物理动作适配检查全部通过，最大推理误差0。
- 8项CPU合成测试覆盖双头/阶段屏蔽、无目标泄露、序列一致性、分任务/球种平衡、接触掩码、分段损失、Basket辅助损失关闭、保存续训及LR恢复。
- 八个条件均完成真实MuJoCo/EGL渲染和320策略步接口检查，合计2560步；四任务均验证终止指标存在，Basket阶段0和1均出现。
- 检查时执行教师动作以验证接口，不是学生成功率评估，未进行正式训练。
- 接触检测覆盖object刚体全部geom，兼容YCB视觉geom与独立解析碰撞球体。
- 训练日志包含各任务、各球种模仿损失及Basket导航/投球阶段帧数，便于观察数据是否覆盖两个阶段。

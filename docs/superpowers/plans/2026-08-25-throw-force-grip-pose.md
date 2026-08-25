# throw_force Grip Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `catch_it_copy` 的 `throw_force` 初始状态改为掌心朝下、手指朝前下方并夹住小球的可验证扔球准备姿态。

**Architecture:** 复用现有 XArm6 IK 和 `link6` 局部坐标定义。先由姿态检查脚本验证目标旋转矩阵和 FK 结果，再把 IK 结果写入 throw_force 专用配置；环境 reset/持球逻辑继续使用同一掌心法线，避免初始化和持球阶段出现坐标不一致。

**Tech Stack:** Python 3.8+, NumPy, SciPy Rotation, MuJoCo, 现有 `MJ_DCMM` IK。

---

### Task 1: 建立姿态检查入口

**Files:**
- Create: `catch_it_copy/verify_throw_force_pose.py`

- [ ] **Step 1: Write the failing check**

创建无 viewer 检查脚本，加载 `DcmmCfg.throw_force_arm_joints`，执行 FK，并断言 `link6` Y 轴的世界 Z 分量小于 `-0.5`、Z 轴的世界 Z 分量小于 `-0.2`。初始配置应因当前姿态不满足而失败，证明检查有效。

- [ ] **Step 2: Run the check and verify the expected failure**

Run from `catch_it_copy`:

```powershell
python verify_throw_force_pose.py
```

Expected: `AssertionError` 指出掌心法线或手指方向仍未朝下，而不是导入错误或模型加载错误。

- [ ] **Step 3: Keep the check focused**

检查脚本同时打印 FK 后的 `link6` 三轴、球心与掌心的向量及其距离，球心距离应接近 `throw_force_radius + 0.02`。不在脚本中修改模型状态或训练参数。

- [ ] **Step 4: Re-run after implementation**

实现后再次运行同一命令，Expected: `throw_force pose checks passed`。

### Task 2: 修正 IK 目标并更新机械臂初始姿态

**Files:**
- Modify: `catch_it_copy/compute_throw_force_ik.py`
- Modify: `catch_it_copy/configs/env/DcmmCfg.py`

- [ ] **Step 1: Change the IK target**

在 IK 辅助脚本中将目标轴定义为：

```python
palm_normal = np.array([0.0, 0.0, -1.0])
finger_dir = np.array([0.0, 0.8, -0.6])
```

保持轴正交化和 `R.from_matrix(...).as_quat()` 流程不变，并更新 docstring 与输出说明。

- [ ] **Step 2: Run IK and inspect reachability**

```powershell
python compute_throw_force_ik.py
```

Expected: 输出 `IK 成功`，并给出六个关节角；若严格目标不可达，则按设计中的降级策略仅降低 `finger_dir` 的前倾量后重试，不能直接恢复掌心朝上。

- [ ] **Step 3: Update the dedicated configuration**

将 IK 输出写入 `DcmmCfg.throw_force_arm_joints`，保持 `arm_joints` 和其他运动模式配置不变。

- [ ] **Step 4: Run the focused pose check**

```powershell
python verify_throw_force_pose.py
```

Expected: 轴向断言通过，球心位于掌心法线方向。

### Task 3: 调整初始手指夹球姿态与球位

**Files:**
- Modify: `catch_it_copy/gym_dcmm/envs/DcmmVecEnv.py`

- [ ] **Step 1: Preserve throw_force-only branching**

只修改 `self.object_motion == "throw_force"` 分支，保留三指 MCP/PIP/DIP 和拇指的专用预弯曲逻辑，不影响 `throw_basket`、`roll`、`bounce` 与普通 `throw`。

- [ ] **Step 2: Define one shared ball offset**

在 reset 和持球阶段使用相同表达式：

```python
palm_pos = ee_xpos + palm_normal * (ball_radius + 0.02)
```

其中 `palm_normal = ee_xmat[:, 1]`。若 FK 检查显示球仍未进入手指夹持区域，只调整 throw_force 专用手指弯曲值或间隙，不改公共坐标约定。

- [ ] **Step 3: Verify initial hand state**

在无 viewer 环境中 reset 一个 episode，确认 `qpos[21:37]` 使用 throw_force 专用值，且球位置计算与持球循环一致；不提前释放球。

### Task 4: 完成回归验证

**Files:**
- Verify: `catch_it_copy/verify_throw_force_pose.py`
- Verify: `catch_it_copy/compute_throw_force_ik.py`
- Verify: `catch_it_copy/gym_dcmm/envs/DcmmVecEnv.py`

- [ ] **Step 1: Run syntax/import checks**

```powershell
python -m py_compile compute_throw_force_ik.py verify_throw_force_pose.py gym_dcmm/envs/DcmmVecEnv.py configs/env/DcmmCfg.py
```

Expected: 命令成功退出且无 traceback。

- [ ] **Step 2: Run the pose/FK behavior check**

```powershell
python verify_throw_force_pose.py
```

Expected: 掌心法线朝下、手指方向朝下、球心偏移距离正确。

- [ ] **Step 3: Run one manual MuJoCo episode if available**

```powershell
python gym_dcmm/envs/DcmmVecEnv.py --viewer --object_motion throw_force
```

Expected: 初始画面中手背在上、掌心在下、手指向下夹球；持球阶段球不掉落，释放后向前飞出。若 Windows OpenGL/viewer 依赖不可用，记录错误并保留无 viewer 检查结果。
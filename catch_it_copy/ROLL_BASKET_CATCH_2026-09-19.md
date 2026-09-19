# Roll / Basket Catching 调整（2026-09-19）

## Roll

实际入口仍为 DcmmVecEnv_roll_645edc4.py。Tracking 单任务奖励不变；Catching 使用真实 16 维手关节，而非错误的 obs['hand']['joints']。

掌坐标系以 link6 局部 Y 为法向，球距掌平面投影中心不超过 0.09 m、法向位置在 [-0.02, 0.12] m 时鼓励闭合，否则鼓励留出入口。角度目标分别为 0.8 / 0.15 rad；姿态奖励为 2 exp(-平均角度误差平方/0.25)，掌内增加有限幅度的关节协调惩罚和接触包裹奖励。参数位于 configs/env/DcmmCfg_roll_645edc4.py 的 roll_grasp_*。

移除 Catching 跟踪阶段旧的轻弯/静止奖励以免与新时序冲突。阶段切换允许满足原有距离和朝向条件的手掌接触；任意手指碰撞不单独触发切换。成功、失败终止条件未放宽。info['roll_grasp'] 包含局部球位置、门控和奖励分项。

这些几何阈值是初始实验值，需要结合 viewer 验证掌心入口位置。没有冻结 PPO 跟踪分支；本轮仅调整环境奖励与阶段衔接。

## Basket

停车期间仍有脚本辅助持球；进入 preparing 后仍完全由物理接触驱动，不赋予发射速度。

取消动作幅度本身的正奖励；张手奖励乘以弹道速度质量，并只在接触时发放。加速项改为三维速度误差改善：w * clip(1 - ||v-v_ref|| / ||v_ref||, -1, 1)，仅持球且存在可行参考速度时有效。参考速度仍由可下降过框的弹道候选选取，包含竖直分量。过框成功和释放事件逻辑不变。

## 训练与验证

使用原有 Catching_TwoStage 命令和可用的 checkpoint_tracking，从新的 Catch 模型开始对照训练；原命令任务名、观测和动作维度不变。Basket 开启 basket_log=true 检查 contact、ball_velocity、reference_velocity、arm_joint_speed 与 IK 成功次数。

本地可执行纯奖励单元测试、Basket 回归测试和 Python 语法检查；这些不等价于长程训练或 viewer 效果验证。Bounce 专用代码未修改。

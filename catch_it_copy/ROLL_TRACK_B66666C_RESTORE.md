# Roll Tracking 恢复到 b66666c

来源：b66666cb9cf918d858cda2c8c0c84f789bb67f3e，2026-09-18 16:04:30 +0800。

仅 `object_motion=roll, task=Tracking` 使用历史环境、配置、机器人封装和等待目标辅助函数。复制历史代码，唯一修改是导入名称，避免与当前 Catching 共用可变配置。文件及校验值见 ROLL_TRACK_B66666C_MANIFEST.json。

历史版本包括：桌高 0.70 m、等待高度 0.50 m、桌外预测落点 XY 跟踪、link6 原点参考、线性截断掌心朝上奖励、跟踪时轻弯与手指速度惩罚，以及原版成功/终止、初始化、观测和动作控制。没有后来的平方朝向奖励、桌沿侧等待修正和掌中心接球点修正。

Roll PPO、共享控制工具和机器人资源在该提交至当前版本的相关检查中没有差异；训练入口继续使用 ppo_dcmm_645edc4。当前 Roll Catching 和 Basket/Bounce 不回退。Catching 内部的 tracking 阶段也仍使用当前 Catching 环境，不等同于独立 Tracking 任务。

训练：

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll
```

测试：

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 object_motion=roll checkpoint_tracking=/path/to/roll_track.pth viewer=true roll_log=true
```

启动标识应包含 `environment/config=b66666c`。建议新开训练，旧检查点评估必须记录其训练版本。

四个历史副本在逆转导入重命名后与 git 源码逐字节一致；已验证两个任务路由及 Python 语法。未执行 MuJoCo 实际训练，代码逻辑相同不保证不同种子、依赖版本或硬件下得到相同训练曲线。

# 当前 Roll Tracking：07c176f

来源提交：07c176fc374709e8ce2721f55e721342671760b3（2026-09-17 13:21:23 +0800）。

独立 task=Tracking、object_motion=roll 路由至 DcmmVecEnv_roll_track_07c176f.py。环境、配置、机器人封装、roll_waiting、roll_rewards 从历史提交复制，仅替换导入名称以隔离当前 Catching。来源校验值见 ROLL_TRACK_07C176F_MANIFEST.json。

保留该版初始化、观测、控制、奖励和终止逻辑：桌高 0.70 m，等待高度 0.50 m，桌外预测落点 XY 跟踪。没有 b66666c 新增的舀水手势奖励，也没有后续掌中心参考点和桌沿侧等待修正。Roll PPO 与该提交相同，继续共用 ppo_dcmm_645edc4。

当前 Roll Catching（含其内部 tracking 阶段）、Basket 和 Bounce 不变；b66666c 副本仅作历史保留，不再作为独立 Tracking 的入口。

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=roll
python3 train_DCMM.py test=True task=Tracking num_envs=1 object_motion=roll checkpoint_tracking=/path/to/model.pth viewer=true roll_log=true
```

启动应显示 environment/config=07c176f。建议从头训练进行版本对照。

源码检查：逆转导入名称后，五个副本与历史提交逐字节一致；验证两条任务路由、Python 语法。不代表已验证仿真收敛或不同依赖环境下的数值复现。

# Roll Track / Bounce 环境迁入主文件

实际环境代码现在均位于 `gym_dcmm/envs/DcmmVecEnv.py`：

| 任务 | 主文件中的类 | 行为来源 |
|---|---|---|
| Roll 独立 Tracking | RollTrackingEnv | 07c176f |
| Bounce Tracking / Catching | BounceEnv | 5fe75d5f |
| Basket / Throw | DcmmVecEnv | 当前实现 |

Roll Catch 仍使用当前 `DcmmVecEnv_roll_645edc4.py`，没有回退。`DcmmVecEnv.__new__` 选择主文件内的 RollTrackingEnv/BounceEnv 实例。两套历史类完整迁入，只重命名类及配置/机器人/辅助函数的引用，未重写奖励、观测、控制或终止条件。

以后修改环境逻辑，请在主文件中搜索对应 **class RollTrackingEnv** 或 **class BounceEnv**，不要修改最前面 DcmmVecEnv 类里未执行的同名模式分支。保留独立类是为了避免相同方法名、全局配置和历史逻辑互相污染。

配置、机器人封装、Roll 等待辅助函数及 PPO 仍使用此前历史文件；本次迁移的是环境实现，不是把所有依赖合成一个文件。旧独立环境文件保留作历史对照，标准训练、直接 DcmmVecEnv 构造及兼容入口均不再调用其中的 Roll Track/Bounce 类。

训练和测试命令不变。`roll_log=true` 或 `bounce_log=true` 会打印主文件中的实际类。Bounce 仍只接受 legacy 物理/发射配置；本次没有重新开启 P/L 参数组。

验证：迁入类与原文件逐方法 AST 对比（逆转名称别名），检查标准入口、Bounce 别名、参数透传、P/L 拒绝及 Roll Catch/Basket 路由；另运行 Basket 回归。未验证 MuJoCo 实际训练收敛。

# Bounce正常速度观看验证（2026-10-29）

使用新独立脚本预计算原始物理轨迹后正常速度回放。没有修改业务训练、奖励、物理工厂或默认dt；较粗dt的探测见DT_PROBE_REPORT.md，不能当原模式无损提速。

- 默认模型为Bounce Catch464.31；支持保存配置的legacy与controlled_hold_v1。
- 每2ms记录完整积分状态及qpos/qvel，终局与实际仿真逐位一致。回放从原compiled scene恢复状态，仅mj_forward，不进行第二次积分。
- 原始球seed0、1，网球seed0，棒球seed0、1、2的录制/加载验证通过；网球终局还与700局正式评估记录的同seed终局一致。
- 本机两秒GUI测量：119显示帧，59.44fps，仿真/墙钟比0.9916，正常exit0。窗口关闭后等待UI线程清理，无剩余线程。
- 3项单测通过：真实时间帧索引、逐位状态恢复且禁止mj_step、close后join线程的顺序。Python3.8语法检查通过，未在3.8运行时实测。

这是明确标识的轨迹回放，不是在线策略交互；预计算需要等待，但查看不会人为改变球的物理运动。

使用方法见 ../../catch_it_copy/BOUNCE_REALTIME_VIEWER_2026-10-29.md。

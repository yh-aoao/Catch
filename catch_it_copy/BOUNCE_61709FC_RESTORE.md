# Bounce 恢复到 61709fc

目标提交：`61709fc82bb4e48aa6a95fb722874ddf7ab483d0`。
在当前项目内隔离保存旧源码，直接提交这些文件到 GitHub，服务器拉取后按原命令训练。
不需要另一个 worktree，不需要服务器保留旧 Git 历史来动态提取文件。

## 执行入口

`train_DCMM.py` 的 `object_motion=bounce` 自动选择旧 PPO；
`DcmmVecEnv(...)` 的 Bounce 构造自动选择旧环境，Gym 注册入口及直接运行环境脚本同样适用。
启动训练时应看到：

```text
[bounce-baseline] commit=61709fc82bb4e48aa6a95fb722874ddf7ab483d0 environment/config/PPO=original
```

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 object_motion=bounce

python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=bounce \
  checkpoint_tracking=/绝对路径/本次旧版训练得到的_tracking.pth
```

务必提交新增文件，不能仅上传 train_DCMM.py 或 DcmmVecEnv.py。
为复现两阶段训练，建议先训练旧版 Tracking，再用对应 checkpoint 训练 Catching；
直接加载当前新版 Tracking 的模型，不等于复现当时的完整训练起点。

## 恢复范围与存放位置

- 环境：`gym_dcmm/envs/DcmmVecEnv_bounce_61709fc.py`。
- 物理、奖励、阈值、随机化配置：`configs/env/DcmmCfg_bounce_61709fc.py`。
- 机器人控制封装：`gym_dcmm/agents/MujocoDcmm_bounce_61709fc.py`。
- PPO、网络、归一化、经验缓存：`gym_dcmm/algs/ppo_dcmm_61709fc/`。
- 原始代码内容仅替换了私有配置和机器人模块的 import 路径，方法体未修改。
- 共用 IK、PID、工具、训练 YAML 与提交一致；共用机器人 XML 只有不影响语义的空白差异。
- Roll、Basket 继续使用当前环境、配置和 PPO。

因此恢复的不仅是接触部位，还包括初始化随机采样顺序、奖励、观测与动作、
阶段切换、严格抓取成功条件、failed_control 失败条件、时间限制及旧统计/保存逻辑。
当前 DcmmVecEnv.py 中保留的新版 Bounce 分支不再由标准 Bounce 构造入口执行。
调整此复现实验的参数需要修改旧版私有配置；修改普通 DcmmCfg.py 的 bounce_* 不会生效。

## 旧版行为保留

- Tracking 以手掌触碰设置 step_touch。
- Catching 成功需要手掌接触、连续 5 步速度不超过 0.05 m/s、XY 距离不超过 0.03m、
  MCP 平均闭合角大于 0.3rad，以及旧观测相对高度大于 0.05m。
- 连续手掌接触超过 20 步仍未成功，按旧代码 failed_control 结束。
- 旧 Catching PPO 以 truncates 统计成功，超时可能被统计为成功；这不是实际抓取率。
- 原来的最佳模型选择逻辑、旧渲染器创建行为也保留。
- P0–P8、L0–L4、gentle 等预设不属于该提交，现在传入会报错；删除这些覆盖参数即可。
- bounce_log=true 只额外打印版本身份，不增加后来修改的末步成功/失败诊断。

## 验证与边界

`BOUNCE_61709FC_MANIFEST.json` 记录目标提交、源码哈希、唯一允许的 import 替换，
以及共用依赖哈希。代码换行差异已归一化，XML 以规范化内容校验。
CPU 测试检查源码完整性、Bounce 路由（含别名和位置参数）、旧 PPO 选择、
预设拒绝，以及 Roll/Basket 仍走原入口。

本机没有完成 MuJoCo 训练验证。源码逻辑恢复不保证得到同一个 653.87 模型：
还需对应的 Tracking checkpoint、训练命令、种子、软件版本和硬件执行条件。

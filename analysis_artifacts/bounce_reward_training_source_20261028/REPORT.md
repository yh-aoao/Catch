# Bounce Catch 训练奖励来源核验（2026-10-28）

核验模型：`bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_288.46.pth`。

训练保存配置为 `task=Catching_TwoStage`、`object_motion=bounce`、`bounce_physics=physical_v2`、`bounce_launch=baseball_intercept_v1`。注册入口 `gym_dcmm/BouncePhysicalWorld-v2` 指向 `make_bounce_physical_v2`；factory 对 Catching 选择 `gym_dcmm.envs.DcmmVecEnv.BounceEnv`，对 Tracking 才选择 July22 环境。

factory 当前文件 SHA256 与训练记录完全一致：`baee27209d6390247d2348aecc058833d16c5f61211efc60974fefbf4b686a59`。因此 factory 对 Catching 选择 BounceEnv 的分支也与训练记录对应。

训练记录的 `gym_dcmm/envs/DcmmVecEnv.py` 整文件 SHA256：
`2ff609cf2f2525585c1604d9a5959a2d7243f54fa7ff4338b3cc4958e11090d9`。

已在 Git 提交 `278ca2b4d92489784edee3a55c1fa820181aa525` 中找到与该 SHA256 完全相同的文件，并保存本目录 `DcmmVecEnv.training_snapshot.py`。

将该训练源码与当前文件按 AST 定位 `BounceEnv` 后比较，整个类的源码文本完全相同，36个方法逐一相同（包含 `compute_reward`、`step`、`_get_obs`、`_get_hand_obs`、`_step_mujoco_simulation`）。`compute_reward` 文本 SHA256 为 `4ff85fa8d841af0b1c35f93266589160504c812297f34102e60c53d4da67fa26`，两者相同。

因此，上一轮发现的手指奖励读取错误、未进入最终总奖励的速度/朝向项，以及距离0.25m阶段切换，确实存在于该训练记录对应的 Bounce Catch 环境版本，不是检查了其他 Bounce 接口。整文件当前 SHA256 不同不影响上述类级别等同性结论。

边界：以上证明训练保存的代码指纹与该历史文件匹配、当前 BounceEnv 类相同；不能仅凭此证明全部外部依赖、运行库和训练机物理模型文件均相同。此次未修改业务代码。

机器可读证据：`../bounce_reward_training_source_audit_20261028.json`。

# 脱手后手指奖励 v2（2026-11-12）

## 修改范围与备份

唯一修改的训练代码是 gym_dcmm/utils/basket_fingers.py。配置、控制器、动作映射、观测、停车区域、阶段切换、释放确认、成功/失败条件及PPO更新没有修改。新的tests/test_basket_post_release_reward.py只用于验证。

修改前代码快照：/home/isee/Catch/catch_it_copy/backups/basket_before_postrelease_reward_20261112_115928/code_snapshot.tar.gz
修改前奖励原文件：/home/isee/Catch/catch_it_copy/backups/basket_before_postrelease_reward_20261112_115928/basket_fingers.py.original
原文件SHA256：cac76c3938881227e6348be0b4fa6ded25ae5bd04882fe6b564ad0e59e988dce
28.85模型权重未改动。

回退只恢复本次修改的奖励模块：
```bash
cd /home/isee/Catch/catch_it_copy
bash backups/basket_before_postrelease_reward_20261112_115928/ROLLBACK.sh
```

回退后重新启动测试/训练进程才会加载旧奖励。脚本根据自身目录定位项目，复制到服务器相同目录结构后也可使用。完整快照用于留档，日常回退只需要上面一条命令。

## 新奖励

原奖励各项计算完全保留，确认脱手前新增奖励为零。确认脱手后在0.08秒内平滑增加下列非正项，按实际仿真时间积分：

| 奖励项 | 内容 | 满额权重 |
|---|---|---:|
| finger_post_recovery | 逼近已定义的自然开手姿态 | 6 |
| finger_post_action | 12维动作均方，惩罚持续饱和指令 | 2 |
| finger_post_velocity | 测量关节速度超过3rad/s后逐渐惩罚，5rad/s缩放、平滑有界 | 1 |
| finger_post_sync | 三根长手指协同 | 4 |
| finger_post_coupling | 同指关节耦合 | 2 |
| finger_post_extension | 软限位和过伸 | 4 |

拇指保留独立同步语义，开手参考姿态与原配置一致。没有加入正奖励，没有修改动作或强制关节姿态。权重常量POST_RELEASE及版本post_release_v2记录在该奖励模块；info的finger_coordination内可查看版本。原checkpoint配置格式保持兼容。

这是共享Basket手指奖励模块的更新，其他使用该模块的Basket配置也会采用新增的脱手后奖励。输出目录应单独命名，以区分旧、新奖励曲线。

## 训练

为保持与上一轮的初始化和训练机制一致，继续从37.86固定抛模型初始化移动训练：
```bash
cd /home/isee/Catch/catch_it_copy
python3 train_DCMM.py --config-name basket_mobile_free_stop   test=False num_envs=32 vector_mode=async device_id=0   basket_throw_checkpoint=/home/isee/Catch/11-11/throw_basket_catch_best_reward_37.86.pth   checkpoint_catching='' basket_mixed_source='' basket_warm_start=''   viewer=false basket_log=false basket_self_collision_debug=false   train.ppo.max_agent_steps=1000000000000   output_name=BasketFreeStopPostReleaseV2
```

这是沿用原第二阶段入口，从同一固定抛技能学习移动，并非从28.85继续优化；本次没有添加新的移动checkpoint续训入口。服务器替换路径与GPU号，只需同步修改后的basket_fingers.py，且应先保留服务器上的原文件。没有自动启动训练。

新模型viewer测试用basket_mobile_free_stop及新权重，basket_mobile.fixed_throw_probability=0.0。原28.85仍可加载，但仅替换奖励不会让已经训练好的策略动作立即改变；新手型效果需要训练后评估。

## 验证

31项相关测试通过，包括确认脱手前奖励不变、持续极端动作被惩罚、稳定自然开手没有新增代价、快速关节运动代价、时间积分与消费一次、旧手指与环境奖励回归。

逐文件比对284个备份源文件，仅basket_fingers.py改变。旧奖励模块与新模块100步数值对比：所有原项完全相同；脱手前总奖励完全相同。28.85确定性推理三回合，原始球、网球、棒球的每步手指角度、目标角度和动作完全一致，终止原因及入框结果一致；新增惩罚分别约-3.38、-2.85、-3.28。该检查证明只有奖励改变，不证明新训练后的动作已经改善。没有执行优化器更新。

记录：/home/isee/Catch/analysis_artifacts/basket_post_reward_v2_20261112/comparison.json

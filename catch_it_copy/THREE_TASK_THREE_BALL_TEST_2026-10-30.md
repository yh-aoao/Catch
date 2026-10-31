# 三个任务、三种球的 Viewer 测试

先执行：

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
```

每次运行下面一条命令。

| 任务 | 原始球 | 网球 | 棒球 |
| --- | --- | --- | --- |
| Roll Catch | `bash scripts/test_roll_catch_252.sh original` | `bash scripts/test_roll_catch_252.sh tennis_ball` | `bash scripts/test_roll_catch_252.sh baseball` |
| Bounce Catch | `bash scripts/view_bounce_realtime.sh original` | `bash scripts/view_bounce_realtime.sh tennis_ball` | `bash scripts/view_bounce_realtime.sh baseball` |
| Basket | `bash scripts/test_basket_three_ball.sh original` | `bash scripts/test_basket_three_ball.sh tennis_ball 'basket_fixed_hand_flex=[1.0,0.9,0.9]'` | `bash scripts/test_basket_three_ball.sh baseball` |

固定使用的模型：

- Roll：`/home/isee/Catch/best_model/roll_catch_best_reward_252.54.pth`。
- Bounce：`outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_464.31.pth`，保存配置为 `physical_v2`、`legacy` 抓取、`baseball_intercept_v1` 发射。
- Basket：`outputs/BasketAimCarry/2026-10-29/01:23:05/nn/throw_basket_catch_best_reward_38.38.pth`，读取该次训练保存的配置。

Roll 使用目前的 `physical_v1` 球类材质与解析碰撞，初速范围为 0.8–1.3m/s；使用旧模型权重，奖励为 `legacy`，没有桌前等待的硬约束。脚本末尾可加 Hydra 覆盖，例如 `'roll_speed=[0.6,0.9]'`。

Bounce 为正常速度观看，先计算默认 3 局真实策略轨迹，再在 Viewer 按 1 倍速回放；保留原训练物理步长。这是预计算回放，鼠标扰动不会重新运行策略。可加 `--episodes 10 --seed 10` 多看一些回合；Space 暂停，左右方向键切换回合，Esc 退出。

Basket 原始球走原入口；网球、棒球走专用测试入口，使用不同尺寸、质量和球类物理先验，YCB 网格只作外观。沿用物理托球准备、策略和奖励，准备失败会报错。网球比原始球小，用原训练手姿 `[0.8,0.7,0.7]` 会在准备阶段滑落，所以网球命令显式改为更弯的初始手姿 `[1.0,0.9,0.9]`；原始球和棒球保持原训练手姿。此项只适配初始化，不改变模型权重，不对球施力。Basket 此模型训练只使用原始球，另外两球的表现属于换球测试，其中网球还改变了初始手姿，不能算严格沿用原训练条件的泛化成功率。

Roll 和 Basket 是在线策略测试。Basket 默认测试 30 局，末尾可加 `train.ppo.test_num_episodes=5`；Roll 可用 Ctrl+C 结束。

## 运行检查

Roll 三球已严格加载 252.54 权重与归一化，并分别执行两个策略步。Bounce 464.31 的棒球已完成一整局预计算和 441 帧精确状态恢复，另外两球沿用之前已验证的同一回放入口。

Basket 使用以上脚本、真实 38.38 模型和 `viewer=false`，原始球、棒球各完成一局，显式握姿的网球完成三局，均正常退出。另做网球 seed0–4、棒球 seed0–2 的物理初始化和短步检查，均为自然接触托持，球体驱动力为零。两个针对碰撞体、惯量与配置隔离的单元测试通过。这些是入口运行检查，不是新的成功率统计。

随后按网球命令的默认 30 局做了连续运行，全部完成，没有初始化中断；本次得到 0/30 入篮。这说明初始化适配只让测试可进行，原始球策略尚未适应网球。该批次沿原 PPO 测试的连续随机序列运行，不能和此前逐局固定种子的原始球正式评估直接比较。日志：`/home/isee/Catch/analysis_artifacts/basket_ycb_tennis_cli_30_20261030.log`。

初始化检查记录：`/home/isee/Catch/analysis_artifacts/basket_ycb_initialization_20261030.json`。

# 最新 Bounce、Roll、Basket 模型评估（2026-10-29）

指定模型每个条件100局、独立种子0–99，共700局。CPU、无viewer，确定性均值动作、保存的归一化、原始动作适配，模型权重严格加载。Bounce/Roll固定所选球的标称物理参数，无训练mixed模式的额外质量/半径抖动。

## 主要结果

| 任务 | 球类/设置 | 成功/100 | 95% Wilson区间 |
|---|---|---:|---:|
| Roll Track 离桌拦截 | 原始球 | 82 | 73.3%–88.3% |
| Roll Track 离桌拦截 | 网球 | 81 | 72.2%–87.5% |
| Roll Track 离桌拦截 | 棒球 | 84 | 75.6%–89.9% |
| Bounce Catch 旧版proximity | 原始球 | 38 | 29.1%–47.8% |
| Bounce Catch 旧版proximity | 网球 | 44 | 34.7%–53.8% |
| Bounce Catch 旧版proximity | 棒球 | 40 | 30.9%–49.8% |
| Basket 投篮 | 固定近目标、训练球 | 97 | 91.5%–99.0% |


## 指标边界

Roll主指标为PPO使用的roll_eval_success：球完全离开桌子前缘、无桌面/地面支撑后，手部曾完成拦截。不要求稳定抓住，且这个事件一旦发生后续失败也可能保留为真。原始info.success是另一套判定，三球分别为64/100、54/100、65/100；不是主指标。Track接触后常结束，不能用接触时长为0认定它的Catch能力。

Bounce模型464.31的保存配置没有controlled_hold_v1，实际评估legacy旧PPO/旧BounceEnv。主指标为bounce_catch_proximity_v2：到达2.5秒时限、未失败、处于grasping且末端距球<0.25m；不是严格受控离地保持。

| Bounce球类 | 曾接触手 | 任意连续手接触≥0.3s | 主成功且连续接触≥0.3s | ball_left失败 |
|---|---:|---:|---:|---:|
| 原始球 | 90 | 45 | 35 | 56 |
| 网球 | 86 | 39 | 37 | 45 |
| 棒球 | 90 | 44 | 38 | 60 |


辅助接触指标来自2ms控制tick，不验证法向力、离地、多指受控或关节正常，也不是每125us物理子步连续测量；不可当实机稳定抓握率。

Basket主结果为真实自由飞行穿过篮圈、再经过0.3秒安全跟随的basket_score。篮圈中心[0,1.05,0.55]m、半径0.20m、倾角25°，使用训练时固定底盘/物理准备/固定手指设置，没有额外评估辅助投球或粘球。100次确认出手，97次真实命中与安全入篮，危险机器人碰撞0；失败为not_a_throw1、ball_on_floor2。此结果不代表随机远目标泛化。

Carry前移6cm仅是奖励目标，并非该版成功硬门槛。100局达到过6cm的次数为0；最后前移中位约1.29cm、最大约3.21cm。因此97%不代表完整Carry动作达标。详见[basket报告](basket/REPORT.md)。

## 与上次指定模型对照

两次Bounce/Roll保存的配置和物理provenance逐字一致，当前物理/环境路径与训练历史一致，推理loop相同。各球100局的初始状态逐局比较见previous_model_comparison.json。下表为同条件描述性对照，不对小幅差异作显著性承诺。

| 任务/球类 | 上次模型 | 本次模型 | 变化百分点 |
|---|---:|---:|---:|
| Roll 原始球 | 65% | 82% | +17 |
| Roll 网球 | 53% | 81% | +28 |
| Roll 棒球 | 63% | 84% | +21 |
| Bounce 原始球 | 24% | 38% | +14 |
| Bounce 网球 | 34% | 44% | +10 |
| Bounce 棒球 | 8% | 40% | +32 |


上次为Roll23.86、Bounce288.46，本次为Roll27.44、Bounce464.31。两项均提升，但Bounce主指标仍仅约四成，持续控制依然需要改善。

## 来源与复现

[source_audit.json](source_audit.json)记录训练版本核对：BounceEnv、RollTrackingEnv及相关PPO/配置/机器人/物理组件与训练历史匹配，Bounce工厂变化仅为新版opt-in接线，默认legacy行为保留。Basket训练目录没有源码指纹，代表当前本地对应保存配置的评估，无法证明全部历史依赖逐字相同。

每个正式子目录保存provenance.json、episodes.jsonl、summary.json、COMPLETE。Basket首轮连续随机流预检96%保存在basket_stream_seeded_precheck，已排除；正式basket为逐局同时重设NumPy/Python/Torch/Gym种子的97%。basket_startup_failed也不计入700局。

使用的模型：
- `/home/isee/Catch/catch_it_copy/outputs/roll_three_ball_tracking_legacy_reward_v1_seed123/2026-10-28/11:36:06/nn/roll_track_best_reward_27.44.pth`，SHA256 `ea8d3dd09238f2c97a605ef495402b2ce37038a07e8bb72fe0a7fabf06759add`。
- `/home/isee/Catch/catch_it_copy/outputs/bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32/nn/bounce_catch_best_reward_464.31.pth`，SHA256 `a2129531e56a326238d5129d5d6dbb383936bf97dc27f404c4bc6fedd49ae2ed`。
- `/home/isee/Catch/catch_it_copy/outputs/BasketAimCarry/2026-10-29/01:23:05/nn/throw_basket_catch_best_reward_38.38.pth`，SHA256 `6bf1a3f9b3d1fca09d0d697a0d06f53ca6f7c99a40d67d6d53e10416fc3094ce`。

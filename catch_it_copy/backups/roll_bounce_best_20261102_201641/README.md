# Roll / Bounce 当前最佳基线备份（2026-11-02）

用户指定：当前这一版效果最好，作为后续修改前的可回退基线。此称谓是用户选定基线，不代表所有指标均超过所有历史模型。

备份位置：`/home/isee/Catch/catch_it_copy/backups/roll_bounce_best_20261102_201641`。

## 备份内容

- `snapshot/models/`：Roll Catch250.65、Bounce Catch-0.73及本轮Track30.07、20.04四个原始权重，逐字节复制。
- `snapshot/project/`：完整gym_dcmm、configs、ycb_objects、scripts、tests，以及assets（XML、机器人网格、YCB网格/纹理）、顶层Python和相关文档。无运行目录/历史大备份递归。
- `snapshot/evaluation/`：本轮600局评估、60局手部诊断、源哈希、逐回合数据和与旧252的对照。
- `MANIFEST.json`：全部snapshot文件的大小与SHA256、当前Git版本与状态。
- `CHECKPOINTS.json`：权重字段和可直接读出的训练元数据。
- `ROLL_REWARD_AND_CONTROL_CONSTANTS.json`、`BOUNCE_GRASP_EFFECTIVE.json`：当前对应配置常量及展开的Bounce奖励/控制默认值。
- `*_training_recipe_reconstructed.yaml`：按已给训练命令重建；Bounce优化调度以checkpoint实际值覆盖。不是服务器原始resolved_config.yaml。
- `RUNTIME.json`、`DEPENDENCIES.txt`：当前验证环境版本。

## 可确认信息与缺口

精确保存模型文件及已测试的本地代码/资产，足以恢复本次已验证行为。服务器没有一并提供原始resolved_config.yaml、训练日志和源码哈希，因此不能证明服务器历史源码/所有奖励覆盖值与本地完全相同。Roll权重只含模型与归一化参数，没有训练步数/优化器。Bounce保存受控保持版本、优化器、调度、随机数状态，步数751616。

**Bounce实际训练方式以权重为准**：`freeze_track_steps=0`、`track_lr_scale=1.0`、`freeze_track_normalization=false`。即从Catch开始联合更新，不是脚本默认的冻结50万步+0.1倍率。先前按默认脚本叙述不适用于这份权重。评估配置中的默认优化调度不会改变固定权重的确定性推理结果。

总训练步数上限历史实际值未知（曾给2500万及10亿两种命令）；权重文件名中的奖励不是成功率。三球均分及Bounce四个手指奖励覆盖按会话训练命令记录，checkpoint不保存完整奖励覆盖，需服务器原配置才能进一步认证。

## 实际代码入口

- Roll Catch：`gym_dcmm/envs/roll_physical.py` -> `gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py`，配置`configs/env/DcmmCfg_roll_645edc4.py`。不要误读主DcmmVecEnv.py里的RollTrackingEnv作为Catch奖励。
- Roll PPO：`gym_dcmm/algs/ppo_dcmm_645edc4/ppo_dcmm_catch_two_stage.py`与models_catch.py。
- Bounce Catch：`bounce_physical_v2.py` -> `bounce_grasp.py`，继承主环境BounceEnv并替换Catch奖励/控制/成功判定；PPO为`ppo_dcmm_5fe75d5f/ppo_dcmm_catch_hold.py`。

## Roll奖励与训练逻辑

加载Track30.07的actor、动作分布参数及观测归一化，Catch分支新初始化；默认联合更新Track、Catch、critic，不存在Bounce专用冻结调度。并行16环境，seed123，每回合三球等概率重采样，有限批次并非严格等量。physical_v1，解析球、各自质量半径惯量及桌面摩擦/滚阻，速度0.8–1.3m/s；roll_reward_version=legacy、roll_wait_guard=false，没有启用adaptive_wait_v1。

注意legacy是配置枚举名，不等于无任何历史等待/预测项；当前Catch源码仍有历史roll_wait_target/roll_wait_shaping、桌边净空及已有grip_v1。这一版全部原样冻结保存，后续不要依名称猜测奖励内容。

Catch接触前：位置/预测目标靠近与高度/桌边项 + roll_scoop与手部反馈 - 手动作控制代价 + 碰撞/机械臂约束代价 + 触球项。
接触后：上述位置与手部反馈 + `10*exp(-50*d²)`精度 + 常数朝向1 - 底盘/机械臂控制代价 + 碰撞/约束/触球 + `2*exp(-(相对速度/0.15)²)`（真实手接触且离桌离地时）。Roll不使用旧Bounce那段无界手指闭合奖励。

roll_scoop包含掌心朝上非线性与改善增量；grip_v1包含接触前姿态/协调、capturing/holding分阶段、手部速度/加速度正则、目标增量平滑、多指真实接触包球、保持时长进展和滑落惩罚。接触后撤去预设关节姿态项，允许按球形状包球。保持进展上限0.8秒、权重8；包球权重2；丢失代价3。完整实现与全部常量以快照为准，不能仅靠此摘要重写。

Roll主要评估成功=region_retention_v5：真实触手启动、已离桌且无桌/地支撑、结束时在世界竖直掌心区域且无失败；没有最短时长/低速要求。stable_hold另计0.3秒、相对速度<=0.15m/s，不能混同。日志legacy_success另保留旧判据。

## Bounce奖励与训练逻辑

由Track20.04初始化，controlled_hold_v1；physical_v2 + baseball_intercept_v1（只针对棒球特殊发射），三球1:1:1。按checkpoint实际：Track/Catch从开始联合训练，同倍率学习率并更新Track归一化。

奖励`gamma*Phi_next-Phi_previous`，终止Phi=0。Phi含位置2、精度4、掌面1、速度1、准备姿态0.3、真实接触2、包球2、保持进展8；成功+25、失败-25、每步-0.01，以及动作幅值/平滑、关节越限、底盘/臂控制、碰撞等代价。这里是替换旧proximity Catch奖励，不叠加旧的触球后随时间无限增长奖励。

会话中的新增finger_coordination_v2覆盖：反折代价0.15、接触前协调代价0.05、普通手指覆盖势能1.0、拇指对向参与势能0.5。前两项近球加权、协调项随接触减弱；拇指不强制与普通三指同角。详细参数见BOUNCE_GRASP_EFFECTIVE.json。

控制：合法关节目标限幅、每步目标增量上限0.08rad、目标跟踪误差上限0.35rad、目标内侧余量0.02rad、PID扭矩上限1Nm（并遵守XML）、积分限幅及关节限位。不是通过改写物理qpos把手强行摆正。

成功需连续0.3秒有效捕获、多接触组>=2、接触分布>=0.15、相对速度<=0.35m/s、离地且无外部支撑、实际关节越限<=0.08rad，结束时仍有效且无失败；成功立即终止。严重越限>0.25rad失败。不是旧464的靠近成功口径，不证明0.3秒以后的长期持球。

## 本次测试结果

每格100局，种子0–99：

|球|Roll区域保持|Roll曾稳定保持|Bounce受控保持|
|---|---:|---:|---:|
|原始球|69%|21%|61%|
|网球|73%|23%|62%|
|棒球|85%|27%|74%|

Roll存在接触后滑落、稳定性不足及小样本0.20–0.23rad关节越限；Bounce主要是接触后包球/相对速度，补查全程越限约0.03rad。60局只读手部诊断与主评估对应轨迹完全相同。完整报告在snapshot/evaluation/REPORT.md。

## 校验与恢复

仅校验（不修改项目）：

```bash
python "/home/isee/Catch/catch_it_copy/backups/roll_bounce_best_20261102_201641/verify_restore.py"
```

恢复到全新目录（目标存在会拒绝，保留当前工作区）：

```bash
python "/home/isee/Catch/catch_it_copy/backups/roll_bounce_best_20261102_201641/verify_restore.py" --restore-to /home/isee/Catch/roll_bounce_baseline_restored_20261102
```

新目录中baseline_models是四个权重，baseline_evaluation保存重建测试配置。激活dcmm后从新目录运行脚本，并显式提供baseline_models里的模型路径。训练/测试脚本自身按所在目录寻找物理资产。

Roll继续训练只能恢复策略/归一化，不是精确优化器续训；旧权重无优化器/步数/RNG。Bounce checkpoint含优化器、步数、调度及RNG，可按PPO_Catch_Hold恢复；务必显式保留实际调度0/1.0/false，并使用相同环境/奖励配置。并行环境本身状态未保存，不保证位级连续轨迹。

重新从Track训练Bounce时，在equal-v2命令后添加：

```bash
bounce_grasp_training.freeze_track_steps=0 \
  bounce_grasp_training.track_lr_scale=1.0 \
  bounce_grasp_training.freeze_track_normalization=false
```

后续修改使用新输出名称，勿覆盖这四个基线权重。snapshot是本地大文件备份，通过本目录.gitignore排除Git，避免再次触发大文件推送限制；跨机器保存时需另行复制整个备份目录。当前只有本机备份，不代表已上传远端。

## 备份验证结果

全部644个snapshot文件SHA256校验通过；实际恢复到独立/tmp目录，两模型分别在恢复代码/资产上运行原始球种子0完整回合，均正常结束，initial/final/reward/success与主测试逐项完全一致。没有改动当前训练代码。TRAIN_COMMANDS.sh记录从Track重新训练的命令，包含Bounce实际联合训练覆盖。

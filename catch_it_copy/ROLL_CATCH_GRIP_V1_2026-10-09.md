# Roll Catch 抓取实验 v1（2026-10-09）

## 基线与回退

修改前 HEAD：3b8815f8ca9f7ff47f191f386f45acc9324cfe4e。已保存字节级备份及 SHA256：backups/roll_catch_before_grip_v1_20261009_163717/manifest.json。

保存了 Catch 环境、配置、roll_grasp、评估模块与原测试。restore.py 默认恢复本次改动的环境、配置、grasp 和测试；不恢复只读评估，避免覆盖后续评估修正。恢复前自动把当前实验文件复制到 replaced_时间戳 子目录，校验备份后再覆盖。不会操作 Git 历史。

在 catch_it_copy 目录回退：

```bash
python3 backups/roll_catch_before_grip_v1_20261009_163717/restore.py
```

模型文件未覆盖，旧模型继续保留；Git 提交与推送尚未执行。

## 本次变化

仅 Roll Catch 奖励和手指目标过滤：

- 实际手部接触后，至少 0.25 秒快速捕获窗口，即使球速很低也不立即降低闭合速度；最近接触中断容忍 0.20 秒。窗口后且相对速度 <=0.25 m/s 转为 holding，使用原低速平滑。
- 捕获期目标速率上限 4 rad/s、平滑系数 0.95（原捕获 3 rad/s、0.8）；目标变化及实际手速/加速度正则乘 0.25。没有脚本闭合指令，动作符号与大小仍由策略输出。
- 接触后撤销固定关节角度目标与链角度协同塑形，避免任意角度压过真实控球。保留等待时原开口姿态塑形。
- 按掌部根 body 下的不同手指运动学分支统计接触，不按碰撞 mesh 数量刷分。两支接触达到该项最大值，不要求所有手指接触；乘相对速度质量。该项最大每策略步 2。
- 连续有效手部接触且球距离掌部参考点 <= 原 roll_drop_distance=0.25 m 时，给持球时长历史最好改善奖励：8/秒、最多 0.8 秒，共 6.4。反复接触不重置奖励额度。
- 有效承接后连续失去有效接触 0.20 秒，一轮最多扣一次 3 分。它是奖励事件，不改变现有掉落终止。
- 终端新增 [roll-grip]：phase、finger_contacts、持球信用状态、原始/过滤动作、实际手指速度与关节目标误差。

本次奖励持球条件使用实际接触与原距离约束，不引用尚需校准的掌面法向门槛。手指分组依赖 palm geom 对应根 body，首次实机/仿真运行需检查 finger_contacts 是否合理；若始终为 0，先审计模型树，不提高权重。

Roll Track、成功率评估、回合时限、初始化和底座/机械臂控制保持原样。Catch 模型中的臂/底座是否学习沿用原设置，没有新增冻结。新旧总 reward 定义不同，不直接比较数值。

## 推荐微调

在 catch_it_copy 目录，使用现有较好 Catch 模型加载完整分支及归一化，降低学习率，输出新目录：

```bash
python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 object_motion=roll checkpoint_catching=best_model/roll_catch_best_reward_252.54.pth checkpoint_tracking='' train.ppo.learning_rate=0.0001 train.ppo.lr_schedule=fixed output_name=RollCatchGripV1 viewer=false imshow_cam=false roll_log=false
```

这是从模型与归一化初始化的 PPO 微调，不是恢复旧优化器、训练步数的完全续训。若有另一份效果更好的 Catch，替换路径。无需重新训练 Track。

测试新模型（替换实际路径）：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=roll checkpoint_catching=/absolute/path/new_model.pth checkpoint_tracking='' viewer=true imshow_cam=false roll_log=true
```

关注同一组发射/种子下主成功率、结束仍持球、接触后滑落率，以及 [roll-grip] 中实际手指接触与指令执行。仅看手指弯曲角度或 best reward 不足以证明改善。旧模型在新过滤下也可能立即改变动作，应以微调后闭环结果对照。

## 验证

9 项 grasp 测试、13 项只读评估测试通过；语法及 diff 检查通过。覆盖早期低速接触仍允许快速捕获、多 mesh 不重复算手指、持球信用上限与滑落一次惩罚、原评估行为。

本机此前 MuJoCo DLL 导入不可用，未运行完整训练或 viewer。本版是可回退的实验，未声称闭环效果已提高。

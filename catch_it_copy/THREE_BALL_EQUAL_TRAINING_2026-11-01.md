# Roll / Bounce三球等概率训练与手指问题处理

## 视频与状态证据

已查看`/home/isee/视频/录屏/bounce1.webm`、`bounce2.webm`和两段Roll录屏。Bounce录屏可见指节向下反折，球接触后也有脱离/弹走回合。单靠画面不能判定每个关节是否越限；同时找到录屏时间附近的精确回放缓存，确认其模型为464.31、抓取版本legacy。

读取网球/棒球各三局缓存的真实qpos与XML关节范围：网球第二指远端最大越限3.269rad，棒球第一指远端最大越限3.854rad。这是旧控制下实际关节越限，不能只靠奖励三指同角度来解决。分析帧和逐关节检查见`/home/isee/Catch/analysis_artifacts/ball_training_revision_20261101/`。

## 本轮调整

- 三种球使用权重`{original:1,tennis_ball:1,baseball:1}`，归一化后每种1/3概率。每回合reset重采样，有限批次不是严格等量；这是等概率混合训练。
- Roll保留现有physical_v1、legacy奖励、关闭硬等待约束、0.8–1.3m/s初速。仅新入口采样比例和输出名称变化，既有Roll环境/奖励源码未改。
- Bounce Track保留原跟踪/触球奖励，只改采样。它输出底盘与机械臂动作，手指专项学习在第二阶段Catch。
- Bounce Catch沿用此前controlled_hold_v1的手目标限幅、手部PID/扭矩限幅和真实保持判据，显式启用新奖励配置`finger_coordination_v2`。没有改球材质、发射、成功阈值或引入球体助力。

新的手指项：

1. **反折代价，权重0.15。** 三根普通手指指尖相对于各自根部，若落到掌面+Z的背侧超过1cm，则按超出量给有上限的代价；只在球靠近时明显生效。使用掌心局部坐标，不按世界向上/向下判断。手臂翻转不会改变判定。拇指不包含在这个三指代价中。
2. **接触前协调代价，权重0.05。** 三指对应关节相对中位数偏差超过0.35rad才惩罚，接触比例增加时逐渐撤销。允许不对称包球，不强制三指角度相等。
3. **真实普通手指接触覆盖，势能权重1.0。** 用不同手指分组计数，两个普通手指接触达到该项上限，不按同一手指的碰撞网格数量刷分，不要求三指全接触才成功。
4. **拇指对向参与，势能权重0.5。** 在捕获区内，拇指与至少一根普通手指同时有真实有效接触才获得分数，不强迫拇指复制三指关节角。

接触覆盖与拇指分数通过原有gamma*Phi_next-Phi_previous发放；反折和准备协调是直接小额代价。原位置/速度/掌面接近、真实接触、包球/保持、成功+25、失败-25以及动作/越限代价继续保留。四个新权重默认都为0，只有新等概率Catch入口显式打开，所以旧controlled_hold_v1入口可用作无新增手指项的对照。

控制与奖励联合处理是新训练候选。短检查可以验证约束和训练链路，不能证明正式训练后成功率已经提高。

## 先训练Track

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
```

Roll：

```bash
bash scripts/train_roll_three_ball_equal_v2.sh device_id=0 num_envs=16
```

Bounce：

```bash
bash scripts/train_bounce_three_ball_equal_v2.sh device_id=0 num_envs=16
```

两条命令默认从头训练Track，seed123、2500万环境步，不加载旧Catch权重。默认输出：

- `outputs/roll_three_ball_tracking_equal_v2_seed123/日期/时间/`
- `outputs/bounce_three_ball_tracking_equal_v2_seed123/日期/时间/`

每条命令是一个独立训练进程。GPU编号和环境数可按机器覆盖。

## 新Track训练完再进入Catch

请将下面示例中的路径换成**本轮新Track的实际权重**。

```bash
bash scripts/train_roll_three_ball_equal_v2.sh \
  task=Catching_TwoStage \
  checkpoint_tracking=/本轮Roll训练目录/nn/实际roll_track模型.pth \
  device_id=0 num_envs=16
```

```bash
bash scripts/train_bounce_three_ball_equal_v2.sh \
  task=Catching_TwoStage \
  checkpoint_tracking=/本轮Bounce训练目录/nn/实际bounce_track模型.pth \
  device_id=0 num_envs=16
```

新的Bounce入口要求显式提供Track（或Catch续训权重），避免不小心沿用旧6:2:2的19.31 Track。Catch输出为`bounce_three_ball_catching_equal_posture_v2_seed123`。仍保留前50万环境步冻结Track、之后0.1倍Track学习率、固定Track归一化。成功率使用连续0.3s真实受控保持，应与旧464.31的proximity指标分别记录。

## 回退与对照

原6:2:2训练脚本未修改；原始模型也未覆盖。编辑前源码与SHA256备份：
`backups/before_equal_posture_20261101_174513/manifest.json`。

若要保留三球均等，但只使用原controlled_hold_v1奖励：

```bash
bash scripts/train_bounce_three_ball_hold_v1.sh \
  checkpoint_tracking=/本轮Bounce训练目录/nn/实际bounce_track模型.pth \
  'bounce_ball_weights={original:1,tennis_ball:1,baseball:1}' \
  output_name=bounce_equal_hold_base_ablation_seed123
```

## 验证

- 新增手指数学/FK/三球运行测试5项通过；原controlled-hold回归15项、PPO冻结/恢复测试5项通过，共25项。
- 四条Track/Catch配置入口解析通过，均为1:1:1权重，Bounce Catch明确保存四个新权重。
- 实际完成Roll Track、Bounce Track、以及由该短Bounce Track初始化的新Catch各64环境步的短PPO训练，均正常结束并保存有限值权重。实际核对Catch加载的Track actor和冻结归一化与短Track权重逐项完全一致。
- 这是训练链路与约束检查；未启动2500万步正式训练，没有新版抓取成功率结论。
- 检查记录在`analysis_artifacts/ball_training_revision_20261101/`。训练配置和provenance记录reward_profile、具体权重及源码哈希。

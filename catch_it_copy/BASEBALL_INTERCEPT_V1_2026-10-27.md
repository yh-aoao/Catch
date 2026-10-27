# 棒球独立发射配置

启用：`bounce_physics=physical_v2 bounce_launch=baseball_intercept_v1`。
只在当前球为 baseball 时生效，包括 mixed 训练；网球和原始球的发射条件保持原样。
默认 bounce_launch 仍为 legacy，去掉该覆盖即可回退。

参数：球心释放高度 1.2–1.4 m，初始世界 y=1.5–1.8 m，水平速度 1.3–1.7 m/s，
初始竖直速度 -3.0 至 -2.5 m/s。其余方向和自旋仍来自 bounce_v2。
这是一次性初始条件，没有持续补速或向上力；球—地面材质与奖励函数均不改变。
棒球采用文献近似 e=0.5 的材质文件仍需显式传入，不会通过发射开关自动选择材质。
这些是命名预设固定范围；通用 bounce_v2.height/distance/speed/vertical_speed 不覆盖
该预设中的四个范围。需要恢复通用可调范围时选择 bounce_launch=legacy。

## 验证

固定文献材质，10 个种子各比较原发射与新发射，并以半物理步长作参考。
独立球—地面仿真（不含机器人碰撞）：新配置首跳球心峰值 0.418–0.479 m，
峰值世界 y=0.623–1.026 m，平均峰值时刻约 0.591 s。
与半步长比较的最大首跳峰值差约 0.00693 m。
这是轨迹与数值检查，不是策略成功率或完整动态可达性证明。
记录：`/home/isee/Catch/analysis_artifacts/baseball_intercept_v1_validation_20261027.json`。
Track/Catch 材质不变、棒球单独切换、网球同种子轨迹初始条件不变已纳入回归。
主测试入口也已加载旧 checkpoint 并完成短步进，尚未重训。

## Viewer 测试

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
python train_DCMM.py \
  test=True task=Catching_TwoStage num_envs=1 \
  object_motion=bounce bounce_physics=physical_v2 \
  bounce_launch=baseball_intercept_v1 \
  test_object=ycb:baseball ycb_collision=sphere \
  bounce_material_file=/home/isee/Catch/catch_it_copy/ycb_objects/calibration_data/balls.hard_floor_literature_v1.json \
  object_eval=false viewer=true \
  checkpoint_catching=/home/isee/Catch/best_model/bounce_catch_best_reward_653.87.pth
```

日志出现 `[bounce-launch] mode=baseball_intercept_v1` 表示该预设已被选择。
旧模型不一定能适应新的发射轨迹，抓取能力需要单独评估和重训。

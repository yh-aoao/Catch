# 水果释放位置后移（2026-11-04完成）

原因：旧y=1.25–1.45m过近，机器人尚未到位就离桌。现改为y=2.0–2.2m，距y=0.8m桌边约1.2–1.4m。为减少长距离停滚，同时按水果设置初始速度；材质、几何、摩擦、滚阻和奖励不变，释放后不施持续力。

|水果|默认初速|2.5秒内自然离桌/100|离桌时间范围|
|---|---|---:|---|
|orange|1.1–1.4m/s|100|1.11–2.11s|
|apple|1.6–1.9m/s|95|0.88–1.68s|
|lemon|1.4–1.7m/s|95|0.97–2.49s|

被动运动检查，不是抓取成功率；个别不规则姿态仍会停滚或未在2.5秒内完全离桌。没有过滤后重抽样，全部100种子计入分母。原近释放参数对应的历史验证保留在object_benchmark_20261103，不能混作当前结果。

view_object_benchmark.sh和train_object_benchmark.sh现传roll_speed=null，使用每目标默认速度；显式命令行roll_speed仍优先。旧版训练若恢复保存的配置并写死roll_speed，会继续覆盖新默认，需按实际实验明确设置。

Bounce所有目标、圆柱罐参数逐项对比保持不变。catalog_before.json为修改前备份，catalog_after.json为当前配置。新增目标运行检查覆盖远释放、每目标速度与显式速度覆盖。

重新启动原viewer命令即可：

```bash
conda activate dcmm
cd /home/isee/Catch/catch_it_copy
bash scripts/view_object_benchmark.sh roll orange
bash scripts/view_object_benchmark.sh roll apple
bash scripts/view_object_benchmark.sh roll lemon
```

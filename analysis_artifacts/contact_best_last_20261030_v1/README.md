# Contact best / last 单种子对比

按用户要求，比较根目录的两份Contact权重：`student_best.pth` 为6000轮，`student_last.pth` 为10000轮。两份配置及284项源码哈希清单完全相同。

正式协议：环境评估种子20261029，Throw、Roll、Bounce每项每模型300回合，共1800回合；策略推理batch=1。逐回合保存初始物理/深度哈希、成败及原始任务诊断。最终结果生成在 `report/REPORT.md`。

此前的五组种子任务已按用户要求停止，数据保存在 `../contact_multiseed_20261030_v1/`，不计入本次1800回合。

文件说明：

- `source_audit.json`：两份检查点SHA256、训练轮次、历史源码逐文件哈希。
- `PLAN.json`：单种子比较计划。
- `formal/{best,last}_{throw,roll,bounce}/`：正式逐回合记录和协议。
- `evaluation_code/`：实际使用的脚本快照。
- `EVALUATION_COMPLETE`：仅在全部六项任务成功完成后写入。

源码恢复依赖此前保存的 `../distill_best_multiseed_20261029_v1/historical_sources_and_assets.tar.gz`；恢复脚本为 `/home/isee/Catch/analysis/prepare_contact_eval_20261030.py`，历史环境目录为 `/tmp/catch_contact_eval_sources_20261030/contact`。恢复过程逐项核验284个清单文件及57个额外历史资源，当前项目训练代码不作为环境来源。

运行评估（dcmm环境、GPU 0 / EGL 0）：

```bash
cd /home/isee/Catch
/home/isee/miniconda3/envs/dcmm/bin/python analysis/run_contact_best_last_20261030.py
```

运行报告（无需GPU，要求全部正式回合完成）：

```bash
cd /home/isee/Catch
MPLCONFIGDIR=/tmp/catch-contact-report-mpl \
  /home/isee/miniconda3/envs/dcmm/bin/python analysis/report_contact_best_last_20261030.py
```

原训练best的选择指标是三个任务中最低成功率，因此报告同时给出三任务平均和最差任务率。Roll的主指标是区域保持，稳定抓持另列。本次单种子结果用于比较这两份具体权重，不代表跨环境种子或跨训练种子的普遍排名。

# Belief / Future 一万轮模型重复评估

最终结果在 [report/REPORT.md](report/REPORT.md)。正式数据仅来自 `formal/` 六个目录，每模型每任务 5 个环境种子 × 300 回合，总计 9000 回合。

- Belief：`distill_best_ckpt/belief_student_best.pth`，iteration 10000。
- Future：`distill_best_ckpt/future_student_last.pth`，iteration 10000。Future best 是 6000 轮，没有混入本次对比。
- `FINAL_VERIFICATION.json` 记录最终回合数、逐回合计数和协议哈希核验。
- `source_audit.json` 记录历史源码及资源来源与哈希；Future 有一个未使用训练启动器的例外，详见报告。
- `evaluation_code/` 保存本次运行的评估器、只读状态采集、调度器、统计和图像诊断脚本。
- `depth_diagnostic/` 是独立的 10 回合图像检查，不计入正式成功率。
- `pilot/`、`serial_check/`、`parallel_single_inference_check/`、`formal_batched_discarded/` 均为验证或废弃数据，不能与正式数据合并。

评估使用原配置的任务判定；Roll 主成功率是区域保持，稳定抓持另列。两个已训练策略各只有一份，5 个种子是环境评估种子，不是独立训练重复。当前本地运行库版本保存在每个正式目录的 `protocol.json`。

历史源码与所需资源已完整归档到 `historical_sources_and_assets.tar.gz`，避免 `/tmp` 被清理后丢失。需要恢复时：

```bash
cd /home/isee/Catch
mkdir -p /tmp/catch_distill_eval_sources_20261029
tar -xzf analysis_artifacts/distill_best_multiseed_20261029_v1/historical_sources_and_assets.tar.gz \
  -C /tmp/catch_distill_eval_sources_20261029
```

仅重新生成报告，不需要 GPU：

```bash
cd /home/isee/Catch
MPLCONFIGDIR=/tmp/catch-report-mpl \
  /home/isee/miniconda3/envs/dcmm/bin/python analysis/report_distill_multiseed_20261029.py
```

本次调度命令为 `python analysis/run_distill_multiseed_20261029.py`（dcmm 环境、GPU 0 / EGL 0）。已完成的正式目录会跳过；不应删除完成标记来覆盖本次数据。若要开展新评估，应使用新的输出目录与明确的新协议。

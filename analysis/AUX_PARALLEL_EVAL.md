# Future / Contact 并行环境评估

`evaluate_aux_parallel.py` 使用 `--num-envs` 控制单项任务的并行仿真环境数，深度图和策略动作按批次推理。Throw、Roll、Bounce依次运行，因此32表示当前任务32个环境，不是三个任务共96个。`--episodes 300` 是每任务总计300回合；32个环境分别分配9或10回合，按固定配额收集，避免只统计先结束的环境。

默认模型是Future last（10000轮）与Contact best（6000轮）；可用 `--checkpoint` 指定同变体、同审计源码版本的检查点。脚本会恢复缺失的历史环境源码并核验哈希，依赖已有 `analysis_artifacts/distill_best_multiseed_20261029_v1/historical_sources_and_assets.tar.gz` 和审计JSON。不会替换已存在但哈希不同的历史文件。

示例（在项目根目录、dcmm环境中运行）：

```bash
python analysis/evaluate_aux_parallel.py --variant future \
  --num-envs 32 --episodes 300 --seed 20261029 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 \
  --output analysis_artifacts/future_parallel_new

python analysis/evaluate_aux_parallel.py --variant contact \
  --num-envs 32 --episodes 300 --seed 20261029 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 \
  --output analysis_artifacts/contact_parallel_new
```

每次选择新的输出目录。结果为输出根目录的 `student_evaluation.json` 及每任务子目录的 `summary.json`、`episodes.jsonl`；运行过程保存在 `progress.json`、`episodes_partial.jsonl`。程序不覆盖已有评估协议，也暂不支持从中断的部分回合续评。

一个主种子控制全部环境：第i个环境构造种子为 `seed+1000000+task_id*10000+i*1009`，其第j回合重置种子再加 `j*100003`。增加并行环境数会改变采样的物理参数和回合序列。比较两个模型时，应统一主种子、环境数、回合数及推理批大小；并行版与之前单环境版的百分比不保证完全相同。

默认 `--inference-batch-size 0` 对所有活跃环境一起推理；设为1则逐环境前向计算，环境仿真仍并行。每个环境的GRU记忆、上一动作、奖励和回合计数独立维护，仅对结束回合的环境清零。

本地32GB内存、8GB显存时可先尝试8或16环境；32环境需要同时容纳32套仿真/渲染进程。每个任务是一个模型进程加N个环境子进程。

已做CPU合成测试：固定配额、快慢环境样本平衡、异步终止时独立GRU/动作重置、总回合少于环境数、超时回合清理、源码恢复哈希及拒绝覆盖修改文件；未启动正式模型评估或32环境GPU测试。

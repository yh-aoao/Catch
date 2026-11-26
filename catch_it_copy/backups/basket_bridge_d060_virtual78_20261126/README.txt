已认可候选：原小物理篮筐桥接模型；100次测试，0.6m虚拟区域78%，原进框率13%。
权重SHA256: e00e4fb57fdd3db35e2e8944f008f9d53a5925c18789549b603f56890805c552
不是“物理0.6m篮筐已训练78%”的模型。

candidate_full_snapshot.tar.gz包含匹配测试权重、训练/测试代码和配置、MuJoCo资产、gym_dcmm环境、
100回合逐回合测试结果及汇总。manifest.json记录全部文件和归档哈希。
系统Python/Conda/CUDA/驱动依赖不在归档内，使用原dcmm环境。

回退不覆盖现有工作目录，先还原到新目录：
python backups/basket_bridge_d060_virtual78_20261126/restore_candidate.py --output /home/isee/Catch/restored_bridge_78
cd /home/isee/Catch/restored_bridge_78
python -m basket_mobile_bridge.eval_regions --checkpoint basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth --output basket_mobile_bridge/evaluations/recheck_78 --episodes 100 --diameters 0.6 0.8 1.0 1.2 1.5 --seed 900123 --device cpu

正式测试可由用户自行运行；未自动重跑100回合。

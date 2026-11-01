# Basket 联合训练稳定性修正

针对Mobile SR与Replay SR同时持续下降，本次修正更新逻辑；不修改投进、有效释放或安全判定来提高表面成功率。没有启动训练。

## 修正内容

1. KL恢复为动作维度求和：底座2维、抛球18维分别统计。默认固定学习率：底座/critic 1e-4，机械臂/手指1e-5。若显式选择kl调度，使用两组独立的有上限调度，绝不超过各自初始学习率。
2. 每次采样后的PPO更新始终对比采样时的old_mu、old_sigma和old_logp；不再在优化过程中覆盖old_mu/old_sigma。底座归一化统计在更新期间保持不变，全部更新完成后只用采样数据更新一次。抛球归一化继续沿用原固定模型。
3. 单次PPO更新内，底座KL超过0.02或抛球KL超过0.01，就停止该actor在本轮余下的更新，另一actor和critic仍可更新。下一次采样后重新判断。分别裁剪两组actor和critic梯度，避免critic的大梯度压低actor更新。
4. 按导航、移动抛球、固定复习抛球三组分别标准化优势。抛球RL损失在整份采样数据上按移动/复习各50%加权，与两类回合长短和样本在minibatch中的聚集无关。缺少某类样本时，只用实际存在的样本，不构造伪数据。
5. 固定复习回合加入0.05权重的参考策略KL损失，约束当前抛球分布不要偏离原固定模型过快；约束同时涉及动作均值和探索标准差。参考网络只产生损失目标，不执行机器人控制。实际机械臂/手指actor始终可训练，参数在优化器中。
6. 增加与当前观测对应的rehearsal元数据，并单独存入rollout。该标签不进入策略输入，策略仍为51维观测/20维动作；不从向量自动重置后的终端info猜测当前回合类别。
7. 日志增加Park SR、Release SR、Base LR、Throw LR、Base KL、Throw KL。TensorBoard还记录参考KL、两组本轮停止标记、导航/移动抛球/复习抛球样本数。

本次代码检查确认之前实现存在KL尺度与调度不匹配等风险，但没有服务器的实际学习率曲线，不能声称已证明该轮退化完全由某一个因素导致。稳定性改进也需要新训练验证长期效果。

## 重新训练

建议从最初的固定底座41.30或对应41.33重新开始第二阶段，保留旧实验作对照，不接着退化的移动last模型训练。新来源仍然是原固定底座模型，两个actor一起训练。

```bash
cd ~/Catch/catch_it_copy
bash scripts/train_basket_mobile41.sh /服务器上原固定底座模型的完整路径.pth
```

本机默认模型存在时：

```bash
bash scripts/train_basket_mobile41.sh ../11-1/throw_basket_catch_best_reward_41.30.pth
```

默认32环境、GPU0、异步、1万亿步上限；不使用.inf。新目录：`outputs/BasketMobile41JointStable/日期/时间/nn/`。

启动日志应出现：

```text
[basket-joint-stable] schedule=fixed; LR ceilings: base=0.0001 throw=1e-05
```

仍然30%回合复习、70%回合移动；这是回合抽样比例，抛球损失权重单独平衡。不要将原固定模型填入checkpoint_catching，脚本将其传给basket_throw_checkpoint。

## 验证

- 104项Basket测试通过，包含KL求和/相同分布KL为0、复习样本聚集时损失总权重不变、学习率上限、阶段独立停止、统计量固定、采样分布锚点不被覆盖，以及原有任务回归。
- 合成数据运行实际更新函数，但optimizer.step替换为空操作；确认两个actor均获得梯度，参考网络不获梯度，真实参数未更新。
- 真实41.30加载、20动作和51输入存储、源actor一致、内嵌checkpoint保存重载、向量环境推理检查通过。
- 两个spawn异步环境核对rehearsal标签与实际回合一致，并验证观测/元数据写入rollout，执行有限推理步；没有训练。
- 新结构只推理测试固定复习5回合，5/5安全投进。该结果仅验证初始化/加载，不是新移动策略训练成功率。
- Python编译、shell语法及本次代码范围的diff检查通过。

记录：`/home/isee/Catch/analysis_artifacts/basket_joint_stable_20261101/`。

## 服务器同步

需同步本次新增或修改的文件：

- configs/basket_mobile41.yaml
- configs/config.yaml
- gym_dcmm/algs/ppo_dcmm/basket_joint_stability.py（新增）
- gym_dcmm/algs/ppo_dcmm/basket_mobile_joint.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_basket_mobile_joint.py
- gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py
- gym_dcmm/envs/DcmmVecEnv.py
- scripts/train_basket_mobile41.sh
- tests/test_basket_joint_stability.py、tests/test_basket_joint_update.py（新增）及tests/test_basket_mobile_joint.py

环境文件和训练器需一起同步，否则元数据/观测检查会报错。

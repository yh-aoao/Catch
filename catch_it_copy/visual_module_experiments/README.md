# C1–C5：B6 位置辅助预测之上的独立模块实验

五组各自从头训练，不叠加其他 C 组模块。均保留 B6 的位置监督（权重 0.1），速度监督为 0；学生无任务标签、无小球真值输入。默认 top/wrist 两路 160×160 深度、seed=123、每任务4环境、10000轮、每2000轮每任务评估300回合。Track/Catch是分别训练的独立学生，不自动串联。

| 编号 | 目录 | 改动 |
|---|---|---|
| C1 | distillation_aux_future | GRU→128维MLP→未来1/3/5策略步位移（9维），线性映射回GRU维度后残差融合到动作头；未来SmoothL1权重0.1 |
| C2 | distillation_aux_belief | 上一GRU状态＋上一步实际动作预测当前视觉特征；当前视觉与历史状态预测每相机可见性置信度，融合当前/预测特征后输入GRU；可见性BCE与预测特征SmoothL1各0.05 |
| C3 | distillation_aux_spatial | 每相机末层空间特征图增加1×1卷积热图，用softmax汇聚局部特征及二维坐标，与原全局embedding残差相加；可见像素中心的高斯热图交叉熵权重0.05 |
| C4 | distillation_aux_contact | GRU预测未来3步内接触、上一步接触、过去3步均接触三个概率，调制动作头输入；BCE权重0.1，调制幅度0.1，初始为恒等映射 |
| C5 | distillation_aux_camera_gate | B6＋原B3式双相机softmax门控，乘2使初始等权对应原拼接；无额外模块监督，module_loss为0正常 |

## 标签与边界

- C1标签为世界坐标系物体几何中心相对当前时刻的位移（米），不混用各时刻移动的机器人坐标。按完整回合生成，不跨reset、不补造终止后的未来。屏蔽已经出现手部接触的预测区间；地面反弹仍保留，以学习非弹道变化。标签描述采样轨迹，不声称是任意动作下的反事实预测。未来步长对应实际策略控制间隔，不能直接当作固定秒数。
- C2/C3仅训练采样额外渲染两路实例分割，使用同一个renderer、同一仿真时刻；mask不输入策略。恢复深度渲染模式后才继续正常采样。可见性定义为至少一个小球像素可见；不是标定过的位置不确定性。C3监督可见部分的像素质心，不是假设被遮挡中心仍可见。完全遮挡时不计算空间热图监督。
- C2特征预测目标stop-gradient；无历史的首帧不计算预测损失。GRU逐步融合保证整序列、分块、在线一步推理一致。没有额外添加随机遮挡训练；鲁棒性须另用受控遮挡实验验证。
- C4包装原环境的`_get_contacts`记录子步中的物体—掌/手指接触，保留原函数返回值；策略步末再兜底检测。当前/历史接触来自先前策略区间；三个区间均出现接触不等于物理上连续保持0.3秒，更不替代成功标准。终止前不足3步且没有接触的负样本为右删失，屏蔽未来分类标签。
- 所有任务原物理、奖励、成功标准保持原实现。附加标签不会进入`student_step/forward`。eval关闭分割标签渲染；视觉输入不使用mask或任务ID。
- 默认跳过教师基线评估，仅跳过成功率测试，不跳过教师动作标注。best按三个任务最低成功率选择，不按均值选择。最终轮次请查看`eval_010000.json`，不要混用`student_best.pth`的结果。

## 运行

在`catch_it_copy`目录内运行。每个入口支持check、camera-check、teacher-eval、train、eval与resume。五个目录拥有独立schema、配置、运行输出；共用`visual_module_experiments`实现并追踪来源哈希，不修改旧B0–B6。

```bash
# Linux bash：顺序运行五组Catch实验（每组每任务4环境，共12环境）
for model in future belief spatial contact camera_gate; do
  python -m distillation_aux_${model}.run train \
    --stage catch --tasks throw roll bounce --device cuda:0 --gl-backend egl \
    --iterations 10000 --num-envs 4 --seed 123 \
    --eval-every 2000 --episodes 300 --skip-teacher-baseline \
    --output distillation_aux_${model}/runs/catch_10000_seed123 || break
done
```

Track使用`--stage track`，输出目录也改为`track_10000_seed123`。如改用8环境请五组统一修改。Torch与EGL设备编号不一定相同，按服务器实际映射指定`--egl-device-id`。首次同步请先运行小测：

```bash
for model in future belief spatial contact camera_gate; do
  python -m distillation_aux_${model}.run train --stage catch \
    --device cuda:0 --gl-backend egl --num-envs 1 --iterations 5 \
    --eval-every 5 --episodes 3 --skip-teacher-baseline \
    --output distillation_aux_${model}/runs/catch_smoke || break
done
```

测试/续训必须使用原目录对应的入口，不能把B6或其他C组checkpoint作为resume。新配置也不能覆盖已有实验目录。

## 本地验证

```bash
python -m unittest visual_module_experiments.test_modules
```

覆盖五组×Track/Catch的在线/整序列一致性、梯度、标签隔离、未来/接触标签边界、保存/评估/续训，采用CPU合成环境。不能把合成环境成功率当作机器人结果；MuJoCo/EGL服务器实测仍需上述smoke，尤其C2/C3的分割渲染路径。

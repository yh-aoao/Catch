# 无标签双相机视觉蒸馏 v2（Track / Catch）

从 `catch_it_copy` 目录运行。本版本新增于 `distillation_visual/`，原有状态 MLP 蒸馏继续保留。
Track 和 Catch 仍是两个独立训练实验、两个独立学生，各自融合 throw / roll / bounce 三个教师。

**v2默认双相机。旧v1单相机checkpoint不能直接eval或resume到v2，需新建输出目录重新训练；如需复现旧实验，请使用对应旧版代码。** 相机顺序、每路深度范围随checkpoint保存，训练和评估使用同一配置；没有修改相机安装姿态。

配置 `vision.cameras` 是实际模型输入顺序与归一化范围的来源；`vision.camera/near/far` 为旧配置兼容及相机检查预览保留。双路输入布局固定为 `[top,wrist]`，每路CNN不共享参数。

## 输入和网络

```text
top 相机深度图（1×112×112） → 独立CNN → 128维特征 ─┐
wrist 相机深度图（1×112×112） → 独立CNN → 128维特征 ┤
机器人自身状态：Track 12维 / Catch 24维 ──────────┤
上一步实际发送的归一化动作：Track 8 / Catch 20维 ──┤
                                                ↓
                                      单层 GRU，隐藏维度256
                                                ↓
                                      MLP 512→256→128，ELU
                                                ↓
                                      tanh → 8 / 20维动作
```

每路CNN通道为[16,32,32]，卷积核[5,4,3]，步幅[2,2,1]，末尾MaxPool2，再经全连接与ELU得到128维。双路特征拼接为256维，加入状态和上一步动作后，GRU输入为Track 276维 / Catch 300维。

结构参考 [Robot Parkour Learning 附录B、表11](https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf)。
这是适配本项目的实现，不是论文完整复现：图像尺寸、相机、控制频率、动作维度、分组损失权重、回放与优化设置不同；尚未加入论文的视觉噪声、延迟与相机随机化。

- **学生没有任务 one-hot，也没有小球位置/速度六维真值。** `proprioception()` 用白名单只拼接底盘速度、末端位置/四元数/速度，Catch 另加12维手部状态。`Student.forward()` 不接受任务编号、物体状态或终止信息。
- 只有采样调度器知道任务，使用对应的冻结教师为当前状态提供动作标签。教师仍使用原18/30维观测及原归一化统计。
- 三个任务使用同一个物理动作缩放；启动时强制检查，避免部署控制器暗中根据任务更换动作含义。
- 每个环境分别保留 GRU 隐藏状态；新 episode 将隐藏状态及上一步动作清零。上一步动作指发送给控制器的命令，不是关节实测运动。
- 仿真外部状态仍用于教师监督与评估统计，绝不拼到学生输入。`teacher_obs` 是进程通信中的原始观测容器，学生只从中提取上述机器人字段。

## 相机与物理环境

使用项目已有 `top` 和 `wrist` 两台相机，不使用追踪小球的全知相机，不读取物体分割或真值投影坐标。
旧环境 `render()` 在无头模式会跳过渲染，并且部分深度后处理使用环境随机数；新版本在独立实例中停用旧的诊断图像输出，使用 `mujoco.Renderer` 在当前 `Dcmm.model/data` 上采集真实深度。
每个策略步同步采集两路图像，两次渲染之间不推进仿真；reset 后采集新回合图像；采图不推进仿真、不读取小球真值、不消耗 NumPy 物理随机数。
top默认固定深度范围0.1～5米，wrist默认0.01～1.5米，裁剪后映射到[0,1]，以float16存储；不逐帧归一化，避免丢失距离含义。
只对机器人状态做一次均值/标准差校准，之后冻结；上一步动作不做标准化。

Linux 无头服务器默认使用 EGL。`--device cuda:0` 选择 PyTorch 计算设备，不承诺同时选择 EGL 渲染设备；多GPU机器需要按服务器 EGL 枚举情况设置 `MUJOCO_EGL_DEVICE_ID`。
有桌面的 viewer 评估使用 `--gl-backend glfw --viewer`。真正的视觉训练必须能够渲染，不能用空图/零图绕过 OpenGL 故障。
相机位置来自各任务自身资产，真实物体可见性需要检查下面导出的图像。原项目有真机视觉系统，不等于这一新增原始深度策略已经完成真机迁移。

## 教师与成功标准

`configs/catch.json` / `configs/track.json` 是独立、可编辑的视觉训练配置，已复制当前经过检查的教师文件名、SHA256、环境参数和成功标准，后续更换教师请修改相应文件。

| 阶段 | throw | roll | bounce |
|---|---|---|---|
| Catch | `best_model/throw_catch_best_reward_497.09.pth` | `best_model/roll_catch_best_reward_252.54.pth` | `best_model/bounce_catch_best_reward_653.87.pth` |
| Track | `best_model_track/throw_track_best_reward_58.56.pth` | `best_model_track/roll_track_best_reward_33.45.pth` | `best_model_track/bounce_track_best_58.18.pth` |

沿用原蒸馏的教师严格加载、原始推理一致性检查、动作适配器与成功统计：throw/bounce读取各自环境的 `success`；roll读取 `roll_eval_success`。没有修改奖励、接球标准或原环境文件。

## 在线训练流程

1. 校验教师，然后在视觉环境适配器上重新评估教师基线，默认每任务100回合。相机路径也会被实际执行。
2. 初始学生执行动作，收集128步/环境的机器人状态，拟合并冻结状态归一化。不是离线BC，不执行教师动作预热。
3. 每轮每任务每环境收集64步。教师根据同一时刻真实状态生成标签；环境执行学生动作。
4. 按**完整 episode**存入 CPU 回放，每任务上限8192帧；满后移除最旧完整回合。未结束回合暂存在采样器，第一次优化前等待每任务至少一个完整回合。
5. 每次更新每任务抽2个完整回合，默认每轮更新8次。每回合 GRU 从零开始；按16步截断反向传播，前向隐藏状态仍贯穿完整回合。补齐帧不计损失，任务等权、抽样回合等权，动作组等权。
6. 默认使用论文启发的连续动作 BCE：教师动作映射到[0,1]，学生以 `BCEWithLogits(2*logits)` 稳定计算，再乘2。它是动作监督，不是分类识别任务；配置可切换 `loss: mse`。BCE 的最小值通常不为0，不能和旧MSE数值直接比较。
7. 每次优化结束，用更新后的网络重新计算尚未结束回合的历史，刷新采样隐藏状态，避免旧网络产生的隐藏状态继续传入新网络。
8. 默认训练1000轮，每200轮及最后一轮评估，每任务100回合。按“最差任务相对教师的成功率差”保存 best，与之前选择原则一致。

图像回放默认约1.15 GiB（仅深度数据），另有状态、尚未结束回合、训练激活和进程开销。`resume_last.pth` 包含回放，可能超过1.2 GB。视觉版还增加相机渲染、CNN和时序反向传播，不能沿用旧MLP每轮耗时估计。

## 先检查再训练

所有命令在 `catch_it_copy` 目录执行。每次新训练使用新输出目录；失败后已有 `resolved_config.json` 也应换目录，以免混淆记录。

教师接口检查（不创建仿真）：

```bash
python -m distillation_visual.run check --stage catch --output distillation_visual/runs/check_catch
python -m distillation_visual.run check --stage track --output distillation_visual/runs/check_track
```

实际相机检查（教师驱动轨迹，不是成功率测试）：

```bash
python -m distillation_visual.run camera-check --stage catch --tasks throw roll bounce --device cuda:0 --gl-backend egl --camera-steps 80 --output distillation_visual/runs/camera_catch
python -m distillation_visual.run camera-check --stage track --tasks throw roll bounce --device cuda:0 --gl-backend egl --camera-steps 80 --output distillation_visual/runs/camera_track
```

输出每任务 `*_depth.npz`（形状N×2×H×W，归一化深度序列、camera_names、每帧动作后的done、每路near/far）、`*_depth.png`（抽样图像拼图，亮处近、暗处远）、`camera_report.json`。第c路还原裁剪后的米制深度：`depth_normalized[:,c]*(far[c]-near[c])+near[c]`。每帧都是动作执行前的观测；done为真后下一帧属于新回合。图像统计正常不代表小球一定可见，需查看序列。

5轮流程测试：

```bash
python -m distillation_visual.run train --stage catch --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5 --num-envs 1 --baseline-episodes 3 --episodes 3 --allow-zero-teacher-success --output distillation_visual/runs/catch_visual_v2_smoke
python -m distillation_visual.run train --stage track --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5 --num-envs 1 --baseline-episodes 3 --episodes 3 --allow-zero-teacher-success --output distillation_visual/runs/track_visual_v2_smoke
```

`--allow-zero-teacher-success` 只适合这里的小样本流程检查，正式训练不用。5轮只验证流程，不期待已收敛。正式训练必须新建运行，不要续训这两个修改过评估设置的 smoke 实验。

正式训练（每200轮评估，默认100回合/任务）：

```bash
python -m distillation_visual.run train --stage catch --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 1000 --num-envs 2 --output distillation_visual/runs/catch_visual_v2_1000
python -m distillation_visual.run train --stage track --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 1000 --num-envs 2 --output distillation_visual/runs/track_visual_v2_1000
```

`--num-envs 2` 是每任务2个，共6个仿真/渲染进程，不建议未经测量直接套用旧单任务32环境。两个正式命令可分别运行，不必同时占用同一GPU。

独立评估（`--seed` 可换一组回合，任务参数仅决定测试环境，不进入学生网络）：

```bash
python -m distillation_visual.run eval --checkpoint distillation_visual/runs/catch_visual_v2_1000/student_best.pth --tasks throw roll bounce --device cuda:0 --gl-backend egl --episodes 300 --seed 456 --output distillation_visual/runs/catch_visual_v2_eval
```

有图形桌面的 viewer：

```bash
python -m distillation_visual.run eval --checkpoint distillation_visual/runs/catch_visual_v2_1000/student_best.pth --tasks bounce --device cuda:0 --gl-backend glfw --viewer --episodes 10 --output distillation_visual/runs/catch_visual_view
```

断点续训（训练配置自动从checkpoint读取，可修改总轮数及PyTorch设备）：

```bash
python -m distillation_visual.run train --resume distillation_visual/runs/catch_visual_v2_1000/resume_last.pth --iterations 2000 --device cuda:0 --output distillation_visual/runs/catch_visual_v2_1000
```

`student_last.pth` 每轮保存；`student_best.pth` 在评估改进后保存；完整 `resume_last.pth` 默认每200轮及末轮保存。恢复优化器、完整回合回放和随机数，但重新开始仿真回合，不是逐位连续恢复。未结束回合不保存在断点中。
评估会校验源码/资产哈希，防止在变更成功标准后误比较旧结果。部署文件自带网络、相机配置和归一化统计，不需要加载教师。

## 验证记录与限制

### top / wrist 同步深度检查

```bash
python -m distillation_visual.camera_pair --stage catch --tasks throw roll bounce --device cuda:0 --gl-backend egl --steps 200 --output distillation_visual/runs/camera_pair_catch
```

Track 改为 `--stage track`，并使用另一输出目录。采集由原教师控制机器人。
每个任务子目录下有 `top_left_wrist_right.png` 拼图、`frames/` 中每一步的并排PNG（左top、右wrist）、`synchronized_depth.npz` 和 `metadata.json`。
两台相机渲染之间不推进物理，采集后才执行教师动作；每对图有相同的仿真时间及回合编号。
NPZ 的 `top_depth_m` / `wrist_depth_m` 是未做应用层裁剪的float32米制深度，形状 `[帧数,H,W]`，`sim_time` / `episode_id` 用于对齐，`done_after_action` 表示本帧之后的动作是否结束回合。
PNG仅用于预览：top默认0.1～5米，wrist默认0.01～1.5米（可用 `--wrist-near` / `--wrist-far` 调整）；两路灰度不能直接当作相同距离比较。MuJoCo自身的近远裁剪仍然生效，预览范围不能恢复渲染器已裁剪掉的几何。
此命令只导出诊断数据；当前v2学生训练已默认同时使用top和wrist。

- 本机六个教师全部通过原推理、观测、动作一致性检查，最大推理误差0。
- `python -m unittest discover -s distillation_visual/tests -v`：覆盖无真值泄漏、GRU逐帧一致性/重置、padding梯度屏蔽、BCE稳定性、完整回放、不同任务动作缩放拒绝、相机序列导出与回合图像对齐、Track/Catch模拟训练-评估-续训。
- 本机真实相机命令在加载 MuJoCo DLL 时失败（Windows `WinError 1114`），尚未完成实际仿真渲染或接球闭环验证。需要在原 Linux dcmm 环境执行相机检查和smoke命令；不能将模拟测试的成功率当成真实接球结果。
- 当前是仿真双相机视觉蒸馏版；没有完成相机噪声/遮挡/延迟增强或真机视觉接口适配，不承诺仅因原项目做过真机就能直接部署。


## 160×160 双相机对照版本

新增 `configs/track_160.json` 和 `configs/catch_160.json`，使用同一训练入口。
相对于对应的 112 配置，仅修改 vision.height、vision.width 和 env_kwargs.img_size；
实际两路深度渲染均使用 160×160。教师、成功标准、相机姿态、深度范围、训练参数保持一致。
没有加入辅助预测、任务标签或小球真值输入。
CNN 自动根据分辨率计算展平尺寸，GRU 与动作头维度不变；CNN 全连接参数量会增加，
因此这是分辨率及其自然计算开销的对照，并非严格等参数量比较。
图像缓存像素量约为原来的 2.04 倍。

从 catch_it_copy 运行（先验证 5 轮，再新建目录正式训练）：

```bash
python -m distillation_visual.run train --config distillation_visual/configs/catch_160.json --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5 --num-envs 2 --baseline-episodes 3 --episodes 3 --eval-every 5 --output distillation_visual/runs/catch_dualcam160_smoke
python -m distillation_visual.run train --config distillation_visual/configs/track_160.json --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5 --num-envs 2 --baseline-episodes 3 --episodes 3 --eval-every 5 --output distillation_visual/runs/track_dualcam160_smoke

python -m distillation_visual.run train --config distillation_visual/configs/catch_160.json --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5000 --num-envs 8 --baseline-episodes 200 --episodes 200 --eval-every 500 --output distillation_visual/runs/catch_dualcam160_5000
python -m distillation_visual.run train --config distillation_visual/configs/track_160.json --tasks throw roll bounce --device cuda:0 --gl-backend egl --iterations 5000 --num-envs 8 --baseline-episodes 200 --episodes 200 --eval-every 500 --output distillation_visual/runs/track_dualcam160_5000
```

评估自动从 checkpoint 读取分辨率与阶段：

```bash
python -m distillation_visual.run eval --checkpoint distillation_visual/runs/catch_dualcam160_5000/student_best.pth --tasks throw roll bounce --device cuda:0 --gl-backend glfw --viewer --episodes 10 --output distillation_visual/runs/catch_dualcam160_view
```

160 模型应从头训练；不能用 112 checkpoint 的 --resume 改分辨率，resume 会继续使用旧配置。
请保留旧实验对应代码版本：当前 source/assets 严格校验包含配置文件，新增配置也会改变来源清单，
旧 checkpoint 在新代码版本下可能被拒绝。请在旧代码副本评估/续训旧模型，不要绕过校验。
本地网络验证不替代服务器上的真实 MuJoCo 双相机测试。

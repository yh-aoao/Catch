# Slide：斜坡滑落接取 v1

独立环境，复用机器人观测、IK、PID与动作缓冲；不使用旧Roll的step、奖励或终止判定。不改已有Roll/Bounce/Throw环境路由。

默认20度斜坡、坡面长1.2m宽0.7m，前缘世界y=0.9m、z=0.7m。小方块半尺寸0.03/0.03/0.02m，质量0.05kg，坡面与方块滑动摩擦系数0.15，从坡面70%长度处静止释放。物体由重力和碰撞运动，没有延时发射、外力托举或注入速度。用方块是为了区分滑落与球体滚落；这些参数是待验证的仿真初始配置，不是实测材料参数。

## 目标、奖励与成功

- 出坡前用坡度/摩擦预测出坡速度与下落位置，手掌在坡口外侧等待；出坡后使用实际物体位置/速度更新落点。预测只影响奖励目标，不驱动物体。
- 奖励以掌面几何中心到目标的XY误差为主，辅以高度、接触、保持与动作平滑；惩罚手进入坡口后方。
- Track：物体中心加保守包围半径越过前缘后，与掌面或手指接触成功。接触坡上物体不算成功。
- Catch：出坡后接触手，并在掌面附近保持0.3秒；允许最长0.08秒短暂接触间断。保持区域XY半径0.13m，世界Z相对掌面为[-0.02,0.16]m；这是本版固定小方块标准，不能直接声称适用于任意形态。
- 底座碰撞、手撞坡、物体落地、物体越界均失败；默认5秒超时失败。同一步碰地和碰手时失败优先。
- 默认释放X随机范围±0.08m，其余坡度/形态固定；先验证场景再扩大分布。

## 场景检查

从`catch_it_copy`目录运行：

```bash
python -m motion_slide.probe --episodes 3
python -m motion_slide.probe --viewer --episodes 3
```

这是零动作检查：确认initial_velocity全零，物体沿坡滑下、departed=true；未接住而落地是预期，不代表训练失败。确认机器人初始姿态无自碰撞、坡口与手有接取空间之后再训练。无界面版本不需要GL窗口。

## 训练

```bash
python3 train_DCMM.py test=False task=Tracking num_envs=32 \
  object_motion=slide slide_log=false device_id=0 \
  output_name=SlideTrackV1

python3 train_DCMM.py test=False task=Catching_TwoStage num_envs=32 \
  object_motion=slide slide_log=false device_id=0 \
  checkpoint_tracking=/absolute/path/to/slide_track.pth \
  output_name=SlideCatchV1
```

Catch命令中的checkpoint必须换成新Slide Track实际保存的模型路径。不要将旧Roll模型当作已验证的Slide教师。现有PPO负责训练，学生观测和动作维数沿用18/8与30/20接口；本轮尚未扩展多任务视觉蒸馏的任务列表。

## 测试

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 \
  object_motion=slide slide_log=true viewer=true object_eval=false \
  checkpoint_tracking=/absolute/path/to/slide_track.pth

python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 \
  object_motion=slide slide_log=true viewer=true object_eval=false \
  checkpoint_catching=/absolute/path/to/slide_catch.pth
```

自定义参数可写JSON并传`slide_config=/absolute/path/config.json`，场景检查使用`--config`。字段见`geometry.py::SlideConfig`；例如`{"angle_deg":25,"friction":0.15,"release_x_range":0.0}`。未指定字段沿用默认值。

## 验证状态

已做纯几何、XML修改、控制转换、模拟子步成功/失败逻辑测试与语法检查。当前Windows本机MuJoCo DLL不可用，尚未运行真实场景或训练，不能保证坡口/初始姿态无需调整。请保留服务器probe日志作为下一步闭环验证依据。

# 多形态测试入口

`train_DCMM.py`新增`test_object`。仅允许test=True、object_eval=false，支持throw/roll/bounce的Tracking、Catching_TwoStage和既有单阶段测试入口。默认default完全沿用原来的路由；非默认走独立工厂，选择对应历史环境类，通过子类覆盖物体XML及重置后的物理常量，不修改原冻结文件。

| 值 | 几何参数（米，MuJoCo半尺寸） | 质量kg |
|---|---|---:|
| sphere | 半径.04 | .05 |
| tennis_ball | 半径.0335 | .058 |
| pingpong_ball | 半径.02 | .0027 |
| box | .03,.03,.03 | .05 |
| cylinder | 半径.03，半高.04 | .05 |
| ellipsoid | 半轴.03,.04,.05 | .05 |
| capsule | 半径.025，直段半长.03 | .05 |

网球/乒乓球仅为尺寸质量预设：外观未导入YCB纹理，摩擦、接触弹性和阻尼沿用任务原配置，惯量按均匀几何体计算，不代表真实空心乒乓球。要做真实材料泛化仍需标定。XML移除旧惯量让MuJoCo按新形状质量计算；每次reset恢复尺寸、质量、惯量，避免被Bounce的历史半径/质量逻辑覆盖。保持单个目标碰撞geom，原接触ID统计继续适用。

## 命令

```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 object_motion=throw test_object=tennis_ball object_eval=false viewer=true checkpoint_tracking=/absolute/path/to/throw_track.pth
python3 train_DCMM.py test=True task=Tracking num_envs=1 object_motion=roll test_object=cylinder object_eval=false viewer=true checkpoint_tracking=/absolute/path/to/roll_track.pth
python3 train_DCMM.py test=True task=Tracking num_envs=1 object_motion=bounce test_object=pingpong_ball object_eval=false viewer=true checkpoint_tracking=/absolute/path/to/bounce_track.pth
```

Catch沿用该模型原先的测试task与checkpoint配置，添加相同test_object参数。例如两阶段：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=roll test_object=cylinder object_eval=false viewer=true checkpoint_catching=/absolute/path/to/roll_catch.pth
```

本轮未扩展distillation_*的独立学生测试CLI，不能直接把Hydra参数拼到那些命令后。YCB网格未接入。

## 限制和验证

既有奖励及成功判定保持不变，非球形时其中半径/接球区域等近似仍未全面泛化，非球形成绩仅作探索性对照；不应直接当正式多形态成功率。Roll从桌面释放是否有合适支撑、异形物体是否能够滚出桌边也需viewer验证。不要把物体未滚下桌面直接解释为抓取策略失败。

已做7个预设的XML检查、六个任务路径对应的5个具体类/方法静态检查与语法检查。本机MuJoCo DLL不可加载，尚无真实reset/动力学集成验证。服务器启动应看到[test-object]记录实际preset、质量、尺寸和环境类，先num_envs=1观察再批量评估。

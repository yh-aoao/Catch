# Roll 修改前快照

创建于2026-10-27（Asia/Shanghai），代码基线commit：
`90d9d1474fb89b84885d08af50d2055d163cc40f`。
这是修改自适应等待奖励前的实际文件副本，不依赖Git能否重建历史状态。

- `snapshot/`：164个文件，含gym_dcmm源码、configs、ycb_objects、已有测试、机器人XML及Roll模型。
- `manifest.json`：逐文件SHA256、字节数、原路径、Git状态及需要还原的三个集成文件。
- `restore.py`：默认只检查；显式`--apply`才复制旧文件。恢复前自动另存当前三个文件。
- 不包含完整runs、全部资源网格或外部Python环境。本次没有修改这些资源。

优先通过`roll_reward_version=legacy`切回旧奖励。新版单独封装，没有修改旧环境类、
旧roll_waiting.py、原PPO、旧物理参数或旧模型。

检查快照：

```bash
python3 backups/roll_before_adaptive_wait_20261027_001703/restore.py
```

确实需要恢复修改前入口时，在catch_it_copy下执行：

```bash
python3 backups/roll_before_adaptive_wait_20261027_001703/restore.py --apply
```

还原文件仅为`train_DCMM.py`、`configs/config.yaml`、`gym_dcmm/__init__.py`。
新模块、输出和文档不会删除；还原旧入口后不会选择新模块。
重新启动Python进程后生效。迁移项目路径时可追加`--root /实际路径/catch_it_copy`。
若以后在这三个文件做了其他修改，完整文件恢复也会回退它们；优先用legacy开关，
需要完整恢复时可从`before_restore_*`目录取回恢复前的文件。

旧Catch模型：`snapshot/best_model/roll_catch_best_reward_252.54.pth`。
SHA256：`40020e1b13a29ad49de9e9b6ad2e6316678f418bdd8712be898b7a257af3d0ba`。
完整旧奖励、训练流程与新版说明见项目`ROLL_ADAPTIVE_WAIT_V1_2026-10-27.md`。

# PPO → SAC 算法替换设计文档

**日期**: 2026-07-30  
**目标**: 复制 `catch_it_copy` 为新项目 `catch_it_sac`，将其中的 PPO 强化学习算法替换为 SAC

---

## 一、背景

`catch_it_copy` 是基于 ICRA 2025 "Catch It!" 论文实现的分支，使用 PPO 算法训练移动灵巧手抓取飞行物体。支持四种运动模式：throw / roll / bounce / throw_basket，采用两阶段训练（Tracking → Catching）。

本次改动：将 PPO 替换为 SAC（Soft Actor-Critic），验证 off-policy 算法在此任务上的表现。

## 二、目录结构

`catch_it_sac` = `catch_it_copy` 的完整物理拷贝（不是软链接），改动仅限以下文件：

```
catch_it_sac/
├── gym_dcmm/algs/
│   ├── __init__.py                      # 不变
│   └── sac_dcmm/                        # 新建，替换 ppo_dcmm/
│       ├── __init__.py
│       ├── sac_dcmm_track.py            # SAC_Track（阶段1 Tracking）
│       ├── sac_dcmm_catch_two_stage.py  # SAC_Catch_TwoStage（阶段2 Catching）
│       ├── sac_dcmm_catch_one_stage.py  # SAC_Catch_OneStage（备选）
│       ├── models_track.py              # Actor + TwinQ，单分支
│       ├── models_catch.py              # Actor + TwinQ，双分支（track冻结+catch训练）
│       ├── replay_buffer.py             # Off-policy ReplayBuffer（替换 ExperienceBuffer）
│       └── utils.py                     # 复用，不变
├── configs/
│   ├── config.yaml                      # defaults 改为 DcmmSAC
│   └── train/
│       └── DcmmSAC.yaml                 # 新建 SAC 超参数配置
├── train_DCMM.py                        # import 路径从 ppo_dcmm 改为 sac_dcmm
└── ...（其余所有文件完整复制，不做任何修改）
```

## 三、网络架构

### 3.1 Tracking 阶段（models_track.py）

**Actor 网络**（在 PPO 版本基础上修改）：

```
MLP(obs) → 共享骨干 [256, 128] + ELU
  ├── Linear(out, action_dim) → mu          → tanh → [-1, 1]
  └── Linear(out, action_dim) → log_std     → clamp(log_std_min, log_std_max)
```

与 PPO 的关键区别：
- PPO：σ 是**状态无关**的可学习参数（`nn.Parameter`）
- SAC：log_std 是**状态相关**的，由网络从 obs 计算

**动作采样**（带重参数化和 tanh squash）：

```python
normal = Normal(mu, std)
z = normal.rsample()                    # 重参数化，梯度可传
action = tanh(z)
# log_prob 需要 tanh 修正：
log_prob = normal.log_prob(z) - log(1 - action^2 + ε)
log_prob = log_prob.sum(dim=-1)
```

**Twin Q 网络**（新增，PPO 无此结构）：

```
Q1: MLP(obs + action) → [256, 128] + ELU → Linear(out, 1)
Q2: MLP(obs + action) → [256, 128] + ELU → Linear(out, 1)
Q1_target: 与 Q1 结构相同，独立参数，软更新
Q2_target: 与 Q2 结构相同，独立参数，软更新
```

**温度参数**（可学习）：

```
log_alpha: nn.Parameter (标量，requires_grad=True)
target_entropy = -action_dim * target_entropy_scale
```

### 3.2 Catching 阶段（models_catch.py）

双分支结构，与 PPO 的 `models_catch.py` 同理：

```
Actor:
  obs_t → actor_mlp_t （冻结）→ mu_t (8D) + log_std_t (8D)
  obs_c → actor_mlp_c （训练）→ mu_c (12D) + log_std_c (12D)
  完整 action = concat(mu_t, mu_c)，完整 log_std = concat(log_std_t, log_std_c)

Critic:
  Q1/Q2: MLP(完整 obs + 完整 action) → Q 值
  Q1_target/Q2_target: 同上结构，软更新
```

### 3.3 网络初始化

- MLP 隐藏层：orthogonal 初始化，gain = √2
- mu 输出层：orthogonal，gain = 0.01
- log_std 输出层：orthogonal，gain = 0.01
- Q 输出层：orthogonal，gain = 1.0
- log_alpha：初始化为 0

## 四、Replay Buffer（replay_buffer.py）

替换 PPO 的 `experience.py`（ExperienceBuffer）。

### 结构

```python
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        # 预分配环形缓冲区
        self.obses = torch.zeros(capacity, obs_dim)
        self.actions = torch.zeros(capacity, act_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.next_obses = torch.zeros(capacity, obs_dim)
        self.dones = torch.zeros(capacity, 1)
        self.ptr = 0       # 写入位置
        self.size = 0      # 当前有效数据量

    def add(self, obs, action, reward, next_obs, done):
        # 单条写入（每步调用）

    def sample(self, batch_size):
        # 随机采样 batch_size 条
        return (obs_batch, actions_batch, rewards_batch,
                next_obs_batch, dones_batch)

    def __len__(self):
        return self.size
```

**关键差异**：
- 不复用 PPO 的 batch 迭代器模式（`__getitem__`）
- 不需要 GAE 计算
- 不需要 `update_mu_sigma`
- 批量采样方法更简单：随机索引 + 拼接

## 五、SAC 算法核心（sac_dcmm_track.py + sac_dcmm_catch_two_stage.py）

### 5.1 训练循环

```
1. 随机探索：start_steps 步内动作从 Uniform(-1, 1) 采样
2. 每一步：
   a. actor 采样动作（或随机）
   b. env.step(action)
   c. replay_buffer.add(obs, action, reward, next_obs, done)
   d. if len(replay_buffer) >= batch_size:
        for _ in range(updates_per_step):
          update()
```

### 5.2 更新流程（update 函数）

从 replay buffer 采样 batch_size 条数据，依次计算三个 loss：

**Q-loss**（两个 Q 网络同时优化，取 min 缓解过估计）：

```
with torch.no_grad():
  next_action, next_log_prob = actor(next_obs)
  target_q = min(Q1_target, Q2_target)(next_obs, next_action)
  target_value = target_q - alpha * next_log_prob
  q_target = reward + (1 - done) * gamma * target_value

q1_loss = MSE(Q1(obs, action), q_target)
q2_loss = MSE(Q2(obs, action), q_target)
q_loss = q1_loss + q2_loss
```

**Policy loss**（最大化 Q 值 + 熵正则）：

```
new_action, new_log_prob = actor(obs)
q_min = min(Q1(obs, new_action), Q2(obs, new_action))
policy_loss = (alpha * new_log_prob - q_min).mean()
```

**Temperature loss**（自动调节 α，可选开关）：

```
alpha = exp(log_alpha)
alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()
```

**软更新 Target Q**：

```
for param, target_param in [(Q1, Q1_target), (Q2, Q2_target)]:
    target_param = tau * param + (1 - tau) * target_param
```

### 5.3 与 PPO 类的接口兼容

`SAC_Track` 和 `SAC_Catch_TwoStage` 保持与 `PPO_Track` / `PPO_Catch_TwoStage` 相同的外部接口：

| 方法 | 说明 |
|------|------|
| `__init__(env, output_dir, full_config)` | 初始化，从 config 读取 SAC 超参数 |
| `train()` | 主训练循环 |
| `test()` | 测试循环 |
| `save(name)` | 保存 checkpoint |
| `restore_train(fn)` | 加载训练 checkpoint |
| `restore_test(fn)` | 加载测试 checkpoint |
| `model_act(obs_dict, inference)` | 动作采样 |
| `action2dict(actions)` | 动作反归一化为环境格式 |

## 六、配置文件

### 6.1 DcmmSAC.yaml（新增 SAC 配置）

```yaml
seed: ${..seed}
algo: SAC

network:
  mlp:
    units: [256, 128]
  separate_value_mlp: True

sac:
  name: ${resolve_default:Dcmm,${...experiment}}
  # --- 通用参数（与 PPO 一致） ---
  normalize_input: True
  num_actors: ${resolve_default:1,${...num_envs}}
  reward_scale_value: 1.0
  gamma: 0.99
  learning_rate: 3e-4
  lr_schedule: fixed
  save_best_after: 500
  save_frequency: 2000
  grad_norm: 1.0
  truncate_grads: True
  action_track_denorm: [1.5, 0.025, 0.01]
  action_catch_denorm: [1.5, 0.025, 0.15]
  horizon_length: 64
  max_agent_steps: 25000000
  max_test_steps: 10000
  test_num_episodes: 100
  img_dim: [112, 112]
  num_frames: 2

  # --- SAC 专属参数 ---
  tau: 0.005                    # target Q 软更新速率
  alpha_lr: 3e-4                # 温度参数学习率
  target_entropy_scale: 1.0     # target_entropy = -act_dim * scale
  replay_buffer_size: 1000000   # replay buffer 容量
  batch_size: 256               # 每次更新的采样数
  start_steps: 10000            # 初始随机探索步数
  updates_per_step: 1           # 每步更新次数
  log_std_min: -20              # log_std 下限
  log_std_max: 2                # log_std 上限
  auto_entropy_tuning: True     # 是否自动调节 α
```

### 6.2 config.yaml 改动

仅改一行：`defaults → train: DcmmSAC`

## 七、train_DCMM.py 改动

仅改 import 路径和类名：

```python
# 改前
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track

# 改后
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_two_stage import SAC_Catch_TwoStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_one_stage import SAC_Catch_OneStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_track import SAC_Track

# 改前
SAC = PPO_Track if ... else PPO_Catch_TwoStage else PPO_Catch_OneStage

# 改后
SAC = SAC_Track if ... else SAC_Catch_TwoStage else SAC_Catch_OneStage
```

## 八、Checkpoint 兼容性

SAC Track checkpoint 存储格式：

```python
weights = {
    'model': self.model.state_dict(),              # 完整模型权重（含 log_std_head）
    'actor_mlp_t': self.model.actor_mlp.state_dict(),
    'mu_t': self.model.mu.state_dict(),
    'log_std_t': self.model.log_std.state_dict(),  # SAC 状态相关 log_std 的输出层
    'running_mean_std': self.running_mean_std.state_dict(),
}
```

与 PPO checkpoint 的关键差异：
- PPO 用 `sigma`（单个可学习参数），SAC 用 `log_std_t`（Linear 层的 state_dict）
- Catch 阶段加载时通过 `actor_mlp_t`、`mu_t`、`log_std_t` 三个 key 加载并冻结 tracking 分支
- `model` key 包含完整网络状态，用于整体 save/restore

## 九、不需要改动的部分

以下内容保持不变：
- `gym_dcmm/envs/` — 所有环境代码（DcmmVecEnv.py 等）
- `gym_dcmm/agents/` — MuJoCo 模型、IK 求解器
- `gym_dcmm/utils/` — PID、通用工具、IK 包
- `configs/env/DcmmCfg.py` — 物理参数配置
- `assets/` — 模型文件
- `requirements.txt`, `setup.py`

## 十、实施风险

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| SAC 在连续高维动作空间调参困难 | 中 | 参考原始 SAC 论文推荐超参，逐步调优 |
| Off-policy 采样效率依赖 buffer 质量 | 中 | start_steps 确保足够初始探索；考虑加入 HER 等改进 |
| 两阶段 SAC 训练稳定性 | 低 | 与 PPO 相同结构，冻结机制已验证 |
| tanh squashed 动作的 log_prob 数值问题 | 低 | 使用标准数值稳定公式（已有成熟实现） |

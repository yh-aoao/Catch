# PPO → SAC 算法替换 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复制 `catch_it_copy` 为 `catch_it_sac`，将 PPO 替换为 SAC 强化学习算法

**Architecture:** 新建 `gym_dcmm/algs/sac_dcmm/` 包替换 `ppo_dcmm/`，包含 ReplayBuffer、SAC 版 Actor+TwinQ 网络、SAC_Track/SAC_Catch 训练器。环境层、配置系统、训练入口完全复用

**Tech Stack:** Python 3.8, PyTorch, NumPy, Hydra, MuJoCo

---

## 文件清单

| 操作 | 文件路径 |
|------|---------|
| 复制 | `catch_it_copy/` → `catch_it_sac/`（整个目录物理拷贝） |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/__init__.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/utils.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/replay_buffer.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/models_track.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/models_catch.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_track.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_two_stage.py` |
| 新建 | `catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_one_stage.py` |
| 新建 | `catch_it_sac/configs/train/DcmmSAC.yaml` |
| 修改 | `catch_it_sac/configs/config.yaml` |
| 修改 | `catch_it_sac/train_DCMM.py` |
| 删除 | `catch_it_sac/gym_dcmm/algs/ppo_dcmm/`（整个目录） |

---

### Task 1: 复制项目目录

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/`（整个目录树）

- [ ] **Step 1: 物理拷贝 catch_it_copy → catch_it_sac**

```bash
cp -r f:/yjs/Catch/catch_it_copy f:/yjs/Catch/catch_it_sac
```

- [ ] **Step 2: 删除旧 PPO 算法目录**

```bash
rm -rf f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/ppo_dcmm
```

- [ ] **Step 3: 创建新 SAC 算法目录**

```bash
mkdir -p f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm
```

- [ ] **Step 4: 验证目录结构**

```bash
ls f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/
# 预期输出: __init__.py  sac_dcmm/
ls f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/
# 预期输出: (空目录)
```

---

### Task 2: 创建 `__init__.py` 和 `utils.py`

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/__init__.py`
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/utils.py`

- [ ] **Step 1: 创建 `__init__.py`**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/__init__.py`：

```python
```

文件为空（或仅保留作为 Python 包标记）。

- [ ] **Step 2: 创建 `utils.py`（直接从 ppo_dcmm 复用）**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/utils.py`：

```python
import torch
import torch.nn as nn
import numpy as np


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer('running_mean', torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer('running_var', torch.ones(in_size, dtype = torch.float64))
        self.register_buffer('count', torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis)
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = \
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count, mean, var, input.size()[0])

        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y
```

- [ ] **Step 3: 验证导入**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "from gym_dcmm.algs.sac_dcmm.utils import RunningMeanStd, AverageScalarMeter; print('OK')"
# 预期输出: RunningMeanStd: (1,)  (或类似内容) + OK
```

---

### Task 3: 创建 `replay_buffer.py`

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/replay_buffer.py`

- [ ] **Step 1: 实现 ReplayBuffer 类**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/replay_buffer.py`：

```python
import torch


class ReplayBuffer:
    """环形缓冲区，用于 SAC off-policy 经验回放"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.device = device
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obses = torch.zeros(
            (capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(
            (capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=device)
        self.next_obses = torch.zeros(
            (capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=device)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """存储单步 transition"""
        self.obses[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obses[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机采样 batch_size 条 transition"""
        indices = torch.randint(
            0, self.size, (batch_size,), device=self.device)
        return (
            self.obses[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obses[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
```

- [ ] **Step 2: 验证导入和基本功能**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import torch
from gym_dcmm.algs.sac_dcmm.replay_buffer import ReplayBuffer
buf = ReplayBuffer(capacity=100, obs_dim=18, act_dim=8, device='cpu')
buf.add(torch.zeros(18), torch.zeros(8), torch.tensor([1.0]), torch.zeros(18), torch.tensor([0.0]))
buf.add(torch.ones(18), torch.ones(8), torch.tensor([0.5]), torch.ones(18), torch.tensor([1.0]))
print('size:', len(buf))
obs, act, rew, nobs, dones = buf.sample(2)
print('obs shape:', obs.shape, 'act shape:', act.shape)
print('OK')
"
# 预期输出: size: 2 / obs shape: torch.Size([2, 18]) act shape: torch.Size([2, 8]) / OK
```

---

### Task 4: 创建 `models_track.py`（SAC Track 网络）

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/models_track.py`

- [ ] **Step 1: 实现 Track 阶段的 SAC Actor + TwinQ 网络**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/models_track.py`：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
         for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class QNetwork(nn.Module):
    """Twin Q 网络：输入 (obs + action)，输出 Q 值"""

    def __init__(self, obs_dim, action_dim, units):
        super(QNetwork, self).__init__()
        input_dim = obs_dim + action_dim
        self.q_mlp = MLP(units=units, input_size=input_dim)
        out_size = units[-1]
        self.q_out = nn.Linear(out_size, 1)
        torch.nn.init.orthogonal_(self.q_out.weight, gain=1.0)
        if self.q_out.bias is not None:
            torch.nn.init.zeros_(self.q_out.bias)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = self.q_mlp(x)
        return self.q_out(x)


class SACActor(nn.Module):
    """SAC Actor：输出 state-dependent mu 和 log_std"""

    def __init__(self, obs_dim, action_dim, units,
                 log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.actor_mlp = MLP(units=units, input_size=obs_dim)
        out_size = units[-1]
        self.mu = nn.Linear(out_size, action_dim)
        self.log_std = nn.Linear(out_size, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.log_std.weight, gain=0.01)

    def forward(self, obs):
        x = self.actor_mlp(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        mu = torch.tanh(mu)
        return mu, log_std

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """采样动作（带 tanh squash）"""
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            z = mu
        else:
            z = dist.rsample()

        action = torch.tanh(z)

        # log_prob 计算（含 tanh 修正）
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, std

    def evaluate(self, obs):
        """评估：计算给定 obs 下当前策略的动作、log_prob"""
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, std

    @torch.no_grad()
    def act_inference(self, obs):
        """推理模式：确定性动作"""
        action, _, _, _ = self.act(obs, deterministic=True)
        return action
```

- [ ] **Step 2: 验证导入**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import torch
from gym_dcmm.algs.sac_dcmm.models_track import SACActor, QNetwork
actor = SACActor(obs_dim=18, action_dim=8, units=[256, 128])
q = QNetwork(obs_dim=18, action_dim=8, units=[256, 128])
obs = torch.randn(4, 18)
action, log_prob, mu, std = actor.act(obs)
print('action shape:', action.shape, 'log_prob shape:', log_prob.shape)
q_val = q(obs, action)
print('q_val shape:', q_val.shape)
# 验证 evaluate
a2, lp2, _, _ = actor.evaluate(obs)
print('evaluate action shape:', a2.shape)
print('OK')
"
# 预期输出: action shape: torch.Size([4, 8]) log_prob shape: torch.Size([4, 1])
#           q_val shape: torch.Size([4, 1]) / evaluate action shape: torch.Size([4, 8]) / OK
```

---

### Task 5: 创建 `models_catch.py`（SAC Catch 双分支网络）

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/models_catch.py`

- [ ] **Step 1: 实现 Catch 阶段的双分支 SAC Actor + TwinQ 网络**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/models_catch.py`：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
         for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class QNetwork(nn.Module):
    """Twin Q 网络：输入 (obs + action)，输出 Q 值"""

    def __init__(self, obs_dim, action_dim, units):
        super(QNetwork, self).__init__()
        input_dim = obs_dim + action_dim
        self.q_mlp = MLP(units=units, input_size=input_dim)
        out_size = units[-1]
        self.q_out = nn.Linear(out_size, 1)
        torch.nn.init.orthogonal_(self.q_out.weight, gain=1.0)
        if self.q_out.bias is not None:
            torch.nn.init.zeros_(self.q_out.bias)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = self.q_mlp(x)
        return self.q_out(x)


class SACActor(nn.Module):
    """
    双分支 SAC Actor：
    - tracking 分支：obs_t[0:obs_t_dim] → actor_mlp_t → mu_t(8D) + log_std_t(8D)  [冻结]
    - catching 分支：obs_c[2:]       → actor_mlp_c → mu_c(12D) + log_std_c(12D)  [训练]
    - 完整输出：concat(mu_t, mu_c) = 20D
    """
    def __init__(self, obs_dim, obs_t_dim, action_dim, units,
                 log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.action_dim = action_dim
        self.action_t_dim = action_dim - 12   # 8: base(2) + arm(6)
        self.action_c_dim = action_dim - 8    # 12: hand(12)

        self.actor_mlp_t = MLP(units=units, input_size=obs_t_dim - 12)  # 18-12=6: tracking obs without hand
        self.actor_mlp_c = MLP(units=units, input_size=obs_dim - 2)     # 30-2=28: full obs without base v

        out_size = units[-1]
        self.mu_t = nn.Linear(out_size, self.action_t_dim)
        self.mu_c = nn.Linear(out_size, self.action_c_dim)
        self.log_std_t = nn.Linear(out_size, self.action_t_dim)
        self.log_std_c = nn.Linear(out_size, self.action_c_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.orthogonal_(self.mu_t.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.mu_c.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.log_std_t.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.log_std_c.weight, gain=0.01)

    def forward(self, obs, obs_t, obs_c):
        """
        Args:
            obs:  完整观测 (30D)
            obs_t: tracking 分支输入 (18D, 即 obs[:,:-12])
            obs_c: catching 分支输入 (28D, 即 obs[:,2:])
        Returns:
            mu: 完整动作均值 (20D)
            log_std: 完整动作 log_std (20D)
        """
        x_t = self.actor_mlp_t(obs_t)
        x_c = self.actor_mlp_c(obs_c)

        mu_t = self.mu_t(x_t)
        mu_c = self.mu_c(x_c)
        log_std_t = self.log_std_t(x_t)
        log_std_c = self.log_std_c(x_c)

        log_std_t = torch.clamp(log_std_t, self.log_std_min, self.log_std_max)
        log_std_c = torch.clamp(log_std_c, self.log_std_min, self.log_std_max)

        mu_t = torch.tanh(mu_t)
        mu_c = torch.tanh(mu_c)

        mu = torch.cat([mu_t, mu_c], dim=1)
        log_std = torch.cat([log_std_t, log_std_c], dim=1)

        return mu, log_std

    @torch.no_grad()
    def act(self, obs, obs_t, obs_c, deterministic=False):
        mu, log_std = self.forward(obs, obs_t, obs_c)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            z = mu
        else:
            z = dist.rsample()

        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, std

    def evaluate(self, obs, obs_t, obs_c):
        mu, log_std = self.forward(obs, obs_t, obs_c)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, std

    @torch.no_grad()
    def act_inference(self, obs, obs_t, obs_c):
        action, _, _, _ = self.act(obs, obs_t, obs_c, deterministic=True)
        return action
```

- [ ] **Step 2: 验证导入**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import torch
from gym_dcmm.algs.sac_dcmm.models_catch import SACActor, QNetwork
actor = SACActor(obs_dim=30, obs_t_dim=18, action_dim=20, units=[256, 128])
q = QNetwork(obs_dim=30, action_dim=20, units=[256, 128])
obs = torch.randn(4, 30)
obs_t = obs[:, :-12]
obs_c = obs[:, 2:]
action, log_prob, mu, std = actor.act(obs, obs_t, obs_c)
print('action shape:', action.shape)   # torch.Size([4, 20])
print('mu_t part frozen check: mu[:,:8] shape:', mu[:, :8].shape)
q_val = q(obs, action)
print('q_val shape:', q_val.shape)     # torch.Size([4, 1])
print('OK')
"
# 预期输出: action shape: torch.Size([4, 20]) / mu[:,:8] shape: torch.Size([4, 8]) / q_val shape: torch.Size([4, 1]) / OK
```

---

### Task 6: 创建 `sac_dcmm_track.py`（SAC Track 训练器）

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_track.py`

- [ ] **Step 1: 实现 SAC_Track 完整训练器**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_track.py`：

```python
import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import time
import torch
import numpy as np
import wandb
from tensorboardX import SummaryWriter

from .replay_buffer import ReplayBuffer
from .models_track import SACActor, QNetwork
from .utils import AverageScalarMeter, RunningMeanStd


class SAC_Track(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.sac_config = full_config.train.sac

        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.sac_config['num_actors'])
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.call("act_t_dim")[0]
        print("actions_num: ", self.actions_num)
        self.obs_shape = (self.env.call("obs_t_dim")[0],)
        self.full_action_dim = self.env.call("act_c_dim")[0]

        # ---- SAC 超参数 ----
        self.gamma = self.sac_config['gamma']
        self.tau = self.sac_config['tau']
        self.reward_scale_value = self.sac_config['reward_scale_value']
        self.batch_size = self.sac_config['batch_size']
        self.start_steps = self.sac_config['start_steps']
        self.updates_per_step = self.sac_config['updates_per_step']
        self.auto_entropy_tuning = self.sac_config['auto_entropy_tuning']
        self.target_entropy_scale = self.sac_config['target_entropy_scale']
        self.truncate_grads = self.sac_config['truncate_grads']
        self.grad_norm = self.sac_config['grad_norm']
        self.normalize_input = self.sac_config['normalize_input']
        self.action_track_denorm = self.sac_config['action_track_denorm']
        self.action_catch_denorm = self.sac_config['action_catch_denorm']
        self.horizon_length = self.sac_config['horizon_length']

        # ---- Models ----
        net_units = self.network_config.mlp.units
        log_std_min = self.sac_config['log_std_min']
        log_std_max = self.sac_config['log_std_max']

        self.actor = SACActor(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        self.q1 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self.q1_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self._hard_update(self.q1_target, self.q1)
        self._hard_update(self.q2_target, self.q2)

        # ---- 温度参数 ----
        target_entropy = -self.actions_num * self.target_entropy_scale
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # ---- Observation Normalization ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)

        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Optimizers ----
        self.init_lr = float(self.sac_config['learning_rate'])
        self.alpha_lr = float(self.sac_config['alpha_lr'])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.init_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.init_lr, eps=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], self.alpha_lr, eps=1e-5)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
            capacity=self.sac_config['replay_buffer_size'],
            obs_dim=self.obs_shape[0],
            act_dim=self.actions_num,
            device=self.device,
        )

        # ---- Logging & Metrics ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer
        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(self.sac_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.sac_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.sac_config['test_num_episodes'])

        self.obs = None
        self.epoch_num = 0

        batch_size = self.num_actors
        self.current_rewards = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones(
            (batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.sac_config['max_agent_steps']
        self.max_test_steps = self.sac_config['max_test_steps']
        self.best_rewards = -10000

        self.data_collect_time = 0
        self.rl_train_time = 0

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, source_param in zip(
            target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * source_param.data)

    # ---- Core SAC Update ----
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)

        rewards_batch = rewards_batch * self.reward_scale_value

        # ---- Q 网络更新 ----
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.act(next_obs_batch)
            target_q1 = self.q1_target(next_obs_batch, next_action)
            target_q2 = self.q2_target(next_obs_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = target_q - self.alpha * next_log_prob
            q_target = rewards_batch + (1.0 - dones_batch) * self.gamma * target_value

        q1_pred = self.q1(obs_batch, actions_batch)
        q2_pred = self.q2(obs_batch, actions_batch)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, q_target.detach())
        q2_loss = torch.nn.functional.mse_loss(q2_pred, q_target.detach())
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.truncate_grads:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            torch.nn.utils.clip_grad_norm_(q_params, self.grad_norm)
        self.q_optimizer.step()

        # ---- Policy 更新 ----
        new_action, new_log_prob, _, _ = self.actor.evaluate(obs_batch)
        q1_new = self.q1(obs_batch, new_action)
        q2_new = self.q2(obs_batch, new_action)
        q_min = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        # ---- Alpha 更新 ----
        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (new_log_prob.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        # ---- 软更新 Target Q ----
        self._soft_update(self.q1_target, self.q1, self.tau)
        self._soft_update(self.q2_target, self.q2, self.tau)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item(),
        }

    # ---- 观测/动作转换（与 PPO 相同） ----
    def obs2tensor(self, obs):
        task = self.env.call('task')[0]
        if task == 'Catching':
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
                obs["hand"],
            ), axis=1)
        else:
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
            ), axis=1)
        return torch.tensor(obs_array, dtype=torch.float32).to(self.device)

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        task = self.env.call('task')[0]
        if task == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_track_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        return {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor,
        }

    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        if inference:
            actions = self.actor.act_inference(processed_obs)
            return {'actions': actions}
        else:
            action, log_prob, mu, sigma = self.actor.act(processed_obs)
            return {
                'actions': action,
                'log_probs': log_prob,
                'mus': mu,
                'sigmas': sigma,
            }

    # ---- 数据收集 ----
    def play_step(self, random_action=False):
        obs_tensor = self.obs['obs']
        next_obs_list = []
        action_list = []

        for i in range(self.num_actors):
            if random_action:
                action = torch.rand(self.actions_num, device=self.device) * 2.0 - 1.0
            else:
                single_obs = obs_tensor[i:i+1]
                processed_obs = self.running_mean_std(single_obs)
                action, _, _, _ = self.actor.act(processed_obs)
                action = action.squeeze(0)

            action = torch.clamp(action, -1.0, 1.0)
            action_list.append(action)

        actions_tensor = torch.stack(action_list)
        actions_padded = torch.nn.functional.pad(
            actions_tensor, (0, self.full_action_dim - self.actions_num), value=0)
        actions_dict = self.action2dict(actions_padded)

        obs, r, terminates, truncates, infos = self.env.step(actions_dict)
        next_obs_tensor = self.obs2tensor(obs)

        r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
        dones = terminates | truncates
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 存储 transition
        for i in range(self.num_actors):
            self.replay_buffer.add(
                obs_tensor[i],
                actions_tensor[i],
                r_tensor[i].unsqueeze(0),
                next_obs_tensor[i],
                dones_tensor[i].unsqueeze(0),
            )

        self.obs = {'obs': next_obs_tensor}

        self.current_rewards += r_tensor.unsqueeze(1)
        self.current_lengths += 1
        done_indices = dones_tensor.nonzero(as_tuple=False)
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        success_values = torch.tensor(
            truncates, dtype=torch.float32, device=self.device)[done_indices]
        self.episode_success.update(success_values)

        for k, v in infos.items():
            if isinstance(v, float) or isinstance(v, int) or \
               (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                self.extra_info[k] = v

        not_dones = 1.0 - dones_tensor.unsqueeze(1)
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * (1.0 - dones_tensor)

    # ---- 主训练循环 ----
    def set_eval(self):
        self.actor.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

    def set_train(self):
        self.actor.train()
        if self.normalize_input:
            self.running_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()

        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.num_actors

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            self.set_eval()

            _collect_t = time.time()
            for _ in range(self.horizon_length):
                random_action = self.agent_steps < self.start_steps
                self.play_step(random_action=random_action)
                self.agent_steps += self.num_actors

            self.data_collect_time += (time.time() - _collect_t)

            # SAC 更新
            _train_t = time.time()
            self.set_train()
            loss_info = None
            for _ in range(self.horizon_length * self.updates_per_step):
                loss_info = self.update()
            self.rl_train_time += (time.time() - _train_t)

            # 日志
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (self.horizon_length * self.num_actors) / (time.time() - _last_t)
            _last_t = time.time()

            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e3):04}K | '
                f'FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            print(info_string)

            self._write_stats(loss_info)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()

            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success, self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)

            ckpt_prefix = f"{self.env.call('object_motion')[0]}_track_sac"
            checkpoint_name = (
                f'{ckpt_prefix}_ep_{self.epoch_num}_'
                f'step_{int(self.agent_steps // 1e6):04}m_'
                f'reward_{mean_rewards:.2f}'
            )

            save_freq = self.sac_config['save_frequency']
            if save_freq > 0:
                if (self.epoch_num % save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                prev_best_ckpt = os.path.join(
                    self.nn_dir, f'{ckpt_prefix}_best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(
                    self.nn_dir, f'{ckpt_prefix}_best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def _write_stats(self, loss_info):
        if loss_info is None:
            return
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/q1_loss': loss_info['q1_loss'],
            'losses/q2_loss': loss_info['q2_loss'],
            'losses/policy_loss': loss_info['policy_loss'],
            'losses/alpha_loss': loss_info['alpha_loss'],
            'info/alpha': loss_info['alpha'],
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v
        wandb.log(log_dict, step=self.agent_steps)
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    # ---- Checkpoint ----
    def save(self, name):
        weights = {
            'model': self.actor.state_dict(),
            'actor_mlp_t': self.actor.actor_mlp.state_dict(),
            'mu_t': self.actor.mu.state_dict(),
            'log_std_t': self.actor.log_std.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
        }
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def _load_compatible_model_state(self, checkpoint_state):
        model_state = self.actor.state_dict()
        loaded, expanded, skipped = [], [], []
        for name, value in checkpoint_state.items():
            if name not in model_state:
                skipped.append(name)
                continue
            target = model_state[name]
            if value.shape == target.shape:
                model_state[name] = value
                loaded.append(name)
            elif value.ndim == target.ndim and value.ndim > 0 and \
                 value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                model_state[name] = merged
                expanded.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
            else:
                skipped.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
        self.actor.load_state_dict(model_state)
        if expanded:
            print("[checkpoint] Partially loaded resized tensors:")
            for item in expanded:
                print("  ", item)
        if skipped:
            print("[checkpoint] Skipped incompatible tensors:")
            for item in skipped:
                print("  ", item)

    # ---- 测试 ----
    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        test_steps = self.num_actors

        while test_steps < self.max_test_steps:
            self._play_test_steps()
            test_steps += self.horizon_length * self.num_actors
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print(f"## Sample Length {len(self.episode_test_rewards)} ##")
            print(f"mean_rewards: {mean_rewards}")
            print(f"mean_lengths: {mean_lengths}")
            print(f"mean_success: {mean_success}")

    def _play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            actions = res_dict['actions']
            actions = torch.clamp(actions, -1.0, 1.0)
            actions = torch.nn.functional.pad(
                actions, (0, self.full_action_dim - self.actions_num), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            self.obs = {'obs': self.obs2tensor(obs)}

            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
            self.current_rewards += r_tensor.unsqueeze(1)
            self.current_lengths += 1
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])
            success_values = torch.tensor(
                truncates, dtype=torch.float32, device=self.device)[done_indices]
            self.episode_test_success.update(success_values)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
```

- [ ] **Step 2: 验证导入无语法错误**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
from gym_dcmm.algs.sac_dcmm.sac_dcmm_track import SAC_Track
print('SAC_Track imported successfully')
"
# 预期输出: SAC_Track imported successfully
```

---

### Task 7: 创建 `sac_dcmm_catch_two_stage.py`（SAC Catch 训练器）

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_two_stage.py`

- [ ] **Step 1: 实现 SAC_Catch_TwoStage 训练器**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_two_stage.py`：

```python
import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import time
import torch
import numpy as np
import wandb
from tensorboardX import SummaryWriter

from .replay_buffer import ReplayBuffer
from .models_catch import SACActor, QNetwork
from .utils import AverageScalarMeter, RunningMeanStd


class SAC_Catch_TwoStage(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.sac_config = full_config.train.sac

        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.sac_config['num_actors'])
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.call("act_c_dim")[0]
        print("actions_num: ", self.actions_num)
        self.obs_shape = (self.env.call("obs_c_dim")[0],)
        self.obs_t_shape = (self.env.call("obs_t_dim")[0],)
        self.full_action_dim = self.env.call("act_c_dim")[0]

        # ---- SAC 超参数 ----
        self.gamma = self.sac_config['gamma']
        self.tau = self.sac_config['tau']
        self.reward_scale_value = self.sac_config['reward_scale_value']
        self.batch_size = self.sac_config['batch_size']
        self.start_steps = self.sac_config['start_steps']
        self.updates_per_step = self.sac_config['updates_per_step']
        self.auto_entropy_tuning = self.sac_config['auto_entropy_tuning']
        self.target_entropy_scale = self.sac_config['target_entropy_scale']
        self.truncate_grads = self.sac_config['truncate_grads']
        self.grad_norm = self.sac_config['grad_norm']
        self.normalize_input = self.sac_config['normalize_input']
        self.action_track_denorm = self.sac_config['action_track_denorm']
        self.action_catch_denorm = self.sac_config['action_catch_denorm']
        self.horizon_length = self.sac_config['horizon_length']

        # ---- Models ----
        net_units = self.network_config.mlp.units
        log_std_min = self.sac_config['log_std_min']
        log_std_max = self.sac_config['log_std_max']

        self.actor = SACActor(
            obs_dim=self.obs_shape[0],
            obs_t_dim=self.obs_t_shape[0],
            action_dim=self.actions_num,
            units=net_units,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        self.q1 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self.q1_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self._hard_update(self.q1_target, self.q1)
        self._hard_update(self.q2_target, self.q2)

        # ---- 温度参数 ----
        target_entropy = -self.actions_num * self.target_entropy_scale
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # ---- Observation Normalization ----
        self.running_mean_std_track = RunningMeanStd(self.obs_t_shape).to(self.device)
        self.running_mean_std_hand = RunningMeanStd((12,)).to(self.device)

        # ---- 加载 Tracking 模型 ----
        self.load_tracking_model(
            full_config.checkpoint_tracking, full_config.checkpoint_catching)

        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Optimizers ----
        self.init_lr = float(self.sac_config['learning_rate'])
        self.alpha_lr = float(self.sac_config['alpha_lr'])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.init_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.init_lr, eps=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], self.alpha_lr, eps=1e-5)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
            capacity=self.sac_config['replay_buffer_size'],
            obs_dim=self.obs_shape[0],
            act_dim=self.actions_num,
            device=self.device,
        )

        # ---- Logging & Metrics ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer
        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(
            self.sac_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(
            self.sac_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(
            self.sac_config['test_num_episodes'])

        self.obs = None
        self.epoch_num = 0

        batch_size = self.num_actors
        self.current_rewards = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones(
            (batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.sac_config['max_agent_steps']
        self.max_test_steps = self.sac_config['max_test_steps']
        self.best_rewards = -10000

        self.data_collect_time = 0
        self.rl_train_time = 0

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, source_param in zip(
            target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * source_param.data)

    # ---- 加载 Tracking 模型 ----
    def load_tracking_model(self, checkpoint_tracking, checkpoint_catching):
        print("### Start loading tracking model")
        if checkpoint_catching or not checkpoint_tracking:
            return
        tracking_checkpoint = torch.load(
            checkpoint_tracking, map_location=self.device)
        self.actor.actor_mlp_t.load_state_dict(
            tracking_checkpoint['actor_mlp_t'])
        self._load_compatible_module_state(
            self.actor.mu_t, tracking_checkpoint['mu_t'], 'mu_t')
        self._load_compatible_module_state(
            self.actor.log_std_t, tracking_checkpoint['log_std_t'], 'log_std_t')
        self.running_mean_std_track.load_state_dict(
            tracking_checkpoint['running_mean_std'])
        # 冻结 tracking 分支
        for param in self.actor.actor_mlp_t.parameters():
            param.requires_grad = False
        for param in self.actor.mu_t.parameters():
            param.requires_grad = False
        for param in self.actor.log_std_t.parameters():
            param.requires_grad = False
        print("### Done loading tracking model")

    def _load_compatible_module_state(self, module, checkpoint_state, label):
        module_state = module.state_dict()
        for name, value in checkpoint_state.items():
            if name not in module_state:
                continue
            target = module_state[name]
            if value.shape == target.shape:
                module_state[name] = value
            elif value.ndim == target.ndim and value.ndim > 0 and \
                 value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                module_state[name] = merged
            else:
                print(f"[checkpoint] Skipped {label}.{name}: "
                      f"{tuple(value.shape)} -> {tuple(target.shape)}")
        module.load_state_dict(module_state)

    # ---- SAC Update ----
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)

        rewards_batch = rewards_batch * self.reward_scale_value

        obs_t_batch = obs_batch[:, :-12]
        obs_c_batch = obs_batch[:, 2:]
        next_obs_t_batch = next_obs_batch[:, :-12]
        next_obs_c_batch = next_obs_batch[:, 2:]

        # ---- Q 网络更新 ----
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.act(
                next_obs_batch, next_obs_t_batch, next_obs_c_batch)
            target_q1 = self.q1_target(next_obs_batch, next_action)
            target_q2 = self.q2_target(next_obs_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = target_q - self.alpha * next_log_prob
            q_target = rewards_batch + \
                (1.0 - dones_batch) * self.gamma * target_value

        q1_pred = self.q1(obs_batch, actions_batch)
        q2_pred = self.q2(obs_batch, actions_batch)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, q_target.detach())
        q2_loss = torch.nn.functional.mse_loss(q2_pred, q_target.detach())
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.truncate_grads:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            torch.nn.utils.clip_grad_norm_(q_params, self.grad_norm)
        self.q_optimizer.step()

        # ---- Policy 更新 ----
        new_action, new_log_prob, _, _ = self.actor.evaluate(
            obs_batch, obs_t_batch, obs_c_batch)
        q1_new = self.q1(obs_batch, new_action)
        q2_new = self.q2(obs_batch, new_action)
        q_min = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        # ---- Alpha 更新 ----
        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (new_log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        # ---- 软更新 Target Q ----
        self._soft_update(self.q1_target, self.q1, self.tau)
        self._soft_update(self.q2_target, self.q2, self.tau)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item(),
        }

    # ---- 观测/动作转换 ----
    def obs2tensor(self, obs):
        task = self.env.call('task')[0]
        if task == 'Catching':
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
                obs["hand"],
            ), axis=1)
        else:
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
            ), axis=1)
        return torch.tensor(obs_array, dtype=torch.float32).to(self.device)

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        task = self.env.call('task')[0]
        if task == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_track_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        return {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor,
        }

    def model_act(self, obs_dict, inference=False):
        obs_tensor = obs_dict['obs']
        processed_obs_track = self.running_mean_std_track(obs_tensor[:, :-12])
        processed_obs_hand = self.running_mean_std_hand(obs_tensor[:, -12:])
        processed_obs = torch.cat((processed_obs_track, processed_obs_hand), dim=1)
        obs_t = processed_obs[:, :-12]
        obs_c = processed_obs[:, 2:]

        if inference:
            actions = self.actor.act_inference(processed_obs, obs_t, obs_c)
            return {'actions': actions}
        else:
            action, log_prob, mu, sigma = self.actor.act(processed_obs, obs_t, obs_c)
            return {
                'actions': action,
                'log_probs': log_prob,
                'mus': mu,
                'sigmas': sigma,
            }

    # ---- 数据收集 ----
    def play_step(self, random_action=False):
        obs_tensor = self.obs['obs']
        action_list = []

        for i in range(self.num_actors):
            if random_action:
                action = torch.rand(
                    self.actions_num, device=self.device) * 2.0 - 1.0
            else:
                single_obs = obs_tensor[i:i+1]
                p_track = self.running_mean_std_track(single_obs[:, :-12])
                p_hand = self.running_mean_std_hand(single_obs[:, -12:])
                p_obs = torch.cat((p_track, p_hand), dim=1)
                obs_t = p_obs[:, :-12]
                obs_c = p_obs[:, 2:]
                action, _, _, _ = self.actor.act(
                    p_obs, obs_t, obs_c)
                action = action.squeeze(0)

            action = torch.clamp(action, -1.0, 1.0)
            action_list.append(action)

        actions_tensor = torch.stack(action_list)
        actions_padded = torch.nn.functional.pad(
            actions_tensor,
            (0, self.full_action_dim - self.actions_num),
            value=0)
        actions_dict = self.action2dict(actions_padded)

        obs, r, terminates, truncates, infos = self.env.step(actions_dict)
        next_obs_tensor = self.obs2tensor(obs)

        r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
        dones = terminates | truncates
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=self.device)

        for i in range(self.num_actors):
            self.replay_buffer.add(
                obs_tensor[i],
                actions_tensor[i],
                r_tensor[i].unsqueeze(0),
                next_obs_tensor[i],
                dones_tensor[i].unsqueeze(0),
            )

        self.obs = {'obs': next_obs_tensor}

        self.current_rewards += r_tensor.unsqueeze(1)
        self.current_lengths += 1
        done_indices = dones_tensor.nonzero(as_tuple=False)
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        success_values = torch.tensor(
            truncates, dtype=torch.float32, device=self.device)[done_indices]
        self.episode_success.update(success_values)

        for k, v in infos.items():
            if isinstance(v, float) or isinstance(v, int) or \
               (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                self.extra_info[k] = v

        not_dones = 1.0 - dones_tensor.unsqueeze(1)
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * (1.0 - dones_tensor)

    # ---- 训练循环 ----
    def set_eval(self):
        self.actor.eval()
        if self.normalize_input:
            self.running_mean_std_track.eval()
            self.running_mean_std_hand.eval()

    def set_train(self):
        self.actor.train()
        if self.normalize_input:
            self.running_mean_std_track.train()
            self.running_mean_std_hand.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()

        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.num_actors

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            self.set_eval()

            _collect_t = time.time()
            for _ in range(self.horizon_length):
                random_action = self.agent_steps < self.start_steps
                self.play_step(random_action=random_action)
                self.agent_steps += self.num_actors

            self.data_collect_time += (time.time() - _collect_t)

            _train_t = time.time()
            self.set_train()
            loss_info = None
            for _ in range(self.horizon_length * self.updates_per_step):
                loss_info = self.update()
            self.rl_train_time += (time.time() - _train_t)

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (self.horizon_length * self.num_actors) / \
                (time.time() - _last_t)
            _last_t = time.time()

            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e3):04}K | '
                f'FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            print(info_string)

            self._write_stats(loss_info)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()

            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards,
                self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths,
                self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success,
                self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)

            ckpt_prefix = f"{self.env.call('object_motion')[0]}_catch_sac"
            checkpoint_name = (
                f'{ckpt_prefix}_ep_{self.epoch_num}_'
                f'step_{int(self.agent_steps // 1e6):04}m_'
                f'reward_{mean_rewards:.2f}'
            )

            save_freq = self.sac_config['save_frequency']
            if save_freq > 0:
                if (self.epoch_num % save_freq == 0) and \
                   (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                prev_best_ckpt = os.path.join(
                    self.nn_dir,
                    f'{ckpt_prefix}_best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(
                    self.nn_dir,
                    f'{ckpt_prefix}_best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def _write_stats(self, loss_info):
        if loss_info is None:
            return
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/q1_loss': loss_info['q1_loss'],
            'losses/q2_loss': loss_info['q2_loss'],
            'losses/policy_loss': loss_info['policy_loss'],
            'losses/alpha_loss': loss_info['alpha_loss'],
            'info/alpha': loss_info['alpha'],
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v
        wandb.log(log_dict, step=self.agent_steps)
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    # ---- Checkpoint ----
    def save(self, name):
        weights = {
            'model': self.actor.state_dict(),
        }
        if self.running_mean_std_track:
            weights['running_mean_std_track'] = \
                self.running_mean_std_track.state_dict()
        if self.running_mean_std_hand:
            weights['running_mean_std_hand'] = \
                self.running_mean_std_hand.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        if 'running_mean_std_track' in checkpoint:
            self.running_mean_std_track.load_state_dict(
                checkpoint['running_mean_std_track'])
        else:
            print("[checkpoint] running_mean_std_track not found")
        if 'running_mean_std_hand' in checkpoint:
            self.running_mean_std_hand.load_state_dict(
                checkpoint['running_mean_std_hand'])
        else:
            print("[checkpoint] running_mean_std_hand not found")

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        if self.normalize_input:
            if 'running_mean_std_track' in checkpoint:
                self.running_mean_std_track.load_state_dict(
                    checkpoint['running_mean_std_track'])
            if 'running_mean_std_hand' in checkpoint:
                self.running_mean_std_hand.load_state_dict(
                    checkpoint['running_mean_std_hand'])
        if not fn:
            return
        self._load_compatible_model_state(checkpoint['model'])

    def _load_compatible_model_state(self, checkpoint_state):
        model_state = self.actor.state_dict()
        loaded, expanded, skipped = [], [], []
        for name, value in checkpoint_state.items():
            if name not in model_state:
                skipped.append(name)
                continue
            target = model_state[name]
            if value.shape == target.shape:
                model_state[name] = value
                loaded.append(name)
            elif value.ndim == target.ndim and value.ndim > 0 and \
                 value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                model_state[name] = merged
                expanded.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
            else:
                skipped.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
        self.actor.load_state_dict(model_state)
        if expanded:
            print("[checkpoint] Partially loaded resized tensors:")
            for item in expanded:
                print("  ", item)
        if skipped:
            print("[checkpoint] Skipped incompatible tensors:")
            for item in skipped:
                print("  ", item)

    # ---- 测试 ----
    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        test_steps = self.num_actors

        while test_steps < self.max_test_steps:
            self._play_test_steps()
            test_steps += self.horizon_length * self.num_actors
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print(f"## Sample Length {len(self.episode_test_rewards)} ##")
            print(f"mean_rewards: {mean_rewards}")
            print(f"mean_lengths: {mean_lengths}")
            print(f"mean_success: {mean_success}")

    def _play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            actions = res_dict['actions']
            actions = torch.clamp(actions, -1.0, 1.0)
            actions = torch.nn.functional.pad(
                actions,
                (0, self.full_action_dim - self.actions_num),
                value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            self.obs = {'obs': self.obs2tensor(obs)}

            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
            self.current_rewards += r_tensor.unsqueeze(1)
            self.current_lengths += 1
            dones = terminates | truncates
            self.dones = torch.tensor(
                dones, dtype=torch.uint8, device=self.device)
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(
                self.current_rewards[done_indices])
            self.episode_test_lengths.update(
                self.current_lengths[done_indices])
            success_values = torch.tensor(
                truncates, dtype=torch.float32, device=self.device)[done_indices]
            self.episode_test_success.update(success_values)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
```

- [ ] **Step 2: 验证导入**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_two_stage import SAC_Catch_TwoStage
print('SAC_Catch_TwoStage imported successfully')
"
# 预期输出: SAC_Catch_TwoStage imported successfully
```

---

### Task 8: 创建 `sac_dcmm_catch_one_stage.py`（备选单阶段）

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_one_stage.py`

- [ ] **Step 1: 实现 SAC_Catch_OneStage 训练器**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/gym_dcmm/algs/sac_dcmm/sac_dcmm_catch_one_stage.py`：

```python
"""
SAC Catch OneStage — 单阶段训练（不分 track/catch），
使用完整的 30D 观测和 20D 动作空间。
"""

import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import time
import torch
import numpy as np
import wandb
from tensorboardX import SummaryWriter

from .replay_buffer import ReplayBuffer
from .models_track import SACActor, QNetwork
from .utils import AverageScalarMeter, RunningMeanStd


class SAC_Catch_OneStage(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.sac_config = full_config.train.sac

        self.env = env
        self.num_actors = int(self.sac_config['num_actors'])
        self.actions_num = self.env.call("act_c_dim")[0]
        self.obs_shape = (self.env.call("obs_c_dim")[0],)
        self.full_action_dim = self.env.call("act_c_dim")[0]

        self.gamma = self.sac_config['gamma']
        self.tau = self.sac_config['tau']
        self.reward_scale_value = self.sac_config['reward_scale_value']
        self.batch_size = self.sac_config['batch_size']
        self.start_steps = self.sac_config['start_steps']
        self.updates_per_step = self.sac_config['updates_per_step']
        self.auto_entropy_tuning = self.sac_config['auto_entropy_tuning']
        self.target_entropy_scale = self.sac_config['target_entropy_scale']
        self.truncate_grads = self.sac_config['truncate_grads']
        self.grad_norm = self.sac_config['grad_norm']
        self.normalize_input = self.sac_config['normalize_input']
        self.action_catch_denorm = self.sac_config['action_catch_denorm']
        self.horizon_length = self.sac_config['horizon_length']

        net_units = self.network_config.mlp.units
        log_std_min = self.sac_config['log_std_min']
        log_std_max = self.sac_config['log_std_max']

        self.actor = SACActor(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        self.q1 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self.q1_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self._hard_update(self.q1_target, self.q1)
        self._hard_update(self.q2_target, self.q2)

        target_entropy = -self.actions_num * self.target_entropy_scale
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)

        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        self.init_lr = float(self.sac_config['learning_rate'])
        self.alpha_lr = float(self.sac_config['alpha_lr'])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.init_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.init_lr, eps=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], self.alpha_lr, eps=1e-5)

        self.replay_buffer = ReplayBuffer(
            capacity=self.sac_config['replay_buffer_size'],
            obs_dim=self.obs_shape[0],
            act_dim=self.actions_num,
            device=self.device,
        )

        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer
        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(
            self.sac_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(
            self.sac_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(
            self.sac_config['test_num_episodes'])

        self.obs = None
        self.epoch_num = 0

        batch_size = self.num_actors
        self.current_rewards = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones(
            (batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.sac_config['max_agent_steps']
        self.max_test_steps = self.sac_config['max_test_steps']
        self.best_rewards = -10000

        self.data_collect_time = 0
        self.rl_train_time = 0

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, source_param in zip(
            target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * source_param.data)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)

        rewards_batch = rewards_batch * self.reward_scale_value

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.act(next_obs_batch)
            target_q1 = self.q1_target(next_obs_batch, next_action)
            target_q2 = self.q2_target(next_obs_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = target_q - self.alpha * next_log_prob
            q_target = rewards_batch + \
                (1.0 - dones_batch) * self.gamma * target_value

        q1_pred = self.q1(obs_batch, actions_batch)
        q2_pred = self.q2(obs_batch, actions_batch)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, q_target.detach())
        q2_loss = torch.nn.functional.mse_loss(q2_pred, q_target.detach())
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.truncate_grads:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            torch.nn.utils.clip_grad_norm_(q_params, self.grad_norm)
        self.q_optimizer.step()

        new_action, new_log_prob, _, _ = self.actor.evaluate(obs_batch)
        q1_new = self.q1(obs_batch, new_action)
        q2_new = self.q2(obs_batch, new_action)
        q_min = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha *
                (new_log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        self._soft_update(self.q1_target, self.q1, self.tau)
        self._soft_update(self.q2_target, self.q2, self.tau)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item(),
        }

    def obs2tensor(self, obs):
        obs_array = np.concatenate((
            obs["base"]["v_lin_2d"],
            obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
            obs["arm"]["ee_v_lin_3d"],
            obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
            obs["hand"],
        ), axis=1)
        return torch.tensor(obs_array, dtype=torch.float32).to(self.device)

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        base_tensor = actions[:, :2] * self.action_catch_denorm[0]
        arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
        hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        return {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor,
        }

    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        if inference:
            actions = self.actor.act_inference(processed_obs)
            return {'actions': actions}
        else:
            action, log_prob, mu, sigma = self.actor.act(processed_obs)
            return {
                'actions': action,
                'log_probs': log_prob,
                'mus': mu,
                'sigmas': sigma,
            }

    def play_step(self, random_action=False):
        obs_tensor = self.obs['obs']
        action_list = []

        for i in range(self.num_actors):
            if random_action:
                action = torch.rand(
                    self.actions_num, device=self.device) * 2.0 - 1.0
            else:
                single_obs = obs_tensor[i:i+1]
                processed_obs = self.running_mean_std(single_obs)
                action, _, _, _ = self.actor.act(processed_obs)
                action = action.squeeze(0)

            action = torch.clamp(action, -1.0, 1.0)
            action_list.append(action)

        actions_tensor = torch.stack(action_list)
        actions_dict = self.action2dict(actions_tensor)

        obs, r, terminates, truncates, infos = self.env.step(actions_dict)
        next_obs_tensor = self.obs2tensor(obs)

        r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
        dones = terminates | truncates
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=self.device)

        for i in range(self.num_actors):
            self.replay_buffer.add(
                obs_tensor[i],
                actions_tensor[i],
                r_tensor[i].unsqueeze(0),
                next_obs_tensor[i],
                dones_tensor[i].unsqueeze(0),
            )

        self.obs = {'obs': next_obs_tensor}

        self.current_rewards += r_tensor.unsqueeze(1)
        self.current_lengths += 1
        done_indices = dones_tensor.nonzero(as_tuple=False)
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        success_values = torch.tensor(
            truncates, dtype=torch.float32, device=self.device)[done_indices]
        self.episode_success.update(success_values)

        for k, v in infos.items():
            if isinstance(v, float) or isinstance(v, int) or \
               (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                self.extra_info[k] = v

        not_dones = 1.0 - dones_tensor.unsqueeze(1)
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * (1.0 - dones_tensor)

    def set_eval(self):
        self.actor.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

    def set_train(self):
        self.actor.train()
        if self.normalize_input:
            self.running_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()

        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.num_actors

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            self.set_eval()

            _collect_t = time.time()
            for _ in range(self.horizon_length):
                random_action = self.agent_steps < self.start_steps
                self.play_step(random_action=random_action)
                self.agent_steps += self.num_actors

            self.data_collect_time += (time.time() - _collect_t)

            _train_t = time.time()
            self.set_train()
            loss_info = None
            for _ in range(self.horizon_length * self.updates_per_step):
                loss_info = self.update()
            self.rl_train_time += (time.time() - _train_t)

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (self.horizon_length * self.num_actors) / \
                (time.time() - _last_t)
            _last_t = time.time()

            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e3):04}K | '
                f'FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            print(info_string)

            self._write_stats(loss_info)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()

            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards,
                self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths,
                self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success,
                self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)

            ckpt_prefix = f"{self.env.call('object_motion')[0]}_catch_sac_one"
            save_freq = self.sac_config['save_frequency']
            if save_freq > 0:
                if (self.epoch_num % save_freq == 0) and \
                   (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(
                        self.nn_dir,
                        f'{ckpt_prefix}_ep_{self.epoch_num}_reward_{mean_rewards:.2f}'))
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                prev_best_ckpt = os.path.join(
                    self.nn_dir,
                    f'{ckpt_prefix}_best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(
                    self.nn_dir,
                    f'{ckpt_prefix}_best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def _write_stats(self, loss_info):
        if loss_info is None:
            return
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/q1_loss': loss_info['q1_loss'],
            'losses/q2_loss': loss_info['q2_loss'],
            'losses/policy_loss': loss_info['policy_loss'],
            'losses/alpha_loss': loss_info['alpha_loss'],
            'info/alpha': loss_info['alpha'],
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v
        wandb.log(log_dict, step=self.agent_steps)
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def save(self, name):
        weights = {
            'model': self.actor.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
        }
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def _load_compatible_model_state(self, checkpoint_state):
        model_state = self.actor.state_dict()
        for name, value in checkpoint_state.items():
            if name not in model_state:
                continue
            target = model_state[name]
            if value.shape == target.shape:
                model_state[name] = value
            elif value.ndim == target.ndim and value.ndim > 0 and \
                 value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                model_state[name] = merged
        self.actor.load_state_dict(model_state)

    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        test_steps = self.num_actors

        while test_steps < self.max_test_steps:
            self._play_test_steps()
            test_steps += self.horizon_length * self.num_actors
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print(f"## Sample Length {len(self.episode_test_rewards)} ##")
            print(f"mean_rewards: {mean_rewards}")
            print(f"mean_lengths: {mean_lengths}")
            print(f"mean_success: {mean_success}")

    def _play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            actions = res_dict['actions']
            actions = torch.clamp(actions, -1.0, 1.0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            self.obs = {'obs': self.obs2tensor(obs)}

            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
            self.current_rewards += r_tensor.unsqueeze(1)
            self.current_lengths += 1
            dones = terminates | truncates
            self.dones = torch.tensor(
                dones, dtype=torch.uint8, device=self.device)
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(
                self.current_rewards[done_indices])
            self.episode_test_lengths.update(
                self.current_lengths[done_indices])
            success_values = torch.tensor(
                truncates, dtype=torch.float32, device=self.device)[done_indices]
            self.episode_test_success.update(success_values)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
```

- [ ] **Step 2: 验证导入**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_one_stage import SAC_Catch_OneStage
print('SAC_Catch_OneStage imported successfully')
"
# 预期输出: SAC_Catch_OneStage imported successfully
```

---

### Task 9: 创建 `DcmmSAC.yaml` 并修改 `config.yaml`

**Files:**
- Create: `f:/yjs/Catch/catch_it_sac/configs/train/DcmmSAC.yaml`
- Modify: `f:/yjs/Catch/catch_it_sac/configs/config.yaml`

- [ ] **Step 1: 创建 DcmmSAC.yaml**

将以下内容写入 `f:/yjs/Catch/catch_it_sac/configs/train/DcmmSAC.yaml`：

```yaml
seed: ${..seed}
algo: SAC

network:
  mlp:
    units: [256, 128]
  separate_value_mlp: True

sac:
  name: ${resolve_default:Dcmm,${...experiment}}
  # --- 通用参数 ---
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
  tau: 0.005
  alpha_lr: 3e-4
  target_entropy_scale: 1.0
  replay_buffer_size: 1000000
  batch_size: 256
  start_steps: 10000
  updates_per_step: 1
  log_std_min: -20
  log_std_max: 2
  auto_entropy_tuning: True
```

- [ ] **Step 2: 修改 config.yaml 的 defaults**

读取 `f:/yjs/Catch/catch_it_sac/configs/config.yaml`，将：

```yaml
defaults:
  - train: DcmmPPO
```

改为：

```yaml
defaults:
  - train: DcmmSAC
```

- [ ] **Step 3: 验证配置加载**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
with hydra.initialize_config_dir(config_dir='configs', version_base=None):
    cfg = hydra.compose(config_name='config', overrides=['task=Tracking', 'num_envs=1', 'test=True'])
    print('algo:', cfg.train.sac.get('algo', '?'))
    print('tau:', cfg.train.sac.tau)
    print('batch_size:', cfg.train.sac.batch_size)
    print('OK')
"
# 预期输出: algo: SAC / tau: 0.005 / batch_size: 256 / OK
```

---

### Task 10: 修改 `train_DCMM.py`

**Files:**
- Modify: `f:/yjs/Catch/catch_it_sac/train_DCMM.py`

- [ ] **Step 1: 替换 import 语句和类引用**

将 `f:/yjs/Catch/catch_it_sac/train_DCMM.py` 中的 4 行：

```python
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
```

改为：

```python
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_two_stage import SAC_Catch_TwoStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_one_stage import SAC_Catch_OneStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_track import SAC_Track
```

- [ ] **Step 2: 替换类选择逻辑**

将：

```python
PPO = PPO_Track if config.task == 'Tracking' else \
      PPO_Catch_TwoStage if config.task == 'Catching_TwoStage' else \
      PPO_Catch_OneStage
agent = PPO(env, output_dif, full_config=config)
```

改为：

```python
SAC = SAC_Track if config.task == 'Tracking' else \
      SAC_Catch_TwoStage if config.task == 'Catching_TwoStage' else \
      SAC_Catch_OneStage
agent = SAC(env, output_dif, full_config=config)
```

- [ ] **Step 3: 验证完整导入链路**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_two_stage import SAC_Catch_TwoStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_catch_one_stage import SAC_Catch_OneStage
from gym_dcmm.algs.sac_dcmm.sac_dcmm_track import SAC_Track
print('All imports OK')
print('SAC_Track:', SAC_Track)
print('SAC_Catch_TwoStage:', SAC_Catch_TwoStage)
print('SAC_Catch_OneStage:', SAC_Catch_OneStage)
"
# 预期输出: All imports OK + 三个类对象
```

---

### Task 11: 最终验证 — 合成导入 + smoketest

**Files:** 无新建，验证已有文件

- [ ] **Step 1: 测试 train_DCMM.py 能否被 Python 解析（不实际运行训练）**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import py_compile
py_compile.compile('train_DCMM.py', doraise=True)
print('train_DCMM.py syntax OK')
"
# 预期输出: train_DCMM.py syntax OK
```

- [ ] **Step 2: 测试环境创建 + Agent 初始化（快速 smoketest）**

```bash
cd f:/yjs/Catch/catch_it_sac && python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
import gymnasium as gym
import gym_dcmm
from gym_dcmm.algs.sac_dcmm.sac_dcmm_track import SAC_Track

with hydra.initialize_config_dir(config_dir='configs', version_base=None):
    cfg = hydra.compose(config_name='config', overrides=[
        'task=Tracking', 'num_envs=4', 'test=True',
        'device_id=-1', 'object_motion=throw', 'wandb_mode=disabled',
    ])
    env = gym.make_vec('gym_dcmm/DcmmVecWorld-v0', num_envs=4,
        task='Tracking', camera_name=['top'],
        render_per_step=False, render_mode='rgb_array',
        object_name='object', img_size=[112, 112],
        imshow_cam=False, viewer=False,
        print_obs=False, print_info=False,
        print_reward=False, print_ctrl=False,
        print_contacts=False, object_eval=False,
        env_time=2.5, steps_per_policy=20, object_motion='throw')
    agent = SAC_Track(env, 'outputs/test_smoke', full_config=cfg)
    print('Smoke test PASSED: Agent initialized successfully')
    env.close()
"
# 预期输出 (末尾): Smoke test PASSED: Agent initialized successfully
```

- [ ] **Step 3: 记录进度**

将以下内容追加到 `f:/yjs/Catch/catch_it_sac/进度.txt`：

```text
7.30:
创建 catch_it_sac：将 PPO 替换为 SAC 算法
  - 新增 gym_dcmm/algs/sac_dcmm/ 包（replay_buffer, models_track, models_catch, sac_dcmm_track/catch）
  - 新增 configs/train/DcmmSAC.yaml
  - 修改 configs/config.yaml（defaults → DcmmSAC）
  - 修改 train_DCMM.py（import SAC 类）
  - 删除 gym_dcmm/algs/ppo_dcmm/
待验证：
  - SAC Tracking 训练效果
  - SAC Catching 训练效果
  - 与 PPO 效果对比
```
```

---

## 自审清单

1. **Spec 覆盖**：设计文档每一节均有对应任务 — 目录结构(Task1) / 网络架构(Task4-5) / ReplayBuffer(Task3) / 训练流程(Task6-8) / 配置(Task9) / train_DCMM(Task10) ✓
2. **无占位符**：所有任务包含完整的代码和命令，无 TBD/TODO/占位符 ✓
3. **类型一致性**：SACActor/QNetwork/ReplayBuffer 的接口在 Task3-5 中定义，Task6-8 中使用的参数名和方法签名完全一致 ✓

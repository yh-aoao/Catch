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

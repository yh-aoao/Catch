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

        self.actor_mlp_t = MLP(units=units, input_size=obs_t_dim)  # 18: tracking obs (full obs without hand)
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

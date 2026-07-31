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

"""Simulation-independent student, observation/action contracts and replay."""
import numpy as np
import torch
from torch import nn

TASK_IDS = {"throw": 0, "roll": 1, "bounce": 2}
SCHEMA = "state18_previous_command8_task3_track_v1"


def observation_array(obs):
    parts = [obs['base']['v_lin_2d'], obs['arm']['ee_pos3d'],
             obs['arm']['ee_quat'], obs['arm']['ee_v_lin_3d'],
             obs['object']['pos3d'], obs['object']['v_lin_3d']]
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != 18 or not np.isfinite(result).all():
        raise ValueError('Expected finite 18D Tracking observation')
    return result


def student_input(obs, previous, task):
    state = observation_array(obs)
    label = np.zeros((*state.shape[:-1], 3), dtype=np.float32)
    label[..., TASK_IDS[task]] = 1
    return np.concatenate([state, previous, label], axis=-1)


class ActionAdapter:
    """Track commands: base2 + arm6; hands receive zero commands as in PPO_Track."""
    def __init__(self, config):
        mapping = config.get('action_indices')
        if mapping is None:
            raise ValueError('Unresolved track action mapping: the 6D throw teacher needs its original controller; do not guess or tail-pad.')
        self.indices = list(mapping)
        if (not self.indices or len(set(self.indices)) != len(self.indices) or
                any(type(i) is not int or i < 0 or i >= 8 for i in self.indices)):
            raise ValueError('action_indices must be unique indices in [0,8)')
        if len(self.indices) != config['expected_action_dim']:
            raise ValueError('Mapping length differs from teacher action dimension')
        self.mask = np.zeros(8, np.float32)
        self.mask[self.indices] = 1
        self.scale = np.repeat(np.asarray(config['action_scales'][:2], np.float32), [2, 6])
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Invalid action scales')

    def canonical(self, native):
        if native.shape[-1] != len(self.indices) or not np.isfinite(native).all():
            raise ValueError('Teacher output does not match mapping')
        result = np.zeros((*native.shape[:-1], 8), np.float32)
        result[..., self.indices] = np.clip(native, -1, 1)
        return result

    def executable(self, command):
        if command.shape[-1] != 8 or not np.isfinite(command).all():
            raise ValueError('Invalid track command')
        return np.clip(command, -1, 1) * self.mask

    def action_dict(self, command):
        physical = self.executable(command) * self.scale
        return {'base': physical[..., :2], 'arm': physical[..., 2:8],
                'hand': np.zeros((*physical.shape[:-1], 12), np.float32)}


class Student(nn.Module):
    def __init__(self, hidden=(256, 256, 128)):
        super().__init__()
        self.hidden = list(hidden)
        self.register_buffer('mean', torch.zeros(18))
        self.register_buffer('std', torch.ones(18))
        self.register_buffer('normalizer_fitted', torch.tensor(False))
        layers, n = [], 29
        for width in hidden:
            layers.extend([nn.Linear(n, width), nn.ELU()])
            n = width
        layers.extend([nn.Linear(n, 8), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalizer(self, balanced_states):
        if self.normalizer_fitted.item():
            raise RuntimeError('Student normalizer is frozen after calibration')
        self.mean.copy_(balanced_states[:, :18].mean(0))
        self.std.copy_(balanced_states[:, :18].std(0, unbiased=False).clamp_min(1e-3))
        self.normalizer_fitted.fill_(True)

    def forward(self, x):
        state = ((x[..., :18] - self.mean) / self.std).clamp(-5, 5)
        return self.net(torch.cat([state, x[..., 18:]], -1))


def imitation_loss(pred, target, mask):
    """Call on an equally sized batch per task; average each valid action group."""
    error = (pred - target).square() * mask
    losses = []
    for start, end in [(0, 2), (2, 8)]:
        count = mask[:, start:end].sum(-1)
        if (count == 0).any():
            raise ValueError('Every task needs at least one supervised channel per group')
        losses.append(error[:, start:end].sum(-1) / count)
    return torch.stack(losses).mean()


class Replay:
    """Per-task ring buffers, balanced sampling, CPU storage."""
    def __init__(self, tasks, capacity):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity = capacity
        self.data = {task: {'x': torch.empty(capacity, 29), 'y': torch.empty(capacity, 8),
                            'mask': torch.empty(capacity, 8), 'size': 0, 'pos': 0}
                     for task in tasks}

    def add(self, task, x, y, mask):
        d = self.data[task]
        x, y = torch.as_tensor(x).cpu(), torch.as_tensor(y).cpu()
        mask = torch.as_tensor(mask).cpu().expand(len(x), -1)
        for i in range(len(x)):
            for key, value in [('x', x), ('y', y), ('mask', mask)]:
                d[key][d['pos']].copy_(value[i])
            d['pos'] = (d['pos'] + 1) % self.capacity
            d['size'] = min(d['size'] + 1, self.capacity)

    def sample(self, per_task, device):
        rows = []
        for d in self.data.values():
            if not d['size']:
                raise ValueError('Every task must have replay data')
            idx = torch.randint(d['size'], (per_task,))
            rows.append({key: d[key][idx] for key in ('x', 'y', 'mask')})
        return tuple(torch.cat([r[key] for r in rows]).to(device) for key in ('x', 'y', 'mask'))

    def state_dict(self):
        return {t: {k: v[:d['size']].clone() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()} for t, d in self.data.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.data):
            raise ValueError('Replay tasks changed')
        for task, source in state.items():
            d = self.data[task]
            if not 0 < source['size'] <= self.capacity:
                raise ValueError('Replay capacity changed or checkpoint is empty')
            for key in ('x', 'y', 'mask'):
                d[key][:source['size']].copy_(source[key])
            d['size'], d['pos'] = source['size'], source['pos']

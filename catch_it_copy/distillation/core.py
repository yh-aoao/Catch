"""Simulation-independent student, observation/action contracts and replay."""
import numpy as np
import torch
from torch import nn

TASK_IDS = {"throw": 0, "roll": 1, "bounce": 2}
SCHEMA = "state30_previous_command20_task3_v1"


def observation_array(obs):
    parts = [obs['base']['v_lin_2d'], obs['arm']['ee_pos3d'],
             obs['arm']['ee_quat'], obs['arm']['ee_v_lin_3d'],
             obs['object']['pos3d'], obs['object']['v_lin_3d'], obs['hand']]
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != 30 or not np.isfinite(result).all():
        raise ValueError('Expected finite 30D Catching observation')
    return result


def student_input(obs, previous, task):
    state = observation_array(obs)
    label = np.zeros((*state.shape[:-1], 3), dtype=np.float32)
    label[..., TASK_IDS[task]] = 1
    return np.concatenate([state, previous, label], axis=-1)


class ActionAdapter:
    """Canonical normalized command: base2, arm6 (zxy rotation), hand12.

    Each task retains its original physical scale/controller. Task-conditioned
    normalized commands are not claimed to share a physical scale across tasks.
    """
    def __init__(self, config):
        mapping = config.get('action_indices')
        if mapping is None:
            raise ValueError('Unresolved action mapping: recover the original throw '
                             'controller and configure action_indices; never tail-pad 18D actions.')
        self.indices = list(mapping)
        if (len(set(self.indices)) != len(self.indices) or
                any(type(i) is not int or i < 0 or i >= 20 for i in self.indices)):
            raise ValueError('action_indices must be unique canonical indices in [0,20)')
        if self.indices[-12:] != list(range(8, 20)):
            raise ValueError('The twelve hand channels must map to canonical indices 8..19')
        self.mask = np.zeros(20, dtype=np.float32)
        self.mask[self.indices] = 1
        self.defaults = np.asarray(config.get('action_defaults', [0.] * 20), np.float32)
        self.scale = np.repeat(np.asarray(config['action_scales'], np.float32), [2, 6, 12])
        if (self.defaults.shape != (20,) or not np.isfinite(self.defaults).all() or
                np.max(np.abs(self.defaults)) > 1 or not np.isfinite(self.scale).all() or
                np.any(self.scale <= 0)):
            raise ValueError('Invalid action defaults/scales')

    def canonical(self, native):
        if native.shape[-1] != len(self.indices):
            raise ValueError('Teacher output and configured mapping disagree')
        result = np.broadcast_to(self.defaults, (*native.shape[:-1], 20)).copy()
        result[..., self.indices] = np.clip(native, -1, 1)
        return result

    def executable(self, command):
        if command.shape[-1] != 20 or not np.isfinite(command).all():
            raise ValueError('Invalid command')
        return np.clip(command, -1, 1) * self.mask + self.defaults * (1 - self.mask)

    def action_dict(self, command):
        physical = self.executable(command) * self.scale
        return {'base': physical[..., :2], 'arm': physical[..., 2:8], 'hand': physical[..., 8:]}


class Student(nn.Module):
    def __init__(self, hidden=(256, 256, 128)):
        super().__init__()
        self.hidden = list(hidden)
        self.register_buffer('mean', torch.zeros(30))
        self.register_buffer('std', torch.ones(30))
        self.register_buffer('normalizer_fitted', torch.tensor(False))
        layers, n = [], 53
        for width in hidden:
            layers.extend([nn.Linear(n, width), nn.ELU()])
            n = width
        layers.extend([nn.Linear(n, 20), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalizer(self, balanced_states):
        if self.normalizer_fitted.item():
            raise RuntimeError('Student normalizer is frozen after calibration')
        self.mean.copy_(balanced_states[:, :30].mean(0))
        self.std.copy_(balanced_states[:, :30].std(0, unbiased=False).clamp_min(1e-3))
        self.normalizer_fitted.fill_(True)

    def forward(self, x):
        state = ((x[..., :30] - self.mean) / self.std).clamp(-5, 5)
        return self.net(torch.cat([state, x[..., 30:]], -1))


def imitation_loss(pred, target, mask):
    """Call on an equally sized batch per task; average each valid action group."""
    error = (pred - target).square() * mask
    losses = []
    for start, end in [(0, 2), (2, 8), (8, 20)]:
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
        self.data = {task: {'x': torch.empty(capacity, 53), 'y': torch.empty(capacity, 20),
                            'mask': torch.empty(capacity, 20), 'size': 0, 'pos': 0}
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

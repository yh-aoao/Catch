"""Student-only contract: depth, proprioception, previous command, recurrent state.

No task identifier, object state, terminal info or teacher features enter forward().
Full episodes are replayed from zero recurrent state; padding never contributes loss.
"""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'distillation_moe_v1'


def camera_specs(vision):
    specs = vision.get('cameras', [{'name': vision['camera'], 'near': vision['near'], 'far': vision['far']}])
    names = [s['name'] for s in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError('Camera names must be nonempty and unique; order defines input channels')
    if any(not 0 < s['near'] < s['far'] for s in specs):
        raise ValueError('Each camera requires 0 < near < far')
    return specs


def dimensions(stage):
    if stage not in ('track', 'catch'):
        raise ValueError('stage must be track or catch')
    return (12, 8) if stage == 'track' else (24, 20)


def proprioception(obs, stage):
    parts = [obs['base']['v_lin_2d'], obs['arm']['ee_pos3d'],
             obs['arm']['ee_quat'], obs['arm']['ee_v_lin_3d']]
    if stage == 'catch':
        parts.append(obs['hand'])
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != dimensions(stage)[0] or not np.isfinite(result).all():
        raise ValueError('Invalid robot-only observation')
    return result


def encode_depth(depth_metres, near, far):
    """Global, fixed metric-depth scaling. Never normalize each frame separately."""
    if not 0 < near < far:
        raise ValueError('Depth clipping requires 0 < near < far')
    depth = np.asarray(depth_metres, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('Renderer must return a finite nonnegative HxW metric depth image')
    return ((np.clip(depth, near, far) - near) / (far - near))[None].astype(np.float16)


class Student(nn.Module):
    def __init__(self, stage, vision, model, auxiliary):
        super().__init__()
        self.proprio_dim, self.action_dim = dimensions(stage)
        self.camera_names = [s['name'] for s in camera_specs(vision)]
        self.image_shape = (len(self.camera_names), vision['height'], vision['width'])
        self.hidden_dim = model['gru_hidden']
        self.register_buffer('mean', torch.zeros(self.proprio_dim))
        self.register_buffer('std', torch.ones(self.proprio_dim))
        self.register_buffer('normalizer_fitted', torch.tensor(False))
        self.encoders = nn.ModuleList()
        for _ in self.camera_names:
            layers, channels = [], 1
            for out, kernel, stride in zip(model['cnn_channels'], model['cnn_kernels'], model['cnn_strides']):
                layers += [nn.Conv2d(channels, out, kernel, stride), nn.ELU()]
                channels = out
            layers += [nn.MaxPool2d(2), nn.Flatten()]
            cnn = nn.Sequential(*layers)
            with torch.no_grad():
                flattened = cnn(torch.zeros(1, 1, *self.image_shape[1:])).shape[-1]
            self.encoders.append(nn.Sequential(cnn, nn.Linear(flattened, model['embedding']), nn.ELU()))
        self.gru = nn.GRU(len(self.camera_names)*model['embedding'] + self.proprio_dim + self.action_dim,
                          self.hidden_dim, batch_first=True)
        head, width = [], self.hidden_dim
        for out in model['mlp_hidden']:
            head += [nn.Linear(width, out), nn.ELU()]
            width = out
        head += [nn.Linear(width, self.action_dim)]
        self.head = nn.Sequential(*head)
        self.aux_head = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.ELU(), nn.Linear(128, 6))
        count, width = model['expert_count'], model['expert_width']
        if count < 2 or width < 1:
            raise ValueError('MoE requires >=2 experts and positive expert width')
        self.router = nn.Linear(self.hidden_dim, count)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim,width),nn.ELU(),nn.Linear(width,self.hidden_dim)) for _ in range(count)])
        self.auxiliary = auxiliary

    def initial_state(self, batch):
        return self.mean.new_zeros(1, batch, self.hidden_dim)

    @torch.no_grad()
    def fit_normalizer(self, states):
        if self.normalizer_fitted.item():
            raise RuntimeError('Proprioception normalizer is frozen after calibration')
        states = torch.as_tensor(states, dtype=torch.float32, device=self.mean.device)
        if states.ndim != 2 or states.shape[-1] != self.proprio_dim or not torch.isfinite(states).all():
            raise ValueError('Invalid calibration states')
        self.mean.copy_(states.mean(0))
        self.std.copy_(states.std(0, unbiased=False).clamp_min(1e-3))
        self.normalizer_fitted.fill_(True)

    def forward(self, depth, proprio, previous, hidden=None):
        return self._forward(depth, proprio, previous, hidden, False)

    def forward_aux(self, depth, proprio, previous, hidden=None):
        return self._forward(depth, proprio, previous, hidden, True)

    def _forward(self, depth, proprio, previous, hidden, auxiliary):
        """B,T,C,H,W and B,T,D inputs; return actions, hidden, pre-tanh logits."""
        b, t = depth.shape[:2]
        if tuple(depth.shape[2:]) != self.image_shape:
            raise ValueError('Image shape differs from checkpoint camera contract')
        if proprio.shape != (b, t, self.proprio_dim) or previous.shape != (b, t, self.action_dim):
            raise ValueError('Student accepts only robot proprioception and previous actions')
        frames = depth.reshape(b*t, *self.image_shape).float()
        features = torch.cat([encoder(frames[:, i:i+1]) for i, encoder in enumerate(self.encoders)], -1)
        features = features.reshape(b, t, -1)
        state = ((proprio.float() - self.mean) / self.std).clamp(-5, 5)
        recurrent, hidden = self.gru(torch.cat([features, state, previous.float()], -1), hidden)
        weights = self.router(recurrent).softmax(-1)
        self.router_weights = weights
        expert_features = torch.stack([expert(recurrent) for expert in self.experts], -2)
        fused = recurrent + (expert_features * weights[...,None]).sum(-2)
        logits = self.head(fused)
        if auxiliary:
            return logits.tanh(), hidden, logits, self.aux_head(recurrent)
        return logits.tanh(), hidden, logits


def sequence_loss(logits, target, valid, loss_type='bce', episode_lengths=None):
    # sigmoid(2*z) == (1+tanh(z))/2; stable BCE for bounded continuous targets.
    if loss_type == 'bce':
        error = 2 * F.binary_cross_entropy_with_logits(2 * logits, (target + 1) / 2, reduction='none')
    elif loss_type == 'mse':
        error = (logits.tanh() - target).square()
    else:
        raise ValueError('Unknown imitation loss')
    groups = [(0, 2), (2, 8)] + ([(8, 20)] if target.shape[-1] == 20 else [])
    per_step = torch.stack([error[..., a:b].mean(-1) for a, b in groups]).mean(0)
    # Equal episodes per task, then equal task weight regardless of episode length.
    lengths = valid.sum(1) if episode_lengths is None else episode_lengths
    per_episode = (per_step * valid).sum(1) / lengths.clamp_min(1)
    return per_episode.mean()


class EpisodeReplay:
    keys = ('depth', 'proprio', 'previous', 'target', 'object_target', 'object_valid')

    def __init__(self, tasks, capacity_frames):
        self.capacity = capacity_frames
        self.data = {task: deque() for task in tasks}
        self.sizes = {task: 0 for task in tasks}

    def add(self, task, rows):
        episode = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in self.keys}
        self.add_episode(task, episode)

    def add_episode(self, task, episode):
        length = len(episode['depth'])
        if not 0 < length <= self.capacity:
            raise ValueError('Episode is empty or exceeds replay capacity')
        if any(len(episode[k]) != length for k in self.keys):
            raise ValueError('Inconsistent episode lengths')
        if torch.count_nonzero(episode['previous'][0]):
            raise ValueError('Replay must begin at episode reset with zero previous action')
        while self.sizes[task] + length > self.capacity:
            self.sizes[task] -= len(self.data[task].popleft()['depth'])
        self.data[task].append(episode)
        self.sizes[task] += length

    def ready(self):
        return all(self.data.values())

    def sample(self, per_task, device):
        if not self.ready():
            raise RuntimeError('Each task needs a completed episode before training')
        episodes = [random.choice(self.data[t]) for t in self.data for _ in range(per_task)]
        length = max(len(e['depth']) for e in episodes)
        batch = {}
        for key in self.keys:
            example = episodes[0][key]
            padded = torch.zeros(len(episodes), length, *example.shape[1:], dtype=example.dtype)
            for i, episode in enumerate(episodes):
                padded[i, :len(episode[key])] = episode[key]
            batch[key] = padded.to(device)
        batch['valid'] = torch.zeros(len(episodes), length, device=device)
        for i, episode in enumerate(episodes):
            batch['valid'][i, :len(episode['depth'])] = 1
        return batch

    def state_dict(self):
        return {t: list(episodes) for t, episodes in self.data.items()}

    def load_state_dict(self, data):
        if set(data) != set(self.data):
            raise ValueError('Replay task set changed')
        for task, episodes in data.items():
            for episode in episodes:
                self.add_episode(task, episode)


def auxiliary_losses(prediction, target, mask, valid, config, lengths):
    """Normalized Smooth-L1; equal episode weight; padding/reset velocity excluded.

    Position is the noiseless environment relative position; velocity is its
    finite difference, not an absolute world velocity. No target enters forward.
    """
    scale = prediction.new_tensor(config['scales'])
    safe_target = torch.where(mask.bool(), target, torch.zeros_like(target))
    error = F.smooth_l1_loss(prediction / scale, safe_target / scale, reduction='none')
    result = {}
    for name, sl in [('position', slice(0, 3)), ('velocity', slice(3, 6))]:
        per_step = (error[..., sl] * mask[..., sl]).mean(-1)
        result[name] = ((per_step * valid).sum(1) / lengths.clamp_min(1)).mean()
    return result

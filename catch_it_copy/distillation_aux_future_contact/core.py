"""Observation-only gated Future + Contact fusion; privileged labels train heads only."""
from distillation_visual_aux.core import *
from distillation_visual_aux.core import Student as BaseStudent
from visual_module_experiments.core import EpisodeReplay
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'distillation_aux_future_contact_v1'


class Student(BaseStudent):
    variant = 'future_contact'

    def __init__(self, stage, vision, model, auxiliary):
        super().__init__(stage, vision, model, auxiliary)
        h = self.hidden_dim
        self.forecast = nn.Sequential(nn.Linear(h, 128), nn.ELU(), nn.Linear(128, 9))
        self.fusion = nn.Linear(9, h)
        self.phase = nn.Sequential(nn.Linear(h, 64), nn.ELU(), nn.Linear(64, 3))
        self.modulation = nn.Linear(3, 2*h)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.gate = nn.Sequential(nn.Linear(h, 64), nn.ELU(), nn.Linear(64, 2))
        # Initially both branches contribute with weight 0.5; neither is task-bound.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.extra = {}

    def _forward(self, depth, proprio, previous, hidden, auxiliary):
        b, t = depth.shape[:2]
        if tuple(depth.shape[2:]) != self.image_shape:
            raise ValueError('Image/camera contract changed')
        if proprio.shape != (b,t,self.proprio_dim) or previous.shape != (b,t,self.action_dim):
            raise ValueError('Only robot proprioception and previous commands are policy inputs')
        frames = depth.reshape(b*t,*self.image_shape).float()
        visual = torch.cat([encoder(frames[:,i:i+1])
                            for i,encoder in enumerate(self.encoders)], -1).reshape(b,t,-1)
        state = ((proprio.float()-self.mean)/self.std).clamp(-5,5)
        recurrent, hidden = self.gru(torch.cat([visual,state,previous.float()],-1), hidden)
        future = self.forecast(recurrent)
        phase = self.phase(recurrent)
        scale, shift = self.modulation(phase.sigmoid()).chunk(2,-1)
        contact_delta = .1*recurrent*scale.tanh() + .1*shift.tanh()
        gates = self.gate(recurrent).sigmoid()
        features = recurrent + gates[...,:1]*self.fusion(future) + gates[...,1:]*contact_delta
        self.extra = dict(future=future, contact=phase, gates=gates)
        logits = self.head(features)
        if auxiliary:
            return logits.tanh(), hidden, logits, self.aux_head(recurrent)
        return logits.tanh(), hidden, logits

    def extra_loss(self, batch, begin, end, lengths):
        valid = batch['valid'][:,begin:end]
        def reduce(error, mask):
            return ((error*mask).sum(1)/lengths.clamp_min(1)).mean()
        future_error = F.smooth_l1_loss(
            self.extra['future'].reshape(*valid.shape,3,3),
            batch['future_target'][:,begin:end], reduction='none').mean(-1)
        future_mask = batch['future_valid'][:,begin:end]*valid[...,None]
        contact_error = F.binary_cross_entropy_with_logits(
            self.extra['contact'], batch['contact_target'][:,begin:end], reduction='none')
        contact_mask = batch['contact_valid'][:,begin:end]*valid[...,None]
        # Preserve each original branch's 0.1 loss weight, independently of its gate.
        return (.1*sum(reduce(future_error[...,k],future_mask[...,k]) for k in range(3))/3
                + .1*sum(reduce(contact_error[...,k],contact_mask[...,k]) for k in range(3))/3)

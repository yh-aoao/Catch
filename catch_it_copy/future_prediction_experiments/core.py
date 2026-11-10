"""Future prediction with optional learned diagonal-Gaussian uncertainty."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from distillation_visual_aux.core import Student as BaseStudent, EpisodeReplay as BaseReplay
from visual_module_experiments.core import Student as OriginalStudent


def phase_labels(episode):
    """Offline diagnostic only: past vertical reversal is a rebound PROXY, not contact truth."""
    n = len(episode['depth'])
    phase = torch.zeros(n, dtype=torch.long)
    z = episode['world_position'][:,2]
    contact = episode['contact_after'].reshape(n).bool()
    rebound = False
    hand_seen = False
    for j in range(n):
        if j >= 2 and z[j-1]-z[j-2] < -1e-4 and z[j]-z[j-1] > 1e-4:
            rebound = True
        if j > 0:
            hand_seen = hand_seen or bool(contact[j-1])
        phase[j] = 3 if hand_seen else (2 if episode['object_target'][j,:3].norm() < .15 else (1 if rebound else 0))
    return phase


class EpisodeReplay(BaseReplay):
    keys = BaseReplay.keys + ('world_position','visible','uv','contact_after')

    def __init__(self, tasks, capacity_frames, horizons=(1,3,5)):
        super().__init__(tasks, capacity_frames)
        self.horizons = tuple(horizons)

    def add_episode(self, task, episode):
        n = len(episode['depth'])
        target = torch.zeros(n,len(self.horizons),3)
        mask = torch.zeros(n,len(self.horizons))
        contact = episode['contact_after'].reshape(n).bool()
        cumulative = torch.cat([torch.zeros(1,dtype=torch.long),contact.long().cumsum(0)])
        for k, offset in enumerate(self.horizons):
            if n > offset:
                target[:-offset,k] = episode['world_position'][offset:]-episode['world_position'][:-offset]
                # At time j predict to j+offset; contact intervals [0, j+offset) must be empty.
                mask[:-offset,k] = (cumulative[offset:n] == 0).float()
        episode.update(future_target=target, future_valid=mask, forecast_phase=phase_labels(episode))
        super().add_episode(task,episode)

    def sample(self, per_task, device):
        original = self.keys
        self.keys = original + ('future_target','future_valid','forecast_phase')
        try:
            return super().sample(per_task,device)
        finally:
            self.keys = original


class StableStudent(OriginalStudent):
    # Exactly the original Future architecture, initialization and auxiliary loss.
    variant = 'future'


class UncertainStudent(BaseStudent):
    def __init__(self, stage, vision, model, auxiliary):
        super().__init__(stage,vision,model,auxiliary)
        cfg = model['future_prediction']
        self.horizons = tuple(cfg['horizons'])
        self.target_scale = cfg['target_scale_m']
        self.logvar_min, self.logvar_max = cfg['logvar_bounds']
        n = len(self.horizons)*3
        self.forecast = nn.Sequential(nn.Linear(self.hidden_dim,128),nn.ELU(),nn.Linear(128,n))
        self.uncertainty = nn.Sequential(nn.Linear(self.hidden_dim,128),nn.ELU(),nn.Linear(128,n))
        nn.init.zeros_(self.uncertainty[-1].weight)
        # Initial normalized variance 1 (sigma = target_scale metres).
        p = -self.logvar_min/(self.logvar_max-self.logvar_min)
        nn.init.constant_(self.uncertainty[-1].bias, math.log(p/(1-p)))
        self.fusion = nn.Linear(2*n,self.hidden_dim)
        self.extra = {}

    def _forward(self, depth, proprio, previous, hidden, auxiliary):
        b,t = depth.shape[:2]
        if tuple(depth.shape[2:]) != self.image_shape:
            raise ValueError('Image/camera contract changed')
        if proprio.shape != (b,t,self.proprio_dim) or previous.shape != (b,t,self.action_dim):
            raise ValueError('Only robot state and previous commands are policy inputs')
        frames = depth.reshape(b*t,*self.image_shape).float()
        visual = torch.cat([encoder(frames[:,i:i+1]) for i,encoder in enumerate(self.encoders)],-1).reshape(b,t,-1)
        state = ((proprio.float()-self.mean)/self.std).clamp(-5,5)
        h,hidden = self.gru(torch.cat([visual,state,previous.float()],-1),hidden)
        mu = self.forecast(h)  # Metres, same target convention as original Future.
        raw = self.uncertainty(h)
        logvar = self.logvar_min + (self.logvar_max-self.logvar_min)*raw.sigmoid()
        # Predictive sigma is trained ONLY by the proper NLL, not a confidence shortcut for imitation.
        sigma = (.5*logvar.detach()).exp()*self.target_scale
        reliability = 1/(1+sigma/self.target_scale)
        encoded = torch.cat([mu*reliability,sigma],-1)
        logits = self.head(h+self.fusion(encoded))
        self.extra = dict(future=mu,logvar=logvar,sigma=sigma)
        if auxiliary:
            return logits.tanh(),hidden,logits,self.aux_head(h)
        return logits.tanh(),hidden,logits

    def extra_loss(self,batch,begin,end,lengths):
        valid = batch['future_valid'][:,begin:end]*batch['valid'][:,begin:end,None]
        shape = (*valid.shape,3)
        target = batch['future_target'][:,begin:end]
        # Invalid labels never affect arithmetic, even if a diagnostic caller puts NaNs there.
        target = torch.where(valid[...,None].bool(),target,torch.zeros_like(target))
        error = (self.extra['future'].reshape(shape)-target)/self.target_scale
        logvar = self.extra['logvar'].reshape(shape)
        nll = .5*(logvar + error.square()*(-logvar).exp() + math.log(2*math.pi))
        per_frame = nll.mean(-1)*valid
        # Preserve equal episode weighting and fixed horizon weighting, like original Future.
        return .1*(per_frame.sum(1)/lengths.clamp_min(1)[:,None]).mean()

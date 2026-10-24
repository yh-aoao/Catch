"""Label-free policies. Privileged labels are consumed only by extra_loss()."""
import torch
from torch import nn
from torch.nn import functional as F
from distillation_visual_aux.core import Student as BaseStudent, EpisodeReplay as BaseReplay


class Student(BaseStudent):
    variant = None

    def __init__(self, stage, vision, model, auxiliary):
        super().__init__(stage, vision, model, auxiliary)
        h, e, c = self.hidden_dim, model['embedding'], len(self.camera_names)
        self.extra = {}
        if self.variant == 'future':
            self.forecast = nn.Sequential(nn.Linear(h, 128), nn.ELU(), nn.Linear(128, 9))
            self.fusion = nn.Linear(9, h)
        elif self.variant == 'belief':
            self.prior = nn.Sequential(nn.Linear(h + self.action_dim, 128), nn.ELU(), nn.Linear(128, c*e))
            self.confidence = nn.Linear(c*e + h, c)
        elif self.variant == 'spatial':
            channels = model['cnn_channels'][-1]
            self.heatmaps = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in range(c)])
            self.local = nn.ModuleList([nn.Linear(channels + 2, e) for _ in range(c)])
        elif self.variant == 'contact':
            self.phase = nn.Sequential(nn.Linear(h, 64), nn.ELU(), nn.Linear(64, 3))
            self.modulation = nn.Linear(3, 2*h)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        elif self.variant == 'camera_gate':
            self.gate = nn.Sequential(nn.Linear(c*e + self.proprio_dim + self.action_dim, 64), nn.ELU(), nn.Linear(64, c))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            raise ValueError('Unknown experimental variant')

    def _forward(self, depth, proprio, previous, hidden, auxiliary):
        b, t = depth.shape[:2]
        if tuple(depth.shape[2:]) != self.image_shape:
            raise ValueError('Image/camera contract changed')
        if proprio.shape != (b,t,self.proprio_dim) or previous.shape != (b,t,self.action_dim):
            raise ValueError('Only robot proprioception and previous commands are policy inputs')
        frames = depth.reshape(b*t,*self.image_shape).float()
        features, heat = [], []
        for i, encoder in enumerate(self.encoders):
            if self.variant == 'spatial':
                fmap = encoder[0][:-1](frames[:,i:i+1])
                global_feature = encoder[2](encoder[1](fmap.flatten(1)))
                logits = self.heatmaps[i](fmap).flatten(1)
                weights = logits.softmax(-1)
                yy, xx = torch.meshgrid(torch.linspace(-1,1,fmap.shape[-2],device=fmap.device),
                                       torch.linspace(-1,1,fmap.shape[-1],device=fmap.device), indexing='ij')
                xy = torch.stack([xx.flatten(), yy.flatten()],-1)
                pooled = (fmap.flatten(2)*weights[:,None]).sum(-1)
                features.append(global_feature + self.local[i](torch.cat([pooled,weights@xy],-1)))
                heat.append((logits.reshape(b,t,-1), xy))
            else:
                features.append(encoder(frames[:,i:i+1]))
        visual = torch.cat(features,-1).reshape(b,t,-1)
        state = ((proprio.float()-self.mean)/self.std).clamp(-5,5)
        self.extra = {}
        if self.variant == 'camera_gate':
            weights = self.gate(torch.cat([visual,state,previous.float()],-1)).softmax(-1)
            self.extra['camera_weights'] = weights
            visual = (visual.reshape(b,t,len(features),-1)*weights[...,None]*len(features)).flatten(2)
        if self.variant == 'belief':
            hidden = self.initial_state(b) if hidden is None else hidden
            outputs, confs, priors = [], [], []
            for j in range(t):
                prior = self.prior(torch.cat([hidden[0],previous[:,j].float()],-1))
                conf = self.confidence(torch.cat([visual[:,j],hidden[0]],-1))
                weight = conf.sigmoid().repeat_interleave(visual.shape[-1]//len(features),-1)
                fused = weight*visual[:,j]+(1-weight)*prior
                out,hidden = self.gru(torch.cat([fused,state[:,j],previous[:,j].float()],-1)[:,None],hidden)
                outputs.append(out); confs.append(conf); priors.append(prior)
            recurrent = torch.cat(outputs,1)
            self.extra.update(confidence=torch.stack(confs,1), prior=torch.stack(priors,1), visual=visual)
        else:
            recurrent,hidden = self.gru(torch.cat([visual,state,previous.float()],-1),hidden)
        action_features = recurrent
        if self.variant == 'future':
            predicted = self.forecast(recurrent)
            self.extra['future'] = predicted
            action_features = recurrent + self.fusion(predicted)
        if self.variant == 'spatial':
            self.extra['heat'] = heat
        if self.variant == 'contact':
            phase = self.phase(recurrent)
            self.extra['contact'] = phase
            scale,shift = self.modulation(phase.sigmoid()).chunk(2,-1)
            action_features = recurrent*(1+.1*scale.tanh()) + .1*shift.tanh()
        logits = self.head(action_features)
        if auxiliary:
            return logits.tanh(),hidden,logits,self.aux_head(recurrent)
        return logits.tanh(),hidden,logits

    def extra_loss(self, batch, begin, end, lengths):
        valid = batch['valid'][:,begin:end]
        def reduce(error, mask):
            while error.ndim > 2:
                error = error.mean(-1)
            return ((error*mask).sum(1)/lengths.clamp_min(1)).mean()
        zero = self.mean.sum()*0
        if self.variant == 'future':
            target = batch['future_target'][:,begin:end]
            mask = batch['future_valid'][:,begin:end]*valid[...,None]
            error = F.smooth_l1_loss(self.extra['future'].reshape(*valid.shape,3,3),target,reduction='none').mean(-1)
            return .1*sum(reduce(error[...,k],mask[...,k]) for k in range(3))/3
        if self.variant == 'belief':
            visible = batch['visible'][:,begin:end]
            reliability = reduce(F.binary_cross_entropy_with_logits(self.extra['confidence'],visible,reduction='none'),valid)
            prior_error = F.smooth_l1_loss(self.extra['prior'],self.extra['visual'].detach(),reduction='none')
            prior_error = prior_error.reshape(*valid.shape,len(self.camera_names),-1).mean(-1)*visible
            # The first frame has no history; exclude it from prior training.
            mask = valid.clone()
            if begin == 0: mask[:,0] = 0
            return .05*reliability + .05*reduce(prior_error,mask)
        if self.variant == 'spatial':
            total = zero
            for i,(logits,xy) in enumerate(self.extra['heat']):
                uv = batch['uv'][:,begin:end,i]
                target = (-((xy[None,None]-uv[:,:,None])**2).sum(-1)/(2*.12**2)).softmax(-1)
                error = -(target*logits.log_softmax(-1)).sum(-1)
                total = total + reduce(error,valid*batch['visible'][:,begin:end,i])
            return .05*total/len(self.camera_names)
        if self.variant == 'contact':
            error = F.binary_cross_entropy_with_logits(self.extra['contact'],batch['contact_target'][:,begin:end],reduction='none')
            return .1*sum(reduce(error[...,k],valid*batch['contact_valid'][:,begin:end,k]) for k in range(3))/3
        return zero


class EpisodeReplay(BaseReplay):
    keys = BaseReplay.keys + ('world_position','visible','uv','contact_after')

    def add_episode(self, task, episode):
        # Derived labels are recomputed deterministically when loading resume data.
        n = len(episode['depth'])
        future = torch.zeros(n,3,3)
        mask = torch.zeros(n,3)
        contact = episode['contact_after'].float().reshape(n)
        for k,offset in enumerate((1,3,5)):
            if n > offset:
                future[:-offset,k] = episode['world_position'][offset:] - episode['world_position'][:-offset]
                mask[:-offset,k] = 1
                # Forecast only intervals before any observed hand contact.
                for j in range(n-offset):
                    if contact[:j+offset].any(): mask[j,k] = 0
        phase = torch.zeros(n,3)
        phase_valid = torch.ones(n,3)
        for j in range(n):
            phase[j,0] = contact[j:min(j+3,n)].max()
            # Right-censored negative windows cannot establish absence of future contact.
            if j+3 > n and phase[j,0] == 0: phase_valid[j,0] = 0
            phase[j,1] = contact[j-1] if j else 0
            phase[j,2] = float(j >= 3 and bool(contact[j-3:j].all()))
        episode.update(future_target=future,future_valid=mask,contact_target=phase,contact_valid=phase_valid)
        super().add_episode(task,episode)

    def sample(self, per_task, device):
        # Base sample must pad both original and derived fields without requiring
        # future labels in Collector.add().
        original = self.keys
        self.keys = original + ('future_target','future_valid','contact_target','contact_valid')
        try:
            return super().sample(per_task,device)
        finally:
            self.keys = original

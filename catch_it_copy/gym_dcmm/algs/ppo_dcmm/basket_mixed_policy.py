"""Physical-property conditioning and explicit fixed/mobile throw-actor transfer."""
import torch
from torch import nn
from .utils import RunningMeanStd


class MixedNormalizer(nn.Module):
    def __init__(self):
        super().__init__();self.original=RunningMeanStd((35,)).eval()
    def train(self,mode=True):
        super().train(mode);self.original.eval();return self
    def forward(self,raw):
        radius=(raw[:,35:36]-.04)/.01
        mass=torch.log(raw[:,36:37].clamp_min(1e-6)/.05)
        return torch.cat((self.original(raw[:,:35]),radius,mass),dim=-1)


def extract_throw(checkpoint,finger_options,carry_options):
    if (checkpoint.get('basket_fixed_protocol')!='aim_carry_fingers_v1'
            or checkpoint.get('basket_arm_control')!='joint_velocity'
            or checkpoint.get('basket_finger_options')!=finger_options
            or checkpoint.get('basket_carry_options')!=carry_options):
        raise ValueError('Mixed Basket requires the matching 41.x finger/carry throw protocol')
    state=checkpoint['model']
    if checkpoint.get('action_layout')=='arm6_hand12':
        actor={k[len('actor_mlp.'):]:v for k,v in state.items() if k.startswith('actor_mlp.')}
        head={k[len('mu.'):]:v for k,v in state.items() if k.startswith('mu.')}
        sigma=state['sigma'];norm=checkpoint['running_mean_std'];kind='fixed35'
    elif checkpoint.get('action_layout')=='joint_base2_arm18_phase_v1':
        actor={k[len('throw_actor.'):]:v for k,v in state.items() if k.startswith('throw_actor.')}
        head={k[len('throw_mu.'):]:v for k,v in state.items() if k.startswith('throw_mu.')}
        sigma=state['throw_sigma']
        norm={k[len('throwing.'):]:v for k,v in checkpoint['running_mean_std'].items() if k.startswith('throwing.')}
        kind='mobile_joint_throw35'
    else:raise ValueError('Source must be fixed 41.x or a jointly trained mobile Basket throw checkpoint')
    if actor['mlp.0.weight'].shape[1]!=35 or head['weight'].shape[0]!=18 or norm['running_mean'].shape!=(35,):
        raise ValueError('Source throw actor must have 35 observations and 18 actions')
    return actor,head,sigma,norm,kind


def import_throw(model,normalizer,checkpoint,finger_options,carry_options):
    actor,head,sigma,norm,kind=extract_throw(checkpoint,finger_options,carry_options)
    target=model.actor_mlp.state_dict()
    for key,value in actor.items():
        if key=='mlp.0.weight':
            if target[key].shape!=(value.shape[0],37):raise ValueError('Source and target hidden widths differ')
            target[key]=torch.zeros_like(target[key]);target[key][:,:35]=value.to(target[key])
        else:
            if target[key].shape!=value.shape:raise ValueError('Source and target hidden widths differ')
            target[key]=value.to(target[key]).clone()
    model.actor_mlp.load_state_dict(target,strict=True)
    model.mu.load_state_dict(head,strict=True)
    with torch.no_grad():model.sigma.copy_(sigma)
    normalizer.original.load_state_dict(norm,strict=True)
    return kind


def stratified_mean(value,labels,counts,population):
    """Equal loss mass per sampled ball over a whole PPO rollout."""
    present=sum(count>0 for count in counts)
    result=value.sum()*0.
    for index,count in enumerate(counts):
        if count:result=result+(value*(labels==index).to(value.dtype)).sum()/count/max(present,1)
    return result*population/max(len(value),1)


class MacroSuccess:
    def __init__(self,meters):self.meters=meters
    def __len__(self):
        return sum(len(m) for m in self.meters) if all(len(m)>=20 for m in self.meters) else 0
    def get_mean(self):return sum(m.get_mean() for m in self.meters)/len(self.meters)


def single_ball_fixed35(checkpoint,ball,protocol,delivery_version,delivery_options):
    """Exactly absorb constant physical inputs into the first-layer biases."""
    import copy
    if (checkpoint.get('basket_mixed_protocol')!='mixed37_known_radius_mass_v1'
            or checkpoint.get('basket_trained_ball')!=ball
            or checkpoint.get('basket_ball_protocol')!=protocol
            or checkpoint.get('basket_delivery_version')!=delivery_version
            or checkpoint.get('basket_delivery_options')!=delivery_options):
        raise ValueError('Single mobile source must match ball, physics and delivery reward')
    profile=protocol[ball]
    # Match MixedNormalizer's float32 arithmetic, including physical input rounding.
    physical=torch.tensor([[profile['radius_m'],profile['mass_kg']]],dtype=torch.float32)
    constant=torch.cat(((physical[:,:1]-.04)/.01,
                        torch.log(physical[:,1:]/.05)),dim=-1)[0]
    result=copy.deepcopy(checkpoint)
    state=result['model']
    for prefix in ('actor_mlp','value_mlp'):
        key=prefix+'.mlp.0.weight';bias=prefix+'.mlp.0.bias'
        weight=state[key]
        if weight.shape[1]!=37:raise ValueError('Expected single-ball fixed37 checkpoint')
        state[bias]=state[bias]+weight[:,35:]@constant.to(weight)
        state[key]=weight[:,:35].clone()
    result['running_mean_std']={k[len('original.'):]:v
        for k,v in checkpoint['running_mean_std'].items() if k.startswith('original.')}
    return result

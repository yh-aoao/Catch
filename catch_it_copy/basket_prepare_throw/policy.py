"""66-input prepare/throw actor; all 20 action channels remain trainable."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

SCHEMA='basket_prepare_throw_v1'

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        def network(output):return nn.Sequential(nn.Linear(66,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,output))
        self.actor=network(20);self.critic=network(1)
        self.carry_head=nn.Linear(128,20)
        self.log_std=nn.Parameter(torch.full((20,),-.8))
        self.register_buffer('mean',torch.zeros(66));self.register_buffer('variance',torch.ones(66));self.register_buffer('count',torch.tensor(1e-4))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        nn.init.zeros_(self.carry_head.weight);nn.init.zeros_(self.carry_head.bias)
        with torch.no_grad():
            self.actor[0].weight[:,65].zero_();self.critic[0].weight[:,65].zero_()
    def normalize(self,x):
        state=((x[:,:65]-self.mean[:65])/torch.sqrt(self.variance[:65]+1e-6)).clamp(-5,5)
        return torch.cat([state,x[:,65:66]],-1)
    @torch.no_grad()
    def update_normalizer(self,x):
        n=x.shape[0];delta=x.mean(0)-self.mean;total=self.count+n
        variance=x.var(0,unbiased=False)
        self.variance.copy_((self.variance*self.count+variance*n+delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total);self.count.copy_(total)
    def action_mean(self,x):
        features=self.actor[:-1](x)
        return torch.where(x[:,65:66]>.5,self.carry_head(features),self.actor[-1](features))
    def distribution(self,x):
        # Carrying needs smaller arm/finger exploration than a fast throw.
        factor=torch.ones_like(x[:,:20])
        carry=x[:,65:66]
        factor[:,2:]=1.-.90*carry
        return Normal(self.action_mean(x),self.log_std.clamp(-5,1).exp()*factor)
    def load_fixed(self,state):
        converted={k:v.clone() for k,v in state.items()}
        for key in ('actor.0.weight','critic.0.weight'):
            converted[key]=torch.cat([converted[key],torch.zeros(converted[key].shape[0],1)],dim=1)
        for key,value in (('mean',0.),('variance',1.)):
            converted[key]=torch.cat([converted[key],torch.tensor([value])])
        for key in ('carry_head.weight','carry_head.bias'):converted[key]=self.state_dict()[key]
        self.load_state_dict(converted,strict=True)
    def value(self,x):return self.critic(x).squeeze(-1)
    def log_probability(self,distribution,raw,active=None):
        # Numerically stable log(1-tanh(raw)^2).
        correction=2*(np.log(2)-raw-torch.nn.functional.softplus(-2*raw))
        terms=distribution.log_prob(raw)-correction
        return (terms if active is None else terms*active).sum(-1)
    def sample(self,x,active=None):
        distribution=self.distribution(x);raw=distribution.sample()
        return raw.tanh(),raw,self.log_probability(distribution,raw,active),self.value(x)
    @torch.no_grad()
    def act(self,observation):return self.action_mean(self.normalize(observation)).tanh()

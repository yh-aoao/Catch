"""One actor emits all 20 channels on every step; no phase branches or masks."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

SCHEMA='basket_autonomous_teacher_v1'

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        def network(output):return nn.Sequential(nn.Linear(65,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,output))
        self.actor=network(20);self.critic=network(1)
        self.log_std=nn.Parameter(torch.full((20,),-.8))
        self.register_buffer('mean',torch.zeros(65));self.register_buffer('variance',torch.ones(65));self.register_buffer('count',torch.tensor(1e-4))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
    def normalize(self,x):return ((x-self.mean)/torch.sqrt(self.variance+1e-6)).clamp(-5,5)
    @torch.no_grad()
    def update_normalizer(self,x):
        n=x.shape[0];delta=x.mean(0)-self.mean;total=self.count+n
        variance=x.var(0,unbiased=False)
        self.variance.copy_((self.variance*self.count+variance*n+delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total);self.count.copy_(total)
    def distribution(self,x):return Normal(self.actor(x),self.log_std.clamp(-5,1).exp())
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
    def act(self,observation):return self.actor(self.normalize(observation)).tanh()

"""Phase-conditioned PPO with a new base actor and a trainable fixed35 throw actor."""
import torch
from torch import nn

from .models_track import ActorCritic, MLP
from .utils import RunningMeanStd


class JointNormalizer(nn.Module):
    """Learn base statistics while retaining the pretrained throw coordinate system.

    Frozen normalization statistics do not freeze any actor parameters.
    The final column is the raw phase flag, never normalized.
    """
    def __init__(self):
        super().__init__()
        self.navigation = RunningMeanStd((51,))
        self.throwing = RunningMeanStd((35,))
        self.throwing.eval()

    def train(self, mode=True):
        super().train(mode)
        self.throwing.eval()
        return self

    def forward(self, raw):
        fixed = raw[:, :35].clone()
        fixed[:, 34] += .2
        return torch.cat((self.navigation(raw), self.throwing(fixed), raw[:,47:48]), dim=-1)


class JointActorCritic(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.separate_value_mlp = True
        self.navigation = ActorCritic(dict(actor_units=list(units), actions_num=2,
            input_shape=(51,), separate_value_mlp=True))
        self.throw_actor = MLP(list(units), 35)
        self.throw_mu = nn.Linear(units[-1],18)
        self.throw_sigma = nn.Parameter(torch.zeros(18))

    @property
    def sigma(self):
        return torch.cat((self.navigation.sigma,self.throw_sigma))

    def import_throw(self, checkpoint, normalizer, finger_options, carry_options):
        # Validate dimensions, action semantics and reward contract before loading.
        from gym_dcmm.utils.basket_throw_policy import FrozenThrowPolicy
        validated = FrozenThrowPolicy(checkpoint,finger_options,carry_options,
                                      self.throw_sigma.device)
        self.throw_actor.load_state_dict(validated.model.actor_mlp.state_dict(),strict=True)
        self.throw_mu.load_state_dict(validated.model.mu.state_dict(),strict=True)
        with torch.no_grad():self.throw_sigma.copy_(validated.model.sigma)
        normalizer.throwing.load_state_dict(validated.normalizer.state_dict(),strict=True)
        # load_state_dict copies values only: these actor parameters remain trainable.

    def actor_groups(self):
        base = list(self.navigation.actor_mlp.parameters())+list(self.navigation.mu.parameters())+[self.navigation.sigma]
        throw = list(self.throw_actor.parameters())+list(self.throw_mu.parameters())+[self.throw_sigma]
        critic = list(self.navigation.value_mlp.parameters())+list(self.navigation.value.parameters())
        return base,throw,critic

    @staticmethod
    def action_mask(obs):
        throwing = obs[:,-1:] > .5
        return torch.cat(((~throwing).expand(-1,2),throwing.expand(-1,18)),dim=-1)

    def _distribution(self, obs):
        base_mu,_,value = self.navigation._actor_critic(dict(obs=obs[:,:51]))
        throw_mu = torch.tanh(self.throw_mu(self.throw_actor(obs[:,51:86])))
        mask = self.action_mask(obs)
        mu = torch.cat((base_mu,throw_mu),dim=-1).masked_fill(~mask,0.)
        logstd = self.sigma.expand_as(mu).masked_fill(~mask,0.)
        return torch.distributions.Normal(mu,logstd.exp()),value,mask

    @torch.no_grad()
    def act(self, obs_dict):
        distribution,value,mask = self._distribution(obs_dict['obs'])
        actions = distribution.sample().masked_fill(~mask,0.)
        return dict(actions=actions,values=value,
            neglogpacs=-(distribution.log_prob(actions)*mask).sum(-1),
            mus=distribution.mean,sigmas=distribution.stddev)

    @torch.no_grad()
    def act_inference(self, obs_dict):
        return self._distribution(obs_dict['obs'])[0].mean

    def forward(self, input_dict):
        distribution,value,mask = self._distribution(input_dict['obs'])
        return dict(values=value,
            prev_neglogp=-(distribution.log_prob(input_dict['prev_actions'])*mask).sum(-1),
            entropy=(distribution.entropy()*mask).sum(-1),
            mus=distribution.mean,sigmas=distribution.stddev)

    def clear_inactive_gradients(self, processed_observation):
        mask=self.action_mask(processed_observation)
        base,throw,_=self.actor_groups()
        for active,parameters in ((bool(mask[:,:2].any()),base),(bool(mask[:,2:].any()),throw)):
            if not active:
                for parameter in parameters:parameter.grad=None

    def policy_kl(self, obs, mu, sigma, old_mu, old_sigma):
        mask=self.action_mask(obs)
        per_action=(torch.log(old_sigma/sigma+1e-5)
            +(sigma.square()+(old_mu-mu).square())/(2.*old_sigma.square())-.5)
        # Do not dilute a two-action navigation KL by eighteen unused hand actions.
        return ((per_action*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()

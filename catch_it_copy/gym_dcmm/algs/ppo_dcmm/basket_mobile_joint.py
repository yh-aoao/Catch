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
    def __init__(self, conditioned=False):
        super().__init__()
        self.navigation_dim=53 if conditioned else 51
        self.conditioned=conditioned
        self.navigation = RunningMeanStd((self.navigation_dim,))
        if conditioned:
            from .basket_mixed_policy import MixedNormalizer
            self.throwing=MixedNormalizer()
        else:
            self.throwing = RunningMeanStd((35,))
        self.throwing.eval()

    def train(self, mode=True):
        super().train(mode)
        self.throwing.eval()
        return self

    def forward(self, raw):
        fixed = raw[:, :35].clone()
        fixed[:, 34] += .2
        if self.conditioned:
            fixed=torch.cat((fixed,raw[:,51:53]),dim=-1)
        return torch.cat((self.navigation(raw), self.throwing(fixed), raw[:,47:48]), dim=-1)


class JointActorCritic(nn.Module):
    def __init__(self, units, conditioned=False):
        super().__init__()
        self.separate_value_mlp = True
        self.navigation_dim=53 if conditioned else 51
        self.throw_dim=37 if conditioned else 35
        self.navigation = ActorCritic(dict(actor_units=list(units), actions_num=2,
            input_shape=(self.navigation_dim,), separate_value_mlp=True))
        self.throw_actor = MLP(list(units), self.throw_dim)
        self.throw_mu = nn.Linear(units[-1],18)
        self.throw_sigma = nn.Parameter(torch.zeros(18))

    @property
    def sigma(self):
        return torch.cat((self.navigation.sigma,self.throw_sigma))

    def import_throw(self, checkpoint, normalizer, finger_options, carry_options):
        # Validate dimensions, action semantics and reward contract before loading.
        if self.throw_dim==37:
            if (checkpoint.get('basket_fixed_protocol')!='aim_carry_fingers_v1'
                    or checkpoint.get('basket_finger_options')!=finger_options
                    or checkpoint.get('basket_carry_options')!=carry_options
                    or checkpoint.get('basket_arm_control')!='joint_velocity'
                    or checkpoint.get('action_layout')!='arm6_hand12'):
                raise ValueError('Mixed mobile source finger/carry/action protocol mismatch')
            state=checkpoint['model']
            self.throw_actor.load_state_dict({k[len('actor_mlp.'):]:v for k,v in state.items() if k.startswith('actor_mlp.')},strict=True)
            self.throw_mu.load_state_dict({k[len('mu.'):]:v for k,v in state.items() if k.startswith('mu.')},strict=True)
            with torch.no_grad():self.throw_sigma.copy_(state['sigma'])
            normalizer.throwing.load_state_dict(checkpoint['running_mean_std'],strict=True)
            return
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
        base_mu,_,value = self.navigation._actor_critic(dict(obs=obs[:,:self.navigation_dim]))
        throw_mu = torch.tanh(self.throw_mu(self.throw_actor(obs[:,self.navigation_dim:self.navigation_dim+self.throw_dim])))
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
        from .basket_joint_stability import gaussian_kl
        mask=self.action_mask(obs)
        # Inactive channels are constants; sum active dimensions as in original PPO.
        return gaussian_kl(old_mu,old_sigma,mu,sigma).mean()

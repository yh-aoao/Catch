import copy
import unittest

import torch

from gym_dcmm.algs.ppo_dcmm.basket_mixed_policy import MixedNormalizer,single_ball_fixed35
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.utils.basket_delivery import make_options


class SingleMobileTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.model=ActorCritic(dict(actor_units=[16,8],actions_num=18,
            input_shape=(37,),separate_value_mlp=True)).eval()
        self.norm=MixedNormalizer().eval()
        self.norm.original.running_mean.copy_(torch.randn(35))
        self.norm.original.running_var.copy_(torch.rand(35)+.5)
        self.protocol={'tennis_ball':dict(radius_m=.0332757853,mass_kg=.058)}
        self.options=make_options()
        self.source=dict(model=copy.deepcopy(self.model.state_dict()),
            running_mean_std=copy.deepcopy(self.norm.state_dict()),
            basket_mixed_protocol='mixed37_known_radius_mass_v1',
            basket_trained_ball='tennis_ball',basket_ball_protocol=self.protocol,
            basket_delivery_version='ballistic_v2',basket_delivery_options=self.options)

    def test_constant_input_absorption_preserves_actor_and_critic(self):
        snapshot=copy.deepcopy(self.source)
        checkpoint=single_ball_fixed35(self.source,'tennis_ball',self.protocol,
            'ballistic_v2',self.options)
        target=ActorCritic(dict(actor_units=[16,8],actions_num=18,
            input_shape=(35,),separate_value_mlp=True)).eval()
        target.load_state_dict(checkpoint['model'])
        raw=torch.randn(64,35)
        props=torch.tensor([[.0332757853,.058]]).expand(64,2)
        old=self.model._actor_critic(dict(obs=self.norm(torch.cat((raw,props),-1))))
        new=target._actor_critic(dict(obs=self.norm.original(raw)))
        for a,b in zip(old,new):
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        for key,value in snapshot['model'].items():
            torch.testing.assert_close(value,self.source['model'][key])
        self.assertTrue(all(p.requires_grad for p in target.parameters()))

    def test_rejects_wrong_ball_physics_or_reward(self):
        for ball,protocol,version,options in [
            ('baseball',self.protocol,'ballistic_v2',self.options),
            ('tennis_ball',{},'ballistic_v2',self.options),
            ('tennis_ball',self.protocol,'legacy',self.options),
            ('tennis_ball',self.protocol,'ballistic_v2',{})]:
            with self.assertRaises(ValueError):
                single_ball_fixed35(self.source,ball,protocol,version,options)


if __name__=='__main__':
    unittest.main()

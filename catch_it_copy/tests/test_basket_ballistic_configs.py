import unittest
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf


class BallisticConfigTests(unittest.TestCase):
    def test_mixed_is_scratch_equal_sampling_and_shares_single_ball_objective(self):
        OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
        root=Path(__file__).resolve().parents[1]/'configs'
        with initialize_config_dir(config_dir=str(root),version_base='1.1'):
            mixed=compose(config_name='basket_three_ball_ballistic')
            singles=[compose(config_name=name) for name in ('basket_single_original','basket_single_tennis','basket_single_baseball')]
        self.assertEqual(mixed.basket_mixed_ball,'mixed')
        self.assertEqual(list(mixed.basket_mixed_options.weights),[1.,1.,1.])
        self.assertEqual(mixed.basket_delivery_version,'ballistic_v2')
        self.assertEqual(mixed.output_name,'BasketThreeBallBallisticScratch')
        for config in [mixed]+singles:
            self.assertFalse(any([config.basket_mixed_source,config.checkpoint_catching,config.checkpoint_tracking,config.basket_warm_start]))
            self.assertTrue(config.basket_fixed_base_training)
            self.assertEqual(config.basket_delivery_options,mixed.basket_delivery_options)
            self.assertEqual(config.basket_carry_options,mixed.basket_carry_options)
        self.assertEqual([c.basket_mixed_ball for c in singles],['original','tennis_ball','baseball'])

    def test_physical_parameters_reach_actor_without_ball_id_or_image(self):
        import torch
        from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
        from gym_dcmm.algs.ppo_dcmm.basket_mixed_policy import MixedNormalizer
        torch.manual_seed(123)
        model=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(37,),separate_value_mlp=True)).eval()
        normalizer=MixedNormalizer().eval()
        raw=torch.zeros(3,37)
        raw[:,-2:]=torch.tensor([[.04,.05],[.03328,.058],[.03618,.138]])
        output=model.act_inference(dict(obs=normalizer(raw)))
        self.assertGreater(float((output[0]-output[1]).abs().max()),1e-8)
        self.assertGreater(float((output[0]-output[2]).abs().max()),1e-8)

if __name__=='__main__':unittest.main()

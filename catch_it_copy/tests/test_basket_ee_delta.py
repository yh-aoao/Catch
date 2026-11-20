"""New Cartesian Basket semantics; preserve the joint-velocity version."""
import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gym_dcmm.algs.ppo_dcmm.basket_mobile_joint import JointActorCritic,JointNormalizer
from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.utils.basket_carry import make_options as carry_options
from gym_dcmm.utils.basket_fingers import make_options as finger_options
from tests.test_basket_tennis_free_stop import TennisFreeStopTests


class TennisEEWorkflowTests(TennisFreeStopTests):
    def config(self,name):
        names={'basket_tennis_range':'basket_tennis_range_ee',
               'basket_mobile_tennis_free_stop':'basket_mobile_tennis_free_stop_ee'}
        return super().config(names.get(name,name))

    def test_ee_actions_use_measured_pose_ik_and_preserve_hand_mapping(self):
        c=self.config('basket_tennis_range')
        env=self.vector(c,False);self.addCleanup(env.close)
        raw=env.envs[0].unwrapped
        env.reset(seed=123)
        requested=np.array([.001,-.002,.001,.003,-.002,.001])
        command=dict(base=np.zeros(2),arm=requested.copy(),hand=np.zeros(12))
        module=importlib.import_module('gym_dcmm.envs.DcmmVecEnv')
        with patch.object(raw.Dcmm,'move_ee_pose',wraps=raw.Dcmm.move_ee_pose) as ik, patch.object(raw.Dcmm,'ik_arm_solve',wraps=raw.Dcmm.ik_arm_solve) as solver, patch.object(module,'joint_velocity_target',
                   side_effect=AssertionError('ee_delta must not execute joint velocity mapping')):
            obs,reward,done,trunc,info=raw.step(command)
        self.assertGreater(ik.call_count,0)
        for call in ik.call_args_list:
            np.testing.assert_array_equal(call.args[0],requested)
            self.assertTrue(call.kwargs['measured_state'])
        target_pos,target_quat=solver.call_args.args
        np.testing.assert_allclose(target_pos,raw.Dcmm.current_ee_pos+requested[:3],atol=1e-10)
        current=Rotation.from_quat(raw.Dcmm.current_ee_quat[[1,2,3,0]])
        expected=(Rotation.from_euler('zxy',requested[3:])*current).as_quat()[[3,0,1,2]]
        np.testing.assert_allclose(target_quat,expected,atol=1e-10)
        self.assertTrue(np.isfinite(raw.Dcmm.target_arm_qpos).all())
        self.assertTrue(np.isfinite(reward))
        np.testing.assert_array_equal(command['arm'],requested)


class CartesianCheckpointTests(unittest.TestCase):
    def test_imports_matching_ee_checkpoint_and_rejects_joint_semantics(self):
        source=ActorCritic(dict(actor_units=[16,8],actions_num=18,input_shape=(35,),
            separate_value_mlp=True))
        norm=RunningMeanStd((35,)).eval()
        ck=dict(model=source.state_dict(),running_mean_std=norm.state_dict(),
            basket_arm_control='ee_delta',action_layout='arm6_hand12',
            basket_fixed_protocol='aim_carry_fingers_v1',
            basket_finger_options=finger_options(),basket_carry_options=carry_options())
        target=JointActorCritic([16,8]);normalizer=JointNormalizer().eval()
        target.import_throw(ck,normalizer,finger_options(),carry_options(),arm_control='ee_delta')
        for k,v in source.actor_mlp.state_dict().items():
            torch.testing.assert_close(v,target.throw_actor.state_dict()[k])
        with self.assertRaises(ValueError):
            target.import_throw(ck,normalizer,finger_options(),carry_options())
        wrong=copy.deepcopy(ck);wrong['basket_arm_control']='joint_velocity'
        with self.assertRaises(ValueError):
            target.import_throw(wrong,normalizer,finger_options(),carry_options(),arm_control='ee_delta')


if __name__=='__main__':unittest.main()

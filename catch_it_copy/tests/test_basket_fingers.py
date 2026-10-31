import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

import numpy as np

from gym_dcmm.utils import basket_fingers as fingers
from gym_dcmm.utils import basket_carry


class FingerTests(unittest.TestCase):
    def setUp(self):
        self.options = fingers.make_options()
        self.ready = np.r_[np.tile([.8, .7, .7], 3), [.4, .5, .4]]
        self.limits = np.array([[-.314,2.23],[-.506,1.885],[-.366,2.042]]*3+
                               [[-.47,2.443],[-1.2,1.9],[-1.34,1.88]])
        self.state = fingers.new_state(self.ready, self.limits, self.options)

    def terms(self, q, t=.04, dt=.04, released=False, action=None):
        fingers.update(self.state, q, t, dt, released, self.options)
        return fingers.reward(self.state, np.zeros(12) if action is None else action, self.options)

    def test_cup_and_coordinated_opening_are_not_penalized_before_release(self):
        self.assertEqual(sum(self.terms(self.ready).values()), 0.)
        q = self.ready*.5
        q[9:] = [.8, -.4, .2]  # Thumb is independent and within its own limits.
        self.assertEqual(sum(self.terms(q).values()), 0.)

    def test_opposite_fingers_and_distal_backfold_receive_distinct_penalties(self):
        q = self.ready.copy()
        q[:9] *= np.repeat([1., .5, 0.], 3)
        terms = self.terms(q)
        self.assertLess(terms['finger_sync'], 0.)
        self.assertEqual(terms['finger_coupling'], 0.)
        q = self.ready.copy()
        q[2] = -.2
        terms = self.terms(q)
        self.assertLess(terms['finger_coupling'], 0.)
        self.assertLess(terms['finger_extension'], 0.)

    def test_measured_overextension_penalty_does_not_depend_on_target_clip(self):
        q = self.ready.copy()
        q[:9] = -.2
        terms = self.terms(q)
        self.assertEqual(terms['finger_sync'], 0.)  # Coordinated, but still wrong.
        self.assertLess(terms['finger_extension'], 0.)
        self.assertTrue(all(value <= 0. for value in terms.values()))

    def test_severe_disagreement_still_has_a_continuous_improvement_signal(self):
        q = self.ready.copy()
        q[:3] *= 3.
        self.terms(q)
        loss = self.state['metrics']['sync_loss']
        q[:3] = self.ready[:3]*4.
        self.terms(q)
        self.assertGreater(self.state['metrics']['sync_loss'], loss)
        self.assertLess(self.state['metrics']['sync_loss'], 1.)

    def test_release_hand_pose_blends_in_and_does_not_lock_controls(self):
        self.assertEqual(self.terms(self.ready, t=.2, released=True)['finger_posture'], 0.)
        middle = self.terms(self.ready, t=.275, released=True)['finger_posture']
        full = self.terms(self.ready, t=.35, released=True)['finger_posture']
        self.assertAlmostEqual(middle, full*.5)
        self.assertLess(full, 0.)
        self.assertEqual(self.terms(np.zeros(12), t=.4, released=True)['finger_posture'], 0.)

    def test_physics_penalties_are_time_step_invariant_and_pending_is_consumed(self):
        q = self.ready.copy()
        q[2] = -.2
        once = sum(self.terms(q, dt=.04).values())
        self.state = fingers.new_state(self.ready, self.limits, self.options)
        for i in range(20):
            fingers.update(self.state, q, (i+1)*.002, .002, False, self.options)
        split = sum(fingers.reward(self.state, np.zeros(12), self.options).values())
        self.assertAlmostEqual(once, split)
        self.assertEqual(sum(fingers.reward(self.state, np.zeros(12), self.options).values()), 0.)

    def test_action_smoothness_uses_all_twelve_hand_channels(self):
        q = self.ready.copy()
        a = self.terms(q, action=np.ones(12))['finger_smoothness']
        b = self.terms(q, action=-np.ones(12))['finger_smoothness']
        self.assertAlmostEqual(a, -.01)
        self.assertAlmostEqual(b, -.04)
        self.assertEqual(self.terms(q, action=-np.ones(12))['finger_smoothness'], 0.)

    def test_bad_configuration_and_pose_references_rejected(self):
        for options in ({'typo':1}, {'sync_weight':-1}, {'post_release_ramp':0},
                        {'thumb_open':[0,0]}, {'long_open':[0,np.nan,0]}):
            with self.assertRaises(ValueError):
                fingers.make_options(options)
        with self.assertRaises(ValueError):
            fingers.new_state(np.zeros(12), self.limits, self.options)

    def test_new_checkpoint_requires_its_finger_configuration_and_old_actor_can_load(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                      and n.name=='_validate_mobile_checkpoint')
        scope={}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<checkpoint-validation>','exec'),scope)
        obj=N(finger_reward=True,finger_options=self.options,carry_reward=True,
              carry_options=basket_carry.make_options(),fixed_physical=True,mobile_basket=False)
        old=dict(basket_fixed_protocol='aim_carry_v1',basket_carry_options=obj.carry_options)
        scope['_validate_mobile_checkpoint'](obj,old)
        checkpoint=dict(old,basket_fixed_protocol='aim_carry_fingers_v1',basket_finger_options=self.options)
        scope['_validate_mobile_checkpoint'](obj,checkpoint)
        obj.finger_reward=False
        with self.assertRaises(ValueError):scope['_validate_mobile_checkpoint'](obj,checkpoint)
        obj.finger_reward=True
        obj.finger_options=dict(self.options,sync_weight=20.)
        with self.assertRaises(ValueError):scope['_validate_mobile_checkpoint'](obj,checkpoint)


if __name__=='__main__':
    unittest.main()

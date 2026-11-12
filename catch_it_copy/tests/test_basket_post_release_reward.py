import unittest
import numpy as np
from gym_dcmm.utils import basket_fingers as f


class PostReleaseRewardTests(unittest.TestCase):
    def setUp(self):
        self.options=f.make_options()
        self.ready=np.r_[np.tile([.8,.7,.7],3),[.4,.5,.4]]
        self.limits=np.array([[-.314,2.23],[-.506,1.885],[-.366,2.042]]*3+
                            [[-.47,2.443],[-1.2,1.9],[-1.34,1.88]])
    def state(self):return f.new_state(self.ready,self.limits,self.options)
    def test_extra_terms_zero_before_confirmed_release(self):
        s=self.state()
        for i in range(1,10):
            q=self.ready*.5
            f.update(s,q,i*.04,.04,False,self.options)
            r=f.reward(s,np.ones(12),self.options)
            self.assertTrue(all(v==0. for k,v in r.items() if k.startswith('finger_post_')))

    def test_steady_extreme_commands_are_penalized_even_with_zero_smoothness(self):
        s=self.state()
        f.update(s,self.ready,1.,.04,True,self.options)
        f.reward(s,np.ones(12),self.options)
        f.update(s,self.ready,1.2,.04,True,self.options)
        r=f.reward(s,np.ones(12),self.options)
        self.assertEqual(r['finger_smoothness'],0.)
        self.assertLess(r['finger_post_action'],0.)
        self.assertLess(r['finger_post_recovery'],0.)
        self.assertTrue(all(v<=0. for v in r.values()))
        self.assertEqual(f.reward(s,np.zeros(12),self.options)['finger_post_action'],0.)

    def test_settled_open_hand_has_no_new_cost_and_no_state_projection(self):
        s=self.state()
        f.update(s,self.ready,1.,.04,True,self.options)
        f.reward(s,np.zeros(12),self.options)
        q=np.zeros(12)
        f.update(s,q,1.2,.04,True,self.options)
        f.reward(s,np.zeros(12),self.options)
        target=q.copy()
        f.update(s,q,1.24,.04,True,self.options)
        r=f.reward(s,np.zeros(12),self.options)
        self.assertTrue(all(v==0. for k,v in r.items() if k.startswith('finger_post_')))
        np.testing.assert_array_equal(q,target)

    def test_measured_fast_motion_adds_cost(self):
        s=self.state()
        f.update(s,self.ready,1.,.002,True,self.options)
        f.reward(s,np.zeros(12),self.options)
        f.update(s,self.ready,1.2,.002,True,self.options)
        f.reward(s,np.zeros(12),self.options)
        f.update(s,self.ready+.1,1.202,.002,True,self.options)
        r=f.reward(s,np.zeros(12),self.options)
        self.assertLess(r['finger_post_velocity'],0.)

    def test_exposure_is_time_integral_not_policy_step_count(self):
        one=f.post_release_exposure(1.4,.4,1.)
        split=sum(f.post_release_exposure(1.+(i+1)*.002,.002,1.) for i in range(200))
        self.assertAlmostEqual(one,split,places=10)
        self.assertEqual(f.post_release_exposure(.9,.04,1.),0.)
        self.assertEqual(f.post_release_exposure(2.,.04,None),0.)

if __name__=='__main__':unittest.main()

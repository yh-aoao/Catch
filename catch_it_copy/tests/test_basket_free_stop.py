import unittest
from types import SimpleNamespace as N
import numpy as np
from gym_dcmm.envs.basket_free_stop import options,region_state
from gym_dcmm.utils.basket_tracking import parking_state


class FreeStopTests(unittest.TestCase):
    def setUp(self):
        self.cfg=N(basket_track_forward_offset=None,basket_base_front_dist=1.,
            basket_track_position_tolerance=.04,basket_track_speed_tolerance=.04,
            basket_track_slowdown_gain=1.5,basket_track_max_speed=.3)
        self.opts=options({})

    def geometry(self,y,speed=0.):
        base=np.array([0.,y,0.])
        nominal=parking_state(base,[0.,speed],0.,[0.,1.5,.5],self.cfg)
        return nominal,region_state(nominal,base,0.,self.cfg,self.opts)

    def test_different_stopping_points_are_valid_without_center_pull(self):
        for y in (.43,.5,.57):
            nominal,region=self.geometry(y)
            self.assertTrue(region['settled'])
            self.assertEqual(region['distance'],0.)
            np.testing.assert_array_equal(region['desired_velocity'],0.)
            np.testing.assert_array_equal(region['observation'],nominal['observation'])

    def test_outside_region_has_signed_progress_and_speed_gate(self):
        _,near=self.geometry(.3)
        _,far=self.geometry(.7)
        self.assertFalse(near['in_position']);self.assertFalse(far['in_position'])
        self.assertGreater(near['error_world'][1],0.)
        self.assertLess(far['error_world'][1],0.)
        _,moving=self.geometry(.5,.1)
        self.assertTrue(moving['in_position']);self.assertFalse(moving['settled'])

    def test_policy_can_drive_through_region_and_release_requires_inside(self):
        from unittest.mock import patch
        from tests.test_basket_mobile import config
        from gym_dcmm.utils import basket_mobile as mobile
        cfg=config(preserve_fixed_throw=True,hold_position=False,approach_braking=False,
            release_position_tolerance=.08,release_speed_tolerance=.15)
        state=mobile.new_state(1.);state['physics_time']=1.
        _,inside=self.geometry(.5)
        command=mobile.base_command(state,[0.,.2],.04,cfg,inside)
        self.assertGreater(command[1],0.) # no forced stop at the first valid point
        state['unlocked']=True
        for y,expected in ((.5,True),(.41,False)):
            _,parking=self.geometry(y)
            with patch('gym_dcmm.utils.basket_mobile.update_launch'):
                mobile.update_physics(state,{},parking,[0,.5,.7],[0,0,0],[0,1,.55],
                    1.,.002,True,False,cfg)
            self.assertEqual(state['last_contact_qualified'],expected)

    def test_invalid_options(self):
        for value in ({'y_offset_range':[.1,-.1]},{'y_offset_range':[0.,0.]},{'bad':1}):
            with self.assertRaises(ValueError):options(value)

if __name__=='__main__':unittest.main()

"""Mock controller tests; never connects to hardware."""
import unittest
from small_joint_move import execute_move,target_from


class Fake:
    connected=True
    mode=0
    def __init__(self):
        self.q=[0.]*6;self.motion_calls=[];self.stops=[];self.return_code=0
        self.drift=False;self.stuck=False;self.limit=False
    def get_state(self):return 0,2
    def get_err_warn_code(self):return 0,[0,0]
    def get_servo_angle(self,**kw):
        assert kw==dict(is_radian=False,is_real=True)
        q=self.q[:]
        if self.drift and self.motion_calls:q[1]=1
        return 0,q
    def is_joint_limit(self,q,**kw):return 0,self.limit
    def set_servo_angle(self,**kw):
        self.motion_calls.append(kw)
        if not self.stuck:self.q=kw['angle'][:]
        return self.return_code
    def set_state(self,state):self.stops.append(state);return 0


class Tests(unittest.TestCase):
    def run_move(self,arm,current=None):
        ticks=[0.]
        def sleep(dt):ticks[0]+=max(dt,.05)
        return execute_move(arm,current or [0.]*6,[.5,0,0,0,0,0],1,1,2,
                            lambda row:None,clock=lambda:ticks[0],sleep=sleep)
    def test_small_absolute_move(self):
        arm=Fake();self.run_move(arm)
        self.assertEqual(len(arm.motion_calls),1)
        cmd=arm.motion_calls[0]
        self.assertEqual(cmd['angle'],[.5,0,0,0,0,0]);self.assertFalse(cmd['relative'])
        self.assertFalse(cmd['is_radian']);self.assertEqual(arm.stops,[])
    def test_invalid_delta(self):
        for delta in (0,3,float('nan')):
            with self.assertRaises(ValueError):target_from([0.]*6,1,delta)
    def test_changed_pose_refuses_motion(self):
        arm=Fake();arm.q[0]=1
        with self.assertRaises(RuntimeError):self.run_move(arm)
        self.assertEqual(arm.motion_calls,[])
    def test_controller_limit_refuses_motion(self):
        arm=Fake();arm.limit=True
        with self.assertRaises(RuntimeError):self.run_move(arm)
        self.assertEqual(arm.motion_calls,[])
    def test_unselected_joint_drift_stops(self):
        arm=Fake();arm.drift=True
        with self.assertRaises(RuntimeError):self.run_move(arm)
        self.assertEqual(arm.stops,[4])
    def test_failed_command_stops(self):
        arm=Fake();arm.return_code=1
        with self.assertRaises(RuntimeError):self.run_move(arm)
        self.assertEqual(arm.stops,[4])
    def test_timeout_stops(self):
        arm=Fake();arm.stuck=True
        with self.assertRaises(TimeoutError):self.run_move(arm)
        self.assertEqual(arm.stops,[4])
    def test_wrong_mode_refuses_motion(self):
        arm=Fake();arm.mode=1
        with self.assertRaises(RuntimeError):self.run_move(arm)
        self.assertEqual(arm.motion_calls,[])


if __name__=='__main__':unittest.main()

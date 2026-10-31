"""State replay checks do not open a graphical window or reintegrate physics."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from ycb_objects.bounce_viewer_replay import (STATE_SPEC, frame_index, restore_frame,
                                            joined_passive_viewer)


class ExactReplayTests(unittest.TestCase):
    def test_passive_viewer_closes_before_joining_created_ui_threads(self):
        import mujoco.viewer
        from types import SimpleNamespace

        calls = []
        old = object()
        class Thread:
            name = 'viewer-ui'
            def join(self):
                calls.append('join')
            def is_alive(self):
                return False
        new = Thread()
        viewer = SimpleNamespace(close=lambda: calls.append('close'))
        cleanup = {}
        with patch('ycb_objects.bounce_viewer_replay.threading.enumerate',
                   side_effect=[[old], [old, new]]), \
                patch.object(mujoco.viewer, 'launch_passive', return_value=viewer):
            with joined_passive_viewer(None, None, cleanup_stats=cleanup):
                calls.append('body')
        self.assertEqual(calls, ['body', 'close', 'join'])
        self.assertEqual(cleanup['joined_threads'], 1)
        self.assertEqual(cleanup['threads_alive_after_join'], 0)

    def test_frame_selection_preserves_recorded_time_without_interpolation(self):
        times = np.array([5., 5.002, 5.004, 5.006])
        self.assertEqual(frame_index(times, 0.), 0)
        self.assertEqual(frame_index(times, .003), 1)
        self.assertEqual(frame_index(times, .0041), 2)
        self.assertEqual(frame_index(times, 10.), 3)

    def test_replay_restores_exact_pose_velocity_and_does_not_step(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".000125" integrator="RK4"/>
          <worldbody><body pos="0 0 1"><freejoint/>
          <geom type="sphere" size=".04" mass=".058"/></body></worldbody>
        </mujoco>''')
        original = mujoco.MjData(model)
        states, positions, velocities = [], [], []
        for _ in range(4):
            mujoco.mj_step(model, original, nstep=16)
            state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
            mujoco.mj_getState(model, original, state, STATE_SPEC)
            states.append(state)
            positions.append(original.qpos.copy())
            velocities.append(original.qvel.copy())
        trace = dict(state=np.asarray(states), qpos=np.asarray(positions),
                     qvel=np.asarray(velocities))
        replay = mujoco.MjData(model)
        with patch('ycb_objects.bounce_viewer_replay.mujoco.mj_step',
                   side_effect=AssertionError('Replay must not integrate')):
            for index in range(4):
                restore_frame(model, replay, trace, index)
                np.testing.assert_array_equal(replay.qpos, positions[index])
                np.testing.assert_array_equal(replay.qvel, velocities[index])
                self.assertEqual(replay.time, states[index][0])


if __name__ == '__main__':
    unittest.main()

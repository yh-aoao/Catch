import unittest
import numpy as np
from tests.test_basket_mixed import environment


class BasketRangeTests(unittest.TestCase):
    def test_random_targets_restore_baseline_and_remain_static(self):
        env=environment(basket_range=dict(y_offset_range=[-.1,.1],original_probability=0.))
        try:
            baseline=env.basket_cfg.basket_center.copy()
            targets=[]
            for seed in (11,12,13):
                env.reset(seed=seed)
                target=env.basket_center.copy();targets.append(target[1])
                self.assertLessEqual(abs(target[1]-baseline[1]),.100001)
                np.testing.assert_array_equal(env.basket_cfg.basket_center,baseline)
                env.step(dict(base=np.zeros(2),arm=np.zeros(6),hand=np.zeros(12)))
                np.testing.assert_array_equal(env.basket_center,target)
            self.assertGreater(len(set(targets)),1)
            env.range_options['original_probability']=1.
            env.reset(seed=11)
            np.testing.assert_array_equal(env.basket_center,baseline)
            self.assertTrue(env.range_rehearsal)
        finally:env.close()

    def test_rejects_invalid_ranges_before_constructing_physics(self):
        for options in [dict(y_offset_range=[.1,-.1]),dict(y_offset_range=[0]),
                        dict(original_probability=1.1),dict(y_offset_range=[0,float('nan')])]:
            with self.assertRaises(ValueError):environment(basket_range=options)


if __name__=='__main__':unittest.main()

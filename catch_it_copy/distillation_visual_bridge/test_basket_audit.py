import unittest
from .environments import reset_observation

class BasketResetTests(unittest.TestCase):
    def test_basket_seeds_gym_generator_without_changing_catch_reset_api(self):
        class Env:
            def reset(self,**kwargs):self.kwargs=kwargs;return 'observation',{}
        env=Env()
        self.assertEqual(reset_observation(env,'basket',123),'observation')
        self.assertEqual(env.kwargs,{'seed':123})
        reset_observation(env,'roll',123);self.assertEqual(env.kwargs,{})

if __name__=='__main__':unittest.main()

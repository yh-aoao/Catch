"""Exercise the actual XML reset method without requiring MuJoCo DLLs."""
import ast
import unittest
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from distillation.teachers import ROOT


class ThrowObjectTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'gym_dcmm/envs/DcmmVecEnv.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DcmmVecEnv')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_object_throw')
        cfg_tree = ast.parse((ROOT / 'configs/env/DcmmCfg.py').read_text(encoding='utf-8-sig'))
        shape = next(ast.literal_eval(n.value) for n in cfg_tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'throw_object_shape' for t in n.targets))
        self.cfg = SimpleNamespace(throw_object_shape=shape, object_mass=[.035, .075],
            object_damping=[.005, .02], object_size={'sphere': [[.035, .045]], 'box': [[.03, .03]] * 3},
            object_shape=['box', 'sphere'], train_object_filter=['box'], object_mesh=['box_mesh', 'ball_mesh'])
        ns = {'ET': ET, 'np': np, 'DcmmCfg': self.cfg}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), ns)
        self.reset_object = ns['_reset_object_throw']

    def environment(self, training):
        xml = '<mujoco><worldbody><body name="object"><inertial mass=".05"/>' \
              '<joint damping=".01"/><geom name="object" type="mesh" mesh="box_mesh" size=".03 .03 .03"/>' \
              '</body></worldbody></mujoco>'
        return SimpleNamespace(object_motion='throw', object_train=training,
                               Dcmm=SimpleNamespace(model_xml_string=xml))

    def test_sphere_for_primitive_and_mesh_modes_after_repeated_reset(self):
        rng = np.random.get_state()
        try:
            for training in (True, False):
                env = self.environment(training)
                for seed in range(20):
                    np.random.seed(seed)
                    xml = self.reset_object(env)
                    geom = ET.fromstring(xml).find('.//geom')
                    self.assertEqual(geom.get('type'), 'sphere')
                    self.assertNotIn('mesh', geom.attrib)
                    self.assertEqual(len(geom.get('size').split()), 1)
                    self.assertTrue(.035 <= float(geom.get('size')) <= .045)
                    env.Dcmm.model_xml_string = xml
        finally:
            np.random.set_state(rng)

    def test_none_restores_legacy_shape_filter(self):
        self.cfg.throw_object_shape = None
        geom = ET.fromstring(self.reset_object(self.environment(True))).find('.//geom')
        self.assertEqual(geom.get('type'), 'box')


if __name__ == '__main__':
    unittest.main()

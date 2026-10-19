import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from ycb_objects.loader import load, compose, remap_contacts


class YCBTests(unittest.TestCase):
    def test_downloaded_assets(self):
        for name in ('tennis_ball', 'baseball', 'tomato_soup_can'):
            resource = load(name)
            root = ET.fromstring(compose('<mujoco><asset/><worldbody><body name="object"><joint type="free"/><geom name="object" type="sphere" size=".04" mass=".05"/></body></worldbody></mujoco>', resource))
            body = root.find('.//body')
            geoms = body.findall('geom')
            self.assertIsNone(body.find('inertial'))
            self.assertEqual(geoms[0].get('contype'), '0')
            self.assertAlmostEqual(sum(float(g.get('mass')) for g in geoms), resource[1])
            self.assertGreater(len(geoms), 2)
            for asset in root.findall('./asset/*'):
                if asset.get('file'):
                    self.assertTrue(Path(asset.get('file')).is_file())

    def test_multi_piece_contacts(self):
        result = remap_contacts(dict(hand_contacts=[101, 102], base_contacts=[102]),
                                [[101, 4], [5, 102], [101, 102]], [100, 101, 102], 100)
        self.assertEqual(result['object_contacts'].tolist(), [4, 5])
        self.assertEqual(result['hand_contacts'].tolist(), [100, 100])
        self.assertEqual(result['base_contacts'].tolist(), [100])

    def test_reject_path(self):
        with self.assertRaises(ValueError):
            load('../bad')

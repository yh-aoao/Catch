import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from ycb_objects.loader import load, compose, remap_contacts


class YCBTests(unittest.TestCase):
    def test_contact_source_survives_replacement(self):
        for friction, solref, solimp in (
                ('0.95 0.3 0.1', '0.001 1', '0.998 0.998 0.001'),
                ('1.2 0.1 0.1', '0.02 1', '0.9 0.95 0.01'),
                ('0.5 0.05 0.015', '0.008 0.1', '0.95 0.998 0.002')):
            xml = ('<mujoco><worldbody><body name="object"><joint type="free"/>'
                   '<geom name="object" type="sphere" size=".04" friction="{}" '
                   'solref="{}" solimp="{}" condim="4"/></body></worldbody></mujoco>').format(
                       friction, solref, solimp)
            for name in ('tennis_ball', 'baseball', 'tomato_soup_can'):
                root = ET.fromstring(compose(xml, load(name)))
                marker = root.find('.//geom[@name="object"]')
                expected = dict(friction=friction, solref=solref, solimp=solimp, condim='4')
                self.assertEqual({key: marker.get(key) for key in expected}, expected)
                self.assertEqual(marker.get('contype'), '0')
                self.assertEqual(marker.get('conaffinity'), '0')
                for geom in root.findall('.//body/geom'):
                    if geom.get('contype') == '1':
                        self.assertEqual({key: geom.get(key) for key in expected}, expected)

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

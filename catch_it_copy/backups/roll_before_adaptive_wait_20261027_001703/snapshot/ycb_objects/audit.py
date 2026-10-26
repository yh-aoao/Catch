"""Compile isolated YCB bodies and report collision and inertia properties."""
import argparse
import json
import xml.etree.ElementTree as ET

from ycb_objects.loader import ROOT, compose, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+', default=['tennis_ball', 'baseball'])
    args = parser.parse_args()
    import mujoco
    import numpy as np

    source = ET.parse(ROOT.parent / 'urdf/x1_xarm6_leap_right_object.xml')
    original = source.find('.//body[@name="object"]/inertial')
    results = {}
    for name in args.objects:
        resource = load(name)
        xml = compose('<mujoco><worldbody><body name="object"><joint type="free"/>'
                      '<geom name="object" type="sphere" size=".04" friction="1.2 .1 .1" '
                      'solref=".02 1" solimp=".9 .95 .01" condim="4"/>'
                      '</body></worldbody></mujoco>', resource)
        model = mujoco.MjModel.from_xml_string(xml)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'object')
        collision_ids = np.flatnonzero((model.geom_bodyid == body_id) &
                                      ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        assert len(collision_ids) > 0
        assert marker_id not in collision_ids
        np.testing.assert_allclose(model.body_mass[body_id], resource[1])
        assert np.all(np.isfinite(model.body_inertia[body_id]))
        assert np.all(model.body_inertia[body_id] > 0)
        for attribute in ('geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim'):
            for geom_id in collision_ids:
                np.testing.assert_allclose(getattr(model, attribute)[geom_id],
                                           getattr(model, attribute)[marker_id])
        results[name] = dict(mass_kg=float(model.body_mass[body_id]),
                            bounding_radius_m=resource[2],
                            collision_parts=len(collision_ids),
                            inertia_kg_m2=model.body_inertia[body_id].tolist(),
                            center_of_mass_m=model.body_ipos[body_id].tolist(),
                            inertia_quaternion=model.body_iquat[body_id].tolist(),
                            legacy_inertia_kg_m2=[float(value) for value in original.get('diaginertia').split()])
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

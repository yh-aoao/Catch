"""Primitive evaluation presets. Ball materials are not calibrated real-world models."""
import xml.etree.ElementTree as ET

PRESETS = {
    'sphere': ('sphere', (.04,), .05),
    'tennis_ball': ('sphere', (.0335,), .058),
    'pingpong_ball': ('sphere', (.02,), .0027),
    'box': ('box', (.03, .03, .03), .05),
    'cylinder': ('cylinder', (.03, .04), .05),
    'ellipsoid': ('ellipsoid', (.03, .04, .05), .05),
    'capsule': ('capsule', (.025, .03), .05),
}


def object_xml(xml, preset):
    kind, size, mass = PRESETS[preset]
    root = ET.fromstring(xml)
    body = root.find(".//body[@name='object']")
    geom = body.find("geom[@name='object']")
    inertial = body.find('inertial')
    if inertial is not None:
        body.remove(inertial)
    geom.attrib.pop('mesh', None)
    geom.attrib.pop('density', None)
    geom.set('type', kind)
    geom.set('size', ' '.join(map(str, size)))
    geom.set('mass', str(mass))
    return ET.tostring(root, encoding='unicode')

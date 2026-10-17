"""Pure geometry for a ramp descending toward world -Y."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SlideConfig:
    angle_deg: float = 20.
    length: float = 1.2
    width: float = .7
    thickness: float = .04
    edge_y: float = .9
    edge_z: float = .7
    friction: float = .15
    mass: float = .05
    half_size: tuple = (.03, .03, .02)
    release_fraction: float = .7
    release_x_range: float = .08
    wait_z: float = .48
    hold_seconds: float = .3
    contact_gap: float = .08

    def validate(self):
        if not 5 <= self.angle_deg <= 40 or min(self.length, self.width, self.thickness, self.mass, *self.half_size) <= 0:
            raise ValueError('Invalid ramp dimensions, angle or object size/mass')
        if not 0 < self.release_fraction < 1 or not 0 <= self.friction < np.tan(np.radians(self.angle_deg)):
            raise ValueError('Release must lie on ramp; friction must permit gravity sliding')
        if self.release_x_range + self.half_size[0] >= self.width/2:
            raise ValueError('Release range extends outside ramp')
        if not 0 < self.wait_z < self.edge_z or self.hold_seconds <= 0 or self.contact_gap < 0:
            raise ValueError('Invalid catching height or hold criterion')
        return self


def frame(cfg):
    a = np.radians(cfg.angle_deg)
    rotation = np.array([[1., 0., 0.], [0., np.cos(a), -np.sin(a)], [0., np.sin(a), np.cos(a)]])
    edge = np.array([0., cfg.edge_y, cfg.edge_z])
    center = edge + rotation @ np.array([0., cfg.length/2, -cfg.thickness/2])
    quat = np.array([np.cos(a/2), np.sin(a/2), 0., 0.])
    return edge, center, rotation, quat


def release_pose(cfg, x=0.):
    edge, _, rotation, quat = frame(cfg)
    return edge + rotation @ np.array([x, cfg.length*cfg.release_fraction, cfg.half_size[2]+.001]), quat


def intercept_target(position, velocity, cfg, on_ramp, gravity=9.81):
    """Plan only a waiting point; never set the physical object's velocity."""
    p, v = np.asarray(position, float).copy(), np.asarray(velocity, float).copy()
    if on_ramp:
        edge, _, rot, _ = frame(cfg)
        distance = max(0., float((p-edge) @ rot[:, 1]))
        downhill = -rot[:, 1]
        acceleration = gravity*(np.sin(np.radians(cfg.angle_deg))-cfg.friction*np.cos(np.radians(cfg.angle_deg)))
        speed = np.sqrt(max(0., float(v @ downhill))**2 + 2*max(0., acceleration)*distance)
        p = edge + np.array([p[0], 0., cfg.half_size[2]])
        v = downhill*speed + np.array([v[0], 0., 0.])
    dz = max(0., p[2]-cfg.wait_z)
    t = max(0., (v[2]+np.sqrt(v[2]**2+2*gravity*dz))/gravity)
    target = p + v*min(t, 1.)
    target[2] = cfg.wait_z
    # Keep the catching hand outside the lower edge, not under the ramp.
    target[1] = min(target[1], cfg.edge_y-.14)
    target[0] = np.clip(target[0], -.65, .65)
    return target

"""Bounce 候选配置：尚未通过可达性验证。修改此文件即可调整各组。

物理标量为固定值；发球条件为 [min, max]，两端相等表示固定。
legacy 使用 DcmmCfg 中原参数及原始位置、方向、自旋范围。
"""
from copy import deepcopy
from types import SimpleNamespace
import math


PHYSICS_BASE = dict(
    radius=0.040, mass=0.050, friction=[0.50, 0.05, 0.015],
    restitution=0.80, solref_timeconst=0.008, damp_scale=0.50,
    joint_damping=0.0002, hand_friction=[2.0, 0.5, 0.1],
)

# 每组继承 PHYSICS_BASE，仅覆盖下列字段。
PHYSICS_GROUPS = {
    "P0": {},
    "P1": dict(radius=0.039),
    "P2": dict(radius=0.041),
    "P3": dict(mass=0.045),
    "P4": dict(mass=0.055),
    "P5": dict(friction=[0.45, 0.05, 0.015]),
    "P6": dict(friction=[0.55, 0.05, 0.015]),
    "P7": dict(restitution=0.78),
    "P8": dict(restitution=0.82),
}

LAUNCH_BASE = dict(
    x=[0.0, 0.0], y=[2.4, 2.4], height=[0.9, 0.9],
    speed=[1.0, 1.0], vz=[-0.2, -0.2], angle_deg=[0.0, 0.0],
    spin_factor=[1.0, 1.0], spin_z=[0.0, 0.0],
)
LAUNCH_GROUPS = {
    "L0": {},
    "L1": dict(speed=[0.95, 0.95]),
    "L2": dict(speed=[1.05, 1.05]),
    "L3": dict(angle_deg=[-10.0, -10.0]),
    "L4": dict(angle_deg=[10.0, 10.0]),
    # 小幅速度／方向随机化，位置和自旋仍固定。
    "gentle": dict(speed=[0.95, 1.05], angle_deg=[-10.0, 10.0]),
}


def parse_groups(selection, available):
    """Comma-separated IDs; reject typos before constructing the MuJoCo model."""
    if not isinstance(selection, str):
        raise ValueError("Bounce groups must be a comma-separated string, e.g. P0,P1")
    names = [name.strip() for name in selection.split(",")]
    valid = set(available) | {"legacy"}
    if any(name not in valid for name in names) or len(set(names)) != len(names):
        raise ValueError("Invalid/duplicate bounce groups {!r}; choose from {}".format(
            selection, ", ".join(sorted(valid))))
    return names


def episode_config(source, physics_names, launch_names, rng):
    """Return an isolated per-environment config; never mutate DcmmCfg globals."""
    cfg = SimpleNamespace(**deepcopy({
        key: value for key, value in vars(source).items() if key.startswith("bounce_")
    }))
    # Single groups consume no selection RNG draw, preserving legacy sampling order.
    physics = physics_names[0] if len(physics_names) == 1 else str(rng.choice(physics_names))
    launch = launch_names[0] if len(launch_names) == 1 else str(rng.choice(launch_names))
    if physics != "legacy":
        cfg.bounce_inertia = None  # P 组沿用实心球惯量，独立于旧固定基线。
        values = dict(deepcopy(PHYSICS_BASE), **deepcopy(PHYSICS_GROUPS[physics]))
        if set(values) != set(PHYSICS_BASE):
            raise ValueError("Unknown bounce physics field in " + physics)
        for key, value in values.items():
            numbers = value if isinstance(value, (list, tuple)) else [value]
            if not all(math.isfinite(v) and v >= 0 for v in numbers):
                raise ValueError("Invalid bounce physics value {}={!r}".format(key, value))
            if key in ("radius", "mass", "solref_timeconst") and value <= 0:
                raise ValueError(key + " must be positive")
            if key == "restitution" and value > 1:
                raise ValueError("restitution must be in [0, 1]")
            if key in ("friction", "hand_friction") and len(numbers) != 3:
                raise ValueError(key + " must have three components")
            setattr(cfg, "bounce_" + key, value if key == "damp_scale" else [value, value])
    launch_values = dict(
        x=[-0.25, 0.25], y=[2.1, 2.7], height=cfg.bounce_init_height,
        speed=cfg.bounce_init_speed, vz=cfg.bounce_init_vz,
        angle_deg=[-25.0, 25.0], spin_factor=[0.5, 1.5], spin_z=[-3.0, 3.0],
    ) if launch == "legacy" else dict(deepcopy(LAUNCH_BASE), **deepcopy(LAUNCH_GROUPS[launch]))
    for key, value in launch_values.items():
        if len(value) != 2 or not all(math.isfinite(v) for v in value) or value[0] > value[1]:
            raise ValueError("Invalid bounce launch range {}={!r}".format(key, value))
    if launch_values['speed'][0] < 0:
        raise ValueError("Horizontal speed must be nonnegative")
    return cfg, launch_values, physics, launch

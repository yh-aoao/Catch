"""XY interception shaping, with a broad world-height waiting band."""
import numpy as np


def target(position, velocity, radius, cfg, gravity=9.81):
    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    edge = cfg.roll_table_pos[1] - cfg.roll_table_size[1]
    plane = cfg.roll_wait_z + radius
    on_table_side = p[1] + radius >= edge
    waiting = on_table_side or p[2] > plane
    result = p.copy()
    if waiting:
        exit_time = max(0., (edge - p[1]) / v[1]) if v[1] < -.01 and p[1] > edge else 0.
        height = cfg.roll_table_height + radius if exit_time > 0 else p[2]
        vz = 0. if exit_time > 0 else v[2]
        fall = (vz + np.sqrt(vz*vz + 2*gravity*max(0., height-plane))) / gravity
        result[:2] += v[:2] * (exit_time + fall)
        result[1] = min(result[1], edge - cfg.roll_wait_edge_margin)
        result[2] = cfg.roll_wait_z
    return result, waiting


def shaping(ee, previous_ee, goal, waiting, cfg):
    distance = np.linalg.norm(ee[:2] - goal[:2])
    xy = cfg.roll_w_xy / (1 + (distance / cfg.roll_sigma_xy)**2)
    progress = 0. if previous_ee is None else cfg.roll_w_approach * (np.linalg.norm(previous_ee[:2]-goal[:2])-distance)
    # Same current target for both distances: target drift alone earns nothing.
    height = -cfg.roll_wait_w_height * min(1., max(0., abs(ee[2]-cfg.roll_wait_z)-cfg.roll_wait_z_band) / .15) if waiting else 0.
    return float(xy), float(progress), float(height)

"""Basket carry pose and palm-frame measurements; no ball actuation."""
import numpy as np


def cup_joints(cfg=None):
    fingers = np.array([.6,.5,.5] if cfg is None else cfg.mobile_hold_hand_flex, dtype=float)
    thumb = np.array([.4,.4,.4] if cfg is None else cfg.mobile_hold_thumb_flex, dtype=float)
    result = np.zeros(16)
    for first in (0,4,8):
        result[[first,first+2,first+3]] = fingers
    result[13:16] = thumb
    return result


def arm_joints(base, cfg=None):
    return np.asarray(base,dtype=float).copy() + (0. if cfg is None else np.radians(cfg.mobile_hold_arm_offset_deg))


def ball_position(data, cfg=None):
    local = np.array([0.,.03,.07] if cfg is None else cfg.mobile_hold_ball_local,dtype=float)
    body = data.body('link6')
    return body.xpos + body.xmat.reshape(3,3)@local


def palm_coordinates(data, position):
    body=data.body('link6')
    return body.xmat.reshape(3,3).T@(np.asarray(position)-body.xpos)


def validate_joints(model, qpos_start, values):
    for offset,value in enumerate(values):
        matches=np.flatnonzero(model.jnt_qposadr==qpos_start+offset)
        if len(matches)!=1:
            raise ValueError('Expected one scalar joint at qpos '+str(qpos_start+offset))
        joint=int(matches[0])
        if model.jnt_limited[joint] and not (model.jnt_range[joint,0]<=value<=model.jnt_range[joint,1]):
            raise ValueError('Basket carry pose exceeds joint range: '+model.joint(joint).name)

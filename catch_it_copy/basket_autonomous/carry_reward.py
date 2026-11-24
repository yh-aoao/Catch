"""Reward-only shaping. No target posture and no actuator or simulation writes."""
import math
import numpy as np

VERSION='carry_navigation_v2'
# Broad ball-center envelope in the moving palm frame, not a preferred point.
LOW=np.array([-.10,-.08,-.035])
HIGH=np.array([.10,.20,.16])


def carry_terms(local,relative_velocity,touch,released,palm_forward,ball_forward,dt):
    local=np.asarray(local,float);velocity=np.asarray(relative_velocity,float)
    speed=float(np.linalg.norm(velocity))
    # Only reward stable carrying, fade while accelerating a forward throw.
    gate=float(np.clip((.80-max(palm_forward,ball_forward))/.55,0.,1.)) if touch and not released else 0.
    inside=bool(np.all(local>=LOW) and np.all(local<=HIGH))
    stable=.08*dt*gate*float(inside)*math.exp(-(speed/.12)**2)
    slip=-.35*dt*gate*min((speed/.30)**2,1.)
    lower=np.clip(1.-(local-LOW)/.025,0.,1.)
    upper=np.clip(1.-(HIGH-local)/.025,0.,1.)
    outward=float(np.max(lower*np.maximum(-velocity,0.)+upper*np.maximum(velocity,0.)))
    edge=-.50*dt*gate*min(outward/.15,1.)
    return dict(carry_stability=stable,carry_slip=slip,carry_edge=edge),dict(
        slip_speed=speed,carry_gate=gate,in_support_envelope=inside,outward_speed=outward)


def region_distance(relative_goal,x_range,y_range):
    goal=np.asarray(relative_goal,float)
    bounds=np.array([x_range,y_range],float)
    return float(np.linalg.norm(goal[:2]-np.clip(goal[:2],bounds[:,0],bounds[:,1])))


def navigation_progress(best,current,eligible,weight=4.,cap=1.):
    # Best-so-far distance prevents farming by driving backwards and forwards.
    if not eligible:return best,0.
    updated=min(best,max(0.,current))
    return updated,weight*max(0.,min(best,cap)-min(updated,cap))


def purposeful_launch(position,velocity,target,predicted_error):
    delta=np.asarray(target)[:2]-np.asarray(position)[:2]
    direction=delta/max(float(np.linalg.norm(delta)),1e-9)
    forward=float(np.dot(np.asarray(velocity)[:2],direction))
    # No requirement to hit or to be stopped: those have their own score/release criteria.
    return bool(forward>=.60 and (predicted_error is not None or velocity[2]>.10))


def release_distance_reward(distance,bounds,bonus=15.,penalty=25.,margin=.05,penalty_distance=.5):
    """Once per confirmed release. Smooth at both interval boundaries."""
    low,high=bounds
    if low<=distance<=high:
        return bonus*min(1.,max(0.,min(distance-low,high-distance)/margin))
    deviation=max(low-distance,distance-high,0.)
    return -penalty*min(deviation/penalty_distance,1.)

"""Explicit passive aerodynamic prior for the held-out lightweight ball."""
import numpy as np

def sphere_drag(velocity,radius):
    velocity=np.asarray(velocity,dtype=float)
    return -.5*1.225*.47*np.pi*radius**2*np.linalg.norm(velocity)*velocity

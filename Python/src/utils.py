# utils.py
"""
Small helper utilities used across modules.
"""

import numpy as np

def circular_mean(angles):
    """
    angles: iterable of angles in radians
    return circular mean in radians in [-pi, pi]
    """
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return np.arctan2(s, c)

def angular_error_deg(a, b):
    """
    circular angular error between a and b (radians), return degrees absolute
    """
    d = np.angle(np.exp(1j*(a-b)))
    return np.abs(np.rad2deg(d))

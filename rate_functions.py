"""
Communication rate: value and gradient.
Consolidates: avg_data_rate.m, rate_grad.m
"""
import numpy as np
from parameters import P, sigma_0, alpha_0, B, H, snr_const
from models import user_quad_distance


def avg_data_rate(S, s_c, N_total):
    """
    Average downlink communication rate (eq. 7).
    S: (2, N) trajectory points
    s_c: (2,) comm user position
    N_total: total number of waypoints across all stages
    Returns: scalar average rate [bits/s]
    """
    d = user_quad_distance(S, s_c)
    return (B / N_total) * np.sum(np.log2(1 + snr_const / d**2))


def rate_grad(S_stage, s_c, N_total):
    """
    Gradient of average rate w.r.t. waypoints of current stage.
    S_stage: (2, N_stg) current stage trajectory
    s_c: (2,) comm user position
    N_total: total number of waypoints across all stages
    Returns: (N_stg,) grad_x, (N_stg,) grad_y

    Matches rate_grad.m.
    """
    d = user_quad_distance(S_stage, s_c)
    dx = S_stage[0, :] - s_c[0]
    dy = S_stage[1, :] - s_c[1]

    common = -2 * snr_const / (d**2 * (d**2 + snr_const))
    scale = B / (N_total * np.log(2))

    grad_x = common * dx * scale
    grad_y = common * dy * scale

    return grad_x, grad_y

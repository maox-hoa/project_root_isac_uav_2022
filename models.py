"""
Core model functions: distances, channels, energy, velocity, trajectory init.
Consolidates: relative_distance.m, user_quad_distance.m, g_k.m, sigma_k.m,
              calc_velocity.m, calc_real_energy.m, init_trajectory.m
"""
import numpy as np
from parameters import *


# ======================== Distance Functions ========================

def relative_distance(S_hover, s_target):
    """
    Distance from hover points to sensing target (eq. 8).
    S_hover: (2, K) hover positions
    s_target: (2,) target position
    Returns: (K,) distance vector
    """
    dx = S_hover[0, :] - s_target[0]
    dy = S_hover[1, :] - s_target[1]
    return np.sqrt(dx**2 + dy**2 + H**2)


def user_quad_distance(S, s_c):
    """
    Distance from trajectory points to communication user (eq. 3).
    S: (2, N) trajectory positions
    s_c: (2,) comm user position
    Returns: (N,) distance vector
    """
    dx = S[0, :] - s_c[0]
    dy = S[1, :] - s_c[1]
    return np.sqrt(dx**2 + dy**2 + H**2)


# ======================== Channel Functions ========================

def g_k(d_s):
    """Two-way sensing channel power gain (eq. 11)."""
    return beta_0 / (d_s ** 4)


def sigma_k(d_s):
    """Std of distance measurement noise (eq. 14)."""
    return np.sqrt((a * sigma_0**2) / (P * G_p * g_k(d_s)))


# ======================== Velocity ========================

def calc_velocity(S, s_start):
    """
    Velocity matrix from trajectory and starting position (eq. 1).
    S: (2, N) waypoints
    s_start: (2,) starting position
    Returns: (2, N) velocity matrix
    """
    S_ext = np.column_stack([s_start, S])
    V = np.diff(S_ext, axis=1) / T_f
    return V


# ======================== Energy ========================

def propulsion_power(V_norm):
    """UAV propulsion power as function of speed (eq. 29)."""
    term1 = P_0 * (1 + 3 * V_norm**2 / U_tip**2)
    inner = np.sqrt(1 + V_norm**4 / (4 * v_0**4)) - V_norm**2 / (2 * v_0**2)
    term2 = P_I * np.sqrt(np.maximum(inner, 0))
    # Note: MATLAB calc_real_energy.m uses V^2 here, not V^3 as in the paper
    term3 = 0.5 * D_0 * rho * s_rotor * A_rotor * V_norm**2
    return term1 + term2 + term3


def hover_power():
    """Power while hovering (V=0)."""
    return propulsion_power(0.0)


def calc_real_energy(S, s_start):
    """
    Total real energy consumed for a trajectory stage.
    S: (2, N) waypoints
    s_start: (2,) starting position
    Returns: scalar energy [J]
    """
    V = calc_velocity(S, s_start)
    V_norm = np.linalg.norm(V, axis=0)
    Nh = S.shape[1] // mu

    flight_energy = T_f * np.sum(propulsion_power(V_norm))
    hover_energy = T_h * Nh * hover_power()

    return flight_energy + hover_energy


# ======================== Trajectory Initialization ========================

def init_trajectory(s_start, s_end, N):
    """
    Create initial straight-line trajectory (eq. 60).
    s_start: (2,) start position
    s_end: (2,) end position (midpoint target)
    N: number of waypoints
    Returns: (2, N) initial trajectory
    """
    direction = s_end - s_start
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction = np.array([1.0, 0.0])
        dist = 1.0
    unit = direction / dist

    S_init = np.zeros((2, N))
    for i in range(N):
        S_init[:, i] = s_start + V_str * (i + 1) * unit * T_f

    return S_init


# ======================== Hover Point Extraction ========================

def get_hover_indices(N, mu_val=None):
    """
    Get hover point indices within a stage (0-indexed).
    MATLAB: mu:mu:N_stg → Python: [mu-1, 2*mu-1, ..., N-1] if N divisible
    """
    if mu_val is None:
        mu_val = mu
    return list(range(mu_val - 1, N, mu_val))


def get_hover_points(S, mu_val=None):
    """
    Extract hover points from trajectory.
    S: (2, N) trajectory
    Returns: (2, K) hover points, list of indices
    """
    idx = get_hover_indices(S.shape[1], mu_val)
    return S[:, idx], idx


def get_all_hover_indices(n_stages, N_stg_val=None, mu_val=None):
    """
    Get all hover indices across multiple stages in concatenated trajectory.
    Returns list of indices into the concatenated (2, n_stages*N_stg) array.
    """
    if N_stg_val is None:
        N_stg_val = N_stg
    if mu_val is None:
        mu_val = mu

    indices = []
    for m in range(n_stages):
        stage_offset = m * N_stg_val
        stage_hover = get_hover_indices(N_stg_val, mu_val)
        indices.extend([stage_offset + idx for idx in stage_hover])
    return indices

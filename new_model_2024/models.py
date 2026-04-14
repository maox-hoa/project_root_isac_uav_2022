"""
Core model functions — extended for M CUs, K STs.
Distance, channel, energy, velocity, trajectory init, hover extraction.
"""
import numpy as np
from parameters import *


# ======================== Distance Functions ========================

def relative_distance(S_hover, s_target):
    """
    Distance from hover points to ONE sensing target (eq. 10).
    S_hover: (2, K_h)
    s_target: (2,)
    Returns: (K_h,)
    """
    dx = S_hover[0, :] - s_target[0]
    dy = S_hover[1, :] - s_target[1]
    return np.sqrt(dx**2 + dy**2 + H**2)


def user_quad_distance(S, s_c):
    """
    Distance from trajectory points to ONE communication user (eq. 4).
    S: (2, N)
    s_c: (2,)
    Returns: (N,)
    """
    dx = S[0, :] - s_c[0]
    dy = S[1, :] - s_c[1]
    return np.sqrt(dx**2 + dy**2 + H**2)


def all_user_distances(S, comm_users):
    """
    Distance from trajectory to ALL CUs.
    S: (2, N)
    comm_users: (2, M)
    Returns: (M, N) distance matrix
    """
    M_u = comm_users.shape[1]
    N = S.shape[1]
    D = np.zeros((M_u, N))
    for m in range(M_u):
        D[m, :] = user_quad_distance(S, comm_users[:, m])
    return D


# ======================== Channel Functions ========================

def g_k(d_s):
    """Two-way sensing channel power gain (eq. 14)."""
    return beta_0 / (d_s ** 4)


def sigma_k(d_s):
    """Std of distance measurement noise (eq. 16)."""
    return np.sqrt((a * sigma_0**2) / (P * G_p * g_k(d_s)))


# ======================== Velocity ========================

def calc_velocity(S, s_start):
    """
    Velocity matrix from trajectory and starting position (eq. 2).
    S: (2, N), s_start: (2,)
    Returns: (2, N)
    """
    S_ext = np.column_stack([s_start, S])
    return np.diff(S_ext, axis=1) / T_f


# ======================== Energy ========================

def propulsion_power(V_norm):
    """UAV propulsion power (eq. 33)."""
    term1 = P_0 * (1 + 3 * V_norm**2 / U_tip**2)
    inner = np.sqrt(1 + V_norm**4 / (4 * v_0**4)) - V_norm**2 / (2 * v_0**2)
    term2 = P_I * np.sqrt(np.maximum(inner, 0))
    term3 = 0.5 * D_0 * rho * s_rotor * A_rotor * V_norm**3
    return term1 + term2 + term3


def hover_power():
    return propulsion_power(0.0)


def calc_real_energy(S, s_start):
    """Total energy for a trajectory stage."""
    V = calc_velocity(S, s_start)
    V_norm = np.linalg.norm(V, axis=0)
    Nh = S.shape[1] // mu
    return T_f * np.sum(propulsion_power(V_norm)) + T_h * Nh * hover_power()


# ======================== Trajectory Initialization ========================

def init_trajectory(s_start, s_end, N):
    """Straight-line initial trajectory (eq. 60)."""
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
    if mu_val is None:
        mu_val = mu
    return list(range(mu_val - 1, N, mu_val))


def get_hover_points(S, mu_val=None):
    idx = get_hover_indices(S.shape[1], mu_val)
    return S[:, idx], idx

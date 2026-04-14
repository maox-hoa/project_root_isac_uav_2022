"""
Communication rate — extended for M CUs with min-fairness.
Ψ^c(j) = min_m ψ^c_m(j)  (eq. 9, TWC)
Uses LSE smoothing for differentiability (eq. 37, TWC).
"""
import numpy as np
from parameters import P, alpha_0, N_0, B, H, T_f, T_h, mu, M as M_users
from models import user_quad_distance, get_hover_indices


# ======================== Per-User Throughput ========================

def user_throughput(S, s_c, B_m):
    """
    Total transmitted data for ONE CU across trajectory S.
    ψ^c_m = Σ T_f·R_{m,n} + Σ T_h·R_{m,μγ}  (eq. 8)
    S: (2, N), s_c: (2,), B_m: scalar bandwidth
    Returns: scalar [bits]
    """
    d = user_quad_distance(S, s_c)          # (N,)
    sigma2_m = B_m * N_0
    snr = (P * alpha_0) / (d**2 * sigma2_m)
    rate = B_m * np.log2(1 + snr)           # (N,)

    N = S.shape[1]
    hp_idx = get_hover_indices(N)

    psi = T_f * np.sum(rate) + T_h * np.sum(rate[hp_idx])
    return psi


def all_user_throughputs(S, comm_users, B_alloc):
    """
    Throughput for ALL CUs.
    S: (2, N), comm_users: (2, M), B_alloc: (M,)
    Returns: (M,) array of ψ^c_m
    """
    M_u = comm_users.shape[1]
    psi = np.zeros(M_u)
    for m in range(M_u):
        psi[m] = user_throughput(S, comm_users[:, m], B_alloc[m])
    return psi


# ======================== Min-Fairness Metric ========================

def comm_metric(S, comm_users, B_alloc):
    """
    Ψ^c = min_m ψ^c_m  (eq. 9)
    Returns: scalar
    """
    psi = all_user_throughputs(S, comm_users, B_alloc)
    return np.min(psi)


def comm_metric_lse(S, comm_users, B_alloc, t=1.0):
    """
    LSE smooth approximation to min (eq. 37):
    Ψ^c ≈ -1/t · log( Σ exp(-t·ψ^c_m) )
    Returns: scalar
    """
    psi = all_user_throughputs(S, comm_users, B_alloc)
    # Numerically stable log-sum-exp
    max_neg = np.max(-t * psi)
    return -(1.0 / t) * (max_neg + np.log(np.sum(np.exp(-t * psi - max_neg))))


# ======================== Average Rate (backward compatible) ========================

def avg_data_rate(S, s_c, N_total, B_m=None):
    """Average rate for 1 CU (arXiv compatible)."""
    if B_m is None:
        B_m = B / M_users
    d = user_quad_distance(S, s_c)
    sigma2_m = B_m * N_0
    snr = (P * alpha_0) / (d**2 * sigma2_m)
    return (B_m / N_total) * np.sum(np.log2(1 + snr))


# ======================== Rate Gradient (for current stage) ========================

def rate_grad_multiuser(S_stage, comm_users, B_alloc, N_total, t=1.0):
    """
    Gradient of LSE-smoothed min-rate w.r.t. current stage waypoints.
    S_stage: (2, N_stg)
    comm_users: (2, M)
    B_alloc: (M,)
    N_total: total waypoints across all stages
    t: LSE scaling parameter

    Returns: (N_stg,) grad_x, (N_stg,) grad_y
    """
    M_u = comm_users.shape[1]
    N_stg = S_stage.shape[1]

    # Per-user throughput weights (softmin weights)
    # w_m = exp(-t·ψ^c_m) / Σ exp(-t·ψ^c_m)
    # We need full trajectory throughputs, but gradient only on current stage
    # For simplicity, compute weights from current stage throughputs
    psi = np.zeros(M_u)
    for m in range(M_u):
        psi[m] = user_throughput(S_stage, comm_users[:, m], B_alloc[m])

    neg_t_psi = -t * psi
    max_val = np.max(neg_t_psi)
    exp_vals = np.exp(neg_t_psi - max_val)
    weights = exp_vals / np.sum(exp_vals)  # softmin weights

    grad_x = np.zeros(N_stg)
    grad_y = np.zeros(N_stg)

    for m in range(M_u):
        d = user_quad_distance(S_stage, comm_users[:, m])
        dx = S_stage[0, :] - comm_users[0, m]
        dy = S_stage[1, :] - comm_users[1, m]

        sigma2_m = B_alloc[m] * N_0
        snr_const_m = (P * alpha_0) / sigma2_m

        common = -2 * snr_const_m / (d**2 * (d**2 + snr_const_m))
        # --- SỬA ĐỔI Ở ĐÂY ---
        # Sai: scale = B_alloc[m] / (N_total * np.log(2))
        # Đúng: Tổng data không nên chia cho N_total.
        # Nếu mục tiêu là Total Data, gradient là tổng gradient của từng điểm.
        scale = B_alloc[m] / np.log(2)
        # ---------------------

        # Weighted by softmin contribution
        # Lưu ý: Softmin weights tính trên Total Rate, nên gradient cũng nên được scale tương ứng
        grad_x += weights[m] * common * dx * scale
        grad_y += weights[m] * common * dy * scale

    return grad_x, grad_y


def rate_grad(S_stage, s_c, N_total, B_m=None):
    """Single-user gradient (backward compatible)."""
    if B_m is None:
        B_m = B / M_users
    d = user_quad_distance(S_stage, s_c)
    dx = S_stage[0, :] - s_c[0]
    dy = S_stage[1, :] - s_c[1]
    sigma2_m = B_m * N_0
    snr_const_m = (P * alpha_0) / sigma2_m
    common = -2 * snr_const_m / (d**2 * (d**2 + snr_const_m))
    scale = B_m / (N_total * np.log(2))
    return common * dx * scale, common * dy * scale

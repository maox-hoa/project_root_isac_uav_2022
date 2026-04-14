"""
CRB / Fisher Information Matrix — extended for K STs.
Ψ^s(j) = max_k ψ^s_k(j)  (eq. 32, TWC)
Uses LSE smoothing for differentiability (eq. 38, TWC).
"""
import numpy as np
from parameters import factor_CRB, H
from models import relative_distance


# ======================== Fisher Matrix Entries (single target) ========================

def fisher_mat_entry(S_hov, s_target, entry_type):
    """
    Compute a Fisher matrix entry for ONE target (eqs. 26-28, TWC).
    S_hov: (2, K_h), s_target: (2,)
    Returns: scalar
    """
    ds = relative_distance(S_hov, s_target)
    dx = S_hov[0, :] - s_target[0]
    dy = S_hov[1, :] - s_target[1]

    if entry_type == 'theta_a':
        return factor_CRB * np.sum(dx ** 2 / ds ** 6)
    elif entry_type == 'theta_b':
        return factor_CRB * np.sum(dy ** 2 / ds ** 6)
    elif entry_type == 'theta_c':
        return factor_CRB * np.sum(dx * dy / ds ** 6)


# ======================== CRB for ONE target (eq. 31, TWC) ========================

def crb(S_hov, s_target):
    """
    ψ^s_k = (Θa + Θb) / (Θa·Θb - Θc²)
    Returns: scalar
    """
    ta = fisher_mat_entry(S_hov, s_target, 'theta_a')
    tb = fisher_mat_entry(S_hov, s_target, 'theta_b')
    tc = fisher_mat_entry(S_hov, s_target, 'theta_c')
    det = ta * tb - tc**2
    if abs(det) < 1e-30:
        return 1e10
    return (ta + tb) / det


# ======================== Multi-Target Metrics ========================

def all_target_crbs(S_hov, targets_est):
    """
    CRB for ALL targets.
    targets_est: (2, K) estimated target positions
    Returns: (K,) array of ψ^s_k
    """
    K_t = targets_est.shape[1]
    crbs = np.zeros(K_t)
    for k in range(K_t):
        crbs[k] = crb(S_hov, targets_est[:, k])
    return crbs


def sens_metric(S_hov, targets_est):
    """
    Ψ^s = max_k ψ^s_k  (eq. 32)
    Returns: scalar
    """
    return np.max(all_target_crbs(S_hov, targets_est))


def sens_metric_lse(S_hov, targets_est, t=1.0):
    """
    LSE smooth approximation to max (eq. 38):
    Ψ^s ≈ 1/t · log( Σ exp(t·ψ^s_k) )
    Returns: scalar
    """
    crbs = all_target_crbs(S_hov, targets_est)
    max_val = np.max(t * crbs)
    return (1.0 / t) * (max_val + np.log(np.sum(np.exp(t * crbs - max_val))))


# ======================== Fisher Entry Gradient (single target) ========================

def fisher_entry_gradient(S_hov, s_target, entry_type, direction):
    """
    Gradient of Fisher entry w.r.t. hover points of current stage.
    S_hov: (2, K_stg) current stage hover points
    Returns: (K_stg,) gradient
    """
    ds = relative_distance(S_hov, s_target)
    dx = S_hov[0, :] - s_target[0]
    dy = S_hov[1, :] - s_target[1]

    if entry_type == 'theta_a':
        if direction == 'x':
            grad = (factor_CRB * (2 * dx * ds**2 - 6 * dx**3) / ds**8
                    + (16 * dx * ds**2 - 32 * dx**3) / ds**6)
        else:
            grad = -dy * dx**2 * (32 / ds**6 + 6 * factor_CRB / ds**8)
    elif entry_type == 'theta_b':
        if direction == 'x':
            grad = -dx * dy**2 * (32 / ds**6 + 6 * factor_CRB / ds**8)
        else:
            grad = (factor_CRB * (2 * dy * ds**2 - 6 * dy**3) / ds**8
                    + (16 * dy * ds**2 - 32 * dy**3) / ds**6)
    elif entry_type == 'theta_c':
        grad = _theta_c_grad(S_hov, s_target, ds, direction)

    return grad


def _theta_c_grad(S_hov, s_target, ds, direction):
    if direction == 'x':
        re_pos_x = S_hov[0, :] - s_target[0]
        re_pos_y = S_hov[1, :] - s_target[1]
    else:
        re_pos_y = S_hov[0, :] - s_target[0]
        re_pos_x = S_hov[1, :] - s_target[1]
    grad = re_pos_y * (factor_CRB / ds**6 + 8 / ds**4)
    grad -= re_pos_x**2 * re_pos_y * (6 * factor_CRB / ds**8 + 32 / ds**6)
    return grad


# ======================== CRB Gradient (single target) ========================

def crb_grad_single(S_hover_all, s_target, K_stg_val):
    """
    Gradient of CRB for ONE target w.r.t. current stage hover points.
    S_hover_all: (2, K_total)
    s_target: (2,)
    K_stg_val: hover points in current stage
    Returns: (K_stg_val,) grad_x, (K_stg_val,) grad_y
    """
    ta = fisher_mat_entry(S_hover_all, s_target, 'theta_a')
    tb = fisher_mat_entry(S_hover_all, s_target, 'theta_b')
    tc = fisher_mat_entry(S_hover_all, s_target, 'theta_c')
    det = ta * tb - tc**2

    S_grad = S_hover_all[:, -K_stg_val:]
    grad_x = np.zeros(K_stg_val)
    grad_y = np.zeros(K_stg_val)

    for dim, grad_out in [('x', grad_x), ('y', grad_y)]:
        da = fisher_entry_gradient(S_grad, s_target, 'theta_a', dim)
        db = fisher_entry_gradient(S_grad, s_target, 'theta_b', dim)
        dc = fisher_entry_gradient(S_grad, s_target, 'theta_c', dim)
        det_grad = da * tb + db * ta - 2 * dc * tc
        numerator = (db * det - tb * det_grad) + (da * det - ta * det_grad)
        grad_out[:] = numerator / det**2

    return grad_x, grad_y


# ======================== Multi-Target CRB Gradient ========================

def crb_grad_multitarget(S_hover_all, targets_est, K_stg_val, t=1.0):
    """
    Gradient of LSE-smoothed max-CRB w.r.t. current stage hover points.
    targets_est: (2, K)
    Returns: (K_stg_val,) grad_x, (K_stg_val,) grad_y
    """
    K_t = targets_est.shape[1]

    # Softmax weights
    crbs = all_target_crbs(S_hover_all, targets_est)
    t_crbs = t * crbs
    max_val = np.max(t_crbs)
    exp_vals = np.exp(t_crbs - max_val)
    weights = exp_vals / np.sum(exp_vals)

    grad_x = np.zeros(K_stg_val)
    grad_y = np.zeros(K_stg_val)

    for k in range(K_t):
        gx_k, gy_k = crb_grad_single(S_hover_all, targets_est[:, k], K_stg_val)
        grad_x += weights[k] * gx_k
        grad_y += weights[k] * gy_k

    return grad_x, grad_y


# Backward compatible alias
def crb_grad(S_hover_all, s_target, K_stg_val):
    return crb_grad_single(S_hover_all, s_target, K_stg_val)

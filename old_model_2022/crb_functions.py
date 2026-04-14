"""
CRB / Fisher Information Matrix: values and gradients.
Consolidates: crb.m, crb_grad.m, fisher_mat_entry.m, fisher_entry_gradient.m
"""
import numpy as np
from parameters import factor_CRB, H
from models import relative_distance


# ======================== Fisher Matrix Entries (eqs. 23-25) ========================

def fisher_mat_entry(S_hov, s_target, entry_type):
    """
    Compute a Fisher matrix entry (scalar).
    S_hov: (2, K) all hover points
    s_target: (2,) target position
    entry_type: 'theta_a', 'theta_b', or 'theta_c'
    Returns: scalar
    """
    ds = relative_distance(S_hov, s_target)  # (K,)

    if entry_type == 'theta_a':
        dx = S_hov[0, :] - s_target[0]
        val = factor_CRB * np.sum(dx**2 / ds**6) + 8 * np.sum(dx**2 / ds**4)

    elif entry_type == 'theta_b':
        dy = S_hov[1, :] - s_target[1]
        val = factor_CRB * np.sum(dy**2 / ds**6) + 8 * np.sum(dy**2 / ds**4)

    elif entry_type == 'theta_c':
        dx = S_hov[0, :] - s_target[0]
        dy = S_hov[1, :] - s_target[1]
        val = factor_CRB * np.sum(dx * dy / ds**6) + 8 * np.sum(dx * dy / ds**4)

    return val


# ======================== CRB Value (eq. 28) ========================

def crb(S_hov, s_target):
    """
    CRB_xt,yt = (Theta_a + Theta_b) / (Theta_a * Theta_b - Theta_c^2)
    S_hov: (2, K) hover points
    s_target: (2,) target position
    Returns: scalar CRB value
    """
    ta = fisher_mat_entry(S_hov, s_target, 'theta_a')
    tb = fisher_mat_entry(S_hov, s_target, 'theta_b')
    tc = fisher_mat_entry(S_hov, s_target, 'theta_c')
    det = ta * tb - tc**2
    if abs(det) < 1e-30:
        return 1e10
    return (ta + tb) / det


# ======================== Fisher Entry Gradient ========================

def fisher_entry_gradient(S_hov, s_target, entry_type, direction):
    """
    Gradient of a Fisher matrix entry w.r.t. hover point coordinates.
    S_hov: (2, K_stg) hover points of CURRENT stage only
    s_target: (2,) target position
    direction: 'x' or 'y'
    Returns: (K_stg,) gradient vector

    Matches fisher_entry_gradient.m exactly.
    """
    ds = relative_distance(S_hov, s_target)
    dx = S_hov[0, :] - s_target[0]
    dy = S_hov[1, :] - s_target[1]

    if entry_type == 'theta_a':
        if direction == 'x':
            # d(theta_a)/d(x_k)
            grad = (factor_CRB * (2 * dx * ds**2 - 6 * dx**3) / ds**8
                    + (16 * dx * ds**2 - 32 * dx**3) / ds**6)
        elif direction == 'y':
            # d(theta_a)/d(y_k)
            grad = -dy * dx**2 * (32 / ds**6 + 6 * factor_CRB / ds**8)

    elif entry_type == 'theta_b':
        # theta_b is theta_a with x<->y swapped
        if direction == 'x':
            # Same as d(theta_a)/d(y) with x<->y
            grad = -dx * dy**2 * (32 / ds**6 + 6 * factor_CRB / ds**8)
        elif direction == 'y':
            grad = (factor_CRB * (2 * dy * ds**2 - 6 * dy**3) / ds**8
                    + (16 * dy * ds**2 - 32 * dy**3) / ds**6)

    elif entry_type == 'theta_c':
        grad = _theta_c_grad(S_hov, s_target, ds, direction)

    return grad


def _theta_c_grad(S_hov, s_target, ds, direction):
    """Gradient of theta_c entry. Matches c_entry_grad in MATLAB."""
    if direction == 'x':
        re_pos_x = S_hov[0, :] - s_target[0]
        re_pos_y = S_hov[1, :] - s_target[1]
    elif direction == 'y':
        re_pos_y = S_hov[0, :] - s_target[0]
        re_pos_x = S_hov[1, :] - s_target[1]

    grad = re_pos_y * (factor_CRB / ds**6 + 8 / ds**4)
    grad -= re_pos_x**2 * re_pos_y * (6 * factor_CRB / ds**8 + 32 / ds**6)
    return grad


# ======================== CRB Gradient (for optimization) ========================

def crb_grad(S_hover_all, s_target, K_stg_val):
    """
    Gradient of CRB w.r.t. hover points of current stage.
    S_hover_all: (2, K_total) ALL hover points (all stages)
    s_target: (2,) estimated target position
    K_stg_val: number of hover points in current stage
    Returns: (K_stg_val,) gradient for x, (K_stg_val,) gradient for y

    Matches crb_grad.m.
    """
    # Fisher entries using ALL hover points
    ta = fisher_mat_entry(S_hover_all, s_target, 'theta_a')
    tb = fisher_mat_entry(S_hover_all, s_target, 'theta_b')
    tc = fisher_mat_entry(S_hover_all, s_target, 'theta_c')
    det = ta * tb - tc**2

    # Current stage hover points (last K_stg_val columns)
    S_grad = S_hover_all[:, -K_stg_val:]

    grad_x = np.zeros(K_stg_val)
    grad_y = np.zeros(K_stg_val)

    for dim, grad_out in [('x', grad_x), ('y', grad_y)]:
        # Gradients of Fisher entries w.r.t. current stage hover points
        da = fisher_entry_gradient(S_grad, s_target, 'theta_a', dim)
        db = fisher_entry_gradient(S_grad, s_target, 'theta_b', dim)
        dc = fisher_entry_gradient(S_grad, s_target, 'theta_c', dim)

        # Gradient of determinant
        det_grad = da * tb + db * ta - 2 * dc * tc

        # Gradient of CRB = (ta+tb)/det  using quotient rule
        # d/dx[(ta+tb)/det] = [(da+db)*det - (ta+tb)*det_grad] / det^2
        numerator = (db * det - tb * det_grad) + (da * det - ta * det_grad)
        grad_out[:] = numerator / det**2

    return grad_x, grad_y
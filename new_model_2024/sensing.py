"""
Multi-target sensing — echo association + per-target MLE.
Implements Section II-A echo association (eq. 3, TWC) and
Section IV-B MLE estimation (eqs. 52-53, TWC).
"""
import numpy as np
from parameters import H, P, G_p, beta_0, a, sigma_0, L_x, L_y
from models import relative_distance, sigma_k


# ======================== Multi-Target Sensing ========================

def sense_targets(targets_true, S_hover):
    """
    Generate noisy distance measurements for K targets at K_h hover points.
    targets_true: (2, K) true target positions
    S_hover: (2, K_h) hover positions
    Returns: D_meas (K, K_h) — noisy measurements, rows = targets
    """
    K_t = targets_true.shape[1]
    K_h = S_hover.shape[1]
    D_meas = np.zeros((K_t, K_h))

    for k in range(K_t):
        d_true = relative_distance(S_hover, targets_true[:, k])
        noise_std = sigma_k(d_true)
        D_meas[k, :] = d_true + noise_std * np.random.randn(K_h)

    return D_meas


def sense_targets_unlabeled(targets_true, S_hover):
    """
    Generate unlabeled echoes (simulating real scenario where
    echo-to-target association is unknown).
    Returns: D_echoes (K, K_h) — rows randomly permuted per HP
    """
    D_meas = sense_targets(targets_true, S_hover)
    K_t, K_h = D_meas.shape

    D_echoes = np.zeros_like(D_meas)
    for j in range(K_h):
        perm = np.random.permutation(K_t)
        D_echoes[:, j] = D_meas[perm, j]

    return D_echoes


# ======================== Echo Association (eq. 3, TWC) ========================

def associate_echoes(S_hover, D_echoes, targets_est):
    """
    Associate unlabeled echoes with targets using nearest-prediction.
    S_hover: (2, K_h)
    D_echoes: (K, K_h) — unlabeled measurements
    targets_est: (2, K) — current target estimates
    Returns: D_assoc (K, K_h) — associated measurements, row k → target k
    """
    K_t = targets_est.shape[1]
    K_h = S_hover.shape[1]
    D_assoc = np.zeros((K_t, K_h))

    for j in range(K_h):
        # Predicted distances from each target estimate to this HP
        d_pred = np.array([
            np.sqrt((S_hover[0, j] - targets_est[0, k])**2 +
                     (S_hover[1, j] - targets_est[1, k])**2 + H**2)
            for k in range(K_t)
        ])

        echoes = D_echoes[:, j].copy()

        # Greedy assignment: closest prediction-measurement pair
        assigned = np.zeros(K_t, dtype=bool)
        used = np.zeros(K_t, dtype=bool)

        for _ in range(K_t):
            best_cost = np.inf
            best_k, best_e = -1, -1
            for k in range(K_t):
                if assigned[k]:
                    continue
                for e in range(K_t):
                    if used[e]:
                        continue
                    cost = abs(echoes[e] - d_pred[k])
                    if cost < best_cost:
                        best_cost = cost
                        best_k, best_e = k, e

            D_assoc[best_k, j] = echoes[best_e]
            assigned[best_k] = True
            used[best_e] = True

    return D_assoc


# ======================== Per-Target MLE Estimation ========================

def estimate_single_target(S_hover_all, D_meas_k, method='random_gridsearch'):
    """
    MLE estimation for ONE target (eq. 52-53, TWC).
    S_hover_all: (2, K_total)
    D_meas_k: (K_total,) measurements for this target
    Returns: (2,) estimated position
    """
    x_hov = S_hover_all[0, :]
    y_hov = S_hover_all[1, :]

    if method == 'random_gridsearch':
        rng_x = np.random.RandomState(3)
        rng_y = np.random.RandomState(4)
        best_val = 1e12
        pos = np.array([L_x / 2, L_y / 2])

        for _ in range(10):
            x_grid = L_x * rng_x.rand(1000)
            y_grid = L_y * rng_y.rand(1000)
            candidate, val = _grid_search_mle(D_meas_k, x_grid, y_grid,
                                               x_hov, y_hov)
            if val < best_val:
                best_val = val
                pos = candidate
        return pos
    else:
        x_grid = np.linspace(0, L_x, 1000)
        y_grid = np.linspace(0, L_y, 1000)
        pos, _ = _grid_search_mle(D_meas_k, x_grid, y_grid, x_hov, y_hov)
        return pos


def estimate_all_targets(S_hover_all, D_assoc, method='random_gridsearch'):
    """
    MLE estimation for ALL targets.
    D_assoc: (K, K_total) — associated measurements
    Returns: (2, K) estimated positions
    """
    K_t = D_assoc.shape[0]
    est = np.zeros((2, K_t))
    for k in range(K_t):
        est[:, k] = estimate_single_target(S_hover_all, D_assoc[k, :], method)
    return est


def _grid_search_mle(D_meas, x_grid, y_grid, x_hov, y_hov):
    """Grid search MLE (matches get_min_gridsearch.m)."""
    K_h = len(D_meas)
    factor = (P * G_p * beta_0) / (a * sigma_0**2)

    X_t, Y_t = np.meshgrid(x_grid, y_grid)
    total = np.zeros_like(X_t)

    for k in range(K_h):
        P_xy = (X_t - x_hov[k])**2 + (Y_t - y_hov[k])**2 + H**2
        log_Pxy = np.log(P_xy)
        numerator = D_meas[k] - np.sqrt(P_xy)
        fraction = (numerator / P_xy)**2
        total += log_Pxy + factor * fraction

    min_idx = np.unravel_index(np.argmin(total), total.shape)
    best_pos = np.array([x_grid[min_idx[1]], y_grid[min_idx[0]]])
    return best_pos, total[min_idx]


# Backward compatible
def sense_target(s_target, S_hover):
    d_true = relative_distance(S_hover, s_target)
    noise_std = sigma_k(d_true)
    return d_true + noise_std * np.random.randn(len(d_true))

def estimate_target(S_hover_all, D_meas, method='random_gridsearch'):
    return estimate_single_target(S_hover_all, D_meas, method)

"""
Target sensing and MLE estimation.
Consolidates: sense_target.m, estimate_target.m, get_min_gridsearch.m, grid_vectors.m
"""
import numpy as np
from parameters import H, P, G_p, beta_0, a, sigma_0
from models import relative_distance, sigma_k


def sense_target(s_target, S_hover):
    """
    Generate noisy distance measurements at hover points.
    s_target: (2,) true target position
    S_hover: (2, K) hover positions
    Returns: (K,) noisy distance measurements
    """
    d_true = relative_distance(S_hover, s_target)
    noise_std = sigma_k(d_true)
    d_hat = d_true + noise_std * np.random.randn(len(d_true))
    return d_hat


def estimate_target(S_hover_all, D_meas, method='random_gridsearch'):
    """
    Estimate target position via MLE grid search (eq. 57-58).
    S_hover_all: (2, K_total) all hover points across stages
    D_meas: (K_total,) all distance measurements
    method: 'gridsearch' or 'random_gridsearch'
    Returns: (2,) estimated target position
    """
    x_hov = S_hover_all[0, :]
    y_hov = S_hover_all[1, :]

    if method == 'gridsearch':
        x_grid = np.linspace(0, 1500, 1000)
        y_grid = np.linspace(0, 1500, 1000)
        pos = _grid_search_mle(D_meas, x_grid, y_grid, x_hov, y_hov)

    elif method == 'random_gridsearch':
        rng_x = np.random.RandomState(3)
        rng_y = np.random.RandomState(4)
        n_reps = 10
        n_pts = 1000
        best_val = 1e12
        pos = np.array([750.0, 750.0])

        for z in range(n_reps):
            x_grid = 1500 * rng_x.rand(n_pts)
            y_grid = 1500 * rng_y.rand(n_pts)
            candidate, val = _grid_search_mle(D_meas, x_grid, y_grid, x_hov, y_hov)
            if val < best_val:
                best_val = val
                pos = candidate

    return pos


def _grid_search_mle(D_meas, x_grid, y_grid, x_hov, y_hov):
    """
    Grid search for MLE. Matches get_min_gridsearch.m.
    Minimizes the negative log-likelihood over a grid.

    Returns: (2,) best position, scalar best value
    """
    K = len(D_meas)
    factor = (P * G_p * beta_0) / (a * sigma_0**2)

    # Create meshgrid: (ny, nx) shape
    X_t, Y_t = np.meshgrid(x_grid, y_grid)

    total = np.zeros_like(X_t)

    for k in range(K):
        # Distance from grid point to k-th hover point
        P_xy = (X_t - x_hov[k])**2 + (Y_t - y_hov[k])**2 + H**2
        log_Pxy = np.log(P_xy)

        numerator = D_meas[k] - np.sqrt(P_xy)
        fraction = (numerator / P_xy)**2  # (d_hat - d_model)^2 / d_model^4 effectively

        total += log_Pxy + factor * fraction

    # Find minimum
    min_idx = np.unravel_index(np.argmin(total), total.shape)
    best_pos = np.array([x_grid[min_idx[1]], y_grid[min_idx[0]]])
    best_val = total[min_idx]

    return best_pos, best_val

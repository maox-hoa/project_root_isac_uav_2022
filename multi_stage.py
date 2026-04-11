"""
Multi-stage trajectory design algorithm (Algorithm 1).
Translates multi_stage.m.
"""
import numpy as np
from parameters import *
from models import (init_trajectory, calc_real_energy, get_hover_indices,
                    get_hover_points, get_all_hover_indices)
from crb_functions import crb as compute_crb
from rate_functions import avg_data_rate
from sensing import sense_target, estimate_target
from optimization import optimize_m


def multi_stage(setup=None, params_override=None, verbose=True):
    """
    Run the complete multi-stage ISAC trajectory design.

    Parameters:
        setup: dict with keys:
            'base_station_pos', 'comm_user_pos', 'sense_target_pos',
            'est_sense_target', 'total_energy'
        params_override: dict to override default parameters (e.g., {'eta': 0.8})
        verbose: print progress

    Returns:
        results: dict with all stage results
    """
    # ---- Setup ----
    if setup is None:
        setup = {
            'base_station_pos': base_station_pos,
            'comm_user_pos': comm_user_pos,
            'sense_target_pos': sense_target_pos,
            'est_sense_target': est_sense_target,
            'total_energy': total_energy,
        }

    # Override parameters if needed
    global eta, n_iter, w_star
    if params_override:
        if 'eta' in params_override:
            eta_local = params_override['eta']
        else:
            eta_local = eta
        if 'n_iter' in params_override:
            n_iter_local = params_override['n_iter']
        else:
            n_iter_local = n_iter
    else:
        eta_local = eta
        n_iter_local = n_iter

    # Temporarily set global params for optimization module
    import parameters
    old_eta = parameters.eta
    old_niter = parameters.n_iter
    parameters.eta = eta_local
    parameters.n_iter = n_iter_local

    s_b = setup['base_station_pos']
    s_c = setup['comm_user_pos']
    s_t = setup['sense_target_pos']
    s_target_est = setup['est_sense_target'].copy()
    E_total = setup['total_energy']

    M_max = 7  # Max stages
    hover_idx = get_hover_indices(N_stg)

    # ---- Storage ----
    S_opt_stages = []         # List of (2, N_stg) arrays
    S_init_stages = []
    D_meas_stages = []        # List of (K_stg,) measurement arrays
    S_target_est_list = []
    E_used_list = []
    E_remaining_list = []
    V_opt_stages = []
    CRB_opt_list = []
    R_opt_list = []
    J_history_stages = []

    s_start = s_b.copy()
    E_m = E_total
    m = 0  # 0-indexed stage counter

    # Weight for initial trajectory direction
    epsilon = eta_local if eta_local < 1.0 else 0.9

    while E_m > E_min and m < M_max:
        m_display = m + 1
        if verbose:
            print(f"\n=== Stage {m_display} | E_remaining = {E_m/1e3:.1f} KJ ===")

        # ---- End point for initial trajectory ----
        s_end = s_target_est * epsilon + s_c * (1 - epsilon)

        # ---- Initial trajectory ----
        S_init = init_trajectory(s_start, s_end, N_stg)
        S_init_stages.append(S_init.copy())

        # ---- Build total trajectory and hover arrays ----
        # Current stage gets initial trajectory
        all_stage_trajs = S_opt_stages + [S_init]
        S_total = np.hstack(all_stage_trajs)  # (2, (m+1)*N_stg)

        # Extract all hover points
        all_hover_idx = get_all_hover_indices(m + 1)
        S_hover_all = S_total[:, all_hover_idx]  # (2, (m+1)*K_stg)

        # ---- Optimize ----
        if verbose:
            print(f"  Optimizing trajectory (N={N_stg}, K={K_stg})...")

        (S_opt, V_opt, CRB_val, R_val,
         J_hist, CRB_hist, R_hist) = optimize_m(
            E_m, s_c, S_hover_all, S_total, s_target_est, s_start
        )

        S_opt_stages.append(S_opt)
        V_opt_stages.append(V_opt)
        CRB_opt_list.append(CRB_val)
        R_opt_list.append(R_val)
        J_history_stages.append(J_hist)

        # ---- Sense target at hover points ----
        S_opt_hover = S_opt[:, hover_idx]  # (2, K_stg)
        D_meas = sense_target(s_t, S_opt_hover)
        D_meas_stages.append(D_meas)

        # ---- Estimate target using ALL measurements ----
        all_opt_trajs = S_opt_stages
        S_opt_total = np.hstack(all_opt_trajs)
        all_opt_hover_idx = get_all_hover_indices(m + 1)
        S_hover_opt_all = S_opt_total[:, all_opt_hover_idx]
        D_meas_all = np.concatenate(D_meas_stages)

        s_target_est = estimate_target(S_hover_opt_all, D_meas_all, method='random_gridsearch')
        S_target_est_list.append(s_target_est.copy())

        # ---- Energy accounting ----
        E_used = calc_real_energy(S_opt, s_start)
        E_m -= E_used
        E_used_list.append(E_used)
        E_remaining_list.append(E_m)

        # ---- Update start position ----
        s_start = S_opt[:, -1].copy()

        if verbose:
            est_err = np.linalg.norm(s_target_est - s_t)
            print(f"  E_used = {E_used/1e3:.1f} KJ, E_remain = {E_m/1e3:.1f} KJ")
            print(f"  Target est: [{s_target_est[0]:.0f}, {s_target_est[1]:.0f}], "
                  f"error = {est_err:.1f} m")

        m += 1

    M = m  # Total number of completed stages

    # ---- Compute final metrics ----
    S_opt_full = np.hstack(S_opt_stages) if S_opt_stages else np.zeros((2, 0))
    all_hover_idx_final = get_all_hover_indices(M) if M > 0 else []
    S_hover_final = S_opt_full[:, all_hover_idx_final] if M > 0 else np.zeros((2, 0))

    final_crb = compute_crb(S_hover_final, s_t) if M > 0 else np.inf
    final_rate = avg_data_rate(S_opt_full, s_c, S_opt_full.shape[1]) if M > 0 else 0.0
    final_mse = np.sum((s_target_est - s_t)**2) if M > 0 else np.inf

    if verbose:
        print(f"\n=== Final Results ({M} stages) ===")
        print(f"  CRB  = {final_crb:.4f} m²")
        print(f"  MSE  = {final_mse:.4f} m²")
        print(f"  Rate = {final_rate/1e6:.4f} Mbits/s")

    # Restore globals
    parameters.eta = old_eta
    parameters.n_iter = old_niter

    return {
        'M': M,
        'S_opt_stages': S_opt_stages,
        'S_init_stages': S_init_stages,
        'V_opt_stages': V_opt_stages,
        'D_meas_stages': D_meas_stages,
        'S_target_est_list': S_target_est_list,
        'E_used_list': E_used_list,
        'E_remaining_list': E_remaining_list,
        'CRB_opt_list': CRB_opt_list,
        'R_opt_list': R_opt_list,
        'J_history_stages': J_history_stages,
        'final_crb': final_crb,
        'final_rate': final_rate,
        'final_mse': final_mse,
        'S_opt_full': S_opt_full,
        'S_hover_final': S_hover_final,
        's_target_est': s_target_est,
    }

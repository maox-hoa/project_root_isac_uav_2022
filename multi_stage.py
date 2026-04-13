"""
Multi-stage trajectory design (Algorithm 1) with:
  - Proper ending stage (Nlst calculation)
  - Energy validation after each stage
  - Correct trajectory/hover accumulation for CRB and Rate
"""
import numpy as np
import parameters as P
from models import (init_trajectory, calc_real_energy, propulsion_power,
                    hover_power, get_hover_indices, get_all_hover_indices)
from crb_functions import crb as compute_crb
from rate_functions import avg_data_rate
from sensing import sense_target, estimate_target
from optimization import optimize_m


def _calc_nlst(E_remaining):
    """
    Calculate max number of waypoints the remaining energy can support.
    Uses conservative estimate: assume flying at V_str, plus hover overhead.
    Returns Nlst (rounded down to multiple of mu for clean HP extraction),
    and Klst = Nlst // mu.
    """
    # Energy per waypoint: conservative estimate using V_max
    e_fly = P.T_f * propulsion_power(P.V_max)
    e_hover_per_wp = (1.0 / P.mu) * P.T_h * hover_power()
    e_per_wp = e_fly + e_hover_per_wp

    # Extra safety margin (10%) for solver inaccuracy
    e_per_wp *= 1.10

    if e_per_wp <= 0:
        return 0, 0

    Nlst = int(E_remaining / e_per_wp)
    # Round down to nearest multiple of mu (need at least mu for 1 HP)
    Nlst = (Nlst // P.mu) * P.mu
    Nlst = min(Nlst, P.N_stg)  # Never exceed N_stg

    Klst = Nlst // P.mu
    return Nlst, Klst


def _run_stage(m_idx, s_start, E_m, s_c, s_target_est,
               prev_S_opt_stages, prev_D_meas_stages,
               N_m, K_m, eta, nit, verbose):
    """
    Run one stage of the MSTD algorithm.

    m_idx: 0-based stage index
    Returns: (S_opt, D_meas, s_target_est_new, E_used, valid)
    """
    s_t_true = None  # Will be set by caller

    epsilon = eta if eta < 1.0 else 0.9
    s_end = s_target_est * epsilon + s_c * (1 - epsilon)

    # Initial trajectory for this stage
    S_init = init_trajectory(s_start, s_end, N_m)
    hover_idx = get_hover_indices(N_m)

    # Build total trajectory: prev optimal stages + current init
    all_trajs = list(prev_S_opt_stages) + [S_init]
    S_total = np.hstack(all_trajs)

    # Build hover array from S_total
    # Accumulate hover indices across all stages
    n_prev_stages = len(prev_S_opt_stages)
    all_hover_idx = []
    offset = 0
    for si in range(n_prev_stages):
        Ni = prev_S_opt_stages[si].shape[1]
        Ki_indices = get_hover_indices(Ni)
        all_hover_idx.extend([offset + hi for hi in Ki_indices])
        offset += Ni
    # Current stage hover indices
    cur_hover_global = [offset + hi for hi in hover_idx]
    all_hover_idx.extend(cur_hover_global)

    S_hover_all = S_total[:, all_hover_idx]

    # Optimize
    S_opt, V_opt, CRB_val, R_val, J_hist, _, _ = optimize_m(
        E_m, s_c, S_hover_all, S_total, s_target_est, s_start,
        N_stg_m=N_m, K_stg_m=K_m, eta_val=eta, n_iter_val=nit
    )

    # ---- Energy validation ----
    E_used = calc_real_energy(S_opt, s_start)
    if E_used > E_m + 100:  # tolerance 100J
        if verbose:
            print(f" ENERGY OVERFLOW: used={E_used/1e3:.2f}KJ > avail={E_m/1e3:.2f}KJ → skip")
        return None, None, s_target_est, 0, False, J_hist

    return S_opt, hover_idx, s_target_est, E_used, True, J_hist


def multi_stage(setup=None, eta_val=None, n_iter_val=None, verbose=True):
    """
    Run complete MSTD algorithm (Algorithm 1 from paper).

    Implements:
      - Regular stages with N_stg waypoints
      - Ending stage with Nlst <= N_stg waypoints when energy < E_min
      - Energy validation per stage
    """
    if setup is None:
        setup = {
            'base_station_pos': P.base_station_pos,
            'comm_user_pos': P.comm_user_pos,
            'sense_target_pos': P.sense_target_pos,
            'est_sense_target': P.est_sense_target,
            'total_energy': P.total_energy,
        }

    eta = eta_val if eta_val is not None else P.eta
    nit = n_iter_val if n_iter_val is not None else P.n_iter

    s_b = setup['base_station_pos']
    s_c = setup['comm_user_pos']
    s_t = setup['sense_target_pos']   # TRUE target (for sensing + final eval)
    s_target_est = setup['est_sense_target'].copy()
    E_total = setup['total_energy']

    M_max = 10
    s_start = s_b.copy()
    E_m = E_total

    # Storage
    S_opt_stages = []       # list of (2, N_i) optimized trajectories
    D_meas_stages = []      # list of (K_i,) measurement arrays
    S_target_est_list = []
    E_used_list = []
    J_history_stages = []

    # ==========================================
    # Phase 1: Regular stages (N_m = N_stg)
    # ==========================================
    m = 0
    while E_m > P.E_min and m < M_max:
        if verbose:
            print(f"  Stage {m+1} (N={P.N_stg},K={P.K_stg}) | E={E_m/1e3:.1f}KJ", end="")

        epsilon = eta if eta < 1.0 else 0.9
        s_end = s_target_est * epsilon + s_c * (1 - epsilon)
        S_init = init_trajectory(s_start, s_end, P.N_stg)
        hover_idx = get_hover_indices(P.N_stg)

        # Build FULL trajectory and FULL hover arrays
        all_trajs = list(S_opt_stages) + [S_init]
        S_total = np.hstack(all_trajs)

        # Hover indices across all stages
        all_hi = _accumulate_hover_indices(S_opt_stages, P.N_stg, hover_idx)
        S_hover_all = S_total[:, all_hi]

        # Optimize
        S_opt, V_opt, CRB_val, R_val, J_hist, _, _ = optimize_m(
            E_m, s_c, S_hover_all, S_total, s_target_est, s_start,
            N_stg_m=P.N_stg, K_stg_m=P.K_stg, eta_val=eta, n_iter_val=nit
        )

        # Energy validation
        E_used = calc_real_energy(S_opt, s_start)
        if E_used > E_m + 100:
            if verbose:
                print(f" → OVERFLOW ({E_used/1e3:.1f}>{E_m/1e3:.1f}KJ), stop")
            break

        S_opt_stages.append(S_opt)
        E_used_list.append(E_used)
        J_history_stages.append(J_hist)

        # Sense target at hover points
        S_opt_hover = S_opt[:, hover_idx]
        D_meas = sense_target(s_t, S_opt_hover)
        D_meas_stages.append(D_meas)

        # MLE estimation using ALL accumulated hover points and measurements
        S_all = np.hstack(S_opt_stages)
        all_hi_opt = _accumulate_hover_indices_flat(S_opt_stages)
        S_hov_all_opt = S_all[:, all_hi_opt]
        D_all = np.concatenate(D_meas_stages)
        s_target_est = estimate_target(S_hov_all_opt, D_all, method='random_gridsearch')
        S_target_est_list.append(s_target_est.copy())

        E_m -= E_used
        s_start = S_opt[:, -1].copy()

        if verbose:
            err = np.linalg.norm(s_target_est - s_t)
            print(f" → E_used={E_used/1e3:.1f}KJ, err={err:.1f}m")

        m += 1

    # ==========================================
    # Phase 2: Ending stage (Nlst < N_stg)
    # ==========================================
    Nlst, Klst = _calc_nlst(E_m)
    if Nlst >= P.mu and Klst >= 1 and m < M_max:
        if verbose:
            print(f"  Stage {m+1} ENDING (N={Nlst},K={Klst}) | E={E_m/1e3:.1f}KJ", end="")

        epsilon = eta if eta < 1.0 else 0.9
        s_end = s_target_est * epsilon + s_c * (1 - epsilon)
        S_init = init_trajectory(s_start, s_end, Nlst)
        hover_idx = get_hover_indices(Nlst)

        all_trajs = list(S_opt_stages) + [S_init]
        S_total = np.hstack(all_trajs)
        all_hi = _accumulate_hover_indices(S_opt_stages, Nlst, hover_idx)
        S_hover_all = S_total[:, all_hi]

        S_opt, V_opt, CRB_val, R_val, J_hist, _, _ = optimize_m(
            E_m, s_c, S_hover_all, S_total, s_target_est, s_start,
            N_stg_m=Nlst, K_stg_m=Klst, eta_val=eta, n_iter_val=nit
        )

        E_used = calc_real_energy(S_opt, s_start)
        if E_used <= E_m + 100:
            S_opt_stages.append(S_opt)
            E_used_list.append(E_used)
            J_history_stages.append(J_hist)

            S_opt_hover = S_opt[:, hover_idx]
            D_meas = sense_target(s_t, S_opt_hover)
            D_meas_stages.append(D_meas)

            S_all = np.hstack(S_opt_stages)
            all_hi_opt = _accumulate_hover_indices_flat(S_opt_stages)
            S_hov_all_opt = S_all[:, all_hi_opt]
            D_all = np.concatenate(D_meas_stages)
            s_target_est = estimate_target(S_hov_all_opt, D_all, method='random_gridsearch')
            S_target_est_list.append(s_target_est.copy())

            E_m -= E_used
            if verbose:
                err = np.linalg.norm(s_target_est - s_t)
                print(f" → E_used={E_used/1e3:.1f}KJ, err={err:.1f}m")
        else:
            if verbose:
                print(f" → OVERFLOW, skip ending stage")

    # ==========================================
    # Final evaluation
    # ==========================================
    M = len(S_opt_stages)
    S_full = np.hstack(S_opt_stages) if M > 0 else np.zeros((2, 0))
    all_hi_final = _accumulate_hover_indices_flat(S_opt_stages) if M > 0 else []
    S_hov_final = S_full[:, all_hi_final] if M > 0 and len(all_hi_final) > 0 else np.zeros((2, 0))

    # CRB computed at TRUE target position
    final_crb = compute_crb(S_hov_final, s_t) if S_hov_final.shape[1] >= 3 else np.inf
    # Rate computed over ALL trajectory points
    N_total = S_full.shape[1]
    final_rate = avg_data_rate(S_full, s_c, N_total) if N_total > 0 else 0.0
    final_mse = np.sum((s_target_est - s_t)**2) if M > 0 else np.inf

    if verbose:
        print(f"  => {M} stages, {N_total} pts, {S_hov_final.shape[1]} HPs: "
              f"CRB={final_crb:.4f}m², MSE={final_mse:.2f}m², "
              f"Rate={final_rate/1e6:.4f}Mbit/s, E_left={E_m/1e3:.2f}KJ")

    return {
        'M': M, 'S_opt_stages': S_opt_stages, 'D_meas_stages': D_meas_stages,
        'S_target_est_list': S_target_est_list, 'E_used_list': E_used_list,
        'J_history_stages': J_history_stages,
        'final_crb': final_crb, 'final_rate': final_rate, 'final_mse': final_mse,
        'S_opt_full': S_full, 'S_hover_final': S_hov_final,
        's_target_est': s_target_est,
    }


# ==========================================
# Helper: accumulate hover indices
# ==========================================

def _accumulate_hover_indices(prev_stages, N_cur, hover_idx_cur):
    """
    Build list of global hover indices for S_total = [prev_stages... | S_init_cur].
    prev_stages: list of (2, N_i) arrays from completed stages
    N_cur: waypoints in current stage
    hover_idx_cur: hover indices within current stage
    """
    indices = []
    offset = 0
    for S_prev in prev_stages:
        Ni = S_prev.shape[1]
        hi = get_hover_indices(Ni)
        indices.extend([offset + h for h in hi])
        offset += Ni
    # Current stage
    indices.extend([offset + h for h in hover_idx_cur])
    return indices


def _accumulate_hover_indices_flat(S_opt_stages):
    """Get all hover indices from list of optimized stage trajectories."""
    indices = []
    offset = 0
    for S in S_opt_stages:
        Ni = S.shape[1]
        hi = get_hover_indices(Ni)
        indices.extend([offset + h for h in hi])
        offset += Ni
    return indices
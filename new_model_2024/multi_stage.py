"""
Multi-stage trajectory design (Algorithm 2, TWC) — M CUs, K STs.
"""
import numpy as np
import parameters as P
from models import (init_trajectory, calc_real_energy, propulsion_power,
                    hover_power, get_hover_indices)
from crb_functions import sens_metric, all_target_crbs
from rate_functions import comm_metric, all_user_throughputs
from sensing import sense_targets, associate_echoes, estimate_all_targets
from optimization import optimize_trajectory, optimize_bandwidth


def _calc_nlst(E_remaining):
    e_fly = P.T_f * propulsion_power(P.V_max)
    e_hover_per_wp = (1.0 / P.mu) * P.T_h * hover_power()
    e_per_wp = (e_fly + e_hover_per_wp) * 1.10
    if e_per_wp <= 0:
        return 0, 0
    Nlst = int(E_remaining / e_per_wp)
    Nlst = (Nlst // P.mu) * P.mu
    Nlst = min(Nlst, P.N_stg)
    Nlst = max(Nlst, 0)
    Klst = Nlst // P.mu
    return Nlst, Klst


def _accumulate_hover_indices(prev_stages, N_cur, hover_idx_cur):
    indices = []
    offset = 0
    for S_prev in prev_stages:
        Ni = S_prev.shape[1]
        hi = get_hover_indices(Ni)
        indices.extend([offset + h for h in hi])
        offset += Ni
    indices.extend([offset + h for h in hover_idx_cur])
    return indices


def _accumulate_hover_indices_flat(S_opt_stages):
    indices = []
    offset = 0
    for S in S_opt_stages:
        Ni = S.shape[1]
        hi = get_hover_indices(Ni)
        indices.extend([offset + h for h in hi])
        offset += Ni
    return indices


def multi_stage(setup=None, eta_val=None, n_iter_val=None, verbose=True):
    """
    Run complete MSTD algorithm (Algorithm 2, TWC version).
    Multi-CU (min-fairness), multi-ST (max-CRB), bandwidth allocation.
    """
    if setup is None:
        setup = {
            'base_station_pos': P.base_station_pos,
            'comm_user_pos': P.comm_user_pos,
            'sense_target_pos': P.sense_target_pos,
            'est_sense_targets': P.est_sense_targets,
            'total_energy': P.total_energy,
        }

    eta = eta_val if eta_val is not None else P.eta
    nit = n_iter_val if n_iter_val is not None else P.n_iter

    s_b = setup['base_station_pos']            # (2,)
    comm_users = setup['comm_user_pos']         # (2, M)
    targets_true = setup['sense_target_pos']    # (2, K)
    targets_est = setup['est_sense_targets'].copy()  # (2, K)
    E_total = setup['total_energy']

    M_u = comm_users.shape[1]
    K_t = targets_true.shape[1]
    B_alloc = np.full(M_u, P.B / M_u)

    M_max = 10
    s_start = s_b.copy()
    E_m = E_total

    # Storage
    S_opt_stages = []
    D_meas_stages = []         # list of (K, K_h) arrays
    targets_est_list = []
    E_used_list = []
    J_history_stages = []

    # Initial metrics (from coarse estimate)
    # Use a dummy trajectory to get initial Ψ^s, Ψ^c
    prev_sens = None
    prev_comm = None
    def get_initial_metrics(S_init_traj, hover_idx_init, targets_est_curr, comm_users_curr, B_alloc_curr):
        S_hov_init = S_init_traj[:, hover_idx_init]
        curr_sens = sens_metric(S_hov_init, targets_est_curr)
        curr_comm = comm_metric(S_init_traj, comm_users_curr, B_alloc_curr)
        return curr_sens, curr_comm

    def _run_one_stage(N_m, K_m, stage_label):
        nonlocal s_start, E_m, targets_est, B_alloc
        nonlocal prev_sens, prev_comm

        epsilon = eta if eta < 1.0 else 0.9
        # Target midpoint for initial trajectory direction
        target_center = np.mean(targets_est, axis=1)
        comm_center = np.mean(comm_users, axis=1)
        s_end = target_center * epsilon + comm_center * (1 - epsilon)

        S_init = init_trajectory(s_start, s_end, N_m)
        hover_idx = get_hover_indices(N_m)

        # Nếu là Stage đầu tiên (m=0), cập nhật prev_sens và prev_comm dựa trên S_init
        if "Stage 1" in stage_label:
            prev_sens, prev_comm = get_initial_metrics(S_init, hover_idx, targets_est, comm_users, B_alloc)

        # Build full trajectory + hover arrays
        all_trajs = list(S_opt_stages) + [S_init]
        S_total = np.hstack(all_trajs)
        all_hi = _accumulate_hover_indices(S_opt_stages, N_m, hover_idx)
        S_hover_all = S_total[:, all_hi]

        # Bandwidth allocation (P'_2)
        B_alloc = optimize_bandwidth(S_total, comm_users,
                                      S_total.shape[1])

        # Trajectory optimization (P'_1)
        S_opt, V_opt, J_hist, _, _ = optimize_trajectory(
            E_m, comm_users, S_hover_all, S_total,
            targets_est, s_start, B_alloc,
            prev_sens, prev_comm,
            N_stg_m=N_m, K_stg_m=K_m,
            eta_val=eta, n_iter_val=nit
        )

        # Energy validation
        E_used = calc_real_energy(S_opt, s_start)
        if E_used > E_m + 100:
            if verbose:
                print(f"  {stage_label} → OVERFLOW ({E_used/1e3:.1f}>{E_m/1e3:.1f}KJ)")
            return False

        S_opt_stages.append(S_opt)
        E_used_list.append(E_used)
        J_history_stages.append(J_hist)

        # Sense targets
        S_opt_hover = S_opt[:, hover_idx]
        D_echoes = sense_targets(targets_true, S_opt_hover)
        D_assoc = associate_echoes(S_opt_hover, D_echoes, targets_est)
        D_meas_stages.append(D_assoc)

        # MLE — accumulate ALL hover points and measurements
        S_all = np.hstack(S_opt_stages)
        all_hi_opt = _accumulate_hover_indices_flat(S_opt_stages)
        S_hov_all_opt = S_all[:, all_hi_opt]
        D_all = np.concatenate(D_meas_stages, axis=1)  # (K, K_total)

        targets_est = estimate_all_targets(S_hov_all_opt, D_all)
        targets_est_list.append(targets_est.copy())

        # Update metrics for next stage normalization
        prev_sens = sens_metric(S_hov_all_opt, targets_est)
        prev_comm = comm_metric(S_all, comm_users, B_alloc)

        E_m -= E_used
        s_start = S_opt[:, -1].copy()

        if verbose:
            errs = [np.linalg.norm(targets_est[:, k] - targets_true[:, k])
                    for k in range(K_t)]
            crbs = all_target_crbs(S_hov_all_opt, targets_est)
            psis = all_user_throughputs(S_all, comm_users, B_alloc)
            print(f"  {stage_label} → E={E_used/1e3:.1f}KJ | "
                  f"CRBs={[f'{c:.2f}' for c in crbs]} | "
                  f"Errs={[f'{e:.0f}m' for e in errs]} | "
                  f"MinRate={np.min(psis)/1e6:.1f}Mb")

        return True

    # ==========================================
    # Phase 1: Regular stages
    # ==========================================
    m = 0
    while E_m > P.E_min and m < M_max:
        label = f"Stage {m+1} (N={P.N_stg})"
        if verbose:
            print(f"  {label} | E={E_m/1e3:.1f}KJ", end="")
            print()
        ok = _run_one_stage(P.N_stg, P.K_stg, label)
        if not ok:
            break
        m += 1

    # ==========================================
    # Phase 2: Ending stage
    # ==========================================
    Nlst, Klst = _calc_nlst(E_m)
    if Nlst >= P.mu and Klst >= 1 and m < M_max:
        label = f"Stage {m+1} ENDING (N={Nlst})"
        if verbose:
            print(f"  {label} | E={E_m/1e3:.1f}KJ")
        _run_one_stage(Nlst, Klst, label)

    # ==========================================
    # Final evaluation
    # ==========================================
    M_stages = len(S_opt_stages)
    S_full = np.hstack(S_opt_stages) if M_stages > 0 else np.zeros((2, 0))
    all_hi_final = _accumulate_hover_indices_flat(S_opt_stages)
    S_hov_final = S_full[:, all_hi_final] if len(all_hi_final) > 0 else np.zeros((2, 0))

    final_crbs = all_target_crbs(S_hov_final, targets_true) if S_hov_final.shape[1] >= 3 else np.full(K_t, np.inf)
    final_psis = all_user_throughputs(S_full, comm_users, B_alloc) if S_full.shape[1] > 0 else np.zeros(M_u)
    final_errs = [np.sum((targets_est[:, k] - targets_true[:, k])**2) for k in range(K_t)]

    if verbose:
        print(f"\n  => {M_stages} stages, {S_full.shape[1]} pts, "
              f"{S_hov_final.shape[1]} HPs")
        print(f"     CRBs (true): {[f'{c:.4f}' for c in final_crbs]}")
        print(f"     MSEs:        {[f'{e:.2f}' for e in final_errs]}")
        print(f"     Throughputs: {[f'{p/1e6:.2f}Mb' for p in final_psis]}")
        print(f"     Min Rate:    {np.min(final_psis)/1e6:.2f} Mb")
        print(f"     E_left:      {E_m/1e3:.2f} KJ")

    return {
        'M': M_stages,
        'S_opt_stages': S_opt_stages,
        'D_meas_stages': D_meas_stages,
        'targets_est_list': targets_est_list,
        'E_used_list': E_used_list,
        'J_history_stages': J_history_stages,
        'final_crbs': final_crbs,
        'final_rate': np.min(final_psis),
        'final_mses': final_errs,
        'S_opt_full': S_full,
        'S_hover_final': S_hov_final,
        'targets_est': targets_est,
        'B_alloc': B_alloc,
    }

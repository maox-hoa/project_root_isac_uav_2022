"""
REPRODUCE VERSION — CLOSE TO IEEE TWC PAPER
"""

import numpy as np
import cvxpy as cp

from models import get_hover_indices
from crb_functions import sens_metric_lse, crb_grad_multitarget
from rate_functions import comm_metric_lse, rate_grad_multiuser
import parameters as P


# ======================== ENERGY ========================

def _build_energy(V, delta, N, K_h):
    sum_v_sq = cp.sum_squares(V)

    sum_v_cube = 0
    for i in range(N):
        sum_v_cube += cp.power(cp.norm(V[:, i], 2), 3)

    ps1 = P.P_0 * (N + 3.0 / P.U_tip**2 * sum_v_sq)
    ps1 += 0.5 * P.D_0 * P.rho * P.s_rotor * P.A_rotor * sum_v_cube
    ps2 = P.P_I * cp.sum(delta)
    ps3 = K_h * (P.P_0 + P.P_I)

    return P.T_f * (ps1 + ps2) + P.T_h * ps3


# ======================== SOLVER ========================

def _solve(prob):
    for solver in [cp.ECOS, cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return True
        except:
            continue
    return False


# ======================== MAIN ========================

def optimize_trajectory(E_m, comm_users, S_hover_all, S_total,
                        targets_est, s_start, B_alloc,
                        prev_sens_metric, prev_comm_metric,
                        N_stg_m=None, K_stg_m=None):

    N_m = N_stg_m if N_stg_m else P.N_stg
    K_m = K_stg_m if K_stg_m else P.K_stg
    eta = P.eta
    nit = P.n_iter

    hover_idx = get_hover_indices(N_m)

    S_init = S_total[:, -N_m:].copy()

    # ===== normalization =====
    sens_norm = max(abs(prev_sens_metric), 1e-6)
    comm_norm = max(abs(prev_comm_metric), 1e-6)

    delta_trust = 30.0

    J_hist = []

    for it in range(nit):

        # ===== update global =====
        S_hover_all[:, -K_m:] = S_init[:, hover_idx]
        S_total[:, -N_m:] = S_init

        # ===== LSE metrics =====
        Psi_s = sens_metric_lse(S_hover_all, targets_est, t=5.0)
        Psi_c = comm_metric_lse(S_total, comm_users, B_alloc, t=5.0)

        # ===== gradients =====
        crb_gx, crb_gy = crb_grad_multitarget(
            S_hover_all, targets_est, K_m, t=5.0)

        rate_gx, rate_gy = rate_grad_multiuser(
            S_init, comm_users, B_alloc, S_total.shape[1], t=5.0)

        # ===== CVX =====
        S = cp.Variable((2, N_m))
        V = cp.Variable((2, N_m))
        delta = cp.Variable(N_m, nonneg=True)
        xi = cp.Variable(N_m, nonneg=True)

        # ===== incremental objective =====
        dPsi_s = (crb_gx @ (S[0, hover_idx] - S_init[0, hover_idx]) +
                  crb_gy @ (S[1, hover_idx] - S_init[1, hover_idx]))

        dPsi_c = (rate_gx @ (S[0, :] - S_init[0, :]) +
                  rate_gy @ (S[1, :] - S_init[1, :]))

        obj = cp.Minimize(
            eta * (-dPsi_s / sens_norm)
            + (1 - eta) * (dPsi_c / comm_norm)
        )

        cons = []

        # dynamics
        cons.append((S[:, 0] - s_start) / P.T_f == V[:, 0])
        cons.append((S[:, 1:] - S[:, :-1]) / P.T_f == V[:, 1:])

        # speed
        for i in range(N_m):
            cons.append(cp.norm(V[:, i]) <= P.V_max)

        # bounds
        cons += [
            S[0, :] >= 0,
            S[1, :] >= 0,
            S[0, :] <= P.L_x,
            S[1, :] <= P.L_y
        ]

        # energy
        cons.append(E_m >= _build_energy(V, delta, N_m, K_m))

        # trust region
        cons.append(cp.norm(S - S_init, 'fro') <= delta_trust)

        prob = cp.Problem(obj, cons)

        if not _solve(prob):
            break

        S_new = S.value
        obj_val = prob.value

        J_hist.append(obj_val)

        # ===== update =====
        step = 0.3 / (1 + it)
        S_init = S_init + step * (S_new - S_init)

    return S_init, None, np.array(J_hist), None, None
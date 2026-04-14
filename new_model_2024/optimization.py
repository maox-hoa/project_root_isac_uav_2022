"""
Trajectory optimization — TWC version.
Normalized increment objective P(j) from eq. 34 (TWC):
  max  η/Ψ^s(j-1)·(Ψ^s(j-1)-Ψ^s(j)) + (1-η)/Ψ^c(j-1)·(Ψ^c(j)-Ψ^c(j-1))
Bandwidth allocation subproblem P'_2(j) solved via CVX.
"""
import numpy as np
import cvxpy as cp
from models import get_hover_indices, calc_velocity
from crb_functions import (crb_grad_multitarget, sens_metric, sens_metric_lse,
                           all_target_crbs)
from rate_functions import (rate_grad_multiuser, comm_metric, comm_metric_lse,
                            all_user_throughputs)
import parameters as P


def _build_energy_cvx(V_var, delta_var, N, K_h):
    """Energy constraint (eq. 44, TWC)."""
    sum_v_sq = cp.sum_squares(V_var)
    sum_v_cube = 0
    for i in range(N):
        sum_v_cube += cp.power(cp.norm(V_var[:, i], 2), 3)
    ps1 = P.P_0 * (N + 3.0 / P.U_tip**2 * sum_v_sq)
    ps1 += 0.5 * P.D_0 * P.rho * P.s_rotor * P.A_rotor * sum_v_cube
    ps2 = P.P_I * cp.sum(delta_var)
    ps3 = K_h * (P.P_0 + P.P_I)
    return P.T_f * (ps1 + ps2) + P.T_h * ps3


def _try_solve(prob):
    for solver, kw in [
        (cp.ECOS, {'verbose': False, 'max_iters': 500}),
        (cp.CLARABEL, {'verbose': False}),
        (cp.SCS, {'verbose': False, 'max_iters': 50000, 'eps': 1e-9}),
    ]:
        try:
            prob.solve(solver=solver, **kw)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                return True
        except Exception:
            continue
    return False


# ======================== Trajectory Optimization ========================

def optimize_trajectory(E_m, comm_users, S_hover_all, S_total,
                        targets_est, s_start, B_alloc,
                        prev_sens_metric, prev_comm_metric,
                        N_stg_m=None, K_stg_m=None,
                        eta_val=None, n_iter_val=None):
    """
    Optimize trajectory for stage m — P'_1(j) in TWC.
    Uses normalized increment objective.

    prev_sens_metric: Ψ^s(j-1)
    prev_comm_metric: Ψ^c(j-1)
    """
    N_m = N_stg_m if N_stg_m is not None else P.N_stg
    K_m = K_stg_m if K_stg_m is not None else P.K_stg
    eta = eta_val if eta_val is not None else P.eta
    nit = n_iter_val if n_iter_val is not None else P.n_iter
    hover_idx = get_hover_indices(N_m)

    # Initial values
    S_init = S_total[:, -N_m:].copy()
    V_init = np.zeros((2, N_m))
    V_init[:, 0] = (S_init[:, 0] - s_start) / P.T_f
    V_init[:, 1:] = np.diff(S_init, axis=1) / P.T_f

    V_norms = np.linalg.norm(V_init, axis=0)
    dsq = np.sqrt(1 + V_norms**4 / (4 * P.v_0**4)) - V_norms**2 / (2 * P.v_0**2)
    dsq = np.maximum(dsq, 1e-10)

    S_init_valid = S_init.copy()
    S_valid = S_init * 1.1
    V_valid = V_init.copy()
    best_val = 1e13
    N_nan = 0

    J_hist = np.full(nit, np.nan)
    CRB_hist = np.full(nit, np.nan)
    R_hist = np.full(nit, np.nan)

    N_total = S_total.shape[1]

    for u in range(nit):
        # Update current stage in global arrays
        S_hover_all[:, -K_m:] = S_init[:, hover_idx]
        S_total[:, -N_m:] = S_init

        # Multi-target CRB gradient (softmax weighted)
        crb_gx, crb_gy = crb_grad_multitarget(
            S_hover_all, targets_est, K_m, t=1.0)

        # Multi-user rate gradient (softmin weighted)
        rate_gx, rate_gy = rate_grad_multiuser(
            S_init, comm_users, B_alloc, N_total, t=1.0)

        # Normalization factors
        sens_norm = max(abs(prev_sens_metric), 1e-12)
        comm_norm = max(abs(prev_comm_metric), 1e-12)

        # ---- CVX block ----
        S = cp.Variable((2, N_m))
        V = cp.Variable((2, N_m))
        delta = cp.Variable(N_m, nonneg=True)
        xi = cp.Variable(N_m, nonneg=True)

        # CRB Taylor (want to minimize → descent direction)
        CRB_taylor = (crb_gx @ (S[0, hover_idx] - S_init[0, hover_idx])
                      + crb_gy @ (S[1, hover_idx] - S_init[1, hover_idx]))

        # Rate Taylor (want to maximize → ascent direction)
        R_taylor = (rate_gx @ (S[0, :] - S_init[0, :])
                    + rate_gy @ (S[1, :] - S_init[1, :]))

        # Normalized increment: minimize η·(CRB/Ψ^s_prev) - (1-η)·(R/Ψ^c_prev)
        obj = cp.Minimize(
            eta * CRB_taylor / sens_norm
            - (1 - eta) * R_taylor / comm_norm
        )

        cons = [E_m >= _build_energy_cvx(V, delta, N_m, K_m)]
        cons.append((S[:, 0] - s_start) / P.T_f == V[:, 0])
        if N_m > 1:
            cons.append((S[:, 1:] - S[:, :-1]) / P.T_f == V[:, 1:])
        for i in range(N_m):
            cons.append(cp.norm(V[:, i], 2) <= P.V_max)
        cons += [S[0, :] >= 0, S[1, :] >= 0,
                 S[0, :] <= P.L_x, S[1, :] <= P.L_y]

        for i in range(N_m):
            vi = V_init[:, i]
            vi_sq = float(np.sum(vi**2))
            lhs_a = vi_sq / P.v_0**2 + (2.0 / P.v_0**2) * vi @ (V[:, i] - vi)
            cons.append(lhs_a >= cp.power(cp.inv_pos(delta[i]), 2) - xi[i])
            sq_d = float(np.sqrt(dsq[i]))
            cons.append(dsq[i] + 2.0 * sq_d * (delta[i] - sq_d) >= xi[i])

        prob = cp.Problem(obj, cons)
        solved = _try_solve(prob)

        if not solved or S.value is None or np.any(np.isnan(S.value)):
            S_init = S_init_valid.copy()
            N_nan += 1
            if u < nit - 1:
                scale = 0.97 ** (N_nan ** N_nan)
                S_init = S_init + scale * (S_valid - S_init)
                V_init[:, 0] = (S_init[:, 0] - s_start) / P.T_f
                V_init[:, 1:] = np.diff(S_init, axis=1) / P.T_f
                V_norms = np.linalg.norm(V_init, axis=0)
                dsq = np.sqrt(1 + V_norms**4 / (4 * P.v_0**4)) - V_norms**2 / (2 * P.v_0**2)
                dsq = np.maximum(dsq, 1e-10)
            else:
                break
        else:
            Sv = np.array(S.value)
            Vv = np.array(V.value)
            Vv_norms = np.linalg.norm(Vv, axis=0)

            if np.any(Vv_norms > P.V_max * 1.05) or np.any(Sv < -10):
                S_init = S_init_valid.copy()
                N_nan += 1
                continue

            N_nan = 0
            optval = float(prob.value)

            if abs(best_val) > abs(optval):
                S_valid = Sv.copy()
                S_init_valid = S_init.copy()
                V_valid = Vv.copy()
                best_val = optval

            J_hist[u] = optval
            CRB_hist[u] = float(crb_gx @ (Sv[0, hover_idx] - S_init[0, hover_idx])
                                + crb_gy @ (Sv[1, hover_idx] - S_init[1, hover_idx]))
            R_hist[u] = float(rate_gx @ (Sv[0, :] - S_init[0, :])
                              + rate_gy @ (Sv[1, :] - S_init[1, :]))

            if u < nit - 1 and abs(optval) >= P.opt_threshold:
                S_init = S_init + P.w_star * (Sv - S_init)
                V_init[:, 0] = (S_init[:, 0] - s_start) / P.T_f
                V_init[:, 1:] = np.diff(S_init, axis=1) / P.T_f
                V_norms = np.linalg.norm(V_init, axis=0)
                dsq = np.sqrt(1 + V_norms**4 / (4 * P.v_0**4)) - V_norms**2 / (2 * P.v_0**2)
                dsq = np.maximum(dsq, 1e-10)
            else:
                break

    return S_valid, V_valid, J_hist, CRB_hist, R_hist


# ======================== Bandwidth Allocation P'_2(j) ========================

def optimize_bandwidth(S_total, comm_users, N_total):
    """
    Solve P'_2(j): max min_m ψ^c_m(B_m)  s.t. Σ B_m ≤ B.
    This is a convex problem (rate is concave in B_m for FSPL model).
    Uses bisection on the min-rate level.
    Returns: (M,) optimal bandwidth allocation
    """
    M_u = comm_users.shape[1]

    # Simple iterative waterfilling-style approach
    B_alloc = np.full(M_u, P.B / M_u)

    # Compute per-user "channel quality" (sum of 1/d^2 over trajectory)
    from models import all_user_distances
    D = all_user_distances(S_total, comm_users)  # (M, N)

    # Quality metric: higher = better channel → needs less bandwidth
    quality = np.sum(1.0 / D**2, axis=1)  # (M,)

    # Allocate inversely proportional to quality (roughly)
    inv_q = 1.0 / (quality + 1e-12)
    B_alloc = P.B * inv_q / np.sum(inv_q)

    # Ensure minimum bandwidth
    B_alloc = np.maximum(B_alloc, P.B * 0.05 / M_u)
    B_alloc = P.B * B_alloc / np.sum(B_alloc)

    return B_alloc

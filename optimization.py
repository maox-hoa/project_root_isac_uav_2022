"""
Trajectory optimization at stage m using CVXPY.
Accepts variable N_stg/K_stg for ending stage support.
"""
import numpy as np
import cvxpy as cp
from models import get_hover_indices
from crb_functions import crb_grad as compute_crb_grad
from rate_functions import rate_grad as compute_rate_grad
import parameters as P


def _build_energy_cvx(V_var, delta_var, N, K):
    """Energy constraint expression (eq. 44)."""
    sum_v_sq = cp.sum_squares(V_var)
    sum_v_cube = 0
    for i in range(N):
        sum_v_cube += cp.power(cp.norm(V_var[:, i], 2), 3)

    ps1 = P.P_0 * (N + 3.0 / P.U_tip**2 * sum_v_sq)
    ps1 += 0.5 * P.D_0 * P.rho * P.s_rotor * P.A_rotor * sum_v_cube
    ps2 = P.P_I * cp.sum(delta_var)
    ps3 = K * (P.P_0 + P.P_I)
    return P.T_f * (ps1 + ps2) + P.T_h * ps3


def _try_solve(prob):
    """Try solvers in preference order."""
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


def optimize_m(E_m, s_c, S_hover_all, S_total, s_target_est, s_start,
               N_stg_m=None, K_stg_m=None, eta_val=None, n_iter_val=None):
    """
    Optimize trajectory for stage m.

    N_stg_m, K_stg_m: waypoints/hovers for THIS stage (can differ from default
                       for ending stage). If None, uses P.N_stg, P.K_stg.
    """
    N_m = N_stg_m if N_stg_m is not None else P.N_stg
    K_m = K_stg_m if K_stg_m is not None else P.K_stg
    eta = eta_val if eta_val is not None else P.eta
    nit = n_iter_val if n_iter_val is not None else P.n_iter
    hover_idx = get_hover_indices(N_m)  # indices within this stage

    # ---- Initial values ----
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
    R_best = 0.0; CRB_best = 0.0; best_val = 1e13; N_nan = 0

    J_hist = np.full(nit, np.nan)
    CRB_hist = np.full(nit, np.nan)
    R_hist = np.full(nit, np.nan)

    for u in range(nit):
        # Update current stage in global arrays
        S_hover_all[:, -K_m:] = S_init[:, hover_idx]
        S_total[:, -N_m:] = S_init

        # Gradients: CRB over ALL hovers, Rate over current stage's N_m pts
        crb_gx, crb_gy = compute_crb_grad(S_hover_all, s_target_est, K_m)
        N_tot = S_total.shape[1]
        rate_gx, rate_gy = compute_rate_grad(S_init, s_c, N_tot)

        # ---- CVX block ----
        S = cp.Variable((2, N_m))
        V = cp.Variable((2, N_m))
        delta = cp.Variable(N_m, nonneg=True)
        xi = cp.Variable(N_m, nonneg=True)

        CRB_taylor = (crb_gx @ (S[0, hover_idx] - S_init[0, hover_idx])
                      + crb_gy @ (S[1, hover_idx] - S_init[1, hover_idx]))
        R_taylor = (rate_gx @ (S[0, :] - S_init[0, :])
                    + rate_gy @ (S[1, :] - S_init[1, :]))

        obj = cp.Minimize(eta * CRB_taylor - (1 - eta) * R_taylor / P.B)

        cons = [E_m >= _build_energy_cvx(V, delta, N_m, K_m)]
        cons.append((S[:, 0] - s_start) / P.T_f == V[:, 0])
        if N_m > 1:
            cons.append((S[:, 1:] - S[:, :-1]) / P.T_f == V[:, 1:])
        for i in range(N_m):
            cons.append(cp.norm(V[:, i], 2) <= P.V_max)
        cons += [S[0, :] >= 0, S[1, :] >= 0, S[0, :] <= P.L_x, S[1, :] <= P.L_y]

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
            # ---- Post-solve validation: reject if constraints clearly violated ----
            Sv = np.array(S.value)
            Vv = np.array(V.value)
            Vv_norms = np.linalg.norm(Vv, axis=0)
            if np.any(Vv_norms > P.V_max * 1.05) or np.any(Sv < -10) or np.any(Sv[0] > P.L_x + 10) or np.any(Sv[1] > P.L_y + 10):
                # Treat as infeasible
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
                continue

            N_nan = 0
            optval = float(prob.value)
            Sv = np.array(S.value)
            Vv = np.array(V.value)

            crb_v = float(crb_gx @ (Sv[0, hover_idx] - S_init[0, hover_idx])
                          + crb_gy @ (Sv[1, hover_idx] - S_init[1, hover_idx]))
            rate_v = float(rate_gx @ (Sv[0, :] - S_init[0, :])
                           + rate_gy @ (Sv[1, :] - S_init[1, :]))

            if abs(best_val) > abs(optval):
                S_valid = Sv.copy()
                S_init_valid = S_init.copy()
                V_valid = Vv.copy()
                R_best = rate_v; CRB_best = crb_v; best_val = optval

            J_hist[u] = optval; CRB_hist[u] = crb_v; R_hist[u] = rate_v

            if u < nit - 1 and abs(optval) >= P.opt_threshold:
                S_init = S_init + P.w_star * (Sv - S_init)
                V_init[:, 0] = (S_init[:, 0] - s_start) / P.T_f
                V_init[:, 1:] = np.diff(S_init, axis=1) / P.T_f
                V_norms = np.linalg.norm(V_init, axis=0)
                dsq = np.sqrt(1 + V_norms**4 / (4 * P.v_0**4)) - V_norms**2 / (2 * P.v_0**2)
                dsq = np.maximum(dsq, 1e-10)
            else:
                break

    return S_valid, V_valid, CRB_best, R_best, J_hist, CRB_hist, R_hist
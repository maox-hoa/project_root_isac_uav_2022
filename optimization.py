"""
Trajectory optimization at stage m using CVXPY.
Translates optimize_m.m from MATLAB/CVX to Python/CVXPY.
"""
import numpy as np
import cvxpy as cp
from parameters import *
from models import get_hover_indices
from crb_functions import crb_grad as compute_crb_grad
from rate_functions import rate_grad as compute_rate_grad


def calc_constraint_energy_cvx(V_var, delta_var, N):
    """
    Build CVXPY expression for the energy constraint (eq. 44).
    V_var: cp.Variable(2, N)
    delta_var: cp.Variable(N)
    Returns: CVXPY expression (convex)
    """
    K = N // mu

    # sum of ||v_i||^2 = total sum of squares
    sum_v_sq = cp.sum_squares(V_var)

    # sum of ||v_i||^3
    v_cube_terms = [cp.power(cp.norm(V_var[:, i], 2), 3) for i in range(N)]
    sum_v_cube = sum(v_cube_terms)

    power_sum1 = P_0 * (N + 3.0 / U_tip**2 * sum_v_sq)
    power_sum1 += 0.5 * D_0 * rho * s_rotor * A_rotor * sum_v_cube
    power_sum2 = P_I * cp.sum(delta_var)
    power_sum3 = K * (P_0 + P_I)

    E_const = T_f * (power_sum1 + power_sum2) + T_h * power_sum3
    return E_const


def optimize_m(E_m, s_c, S_hover_all, S_total, s_target_est, s_start):
    """
    Optimize trajectory for stage m (Algorithm 1 core loop).
    Translates optimize_m.m.

    Parameters:
        E_m:            Available energy [J]
        s_c:            (2,) communication user position
        S_hover_all:    (2, K_total) all hover points (prev stages + current init)
        S_total:        (2, N_total) all trajectory points (prev + current init)
        s_target_est:   (2,) estimated target position
        s_start:        (2,) start position of this stage

    Returns:
        S_opt:      (2, N_stg) optimized trajectory
        V_opt:      (2, N_stg) optimized velocities
        CRB_opt:    scalar CRB Taylor value
        R_opt:      scalar Rate Taylor value
        J_history:  (n_iter,) objective values per iteration
        CRB_history: (n_iter,) CRB values per iteration
        R_history:  (n_iter,) Rate values per iteration
    """
    hover_idx = get_hover_indices(N_stg)  # 0-indexed hover indices within stage

    # ---- Initial values ----
    S_init = S_total[:, -N_stg:].copy()

    # Initial velocity from S_init
    V_init = np.zeros((2, N_stg))
    V_init[:, 0] = (S_init[:, 0] - s_start) / T_f
    V_init[:, 1:] = np.diff(S_init, axis=1) / T_f

    # delta_square_init (eq. 41 evaluated at V_init)
    V_init_norms = np.linalg.norm(V_init, axis=0)
    delta_sq_init = (np.sqrt(1 + V_init_norms**4 / (4 * v_0**4))
                     - V_init_norms**2 / (2 * v_0**2))
    delta_sq_init = np.maximum(delta_sq_init, 1e-10)

    # Valid solution buffers
    S_init_valid = S_init.copy()
    S_valid = S_init * 1.1
    V_valid = V_init.copy()
    delta_valid = np.sqrt(delta_sq_init)
    xi_valid = delta_sq_init.copy()
    R_opt_valid = 0.0
    CRB_opt_valid = 0.0
    best_val = 1e13
    N_nan = 0

    J_history = np.full(n_iter, np.nan)
    CRB_history = np.full(n_iter, np.nan)
    R_history = np.full(n_iter, np.nan)

    for u in range(n_iter):
        # ---- Update hover and total trajectory with current S_init ----
        S_hover_all[:, -K_stg:] = S_init[:, hover_idx]
        S_total[:, -N_stg:] = S_init

        # ---- Compute gradients at S_init ----
        crb_gx, crb_gy = compute_crb_grad(S_hover_all, s_target_est, K_stg)
        N_total = S_total.shape[1]
        rate_gx, rate_gy = compute_rate_grad(S_init, s_c, N_total)

        # ---- CVXPY Problem ----
        S_var = cp.Variable((2, N_stg))
        V_var = cp.Variable((2, N_stg))
        delta_var = cp.Variable(N_stg)
        xi_var = cp.Variable(N_stg)

        # Taylor expansion of CRB (linear in hover points of S_var)
        crb_taylor = 0
        for j in range(K_stg):
            hi = hover_idx[j]
            crb_taylor += crb_gx[j] * (S_var[0, hi] - S_init[0, hi])
            crb_taylor += crb_gy[j] * (S_var[1, hi] - S_init[1, hi])

        # Taylor expansion of Rate (linear in all waypoints of S_var)
        rate_taylor = 0
        for n in range(N_stg):
            rate_taylor += rate_gx[n] * (S_var[0, n] - S_init[0, n])
            rate_taylor += rate_gy[n] * (S_var[1, n] - S_init[1, n])

        # Objective: minimize eta*CRB_taylor - (1-eta)*R_taylor/B
        objective = cp.Minimize(eta * crb_taylor - (1 - eta) * rate_taylor / B)

        # ---- Constraints ----
        constraints = []

        # Energy constraint
        E_expr = calc_constraint_energy_cvx(V_var, delta_var, N_stg)
        constraints.append(E_m >= E_expr)

        # Velocity definition
        constraints.append((S_var[:, 0] - s_start) / T_f == V_var[:, 0])
        if N_stg > 1:
            constraints.append((S_var[:, 1:] - S_var[:, :-1]) / T_f == V_var[:, 1:])

        # Speed limit
        for i in range(N_stg):
            constraints.append(cp.norm(V_var[:, i], 2) <= V_max)

        # Non-negativity
        constraints.append(delta_var >= 0)
        constraints.append(xi_var >= 0)

        # Boundary constraints
        constraints.append(S_var[0, :] >= 0)
        constraints.append(S_var[1, :] >= 0)
        constraints.append(S_var[0, :] <= L_x)
        constraints.append(S_var[1, :] <= L_y)

        # SCA constraints (51a) and (51b)
        for i in range(N_stg):
            v_init_i = V_init[:, i]
            v_init_norm = np.linalg.norm(v_init_i)

            # (51a): linear_lower_bound(||v||^2/v0^2) >= 1/delta^2 - xi
            lhs_51a = ((v_init_norm / v_0)**2
                       + (2 / v_0**2) * v_init_i @ (V_var[:, i] - v_init_i))
            constraints.append(lhs_51a >= cp.square(cp.inv_pos(delta_var[i])) - xi_var[i])

            # (51b): linear_lower_bound(delta^2) >= xi
            sqrt_dsi = np.sqrt(delta_sq_init[i])
            lhs_51b = delta_sq_init[i] + 2 * sqrt_dsi * (delta_var[i] - sqrt_dsi)
            constraints.append(lhs_51b >= xi_var[i])

        # ---- Solve ----
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=30000,
                eps=1e-7,
                acceleration_lookback=20,
                scale=1.0
            )
        except Exception as e:
            print(f"SCS solver failed: {e}")

        # ---- Handle solution ----
        solved = (prob.status in ['optimal', 'optimal_inaccurate']
                  and S_var.value is not None
                  and not np.any(np.isnan(S_var.value))
                  and not np.any(np.isinf(S_var.value)))

        if not solved:
            # Invalid solution: use last valid, adjust S_init
            S_init = S_init_valid.copy()
            N_nan += 1
            if u < n_iter - 1:
                scale = 0.97 ** (N_nan ** N_nan)
                S_init = S_init + scale * (S_valid - S_init)
                V_init[:, 0] = (S_init[:, 0] - s_start) / T_f
                V_init[:, 1:] = np.diff(S_init, axis=1) / T_f
                V_init_norms = np.linalg.norm(V_init, axis=0)
                delta_sq_init = (np.sqrt(1 + V_init_norms**4 / (4 * v_0**4))
                                 - V_init_norms**2 / (2 * v_0**2))
                delta_sq_init = np.maximum(delta_sq_init, 1e-10)
            else:
                break
        else:
            # Valid solution
            N_nan = 0
            S_sol = S_var.value
            V_sol = V_var.value
            delta_sol = delta_var.value
            xi_sol = xi_var.value

            opt_val = prob.value
            # Evaluate Taylor values at solution
            crb_val = 0.0
            for j in range(K_stg):
                hi = hover_idx[j]
                crb_val += crb_gx[j] * (S_sol[0, hi] - S_init[0, hi])
                crb_val += crb_gy[j] * (S_sol[1, hi] - S_init[1, hi])
            rate_val = 0.0
            for n in range(N_stg):
                rate_val += rate_gx[n] * (S_sol[0, n] - S_init[0, n])
                rate_val += rate_gy[n] * (S_sol[1, n] - S_init[1, n])

            if abs(best_val) > abs(opt_val):
                S_valid = S_sol.copy()
                S_init_valid = S_init.copy()
                V_valid = V_sol.copy()
                delta_valid = delta_sol.copy()
                xi_valid = xi_sol.copy()
                R_opt_valid = rate_val
                CRB_opt_valid = crb_val
                best_val = opt_val

            J_history[u] = opt_val
            CRB_history[u] = crb_val
            R_history[u] = rate_val

            # Update S_init for next iteration (eq. 54-55 analog)
            if u < n_iter - 1 and abs(opt_val) >= opt_threshold:
                S_init = S_init + w_star * (S_sol - S_init)
                V_init[:, 0] = (S_init[:, 0] - s_start) / T_f
                V_init[:, 1:] = np.diff(S_init, axis=1) / T_f
                V_init_norms = np.linalg.norm(V_init, axis=0)
                delta_sq_init = (np.sqrt(1 + V_init_norms**4 / (4 * v_0**4))
                                 - V_init_norms**2 / (2 * v_0**2))
                delta_sq_init = np.maximum(delta_sq_init, 1e-10)
            else:
                break

    return (S_valid, V_valid, CRB_opt_valid, R_opt_valid,
            J_history, CRB_history, R_history)

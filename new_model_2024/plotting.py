"""Plotting — extended for M CUs, K STs."""
import numpy as np
import matplotlib.pyplot as plt
from parameters import L_x, L_y
from models import get_hover_indices


def plot_map(results, setup, save_path=None):
    s_b = setup['base_station_pos']
    comm_users = setup['comm_user_pos']       # (2, M)
    targets_true = setup['sense_target_pos']  # (2, K)
    est_init = setup['est_sense_targets']     # (2, K)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Trajectory stages
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(results['M'], 1)))
    s_prev = s_b.copy()
    for i, S_opt in enumerate(results['S_opt_stages']):
        traj = np.column_stack([s_prev, S_opt])
        ax.plot(traj[0], traj[1], 'o-', color=colors[i], markersize=2,
                label=f'Stage {i+1}' if i < 5 else None)
        h_idx = get_hover_indices(S_opt.shape[1])
        ax.plot(S_opt[0, h_idx], S_opt[1, h_idx], 'ko', markersize=4)
        s_prev = S_opt[:, -1].copy()

    # Target estimates evolution
    K_t = targets_true.shape[1]
    for k in range(K_t):
        est_history = [est_init[:, k]] + [e[:, k] for e in results['targets_est_list']]
        est_arr = np.column_stack(est_history)
        ax.plot(est_arr[0], est_arr[1], 'b-d', markersize=8,
                markerfacecolor='none', linewidth=1,
                label=f'$\\hat{{s}}_{{t,{k+1}}}$' if k == 0 else None)

    # Fixed points
    ax.plot(s_b[0], s_b[1], 'g*', markersize=15, label='Base')
    for k in range(K_t):
        ax.plot(targets_true[0, k], targets_true[1, k], 'r^',
                markersize=12, label=f'ST {k+1}' if k < 3 else None)
    M_u = comm_users.shape[1]
    for m in range(M_u):
        ax.plot(comm_users[0, m], comm_users[1, m], 'cs',
                markersize=12, label=f'CU {m+1}' if m < 3 else None)

    ax.set_xlim([0, L_x]); ax.set_ylim([0, L_y])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    ax.set_title(f'ISAC-UAV Trajectory ({results["M"]} stages, '
                 f'{M_u} CUs, {K_t} STs)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_convergence(J_history_stages, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, J in enumerate(J_history_stages):
        valid = ~np.isnan(J)
        if np.any(valid):
            ax.plot(np.where(valid)[0] + 1, J[valid], 'o-', label=f'm={i+1}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective')
    ax.set_title('Convergence')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax

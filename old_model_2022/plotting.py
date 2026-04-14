"""
Plotting utilities for ISAC-UAV simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from parameters import L_x, L_y, mu, N_stg
from models import get_hover_indices


def plot_map(results, setup, save_path=None, title_str=None):
    """
    Plot the UAV trajectory, targets, users, and hover points.
    """
    s_b = setup['base_station_pos']
    s_c = setup['comm_user_pos']
    s_t = setup['sense_target_pos']
    est_init = setup['est_sense_target']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory stages
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, results['M']))
    s_prev = s_b.copy()
    for i, S_opt in enumerate(results['S_opt_stages']):
        traj = np.column_stack([s_prev, S_opt])
        ax.plot(traj[0], traj[1], 'o-', color=colors[i], markersize=2,
                label=f'Stage {i+1}' if i < 4 else None)
        # Hover points
        h_idx = get_hover_indices(S_opt.shape[1])
        ax.plot(S_opt[0, h_idx], S_opt[1, h_idx], 'ko', markersize=4)
        s_prev = S_opt[:, -1].copy()

    # Estimated target positions
    est_positions = np.column_stack([est_init] + results['S_target_est_list'])
    ax.plot(est_positions[0], est_positions[1], 'b-d', markersize=10,
            markerfacecolor='none', linewidth=1.1, label='$\\hat{s}_t$')

    # Fixed points
    ax.plot(s_b[0], s_b[1], 'g*', markersize=15, label='$s_B$ (base)')
    ax.plot(s_t[0], s_t[1], 'r^', markersize=12, markerfacecolor='r', label='$s_t$ (target)')
    ax.plot(s_c[0], s_c[1], 'cs', markersize=12, markerfacecolor='c', label='$s_c$ (user)')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim([0, L_x])
    ax.set_ylim([0, L_y])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if title_str:
        ax.set_title(title_str)
    else:
        ax.set_title(f'ISAC UAV Trajectory ({results["M"]} stages)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_convergence(J_history_stages, save_path=None):
    """Plot objective function convergence across stages."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, J in enumerate(J_history_stages):
        valid = ~np.isnan(J)
        if np.any(valid):
            ax.plot(np.where(valid)[0] + 1, J[valid], 'o-', label=f'm={i+1}')

    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$\eta \widetilde{CRB} - (1-\eta)\bar{R}/B$')
    ax.set_title('Convergence at different stages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax
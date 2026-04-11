"""
Main entry point for ISAC-UAV simulation.
Translates multi_stage_script.m.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from parameters import *
from multi_stage import multi_stage
from plotting import plot_map, plot_convergence

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_default():
    """Run with default parameters (matches multi_stage_script.m)."""
    np.random.seed(0)

    setup = {
        'base_station_pos': base_station_pos,
        'comm_user_pos': comm_user_pos,
        'sense_target_pos': sense_target_pos,
        'est_sense_target': est_sense_target,
        'total_energy': total_energy,
    }

    print("="*60)
    print("ISAC-UAV Simulation (arXiv version)")
    print(f"  eta={eta}, N_stg={N_stg}, mu={mu}, E_tot={total_energy/1e3:.0f}KJ")
    print(f"  Base: {base_station_pos}, CU: {comm_user_pos}")
    print(f"  ST: {sense_target_pos}, ST_est: {est_sense_target}")
    print("="*60)

    res = multi_stage(setup, verbose=True)

    # Plot trajectory map
    fig1, _ = plot_map(res, setup,
                       save_path=os.path.join(FIGURES_DIR, 'trajectory_map.png'))

    # Plot convergence
    fig2, _ = plot_convergence(res['J_history_stages'],
                               save_path=os.path.join(FIGURES_DIR, 'convergence.png'))

    print(f"\nFigures saved to {FIGURES_DIR}")
    return res


if __name__ == '__main__':
    run_default()

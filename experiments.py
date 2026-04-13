"""
Monte Carlo experiments: energy variation, eta variation.
Translates: monte_carlo.m, var_energy.m, var_eta.m, var_iter.m
"""
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from models import get_all_hover_indices
from crb_functions import crb as compute_crb
from rate_functions import avg_data_rate
from multi_stage import multi_stage
from multiprocessing import Pool, cpu_count
def run_single_mc(args):
    mc, setup_base, params_override = args
    np.random.seed(42 + mc)
    setup = {
        'base_station_pos': setup_base['base_station_pos'],
        'comm_user_pos': setup_base['comm_user_pos'][:, mc],
        'sense_target_pos': setup_base['sense_target_pos'][:, mc],
        'est_sense_target': setup_base['est_sense_target'][:, mc],
        'total_energy': setup_base['total_energy'],
    }

    try:
        res = multi_stage(setup, params_override, verbose=False)
        return res['final_crb'], res['final_rate'], res['final_mse']
    except Exception as e:
        print(f"MC {mc+1} failed: {e}")
        return np.nan, np.nan, np.nan


def monte_carlo(n_mc, setup_base, params_override=None, verbose=False):
    CRB_mc = np.zeros(n_mc)
    Rate_mc = np.zeros(n_mc)
    MSE_mc = np.zeros(n_mc)

    n_cores = 10
    print(f"Using {n_cores} CPU cores...")

    args_list = [(mc, setup_base, params_override) for mc in range(n_mc)]

    with Pool(processes=n_cores) as pool:
        results = pool.map(run_single_mc, args_list)

    for i, (crb_val, rate_val, mse_val) in enumerate(results):
        CRB_mc[i] = crb_val
        Rate_mc[i] = rate_val
        MSE_mc[i] = mse_val

    return {
        'CRB_avg': CRB_mc,
        'Rate_avg': Rate_mc,
        'MSE_avg': MSE_mc
    }

def generate_random_positions(n_mc, seed=None):
    """Generate random CU, ST, and initial estimate positions."""
    if seed is not None:
        np.random.seed(seed)
    return {
        'base_station_pos': base_station_pos,
        'comm_user_pos': np.array([L_x * np.random.rand(n_mc),
                                    L_y * np.random.rand(n_mc)]),
        'sense_target_pos': np.array([L_x * np.random.rand(n_mc),
                                       L_y * np.random.rand(n_mc)]),
        'est_sense_target': np.array([L_x * np.random.rand(n_mc),
                                       L_y * np.random.rand(n_mc)]),
        'total_energy': total_energy,
    }


def var_energy(n_mc=10, energy_vec=None, save_path=None):
    """
    Experiment: vary total energy, run MC simulations.
    Translates var_energy.m.
    """
    if energy_vec is None:
        energy_vec = np.arange(10e3, 36e3, 5e3)

    setup_base = generate_random_positions(n_mc, seed=42)

    CRB_over_energy = np.zeros(len(energy_vec))
    MSE_over_energy = np.zeros(len(energy_vec))
    Rate_over_energy = np.zeros(len(energy_vec))

    for i, E_tot in enumerate(energy_vec):
        print(f"\nEnergy = {E_tot/1e3:.0f} KJ ({i+1}/{len(energy_vec)})")
        setup_base['total_energy'] = E_tot
        res_mc = monte_carlo(n_mc, setup_base)
        CRB_over_energy[i] = np.nanmean(res_mc['CRB_avg'])
        MSE_over_energy[i] = np.nanmean(res_mc['MSE_avg'])
        Rate_over_energy[i] = np.nanmean(res_mc['Rate_avg'])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(energy_vec / 1e3, CRB_over_energy, 'bo-', label='CRB')
    ax1.semilogy(energy_vec / 1e3, MSE_over_energy, 'rs-', label='MSE')
    ax1.set_xlabel('Energy [KJ]')
    ax1.set_ylabel('Estimation Error [m²]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('CRB & MSE vs Energy')

    ax2.plot(energy_vec / 1e3, Rate_over_energy / 1e6, 'go-')
    ax2.set_xlabel('Energy [KJ]')
    ax2.set_ylabel('Avg. Rate [Mbits/s]')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Rate vs Energy')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, energy_vec, CRB_over_energy, MSE_over_energy, Rate_over_energy


def var_eta(n_mc=10, eta_vec=None, save_path=None):
    """
    Experiment: vary eta, run MC simulations.
    Translates var_eta.m.
    """
    if eta_vec is None:
        eta_vec = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    setup_base = generate_random_positions(n_mc, seed=42)

    CRB_over_eta = np.zeros(len(eta_vec))
    MSE_over_eta = np.zeros(len(eta_vec))
    Rate_over_eta = np.zeros(len(eta_vec))

    for i, cur_eta in enumerate(eta_vec):
        print(f"\neta = {cur_eta:.1f} ({i+1}/{len(eta_vec)})")
        res_mc = monte_carlo(n_mc, setup_base, params_override={'eta': cur_eta})
        CRB_over_eta[i] = np.nanmean(res_mc['CRB_avg'])
        MSE_over_eta[i] = np.nanmean(res_mc['MSE_avg'])
        Rate_over_eta[i] = np.nanmean(res_mc['Rate_avg'])

    # Plot tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(eta_vec, CRB_over_eta, 'bo-', label='CRB')
    ax1.semilogy(eta_vec, MSE_over_eta, 'rs-', label='MSE')
    ax1.set_xlabel('η')
    ax1.set_ylabel('Estimation Error [m²]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('CRB & MSE vs η')

    ax2.plot(eta_vec, Rate_over_eta / 1e6, 'go-')
    ax2.set_xlabel('η')
    ax2.set_ylabel('Avg. Rate [Mbits/s]')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Rate vs η')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, eta_vec, CRB_over_eta, MSE_over_eta, Rate_over_eta

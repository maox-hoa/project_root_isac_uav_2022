"""Monte Carlo experiments."""
import numpy as np
import matplotlib.pyplot as plt
import parameters as PM
from multi_stage import multi_stage


def mc_single(setup, eta_val=None, n_iter_val=None):
    try:
        res = multi_stage(setup, eta_val=eta_val, n_iter_val=n_iter_val, verbose=False)
        return res['final_crb'], res['final_rate'], res['final_mse']
    except Exception:
        return np.nan, np.nan, np.nan


def gen_setups(n_mc, seed=42):
    rng = np.random.RandomState(seed)
    return [{
        'base_station_pos': PM.base_station_pos.copy(),
        'comm_user_pos': rng.rand(2) * np.array([PM.L_x, PM.L_y]),
        'sense_target_pos': rng.rand(2) * np.array([PM.L_x, PM.L_y]),
        'est_sense_target': rng.rand(2) * np.array([PM.L_x, PM.L_y]),
        'total_energy': PM.total_energy,
    } for _ in range(n_mc)]


def var_energy(n_mc=5, energy_vec=None, save_path=None):
    if energy_vec is None:
        energy_vec = np.arange(10e3, 36e3, 5e3)
    setups = gen_setups(n_mc)
    CRB_avg, MSE_avg, Rate_avg = [np.zeros(len(energy_vec)) for _ in range(3)]
    for i, E in enumerate(energy_vec):
        print(f"E={E/1e3:.0f}KJ ({i+1}/{len(energy_vec)})")
        cs, rs, ms = [], [], []
        for s in setups:
            s['total_energy'] = E
            c, r, m = mc_single(s)
            cs.append(c); rs.append(r); ms.append(m)
        CRB_avg[i] = np.nanmean(cs); MSE_avg[i] = np.nanmean(ms); Rate_avg[i] = np.nanmean(rs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.semilogy(energy_vec/1e3, CRB_avg, 'bo-', label='CRB')
    a1.semilogy(energy_vec/1e3, MSE_avg, 'rs-', label='MSE')
    a1.set_xlabel('$E_{tot}$ [KJ]'); a1.set_ylabel('Error [m²]'); a1.legend(); a1.grid(True, alpha=.3)
    a2.plot(energy_vec/1e3, Rate_avg/1e6, 'go-')
    a2.set_xlabel('$E_{tot}$ [KJ]'); a2.set_ylabel('Rate [Mbit/s]'); a2.grid(True, alpha=.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    return fig


def var_eta(n_mc=5, eta_vec=None, save_path=None):
    if eta_vec is None:
        eta_vec = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    setups = gen_setups(n_mc)
    CRB_avg, MSE_avg, Rate_avg = [np.zeros(len(eta_vec)) for _ in range(3)]
    for i, et in enumerate(eta_vec):
        print(f"η={et:.2f} ({i+1}/{len(eta_vec)})")
        cs, rs, ms = [], [], []
        for s in setups:
            c, r, m = mc_single(s, eta_val=et)
            cs.append(c); rs.append(r); ms.append(m)
        CRB_avg[i] = np.nanmean(cs); MSE_avg[i] = np.nanmean(ms); Rate_avg[i] = np.nanmean(rs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(Rate_avg/1e6, CRB_avg, 'bo-', ms=8, label='CRB')
    ax.semilogy(Rate_avg/1e6, MSE_avg, 'rs-', ms=8, label='MSE')
    for i, et in enumerate(eta_vec):
        ax.annotate(f'η={et}', (Rate_avg[i]/1e6, MSE_avg[i]), textcoords="offset points", xytext=(8,5), fontsize=9)
    ax.set_xlabel('Rate [Mbit/s]'); ax.set_ylabel('Error [m²]'); ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    return fig
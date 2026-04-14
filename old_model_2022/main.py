"""Main entry point."""
import numpy as np, time, warnings, os, sys
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(__file__))

import parameters as PM
from multi_stage import multi_stage
from plotting import plot_map, plot_convergence

FIGS = os.path.join(os.path.dirname(__file__), '../figures')
os.makedirs(FIGS, exist_ok=True)

if __name__ == '__main__':
    np.random.seed(0)
    t0 = time.time()
    setup = {
        'base_station_pos': PM.base_station_pos.copy(),
        'comm_user_pos': PM.comm_user_pos.copy(),
        'sense_target_pos': PM.sense_target_pos.copy(),
        'est_sense_target': PM.est_sense_target.copy(),
        'total_energy': PM.total_energy,
    }
    print(f"ISAC-UAV: η={PM.eta}, N_stg={PM.N_stg}, E={PM.total_energy/1e3:.0f}KJ")
    res = multi_stage(setup, verbose=True)
    plot_map(res, setup, save_path=os.path.join(FIGS, 'trajectory_map.png'))
    plot_convergence(res['J_history_stages'], save_path=os.path.join(FIGS, 'convergence.png'))
    print(f"Done in {time.time()-t0:.0f}s → {FIGS}")
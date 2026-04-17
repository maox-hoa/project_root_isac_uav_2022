"""
Mô phỏng chính - tái hiện các kết quả trong Section V của bài báo.

Các figure chính ta sẽ tái hiện:
  Fig. 3: Convergence của thuật toán lặp
  Fig. 4: MSE và CRB qua các stage của MSTD
  Fig. 5: C&S performance vs E_tot, so sánh ISAC vs Separate/Straight/Circle
  Fig. 6: Quỹ đạo UAV dưới các E_tot khác nhau
  Fig. 9: Tradeoff C&S khi quét η

Kết quả được lưu vào /home/claude/isac_uav/results/
"""
from __future__ import annotations
import os
import pickle
import time
from pathlib import Path
import numpy as np

from config import DEFAULT, SimulationConfig
from mstd import run_mstd
from baselines import (
    straight_trajectory, circle_trajectory, evaluate_trajectory,
    run_separate_scheme,
)
from trajectory_optimizer import (
    optimize_trajectory_stage, compute_Psi_c, compute_Psi_s,
    init_hypothetical_trajectory, _extract_hps, compute_stage_energy,
)
from mle_estimator import coarse_initial_estimate

RESULTS_DIR = Path("/home/claude/isac_uav/results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SCENARIO generation
# ---------------------------------------------------------------------------
def generate_random_scenario(M: int, K: int,
                             cfg: SimulationConfig = DEFAULT,
                             rng: np.random.Generator | None = None,
                             min_dist: float = 100.0):
    """Sinh ngẫu nhiên M CU và K ST trong vùng [0, Lx] x [0, Ly]."""
    if rng is None:
        rng = np.random.default_rng()

    # CU tập trung ở 1 nửa, ST ở nửa khác để dễ minh hoạ tradeoff
    cus = rng.uniform([0.6*cfg.Lx, 0.3*cfg.Ly],
                       [0.95*cfg.Lx, 0.9*cfg.Ly], size=(M, 2))
    sts = rng.uniform([0.15*cfg.Lx, 0.3*cfg.Ly],
                       [0.75*cfg.Lx, 0.9*cfg.Ly], size=(K, 2))
    return cus, sts


# ---------------------------------------------------------------------------
# FIG. 3: Convergence behaviour của Algorithm 1
# ---------------------------------------------------------------------------
def sim_fig3_convergence(cfg: SimulationConfig = DEFAULT,
                         seed: int = 1, verbose: bool = True):
    """
    Chạy tối ưu cho stage 1 với η=0.5, K=M=2, E_tot=40kJ.
    Ghi lại hàm mục tiêu, Ψ_c và Ψ_s qua các iterations.
    """
    rng = np.random.default_rng(seed)
    cus, sts_true = generate_random_scenario(M=2, K=2, cfg=cfg, rng=rng)
    if verbose:
        print(f"[Fig3] CU: {cus}")
        print(f"[Fig3] ST: {sts_true}")

    # Coarse estimate
    sts_est = np.array([coarse_initial_estimate(sts_true[k], cfg, rng)
                        for k in range(2)])

    start = cfg.base_station
    Nf = cfg.Nstg
    B_alloc = np.full(2, cfg.B / 2)
    S_ref = init_hypothetical_trajectory(start, cfg.ground_area_corner, Nf, cfg)
    HP_ref = _extract_hps(S_ref, cfg.mu)
    Psi_c_prev = compute_Psi_c(S_ref, HP_ref, cus, B_alloc, cfg)
    Psi_s_prev = compute_Psi_s(HP_ref, sts_est, cfg)

    # Chạy iterative: max_iter tăng dần để ghi trạng thái sau mỗi iteration
    # Để tiết kiệm thời gian: không chạy 2 lần (1 full + 15 lần partial),
    # mà tận dụng việc Algorithm 1 đã ghi obj_history. Ta sinh lại Ψc, Ψs
    # bằng cách chạy step-by-step RE-USE cache.
    psi_c_iter = [Psi_c_prev]   # iter 0 = hypothetical
    psi_s_iter = [Psi_s_prev]

    max_it = 12
    for it_target in range(1, max_it + 1):
        r = optimize_trajectory_stage(
            start_point=start, Nf=Nf, remaining_energy=40e3,
            prev_waypoints=np.zeros((0,2)), prev_hover_points=np.zeros((0,2)),
            cus=cus, sts_estimate=sts_est,
            Psi_c_prev=Psi_c_prev, Psi_s_prev=Psi_s_prev,
            bandwidths=B_alloc, eta=0.5, cfg=cfg,
            max_iter=it_target, verbose=False,
        )
        S_it = r.waypoints
        HP_it = r.hover_points
        psi_c_iter.append(compute_Psi_c(S_it, HP_it, cus, B_alloc, cfg))
        psi_s_iter.append(compute_Psi_s(HP_it, sts_est, cfg))
        if r.iter_count < it_target:
            # Đã converged
            break

    # Object history từ lần chạy cuối cùng
    data = {
        "obj_history": r.obj_history,
        "psi_c_per_iter": np.array(psi_c_iter),
        "psi_s_per_iter": np.array(psi_s_iter),
        "cus": cus, "sts_true": sts_true, "sts_est": sts_est,
        "final_waypoints": r.waypoints,
        "final_hover_points": r.hover_points,
    }
    with open(RESULTS_DIR / "fig3_convergence.pkl", "wb") as f:
        pickle.dump(data, f)

    if verbose:
        print(f"[Fig3] Đã lưu vào fig3_convergence.pkl")
        print(f"  Obj history: {[f'{o:.4f}' for o in r.obj_history]}")
        print(f"  Ψc final: {psi_c_iter[-1]/1e9:.3f} Gbits")
        print(f"  Ψs final: {psi_s_iter[-1]:.3e} m²")

    return data


# ---------------------------------------------------------------------------
# FIG. 4: MSE và CRB qua các stage
# ---------------------------------------------------------------------------
def sim_fig4_mstd_per_stage(cfg: SimulationConfig = DEFAULT,
                             seed: int = 2, verbose: bool = True):
    """
    Chạy MSTD với E_tot = 60kJ, η=1 (full sensing) để thấy sự cải thiện
    sensing qua các stage (Fig. 4 bài báo).
    """
    rng = np.random.default_rng(seed)
    cus, sts_true = generate_random_scenario(M=2, K=2, cfg=cfg, rng=rng)
    if verbose:
        print(f"[Fig4] CU: {cus}")
        print(f"[Fig4] ST: {sts_true}")

    # Lặp Monte Carlo để có MSE đáng tin
    num_mc = 5
    crb_list = []    # (num_mc, num_stages, K)
    mse_list = []    # (num_mc, num_stages, K)

    for trial in range(num_mc):
        trial_rng = np.random.default_rng(seed * 100 + trial)
        res = run_mstd(
            cus=cus, sts_true=sts_true, Etot=60e3, eta=1.0, cfg=cfg,
            use_bandwidth_alloc=False, max_stages=6,
            max_iter_per_stage=5, rng=trial_rng, verbose=(verbose and trial==0),
        )
        # Padding nếu ít stage hơn
        crb_list.append(np.array(res.crb_history))
        mse_list.append(np.array(res.mse_history))

    # Align các trial về cùng số stage
    min_stg = min(len(c) for c in crb_list)
    crb_arr = np.array([c[:min_stg] for c in crb_list])   # (num_mc, min_stg, K)
    mse_arr = np.array([m[:min_stg] for m in mse_list])

    data = {
        "crb_per_stage": crb_arr,
        "mse_per_stage": mse_arr,
        "crb_mean": crb_arr.mean(axis=0),   # (min_stg, K)
        "mse_mean": mse_arr.mean(axis=0),
        "cus": cus, "sts_true": sts_true,
    }
    with open(RESULTS_DIR / "fig4_mstd_stages.pkl", "wb") as f:
        pickle.dump(data, f)

    if verbose:
        print(f"[Fig4] Đã chạy {num_mc} MC trials, {min_stg} stages:")
        for j in range(min_stg):
            print(f"  Stage {j+1}: CRB mean = {data['crb_mean'][j]}, "
                  f"MSE mean = {data['mse_mean'][j]}")
    return data


# ---------------------------------------------------------------------------
# FIG. 5 + 6 + 7: performance vs E_tot (ISAC vs baselines)
# ---------------------------------------------------------------------------
def sim_fig5_vs_energy(cfg: SimulationConfig = DEFAULT,
                        seed: int = 3,
                        Etot_list: list | None = None,
                        verbose: bool = True):
    """
    So sánh Ψc và Ψs vs E_tot cho ISAC, ISAC+BA, Separate, Straight, Circle.
    Mỗi scheme cùng tổng năng lượng.
    """
    if Etot_list is None:
        Etot_list = [20e3, 30e3, 40e3, 50e3, 60e3]

    rng = np.random.default_rng(seed)
    cus, sts_true = generate_random_scenario(M=2, K=2, cfg=cfg, rng=rng)

    results = {
        "Etot_list": Etot_list,
        "cus": cus, "sts_true": sts_true,
        "isac": {"psi_c": [], "psi_s": [], "waypoints": [], "hover_points": []},
        "isac_ba": {"psi_c": [], "psi_s": [], "waypoints": [], "hover_points": []},
        "separate": {"psi_c": [], "psi_s": []},
        "straight": {"psi_c": [], "psi_s": []},
        "circle": {"psi_c": [], "psi_s": []},
    }

    for E in Etot_list:
        if verbose:
            print(f"\n===== E_tot = {E/1e3:.0f} kJ =====")
        # --- ISAC (không BA) ---
        t0 = time.time()
        r_isac = run_mstd(cus, sts_true, E, eta=0.5, cfg=cfg,
                           use_bandwidth_alloc=False, max_stages=8,
                           max_iter_per_stage=5,
                           rng=np.random.default_rng(seed*7 + int(E)),
                           verbose=False)
        results["isac"]["psi_c"].append(r_isac.total_transmitted_data[-1])
        results["isac"]["psi_s"].append(r_isac.psi_s_history[-1])
        results["isac"]["waypoints"].append(r_isac.all_waypoints)
        results["isac"]["hover_points"].append(r_isac.all_hover_points)
        if verbose:
            print(f"  ISAC:     Ψc={r_isac.total_transmitted_data[-1]/1e9:.3f} Gbits, "
                  f"Ψs={r_isac.psi_s_history[-1]:.3e} m² ({time.time()-t0:.1f}s)")

        # --- ISAC + BA ---
        t0 = time.time()
        r_isac_ba = run_mstd(cus, sts_true, E, eta=0.5, cfg=cfg,
                              use_bandwidth_alloc=True, max_stages=8,
                              max_iter_per_stage=5,
                              rng=np.random.default_rng(seed*7 + int(E)),
                              verbose=False)
        results["isac_ba"]["psi_c"].append(r_isac_ba.total_transmitted_data[-1])
        results["isac_ba"]["psi_s"].append(r_isac_ba.psi_s_history[-1])
        results["isac_ba"]["waypoints"].append(r_isac_ba.all_waypoints)
        results["isac_ba"]["hover_points"].append(r_isac_ba.all_hover_points)
        if verbose:
            print(f"  ISAC+BA:  Ψc={r_isac_ba.total_transmitted_data[-1]/1e9:.3f} Gbits, "
                  f"Ψs={r_isac_ba.psi_s_history[-1]:.3e} m² ({time.time()-t0:.1f}s)")

        # --- Straight ---
        wp = straight_trajectory(cfg.base_station, cfg.ground_area_corner, E, cfg)
        r_st = evaluate_trajectory(wp, cus, sts_true, cfg,
                                    rng=np.random.default_rng(seed*13 + int(E)),
                                    num_mc=3)
        results["straight"]["psi_c"].append(r_st.psi_c)
        results["straight"]["psi_s"].append(r_st.psi_s)
        if verbose:
            print(f"  Straight: Ψc={r_st.psi_c/1e9:.3f} Gbits, Ψs={r_st.psi_s:.3e} m²")

        # --- Circle ---
        wp = circle_trajectory(E, cfg)
        r_ci = evaluate_trajectory(wp, cus, sts_true, cfg,
                                    rng=np.random.default_rng(seed*17 + int(E)),
                                    num_mc=3)
        results["circle"]["psi_c"].append(r_ci.psi_c)
        results["circle"]["psi_s"].append(r_ci.psi_s)
        if verbose:
            print(f"  Circle:   Ψc={r_ci.psi_c/1e9:.3f} Gbits, Ψs={r_ci.psi_s:.3e} m²")

        # --- Separate ---
        t0 = time.time()
        r_sep = run_separate_scheme(
            cus, sts_true, E, cfg=cfg, use_bandwidth_alloc=False,
            rng=np.random.default_rng(seed*19 + int(E)),
            max_stages=8, max_iter_per_stage=5, verbose=False,
        )
        results["separate"]["psi_c"].append(r_sep["psi_c"])
        results["separate"]["psi_s"].append(r_sep["psi_s"])
        if verbose:
            print(f"  Separate: Ψc={r_sep['psi_c']/1e9:.3f} Gbits, "
                  f"Ψs={r_sep['psi_s']:.3e} m² ({time.time()-t0:.1f}s)")

    with open(RESULTS_DIR / "fig5_vs_energy.pkl", "wb") as f:
        pickle.dump(results, f)
    if verbose:
        print(f"\n[Fig5] Lưu vào fig5_vs_energy.pkl")
    return results


# ---------------------------------------------------------------------------
# FIG. 9: tradeoff C&S khi quét η
# ---------------------------------------------------------------------------
def sim_fig9_tradeoff(cfg: SimulationConfig = DEFAULT,
                       seed: int = 5,
                       eta_list: list | None = None,
                       verbose: bool = True):
    if eta_list is None:
        eta_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    rng = np.random.default_rng(seed)
    cus, sts_true = generate_random_scenario(M=2, K=2, cfg=cfg, rng=rng)
    Etot = 40e3

    results = {
        "eta_list": eta_list,
        "cus": cus, "sts_true": sts_true,
        "isac": {"psi_c": [], "psi_s": []},
        "isac_ba": {"psi_c": [], "psi_s": []},
        "trajectories": {},
    }

    for eta in eta_list:
        if verbose:
            print(f"\n===== η = {eta} =====")
        r = run_mstd(cus, sts_true, Etot, eta=eta, cfg=cfg,
                      use_bandwidth_alloc=False, max_stages=5,
                      max_iter_per_stage=5,
                      rng=np.random.default_rng(seed*7 + int(eta*100)),
                      verbose=False)
        results["isac"]["psi_c"].append(r.total_transmitted_data[-1])
        results["isac"]["psi_s"].append(r.psi_s_history[-1])

        r_ba = run_mstd(cus, sts_true, Etot, eta=eta, cfg=cfg,
                        use_bandwidth_alloc=True, max_stages=5,
                        max_iter_per_stage=5,
                        rng=np.random.default_rng(seed*7 + int(eta*100)),
                        verbose=False)
        results["isac_ba"]["psi_c"].append(r_ba.total_transmitted_data[-1])
        results["isac_ba"]["psi_s"].append(r_ba.psi_s_history[-1])

        # Lưu quỹ đạo cho η = 0.2 và 0.8 để minh hoạ (Fig. 9b, 9c)
        if eta in (0.1, 0.3, 0.7, 0.9):
            results["trajectories"][eta] = {
                "waypoints": r.all_waypoints,
                "hover_points": r.all_hover_points,
            }
        if verbose:
            print(f"  ISAC:    Ψc={r.total_transmitted_data[-1]/1e9:.3f} Gbits, "
                  f"Ψs={r.psi_s_history[-1]:.3e}")
            print(f"  ISAC+BA: Ψc={r_ba.total_transmitted_data[-1]/1e9:.3f} Gbits, "
                  f"Ψs={r_ba.psi_s_history[-1]:.3e}")

    with open(RESULTS_DIR / "fig9_tradeoff.pkl", "wb") as f:
        pickle.dump(results, f)
    if verbose:
        print(f"\n[Fig9] Lưu vào fig9_tradeoff.pkl")
    return results


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    cfg = DEFAULT

    if which in ("3", "all"):
        t0 = time.time()
        sim_fig3_convergence(cfg, seed=1, verbose=True)
        print(f"Fig3 xong trong {time.time()-t0:.1f}s\n")

    if which in ("4", "all"):
        t0 = time.time()
        sim_fig4_mstd_per_stage(cfg, seed=2, verbose=True)
        print(f"Fig4 xong trong {time.time()-t0:.1f}s\n")

    if which in ("5", "all"):
        t0 = time.time()
        sim_fig5_vs_energy(cfg, seed=3,
                            Etot_list=[20e3, 40e3, 60e3],  # rút gọn để chạy nhanh
                            verbose=True)
        print(f"Fig5 xong trong {time.time()-t0:.1f}s\n")

    if which in ("9", "all"):
        t0 = time.time()
        sim_fig9_tradeoff(cfg, seed=5,
                           eta_list=[0.1, 0.5, 0.9],  # rút gọn
                           verbose=True)
        print(f"Fig9 xong trong {time.time()-t0:.1f}s\n")

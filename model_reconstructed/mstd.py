"""
Multi-Stage Trajectory Design (MSTD) - Algorithm 2 của bài báo.

Quy trình:
  Init: ước lượng coarse ST qua 3 điểm gần trạm sạc
  For j = 1, 2, ... :
     - Giải P'_1(j): tối ưu quỹ đạo cho stage j
     - Giải P'_2(j): phân bổ băng thông
     - UAV bay theo S_j + hover tại HP -> đo khoảng cách có nhiễu
     - MLE cập nhật ước lượng ST (dùng toàn bộ HP từ stage 1 tới j)
     - Tính E_j còn lại. Nếu không đủ cho stage đầy đủ -> ending stage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from config import SimulationConfig, DEFAULT
from system_model import uav_propulsion_power
from trajectory_optimizer import (
    optimize_trajectory_stage, optimize_bandwidth, compute_stage_energy,
    init_hypothetical_trajectory, _extract_hps,
    compute_Psi_c as compute_Psi_c_opt,
    compute_Psi_s as compute_Psi_s_opt,
)
from mle_estimator import (
    simulate_distance_measurements, mle_estimate_st, coarse_initial_estimate,
)


@dataclass
class MSTDResult:
    stages: list = field(default_factory=list)   # list of (S_j, V_j, HP_j)
    st_estimates_history: list = field(default_factory=list)   # [u_hat_{k,j}] theo j
    mse_history: list = field(default_factory=list)            # MSE mỗi stage
    crb_history: list = field(default_factory=list)            # CRB mỗi stage
    total_transmitted_data: list = field(default_factory=list) # per-stage Ψc
    psi_s_history: list = field(default_factory=list)          # per-stage Ψs
    all_waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    all_hover_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    final_bandwidths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    total_energy_consumed: float = 0.0
    num_stages: int = 0
    all_distance_measurements: list = field(default_factory=list)


def energy_required_for_full_stage(cfg: SimulationConfig = DEFAULT) -> float:
    """Ước lượng năng lượng tối thiểu E_min cho 1 stage đầy đủ.
    Giả định bay ở tốc độ bay hiệu quả (V ~ 10m/s) và hover tại HP.
    """
    P_cruise = uav_propulsion_power(10.0, cfg)   # ~126W
    P_hover = uav_propulsion_power(0.0, cfg)     # ~169W
    Nf = cfg.Nstg
    Nh = Nf // cfg.mu
    return cfg.Tf * Nf * P_cruise + cfg.Th * Nh * P_hover


def run_mstd(cus: np.ndarray,
             sts_true: np.ndarray,
             Etot: float,
             eta: float = 0.5,
             cfg: SimulationConfig = DEFAULT,
             use_bandwidth_alloc: bool = True,
             max_stages: int = 10,
             max_iter_per_stage: int = 8,
             rng: np.random.Generator | None = None,
             verbose: bool = True) -> MSTDResult:
    """
    Chạy MSTD đầy đủ.
    cus: (M, 2) CU locations.
    sts_true: (K, 2) ST locations (ground truth, dùng để sinh đo).
    Etot: Tổng năng lượng (J).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    K = len(sts_true)
    M = len(cus)
    result = MSTDResult()

    # --- 1. Khởi tạo: ước lượng coarse qua 3 điểm gần trạm sạc ---
    sts_est = np.array([
        coarse_initial_estimate(sts_true[k], cfg, rng)
        for k in range(K)
    ])
    if verbose:
        print(f"[MSTD] Coarse estimate:")
        for k in range(K):
            err = np.linalg.norm(sts_est[k] - sts_true[k])
            print(f"  ST {k}: true={sts_true[k]}, est={sts_est[k]} (err={err:.1f}m)")

    result.st_estimates_history.append(sts_est.copy())

    # Initial bandwidth (đều)
    B_alloc = np.full(M, cfg.B / M)

    # Accumulators
    all_waypoints = np.zeros((0, 2))
    all_hover_points = np.zeros((0, 2))
    all_d_measurements = [[] for _ in range(K)]   # [k][stage][hp]
    current_position = cfg.base_station.copy()
    E_remaining = Etot
    E_min_stage = energy_required_for_full_stage(cfg)

    # --- Chuẩn bị reference Ψ_prev (hypothetical trajectory) ---
    S_ref = init_hypothetical_trajectory(current_position,
                                          cfg.ground_area_corner,
                                          cfg.Nstg, cfg)
    HP_ref = _extract_hps(S_ref, cfg.mu)
    Psi_c_prev = compute_Psi_c_opt(S_ref, HP_ref, cus, B_alloc, cfg)
    Psi_s_prev = compute_Psi_s_opt(HP_ref, sts_est, cfg)

    # --- 2. Vòng lặp các stage ---
    j = 0
    while j < max_stages:
        j += 1

        # Nếu hết năng lượng hoàn toàn -> dừng
        if E_remaining <= 0.5e3:   # < 0.5 kJ
            if verbose:
                print(f"\n[MSTD] E còn {E_remaining/1e3:.2f}kJ, dừng.")
            j -= 1
            break

        # Xác định ending stage?
        if E_remaining < E_min_stage:
            # Ending stage: tính N_lst bảo đảm khả thi (dùng V ở điểm Pmin + margin)
            P_cruise_min = uav_propulsion_power(10.0, cfg)    # ~126W (min của Puav)
            P_hover = uav_propulsion_power(0.0, cfg)          # ~169W
            # Năng lượng mỗi line segment + contribution HP
            per_segment = cfg.Tf * P_cruise_min + (cfg.Th * P_hover) / cfg.mu
            # Safety margin 15% để tránh infeasibility
            Nlst = int(0.85 * E_remaining / per_segment)
            Nlst = max(Nlst, cfg.mu)
            Nlst = min(Nlst, cfg.Nstg)
            Nf = Nlst
            if verbose:
                print(f"\n[MSTD] Stage {j} (ENDING): E={E_remaining/1e3:.1f}kJ, Nf={Nf}")
        else:
            Nf = cfg.Nstg
            if verbose:
                print(f"\n[MSTD] Stage {j}: E={E_remaining/1e3:.1f}kJ, Nf={Nf}")

        # --- (a) Optimize trajectory P'_1(j) ---
        # Theo Algorithm 2 (paper): line 2 reset B_m = B/M trước khi giải P'_1(j).
        # Trajectory LUÔN tối ưu với B uniform; BA chỉ áp dụng post-hoc ở line 3.
        B_uniform = np.full(M, cfg.B / M)
        opt_res = optimize_trajectory_stage(
            start_point=current_position,
            Nf=Nf, remaining_energy=E_remaining,
            prev_waypoints=all_waypoints,
            prev_hover_points=all_hover_points,
            cus=cus, sts_estimate=sts_est,
            Psi_c_prev=Psi_c_prev, Psi_s_prev=Psi_s_prev,
            bandwidths=B_uniform, eta=eta, cfg=cfg,
            max_iter=max_iter_per_stage, verbose=verbose,
        )
        S_j = opt_res.waypoints
        V_j = opt_res.velocities
        HP_j = opt_res.hover_points
        if verbose:
            print(f"  Stage {j} trajectory: iters={opt_res.iter_count}, "
                  f"obj={opt_res.obj_history[-1]:.4f}")

        # --- (b) UAV "bay" theo S_j: sinh đo khoảng cách tại HP_j ---
        for k in range(K):
            d_tilde_stage = simulate_distance_measurements(HP_j, sts_true[k], cfg, rng)
            all_d_measurements[k].append(d_tilde_stage)

        # --- (c) Cập nhật waypoints tích luỹ ---
        all_waypoints = np.vstack([all_waypoints, S_j])
        all_hover_points = np.vstack([all_hover_points, HP_j])

        # --- (d) Optimize bandwidth P'_2(j) -- post-hoc, sau khi trajectory cố định ---
        if use_bandwidth_alloc:
            print("Thực hiện tối ưu băng thông. Hàm optimize_bandwidth")
            B_alloc = optimize_bandwidth(all_waypoints, all_hover_points, cus, cfg)
            print(f"B_alloc: {B_alloc}")
        else:
            print("Không thực hiện tối ưu băng thông")
            B_alloc = B_uniform

        # --- (e) Cập nhật ước lượng ST qua MLE (dùng TOÀN BỘ measurements) ---
        for k in range(K):
            d_all = np.concatenate(all_d_measurements[k])
            sts_est[k] = mle_estimate_st(all_hover_points, d_all, cfg, grid_step=25.0)
        if verbose:
            for k in range(K):
                err = np.linalg.norm(sts_est[k] - sts_true[k])
                print(f"  ST {k}: est={sts_est[k]}, err={err:.2f}m")

        # --- (f) Lưu kết quả stage ---
        result.stages.append({"S": S_j, "V": V_j, "HP": HP_j, "start": current_position.copy()})
        result.st_estimates_history.append(sts_est.copy())

        mse = np.array([np.sum((sts_est[k] - sts_true[k])**2) for k in range(K)])
        result.mse_history.append(mse)

        crb = np.array([compute_Psi_s_opt(all_hover_points, sts_true[k:k+1], cfg)
                        for k in range(K)])
        result.crb_history.append(crb.squeeze())

        Psi_c_j = compute_Psi_c_opt(all_waypoints, all_hover_points, cus, B_alloc, cfg)
        Psi_c_non_ba = compute_Psi_c_opt(all_waypoints, all_hover_points, cus, B_uniform, cfg)
        print(f"Psi_c_j: {Psi_c_j} and Psi_c_non_ba: {Psi_c_non_ba}")
        print(f"Rate gain after BA is: {Psi_c_j - Psi_c_non_ba}")
        Psi_s_j = compute_Psi_s_opt(all_hover_points, sts_est, cfg)
        result.total_transmitted_data.append(Psi_c_j)
        result.psi_s_history.append(Psi_s_j)

        # --- (g) Cập nhật năng lượng còn lại + vị trí hiện tại ---
        Nh_stage = HP_j.shape[0]
        E_stage_used = compute_stage_energy(S_j, current_position, Nh_stage, cfg)
        E_remaining -= E_stage_used
        result.total_energy_consumed += E_stage_used
        current_position = S_j[-1].copy()

        # Cập nhật Psi_prev cho stage kế (chính là Psi hiện tại)
        Psi_c_prev = Psi_c_j
        Psi_s_prev = Psi_s_j

        # Ending check
        if Nf < cfg.Nstg:
            break

    result.all_waypoints = all_waypoints
    result.all_hover_points = all_hover_points
    result.final_bandwidths = B_alloc
    result.num_stages = j

    if verbose:
        print(f"\n[MSTD] Hoàn thành {j} stage, tổng năng lượng dùng = "
              f"{result.total_energy_consumed/1e3:.2f}kJ")
        print(f"  Ψ_c cuối = {result.total_transmitted_data[-1]/1e9:.3f} Gbits")
        print(f"  Ψ_s cuối = {result.psi_s_history[-1]:.4e} m²")

    return result


if __name__ == "__main__":
    cfg = DEFAULT
    rng = np.random.default_rng(42)

    # Scenario test (giống Figure 4 của bài báo)
    cus = np.array([[1200., 1200.], [1300., 800.]])
    sts_true = np.array([[1000., 1050.], [300., 1050.]])

    result = run_mstd(
        cus=cus, sts_true=sts_true,
        Etot=40e3, eta=0.5, cfg=cfg,
        use_bandwidth_alloc=False,
        max_stages=5, max_iter_per_stage=5,
        rng=rng, verbose=True,
    )
    print(f"\n=== TÓM TẮT ===")
    print(f"Số stage: {result.num_stages}")
    for j, (psi_c, psi_s, mse) in enumerate(zip(result.total_transmitted_data,
                                                  result.psi_s_history,
                                                  result.mse_history)):
        print(f"  Stage {j+1}: Ψc={psi_c/1e9:.3f} Gbits, Ψs(CRB)={psi_s:.3e}, "
              f"MSE={mse}")
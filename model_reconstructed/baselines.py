"""
Các baseline schemes để so sánh hiệu năng với ISAC-UAV.

1. "Separate": 2 UAV riêng biệt, mỗi UAV chỉ làm 1 nhiệm vụ (comm hoặc sensing).
   - UAV_comm: η=0, chỉ tối ưu quỹ đạo cho truyền thông
   - UAV_sens: η=1, chỉ tối ưu quỹ đạo cho sensing
   Mỗi UAV có năng lượng = E_tot / 2 (để so sánh công bằng: tổng năng lượng bằng ISAC).

2. "Straight": UAV bay thẳng trạm sạc -> góc đối diện với V_max và quay lại, không tối ưu.

3. "Circle": UAV bay vòng tròn tâm (0.5Lx, 0.5Ly) bán kính 0.25 Lx với V=V_max.

Tất cả các scheme đều thực hiện C&S trong khi bay (tại các HP chọn sẵn).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from config import SimulationConfig, DEFAULT
from system_model import uav_propulsion_power
from trajectory_optimizer import (
    compute_Psi_c, compute_Psi_s, _extract_hps, compute_stage_energy,
    optimize_bandwidth,
)
from mle_estimator import simulate_distance_measurements, mle_estimate_st
from mstd import run_mstd


@dataclass
class BaselineResult:
    waypoints: np.ndarray
    hover_points: np.ndarray
    psi_c: float
    psi_s: float     # CRB với ST ước lượng
    mse: np.ndarray  # MSE per ST
    energy_used: float
    st_estimates: np.ndarray
    bandwidths: np.ndarray


# ---------------------------------------------------------------------------
# STRAIGHT trajectory
# ---------------------------------------------------------------------------
def straight_trajectory(start_point: np.ndarray,
                        target_point: np.ndarray,
                        Etot: float,
                        cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """
    UAV bay thẳng trạm sạc -> [Lx, Ly] với V=Vmax, quay lại, tiếp tục
    qua lại cho đến khi hết năng lượng.
    Trả về ma trận waypoints (Nf, 2).
    """
    P_fly = uav_propulsion_power(cfg.Vmax, cfg)
    P_hover = uav_propulsion_power(0.0, cfg)

    # Ước lượng số waypoint khả thi (mỗi waypoint + 1/mu HP đóng góp hover)
    per_step = cfg.Tf * P_fly + cfg.Th * P_hover / cfg.mu
    Nf_max = int(0.95 * Etot / per_step)

    step_len = cfg.Vmax * cfg.Tf
    direction = target_point - start_point
    total_dist = np.linalg.norm(direction)
    if total_dist < 1e-6:
        return np.tile(start_point, (Nf_max, 1))
    unit = direction / total_dist

    # Số waypoint đi 1 chiều
    Nf_oneway = int(np.ceil(total_dist / step_len))

    waypoints = []
    current = start_point.copy()
    going = True
    while len(waypoints) < Nf_max:
        if going:
            current = current + step_len * unit
            if np.dot(current - start_point, unit) >= total_dist:
                current = target_point.copy()
                going = False
        else:
            current = current - step_len * unit
            if np.dot(current - start_point, unit) <= 0:
                current = start_point.copy()
                going = True
        current = np.clip(current, [0, 0], [cfg.Lx, cfg.Ly])
        waypoints.append(current.copy())

    return np.array(waypoints[:Nf_max])


# ---------------------------------------------------------------------------
# CIRCLE trajectory
# ---------------------------------------------------------------------------
def circle_trajectory(Etot: float,
                      cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """
    UAV bay vòng tròn tâm (0.5Lx, 0.5Ly), bán kính 0.25*Lx, với V=Vmax.
    """
    P_fly = uav_propulsion_power(cfg.Vmax, cfg)
    P_hover = uav_propulsion_power(0.0, cfg)
    per_step = cfg.Tf * P_fly + cfg.Th * P_hover / cfg.mu
    Nf_max = int(0.95 * Etot / per_step)

    center = np.array([0.5 * cfg.Lx, 0.5 * cfg.Ly])
    R = 0.25 * cfg.Lx
    step_len = cfg.Vmax * cfg.Tf
    # Góc đi được mỗi bước: θ = step_len / R
    dtheta = step_len / R

    waypoints = []
    theta = 0.0
    for n in range(Nf_max):
        waypoints.append(center + R * np.array([np.cos(theta), np.sin(theta)]))
        theta += dtheta
    return np.array(waypoints)


# ---------------------------------------------------------------------------
# Đánh giá 1 quỹ đạo đã cho sẵn
# ---------------------------------------------------------------------------
def evaluate_trajectory(waypoints: np.ndarray,
                        cus: np.ndarray,
                        sts_true: np.ndarray,
                        cfg: SimulationConfig = DEFAULT,
                        use_bandwidth_alloc: bool = False,
                        rng: np.random.Generator | None = None,
                        num_mc: int = 1) -> BaselineResult:
    """
    Cho 1 quỹ đạo waypoints, tính Ψ_c, Ψ_s, MSE của việc ước lượng ST.
    num_mc: số lần Monte Carlo (sinh đo khoảng cách).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    hover_points = _extract_hps(waypoints, cfg.mu)
    M = len(cus)
    K = len(sts_true)

    # Băng thông
    if use_bandwidth_alloc and hover_points.shape[0] > 0:
        B_alloc = optimize_bandwidth(waypoints, hover_points, cus, cfg)
    else:
        B_alloc = np.full(M, cfg.B / M)

    # MSE (Monte Carlo)
    mse = np.zeros(K)
    sts_est_avg = np.zeros_like(sts_true)
    for trial in range(num_mc):
        sts_est = np.zeros_like(sts_true)
        for k in range(K):
            d_tilde = simulate_distance_measurements(hover_points, sts_true[k], cfg, rng)
            sts_est[k] = mle_estimate_st(hover_points, d_tilde, cfg, grid_step=25.0)
        sts_est_avg += sts_est / num_mc
        for k in range(K):
            mse[k] += np.sum((sts_est[k] - sts_true[k]) ** 2) / num_mc

    # Ψ_c với các giá trị thực
    psi_c = compute_Psi_c(waypoints, hover_points, cus, B_alloc, cfg)
    # Ψ_s: CRB tính trên sts_true để đánh giá fair
    psi_s = compute_Psi_s(hover_points, sts_true, cfg)

    # Năng lượng dùng
    Nh = hover_points.shape[0]
    # Ước lượng: dùng start_point = waypoint đầu để tính velocities
    # Nhưng baselines thường bắt đầu từ trạm sạc
    E_used = compute_stage_energy(waypoints, cfg.base_station, Nh, cfg)

    return BaselineResult(
        waypoints=waypoints,
        hover_points=hover_points,
        psi_c=psi_c, psi_s=psi_s, mse=mse,
        energy_used=E_used,
        st_estimates=sts_est_avg,
        bandwidths=B_alloc,
    )


# ---------------------------------------------------------------------------
# "Separate" scheme: 2 UAV độc lập, mỗi UAV dùng E_tot / 2
# ---------------------------------------------------------------------------
def run_separate_scheme(cus: np.ndarray,
                        sts_true: np.ndarray,
                        Etot: float,
                        cfg: SimulationConfig = DEFAULT,
                        use_bandwidth_alloc: bool = False,
                        rng: np.random.Generator | None = None,
                        max_stages: int = 8,
                        max_iter_per_stage: int = 5,
                        verbose: bool = False) -> dict:
    """
    Chạy "Separate" scheme:
      - UAV_sens: η = 1.0, chỉ tối ưu sensing, năng lượng = Etot / 2
      - UAV_comm: η = 0.0, chỉ tối ưu comm, năng lượng = Etot / 2
    Kết quả của 2 UAV được gộp lại để so sánh.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    E_half = Etot / 2

    if verbose:
        print("===== [Separate] UAV sensing (η=1) =====")
    sens_result = run_mstd(
        cus=cus, sts_true=sts_true, Etot=E_half, eta=1.0, cfg=cfg,
        use_bandwidth_alloc=False, max_stages=max_stages,
        max_iter_per_stage=max_iter_per_stage,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        verbose=verbose,
    )

    if verbose:
        print("\n===== [Separate] UAV comm (η=0) =====")
    comm_result = run_mstd(
        cus=cus, sts_true=sts_true, Etot=E_half, eta=0.0, cfg=cfg,
        use_bandwidth_alloc=use_bandwidth_alloc, max_stages=max_stages,
        max_iter_per_stage=max_iter_per_stage,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        verbose=verbose,
    )

    return {
        "sens": sens_result,
        "comm": comm_result,
        "psi_c": comm_result.total_transmitted_data[-1],
        "psi_s": sens_result.psi_s_history[-1],
        "energy_used": sens_result.total_energy_consumed + comm_result.total_energy_consumed,
    }


if __name__ == "__main__":
    cfg = DEFAULT
    rng = np.random.default_rng(5)
    cus = np.array([[1200., 1200.], [1300., 800.]])
    sts_true = np.array([[1000., 1050.], [300., 1050.]])
    Etot = 40e3

    # Straight
    wp_straight = straight_trajectory(cfg.base_station,
                                      np.array([cfg.Lx, cfg.Ly]),
                                      Etot, cfg)
    r_st = evaluate_trajectory(wp_straight, cus, sts_true, cfg, rng=rng, num_mc=3)
    print(f"[Straight]  Nf={len(wp_straight)}, Ψc={r_st.psi_c/1e9:.3f} Gbits, "
          f"Ψs(CRB)={r_st.psi_s:.3e}, MSE={r_st.mse}, E={r_st.energy_used/1e3:.1f}kJ")

    # Circle
    wp_circle = circle_trajectory(Etot, cfg)
    r_ci = evaluate_trajectory(wp_circle, cus, sts_true, cfg, rng=rng, num_mc=3)
    print(f"[Circle]    Nf={len(wp_circle)}, Ψc={r_ci.psi_c/1e9:.3f} Gbits, "
          f"Ψs(CRB)={r_ci.psi_s:.3e}, MSE={r_ci.mse}, E={r_ci.energy_used/1e3:.1f}kJ")

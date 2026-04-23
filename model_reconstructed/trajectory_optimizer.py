"""
Tối ưu quỹ đạo UAV cho bài toán P'_1(j) (Algorithm 1 của bài báo).

Các bước chính:
1. Tính hướng gradient (ascent direction) của f(S_j) tại điểm hiện tại.
2. Giải bài toán Q''(j): tối đa hoá xấp xỉ tuyến tính bậc một, kèm ràng buộc
   năng lượng được xấp xỉ lồi bằng SCA.
3. Line search dọc theo hướng S*_j - S^{l-1}_j để tìm stepsize tốt nhất.
4. Lặp đến khi hàm mục tiêu không tăng nữa.

Bài toán phụ P'_2(j) cho phân bổ băng thông cũng được giải (lồi, CVX-friendly).
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Callable

from config import SimulationConfig, DEFAULT
from system_model import (
    uav_propulsion_power, comm_rate,
    compute_theta, crb_sum,
    log_sum_exp_min, log_sum_exp_max,
)


# ---------------------------------------------------------------------------
# Các hàm mục tiêu và đạo hàm
# ---------------------------------------------------------------------------
def compute_psi_c_user(all_waypoints: np.ndarray,
                       all_hover_points: np.ndarray,
                       cu_xy: np.ndarray,
                       bandwidth: float,
                       cfg: SimulationConfig = DEFAULT) -> float:
    """
    ψ^c_m(j) eq. (8): tổng dữ liệu tích luỹ đến ST thứ m qua MỌI waypoint
    và hover point từ stage 1 đến j.
    """
    total = 0.0
    for s in all_waypoints:
        total += cfg.Tf * comm_rate(s, cu_xy, bandwidth, cfg)
    for hp in all_hover_points:
        total += cfg.Th * comm_rate(hp, cu_xy, bandwidth, cfg)
    return total


def compute_Psi_c(all_waypoints: np.ndarray,
                  all_hover_points: np.ndarray,
                  cus: np.ndarray,
                  bandwidths: np.ndarray,
                  cfg: SimulationConfig = DEFAULT,
                  smooth: bool = True,
                  t: float | None = None,
                  scale: float | None = None) -> float:
    """Ψ^c(j) - chỉ số comm (paper eq. 9 và 37).

    Mặc định: Log-Sum-Exp smooth-min (đúng paper):
        Ψ^c = -(1/t) log Σ_m exp(-t · ψ^c_m / scale)   × scale

    `scale` cần thiết để LSE hoạt động thực sự — với ψ_c ~ 10⁸, t nhỏ như 1-10
    sẽ bị underflow thành hard min nếu không normalize. scale mặc định =
    max(|ψ|) nếu không cung cấp, cho phép user chọn t theo thang [1, 50] có ý nghĩa.

    Gradient vẫn đúng vì scale là hằng số trong 1 lần gọi.

    smooth=False → hard min (dùng để đánh giá cuối cùng).
    """
    psi_list = np.array([
        compute_psi_c_user(all_waypoints, all_hover_points,
                           cus[m], bandwidths[m], cfg)
        for m in range(len(cus))
    ])
    if not smooth or len(psi_list) == 1:
        return float(np.min(psi_list))
    if t is None:
        t = cfg.t_lse
    if scale is None:
        scale = max(np.max(np.abs(psi_list)), 1e-12)
    # LSE trên ψ đã normalize, rồi scale trả lại
    return float(scale * log_sum_exp_min(psi_list / scale, t))


def compute_Psi_s(all_hover_points: np.ndarray,
                  sts: np.ndarray,
                  cfg: SimulationConfig = DEFAULT,
                  smooth: bool = True,
                  t: float | None = None,
                  scale: float | None = None) -> float:
    """Ψ^s(j) - chỉ số sensing (paper eq. 32 và 38).

    Mặc định: Log-Sum-Exp smooth-max với normalize (xem compute_Psi_c).

    smooth=False → hard max.
    """
    psi_list = np.array([crb_sum(all_hover_points, sts[k], cfg)
                          for k in range(len(sts))])
    if not smooth or len(psi_list) == 1:
        return float(np.max(psi_list))
    if t is None:
        t = cfg.t_lse
    if scale is None:
        scale = max(np.max(np.abs(psi_list)), 1e-12)
    return float(scale * log_sum_exp_max(psi_list / scale, t))


# ---------------------------------------------------------------------------
# HÀM MỤC TIÊU f(S_j, B_j) - smoothed weighted-sum
# ---------------------------------------------------------------------------
def objective_f(stage_waypoints: np.ndarray,
                stage_bandwidths: np.ndarray,
                prev_waypoints: np.ndarray,
                prev_hover_points: np.ndarray,
                cus: np.ndarray,
                sts_estimate: np.ndarray,
                Psi_c_prev: float,
                Psi_s_prev: float,
                eta: float,
                mu: int,
                cfg: SimulationConfig = DEFAULT) -> float:
    """
    Hàm mục tiêu f(S_j, B_j) theo P(j) eq. 34 trong paper.

    f = η · (Ψ^s(j-1) - Ψ^s(j)) / Ψ^s(j-1)
      + (1-η) · (Ψ^c(j) - Ψ^c(j-1)) / Ψ^c(j-1)

    Dùng LSE smoothing (paper eq. 37-38) với t = cfg.t_lse (user tự chọn).
    """
    stage_hover_points = _extract_hps(stage_waypoints, mu)

    all_waypoints = np.vstack([prev_waypoints, stage_waypoints]) \
        if len(prev_waypoints) > 0 else stage_waypoints
    all_hover_points = np.vstack([prev_hover_points, stage_hover_points]) \
        if len(prev_hover_points) > 0 else stage_hover_points

    # LSE-smoothed Ψc và Ψs theo paper eq. 37, 38.
    # Scale = Ψ_prev (constant trong iteration) → gradient analytical đúng.
    Psi_s = compute_Psi_s(all_hover_points, sts_estimate, cfg,
                          smooth=True, scale=Psi_s_prev)
    Psi_c = compute_Psi_c(all_waypoints, all_hover_points,
                          cus, stage_bandwidths, cfg,
                          smooth=True, scale=Psi_c_prev)

    comm_gain = (Psi_c - Psi_c_prev) / max(Psi_c_prev, 1e-9)
    sens_gain = (Psi_s_prev - Psi_s) / max(Psi_s_prev, 1e-9)
    return eta * sens_gain + (1 - eta) * comm_gain


def _extract_hps(stage_waypoints: np.ndarray, mu: int) -> np.ndarray:
    """Trích HP từ các waypoint: HP tại vị trí µγ (γ = 1,2,...) - eq. 1."""
    Nf = stage_waypoints.shape[0]
    Nh = Nf // mu
    if Nh == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, Nh + 1) * mu - 1   # chỉ số 0-based
    return stage_waypoints[idx]


# ---------------------------------------------------------------------------
# GRADIENT GIẢI TÍCH của f đối với các waypoint.
#
# Hàm mục tiêu:
#   f = η * (Ψ_s_prev - Ψ_s(j)) / Ψ_s_prev
#     + (1 - η) * (Ψ_c(j) - Ψ_c_prev) / Ψ_c_prev
#
# Ψ_c(j) = softmin_m ψ^c_m(j)  -  phụ thuộc cả waypoint và HP qua rate.
# Ψ_s(j) = softmax_k ψ^s_k(j)  -  chỉ phụ thuộc HP qua CRB.
# ---------------------------------------------------------------------------
def _grad_Rm_wrt_s(s_xy: np.ndarray, cu_xy: np.ndarray, bandwidth: float,
                   cfg: SimulationConfig) -> np.ndarray:
    """
    ∂R/∂s với R = B log2(1 + γ/d²), γ = Pt α0 / σ², d² = H² + ||s-u||².
    dR/d(d²) = -B γ / [ln2 * (d⁴ + γ d²)];  ∂(d²)/∂s = 2 (s-u).
    """
    d2 = cfg.H**2 + np.sum((s_xy - cu_xy) ** 2)
    gamma = cfg.Pt * cfg.alpha0 / cfg.noise_power
    dR_dd2 = -bandwidth * gamma / (np.log(2) * (d2**2 + gamma * d2))
    return dR_dd2 * 2 * (s_xy - cu_xy)


def _grad_crb_wrt_hp(hp_xy: np.ndarray,
                     all_hover_points: np.ndarray,
                     st_xy: np.ndarray,
                     cfg: SimulationConfig) -> np.ndarray:
    """
    ∂(ψ^s_k)/∂(hp) với ψ^s_k = (Θa + Θb) / D,  D = Θa Θb - Θc².
    Chỉ HP đang xét ảnh hưởng đến các Θ (đạo hàm tại index này).
    """
    K0 = cfg.Pt * cfg.Gp * cfg.beta0 / (cfg.a * cfg.noise_power)

    # Θ tổng (dùng toàn bộ HP)
    dxA = all_hover_points[:, 0] - st_xy[0]
    dyA = all_hover_points[:, 1] - st_xy[1]
    d2A = cfg.H**2 + dxA**2 + dyA**2
    d4A = d2A**2
    d6A = d2A**3
    Theta_a = np.sum(K0 * dxA**2 / d6A + 8 * dxA**2 / d4A)
    Theta_b = np.sum(K0 * dyA**2 / d6A + 8 * dyA**2 / d4A)
    Theta_c = np.sum(K0 * dxA * dyA / d6A + 8 * dxA * dyA / d4A)
    D = Theta_a * Theta_b - Theta_c**2
    if D <= 1e-30:
        return np.zeros(2)

    # Đạo hàm từ HP này (xh, yh)
    dx = hp_xy[0] - st_xy[0]
    dy = hp_xy[1] - st_xy[1]
    d2 = cfg.H**2 + dx**2 + dy**2
    d4 = d2**2; d6 = d2**3; d8 = d2**4

    # ∂(u²/r⁶)/∂xh = 2u/r⁶ − 6u³/r⁸   (vì ∂r⁶/∂xh = 6u·r⁴)
    # ∂(u²/r⁴)/∂xh = 2u/r⁴ − 4u³/r⁶   (vì ∂r⁴/∂xh = 4u·r²)
    dTa_dxh = K0 * (2*dx/d6 - 6*dx**3/d8) + 8 * (2*dx/d4 - 4*dx**3/d6)
    # ∂(u²/r⁶)/∂yh = −6u²v/r⁸;  ∂(u²/r⁴)/∂yh = −4u²v/r⁶
    dTa_dyh = K0 * (-6 * dx**2 * dy / d8) + 8 * (-4 * dx**2 * dy / d6)
    dTb_dxh = K0 * (-6 * dy**2 * dx / d8) + 8 * (-4 * dy**2 * dx / d6)
    dTb_dyh = K0 * (2*dy/d6 - 6*dy**3/d8) + 8 * (2*dy/d4 - 4*dy**3/d6)
    # ∂(uv/r⁶)/∂xh = v/r⁶ − 6u²v/r⁸;  ∂(uv/r⁴)/∂xh = v/r⁴ − 4u²v/r⁶
    dTc_dxh = K0 * (dy/d6 - 6*dx**2*dy/d8) + 8 * (dy/d4 - 4*dx**2*dy/d6)
    dTc_dyh = K0 * (dx/d6 - 6*dx*dy**2/d8) + 8 * (dx/d4 - 4*dx*dy**2/d6)

    num = Theta_a + Theta_b
    dnum_dxh = dTa_dxh + dTb_dxh
    dnum_dyh = dTa_dyh + dTb_dyh
    dD_dxh = dTa_dxh * Theta_b + Theta_a * dTb_dxh - 2 * Theta_c * dTc_dxh
    dD_dyh = dTa_dyh * Theta_b + Theta_a * dTb_dyh - 2 * Theta_c * dTc_dyh
    df_dxh = dnum_dxh / D - num * dD_dxh / D**2
    df_dyh = dnum_dyh / D - num * dD_dyh / D**2
    return np.array([df_dxh, df_dyh])


def analytical_gradient_f(stage_waypoints: np.ndarray,
                          stage_bandwidths: np.ndarray,
                          prev_waypoints: np.ndarray,
                          prev_hover_points: np.ndarray,
                          cus: np.ndarray,
                          sts_estimate: np.ndarray,
                          Psi_c_prev: float,
                          Psi_s_prev: float,
                          eta: float,
                          mu: int,
                          cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """
    Gradient giải tích của f(S_j) theo S_j.

    f =  (1-η)/Ψc_prev * Ψc(j)  -  η/Ψs_prev * Ψs(j)  + const.
    Ψc  = softmin_m ψ^c_m  -> softmin weights w_c (sum=1).
    Ψs  = softmax_k ψ^s_k  -> softmax weights w_s (sum=1).
    -> ∂Ψc/∂s  =  Σ_m w_c[m] * ∂ψ^c_m/∂s
       ∂Ψs/∂s  =  Σ_k w_s[k] * ∂ψ^s_k/∂s
    """
    Nf = stage_waypoints.shape[0]
    grad_S = np.zeros_like(stage_waypoints)

    stage_hover_points = _extract_hps(stage_waypoints, mu)
    Nh_stage = stage_hover_points.shape[0]
    num_prev_hps = prev_hover_points.shape[0]

    all_waypoints = np.vstack([prev_waypoints, stage_waypoints]) \
        if len(prev_waypoints) > 0 else stage_waypoints
    all_hover_points = np.vstack([prev_hover_points, stage_hover_points]) \
        if len(prev_hover_points) > 0 else stage_hover_points

    M = len(cus)
    K = len(sts_estimate)

    # --- LSE softmax/softmin weights (paper eq. 37, 38) ---
    # Normalize ψ bằng Psi_prev (constant trong iteration) trước khi tính weights.
    # Điều này đảm bảo:
    #   (a) t có ý nghĩa thống nhất không phụ thuộc scale của ψ
    #   (b) gradient analytical đúng (scale là constant)
    # ∂Ψ^c/∂ψ^c_m = exp(-t·ψ^c_m/scale) / Σ exp(-t·ψ^c_m'/scale) = softmin weight
    # ∂Ψ^s/∂ψ^s_k = exp(t·ψ^s_k/scale) / Σ exp(t·ψ^s_k'/scale)   = softmax weight
    psi_c = np.array([
        compute_psi_c_user(all_waypoints, all_hover_points,
                           cus[m], stage_bandwidths[m], cfg)
        for m in range(M)
    ])
    psi_s = np.array([crb_sum(all_hover_points, sts_estimate[k], cfg)
                      for k in range(K)])

    t = cfg.t_lse
    scale_c = max(abs(Psi_c_prev), 1e-12)
    scale_s = max(abs(Psi_s_prev), 1e-12)

    if M > 1:
        # softmin với normalize scale
        psi_c_norm = psi_c / scale_c
        c = np.min(psi_c_norm)
        shifted = -t * (psi_c_norm - c)
        w_c = np.exp(shifted); w_c /= w_c.sum()
    else:
        w_c = np.ones(1)

    if K > 1:
        psi_s_norm = psi_s / scale_s
        c = np.max(psi_s_norm)
        shifted = t * (psi_s_norm - c)
        w_s = np.exp(shifted); w_s /= w_s.sum()
    else:
        w_s = np.ones(1)

    coef_c = (1 - eta) / max(Psi_c_prev, 1e-12)
    coef_s = eta / max(Psi_s_prev, 1e-12)

    hp_stage_idx_in_stage = np.arange(1, Nh_stage + 1) * mu - 1 if Nh_stage > 0 else np.array([], dtype=int)

    # Nhóm các waypoint là HP để dễ lookup
    hp_set = set(hp_stage_idx_in_stage.tolist())

    for n in range(Nf):
        s_n = stage_waypoints[n]
        # (a) gradient qua comm khi bay (T_f * R)
        g_c = np.zeros(2)
        for m in range(M):
            g_c += w_c[m] * cfg.Tf * _grad_Rm_wrt_s(
                s_n, cus[m], stage_bandwidths[m], cfg)

        # (b) nếu n là HP: thêm T_h * R (comm) + CRB (sensing)
        g_s = np.zeros(2)
        if n in hp_set:
            for m in range(M):
                g_c += w_c[m] * cfg.Th * _grad_Rm_wrt_s(
                    s_n, cus[m], stage_bandwidths[m], cfg)
            for k in range(K):
                g_s += w_s[k] * _grad_crb_wrt_hp(
                    s_n, all_hover_points, sts_estimate[k], cfg)

        grad_S[n] = coef_c * g_c - coef_s * g_s

    return grad_S


# ---------------------------------------------------------------------------
# GIẢI Q''(j) - bài toán lồi: max ∇f^T (S - S_{l-1})
# với ràng buộc năng lượng xấp xỉ lồi (SCA)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GIẢI Q''(j) - bài toán lồi: max ∇f^T (S - S_{l-1})
# với ràng buộc năng lượng xấp xỉ lồi (SCA)
# ---------------------------------------------------------------------------
class _QppCache:
    """
    Cache problem Q''(j) đã được build sẵn với CVXPY Parameters.
    Tái sử dụng giữa các iteration của cùng 1 stage và giữa các stage
    có cùng Nf để tránh compile CVXPY lặp đi lặp lại.
    """
    _instances: dict = {}

    def __init__(self, Nf: int, cfg: SimulationConfig):
        self.Nf = Nf
        self.cfg = cfg

        # Biến
        self.S = cp.Variable((Nf, 2))
        self.V = cp.Variable((Nf, 2))
        self.delta = cp.Variable(Nf, nonneg=True)
        self.xi = cp.Variable(Nf, nonneg=True)

        # Parameters (cập nhật trước mỗi lần solve)
        self.P_S_prev = cp.Parameter((Nf, 2))
        self.P_V_prev = cp.Parameter((Nf, 2))
        self.P_delta_prev = cp.Parameter(Nf, nonneg=True)
        self.P_grad = cp.Parameter((Nf, 2))
        self.P_start = cp.Parameter(2)
        self.P_E_rem = cp.Parameter(nonneg=True)
        self.P_Nh_times_hoverpower = cp.Parameter(nonneg=True)
        # Thêm params tiền tính để giữ DPP
        self.P_V_prev_norm2 = cp.Parameter(Nf, nonneg=True)       # ||v_prev_n||²
        self.P_delta_prev_sq = cp.Parameter(Nf, nonneg=True)      # δ_prev_n²

        constraints = []

        # (1) Khu vực bay
        constraints += [self.S[:, 0] >= 0, self.S[:, 0] <= cfg.Lx]
        constraints += [self.S[:, 1] >= 0, self.S[:, 1] <= cfg.Ly]

        # (2) Vận tốc (eq. 2)
        constraints.append(self.V[0] == (self.S[0] - self.P_start) / cfg.Tf)
        constraints.append(self.V[1:] == (self.S[1:] - self.S[:-1]) / cfg.Tf)

        # (3) Vận tốc tối đa
        for n in range(Nf):
            constraints.append(cp.norm(self.V[n], 2) <= cfg.Vmax)

        # (4) Năng lượng
        energy_terms = []
        for n in range(Nf):
            v_norm = cp.norm(self.V[n], 2)
            term = cfg.P0 * (1 + 3 * cp.square(v_norm) / (cfg.Utip ** 2)) \
                   + 0.5 * cfg.D0 * cfg.rho * cfg.s_rotor * cfg.A * cp.power(v_norm, 3)
            energy_terms.append(term)
        propel_energy = cfg.Tf * cp.sum(energy_terms)
        induced_energy = cfg.Tf * cfg.PI * cp.sum(self.delta)
        hover_energy = self.P_Nh_times_hoverpower
        constraints.append(
            propel_energy + induced_energy + hover_energy <= self.P_E_rem
        )

        # (5) SCA (46):  δ^{-2} <= lhs_linear + ξ
        # lhs_linear = -||v_prev||²/v₀² + 2/v₀² * <v_prev, V>
        # Dùng sum element-wise để DPP-friendly.
        for n in range(Nf):
            dot_vprev_V = self.P_V_prev[n, 0] * self.V[n, 0] \
                        + self.P_V_prev[n, 1] * self.V[n, 1]
            lhs_linear = -self.P_V_prev_norm2[n] / cfg.v0**2 \
                + (2.0 / cfg.v0**2) * dot_vprev_V
            constraints.append(cp.power(self.delta[n], -2) <= lhs_linear + self.xi[n])

        # (6) SCA (47):  δ² >= ξ  ->  -δ_prev² + 2 δ_prev δ >= ξ
        for n in range(Nf):
            lhs_linear = -self.P_delta_prev_sq[n] + 2 * self.P_delta_prev[n] * self.delta[n]
            constraints.append(lhs_linear >= self.xi[n])

        # (7) Objective
        obj = cp.sum(cp.multiply(self.P_grad, self.S - self.P_S_prev))
        self.problem = cp.Problem(cp.Maximize(obj), constraints)

    @classmethod
    def get(cls, Nf: int, cfg: SimulationConfig) -> "_QppCache":
        # Key bao gồm Nf và id của cfg (cùng 1 cfg object dùng chung)
        key = (Nf, id(cfg))
        if key not in cls._instances:
            cls._instances[key] = cls(Nf, cfg)
        return cls._instances[key]

    def solve(self,
              S_prev: np.ndarray, V_prev: np.ndarray, delta_prev: np.ndarray,
              grad: np.ndarray, start: np.ndarray, E_rem: float, Nh: int,
              solver: str = "CLARABEL", verbose: bool = False):
        if E_rem <= 0:
            if verbose:
                print(f"[Qpp] E_rem={E_rem} <= 0, không thể solve.")
            return None
        delta_prev_safe = np.maximum(delta_prev, 1e-6)
        self.P_S_prev.value = S_prev
        self.P_V_prev.value = V_prev
        self.P_delta_prev.value = delta_prev_safe
        self.P_grad.value = grad
        self.P_start.value = start
        self.P_E_rem.value = max(E_rem, 1e-6)
        self.P_Nh_times_hoverpower.value = self.cfg.Th * Nh * (self.cfg.P0 + self.cfg.PI)
        self.P_V_prev_norm2.value = np.sum(V_prev**2, axis=1)
        self.P_delta_prev_sq.value = delta_prev_safe**2

        try:
            self.problem.solve(solver=solver, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[Qpp] {solver} failed: {e}, thử SCS")
            try:
                self.problem.solve(solver=cp.SCS, verbose=verbose)
            except Exception as e2:
                if verbose:
                    print(f"[Qpp] SCS cũng fail: {e2}")
                return None

        if (self.S.value is None
            or self.problem.status not in ("optimal", "optimal_inaccurate")):
            if verbose:
                print(f"[Qpp] Status: {self.problem.status}")
            return None
        return self.S.value.copy(), self.V.value.copy(), self.delta.value.copy()


def solve_Qpp(stage_waypoints_prev: np.ndarray,
              stage_velocities_prev: np.ndarray,
              stage_delta_prev: np.ndarray,
              grad: np.ndarray,
              remaining_energy: float,
              start_point: np.ndarray,
              Nh_stage: int,
              mu: int,
              cfg: SimulationConfig = DEFAULT,
              verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Wrapper tương thích cũ - dùng cached parameterized problem.
    """
    Nf = stage_waypoints_prev.shape[0]
    cache = _QppCache.get(Nf, cfg)
    return cache.solve(stage_waypoints_prev, stage_velocities_prev,
                       stage_delta_prev, grad, start_point,
                       remaining_energy, Nh_stage, verbose=verbose)


# ---------------------------------------------------------------------------
# Tính vận tốc và delta từ waypoint
# ---------------------------------------------------------------------------
def waypoints_to_velocities(stage_waypoints: np.ndarray,
                            start_point: np.ndarray,
                            cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """Tính V_j từ S_j theo eq. 2."""
    Nf = stage_waypoints.shape[0]
    V = np.zeros_like(stage_waypoints)
    V[0] = (stage_waypoints[0] - start_point) / cfg.Tf
    V[1:] = (stage_waypoints[1:] - stage_waypoints[:-1]) / cfg.Tf
    return V


def velocities_to_delta(velocities: np.ndarray,
                        cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """Tính δ từ V theo eq. 41:
        δ^2 = sqrt(1 + ||v||^4/(4 v0^4)) - ||v||^2/(2 v0^2)
    """
    v_norm = np.linalg.norm(velocities, axis=1)
    inner = np.sqrt(1 + v_norm**4 / (4 * cfg.v0**4)) - v_norm**2 / (2 * cfg.v0**2)
    inner = np.maximum(inner, 1e-10)
    return np.sqrt(inner)


def compute_stage_energy(stage_waypoints: np.ndarray,
                         start_point: np.ndarray,
                         Nh_stage: int,
                         cfg: SimulationConfig = DEFAULT) -> float:
    """Năng lượng tiêu thụ trong stage hiện tại (đúng công thức eq. 33 + hover)."""
    V = waypoints_to_velocities(stage_waypoints, start_point, cfg)
    v_norms = np.linalg.norm(V, axis=1)
    P_fly = uav_propulsion_power(v_norms, cfg)
    return cfg.Tf * np.sum(P_fly) + cfg.Th * Nh_stage * uav_propulsion_power(0.0, cfg)


# ---------------------------------------------------------------------------
# ALGORITHM 1: Iterative ascent direction search for P'_1(j)
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryOptResult:
    waypoints: np.ndarray          # S_j tối ưu
    velocities: np.ndarray          # V_j tối ưu
    hover_points: np.ndarray        # HP trong stage này
    obj_history: list               # lịch sử hàm mục tiêu f
    iter_count: int                 # số iteration


def init_hypothetical_trajectory(start_point: np.ndarray,
                                 target_point: np.ndarray,
                                 Nf: int,
                                 cfg: SimulationConfig = DEFAULT,
                                 cruise_speed: float | None = None) -> np.ndarray:
    """
    Tạo quỹ đạo giả định ban đầu: bay thẳng từ start tới target với vận tốc
    `cruise_speed` (mặc định = Vmax). Nếu cruise_speed = 0 -> đứng yên tại start
    (hữu ích cho ending stage khi năng lượng ít).
    """
    if cruise_speed is None:
        cruise_speed = cfg.Vmax

    if cruise_speed < 1e-6:
        return np.tile(start_point, (Nf, 1))

    direction = target_point - start_point
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.tile(start_point, (Nf, 1))

    step_len = cruise_speed * cfg.Tf
    unit = direction / dist

    waypoints = np.zeros((Nf, 2))
    current = start_point.copy()
    for n in range(Nf):
        current = current + step_len * unit
        current[0] = np.clip(current[0], 0, cfg.Lx)
        current[1] = np.clip(current[1], 0, cfg.Ly)
        waypoints[n] = current
    return waypoints


def init_feasible_trajectory(start_point: np.ndarray,
                             Nf: int,
                             remaining_energy: float,
                             cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """
    Tạo quỹ đạo ban đầu CHẮC CHẮN FEASIBLE dựa trên năng lượng hiện có.
    Thử giảm dần vận tốc cho đến khi Puav * Tf * Nf + Th * Nh * Phover <= E.
    """
    Nh = Nf // cfg.mu
    P_hover = uav_propulsion_power(0.0, cfg)
    hover_energy = cfg.Th * Nh * P_hover
    E_for_fly = remaining_energy - hover_energy

    if E_for_fly <= 0:
        # Chỉ đủ hover -> đứng yên
        return np.tile(start_point, (Nf, 1))

    # Thử các tốc độ giảm dần, tìm tốc độ khả thi
    target = cfg.ground_area_corner
    for V_try in [cfg.Vmax, 20.0, 15.0, 10.0, 5.0, 2.0, 0.5, 0.0]:
        P_fly = uav_propulsion_power(V_try, cfg)
        if cfg.Tf * Nf * P_fly <= E_for_fly * 0.95:  # 5% safety margin
            return init_hypothetical_trajectory(start_point, target, Nf,
                                                 cfg, cruise_speed=V_try)
    # Fallback: đứng yên
    return np.tile(start_point, (Nf, 1))


def optimize_trajectory_stage(start_point: np.ndarray,
                              Nf: int,
                              remaining_energy: float,
                              prev_waypoints: np.ndarray,
                              prev_hover_points: np.ndarray,
                              cus: np.ndarray,
                              sts_estimate: np.ndarray,
                              Psi_c_prev: float,
                              Psi_s_prev: float,
                              bandwidths: np.ndarray,
                              eta: float,
                              cfg: SimulationConfig = DEFAULT,
                              max_iter: int = 15,
                              tol: float = 1e-4,
                              verbose: bool = False,
                              initial_waypoints: np.ndarray | None = None
                              ) -> TrajectoryOptResult:
    """
    Giải P'_1(j) bằng ascent direction search (Algorithm 1).
    """
    Nh = Nf // cfg.mu

    # --- Khởi tạo ---
    if initial_waypoints is None:
        S = init_feasible_trajectory(start_point, Nf, remaining_energy, cfg)
    else:
        S = initial_waypoints.copy()

    V = waypoints_to_velocities(S, start_point, cfg)
    delta = velocities_to_delta(V, cfg)

    obj_history = [
        objective_f(S, bandwidths, prev_waypoints, prev_hover_points,
                    cus, sts_estimate, Psi_c_prev, Psi_s_prev, eta, cfg.mu, cfg)
    ]

    for it in range(1, max_iter + 1):
        # --- (a) Tính gradient giải tích tại S ---
        grad = analytical_gradient_f(
            S, bandwidths, prev_waypoints, prev_hover_points,
            cus, sts_estimate, Psi_c_prev, Psi_s_prev, eta, cfg.mu, cfg
        )

        # --- (b) Giải Q''(j) để tìm S* (hướng ascent) ---
        result = solve_Qpp(S, V, delta, grad, remaining_energy, start_point,
                           Nh, cfg.mu, cfg, verbose=False)
        if result is None:
            if verbose:
                print(f"  iter {it}: Q'' không giải được, dừng.")
            break
        S_star, _, _ = result

        # --- (c) Line search dọc theo S* - S ---
        direction = S_star - S
        best_obj = obj_history[-1]
        best_S = S
        best_omega = 0.0

        for step in range(1, cfg.delta_omega_steps + 1):
            omega = step / cfg.delta_omega_steps
            S_trial = S + omega * direction
            # Kiểm tra ràng buộc không gian
            S_trial[:, 0] = np.clip(S_trial[:, 0], 0, cfg.Lx)
            S_trial[:, 1] = np.clip(S_trial[:, 1], 0, cfg.Ly)
            # Kiểm tra ràng buộc năng lượng
            stage_energy = compute_stage_energy(S_trial, start_point, Nh, cfg)
            if stage_energy > remaining_energy * 1.001:
                continue

            obj_trial = objective_f(S_trial, bandwidths, prev_waypoints,
                                    prev_hover_points, cus, sts_estimate,
                                    Psi_c_prev, Psi_s_prev, eta, cfg.mu, cfg)
            if obj_trial > best_obj + 1e-9:
                best_obj = obj_trial
                best_S = S_trial
                best_omega = omega

        improvement = best_obj - obj_history[-1]
        if verbose:
            print(f"  iter {it}: obj={best_obj:.6f} (Δ={improvement:+.2e}), "
                  f"ω*={best_omega:.3f}")

        obj_history.append(best_obj)

        if improvement < tol:
            if verbose:
                print(f"  Hội tụ tại iter {it}.")
            break

        S = best_S
        V = waypoints_to_velocities(S, start_point, cfg)
        delta = velocities_to_delta(V, cfg)

    hover_points = _extract_hps(S, cfg.mu)
    return TrajectoryOptResult(
        waypoints=S,
        velocities=V,
        hover_points=hover_points,
        obj_history=obj_history,
        iter_count=len(obj_history) - 1,
    )


# ---------------------------------------------------------------------------
# P'_2(j) - BANDWIDTH ALLOCATION (lồi)
# ---------------------------------------------------------------------------
def optimize_bandwidth(all_waypoints: np.ndarray,
                       all_hover_points: np.ndarray,
                       cus: np.ndarray,
                       cfg: SimulationConfig = DEFAULT) -> np.ndarray:
    """
    Giải P'_2(j): tối đa hoá min_m ψ^c_m(j) theo B_m sao cho Σ B_m ≤ B.

    ψ^c_m tỉ lệ tuyến tính với B_m (khi waypoint/HP cố định) - bài toán LP-ish
    nhưng đi qua log2 -> vẫn lồi (log là concave).

    Dạng chuẩn: max t  s.t. t ≤ ψ^c_m(B_m), Σ B_m ≤ B, B_m ≥ 0.
    """
    M = len(cus)

    # Với mỗi CU, tính hằng số C_m = ψ^c_m / B_m
    # (vì B_m chỉ xuất hiện ở thừa số băng thông trong rate)
    C = np.zeros(M)
    for m in range(M):
        total_rate = 0.0
        for s in all_waypoints:
            d2 = cfg.H**2 + np.sum((s - cus[m])**2)
            h = cfg.alpha0 / d2
            snr = cfg.Pt * h / cfg.noise_power
            # Rate chuẩn hoá theo B = 1
            total_rate += cfg.Tf * np.log2(1 + snr)
        for hp in all_hover_points:
            d2 = cfg.H**2 + np.sum((hp - cus[m])**2)
            h = cfg.alpha0 / d2
            snr = cfg.Pt * h / cfg.noise_power
            total_rate += cfg.Th * np.log2(1 + snr)
        C[m] = total_rate

    # Bài toán: max t, s.t. t ≤ C_m B_m, Σ B_m ≤ B, B_m ≥ 0
    # Giải tường minh: t* = C_m B_m bằng nhau -> B_m = t/C_m, Σ B_m = B
    # -> t (Σ 1/C_m) = B -> t = B / Σ (1/C_m)
    # -> B_m = (1/C_m) / Σ (1/C_m) * B
    if np.any(C <= 0):
        return np.ones(M) * cfg.B / M
    weights = 1.0 / C
    B_alloc = weights / np.sum(weights) * cfg.B
    return B_alloc


if __name__ == "__main__":
    cfg = DEFAULT
    rng = np.random.default_rng(0)

    # Test: kịch bản nhỏ - 1 stage với 60 waypoints, 2 CU + 2 ST
    cus = np.array([[1200., 1200.], [1300., 800.]])
    sts_true = np.array([[1000., 1050.], [300., 1050.]])
    # Giả sử initial estimate hơi lệch
    sts_est = sts_true + rng.normal(0, 50, size=sts_true.shape)

    start = cfg.base_station
    Nf = 60
    E_stage = 40e3

    # Psi_c_prev, Psi_s_prev từ initial estimate (chỉ có coarse sensing trước đó)
    # Để tránh chia 0, dùng giá trị hypothetical trajectory làm reference
    S_hyp = init_hypothetical_trajectory(start, cfg.ground_area_corner, Nf, cfg)
    HP_hyp = _extract_hps(S_hyp, cfg.mu)

    B_alloc = np.full(len(cus), cfg.B / len(cus))
    Psi_c_prev = compute_Psi_c(S_hyp, HP_hyp, cus, B_alloc, cfg)
    Psi_s_prev = compute_Psi_s(HP_hyp, sts_est, cfg)

    print(f"Reference (hypothetical): Ψc = {Psi_c_prev/1e9:.3f} Gbits, "
          f"Ψs = {Psi_s_prev:.4e} m²")

    result = optimize_trajectory_stage(
        start_point=start, Nf=Nf, remaining_energy=E_stage,
        prev_waypoints=np.zeros((0, 2)),
        prev_hover_points=np.zeros((0, 2)),
        cus=cus, sts_estimate=sts_est,
        Psi_c_prev=Psi_c_prev, Psi_s_prev=Psi_s_prev,
        bandwidths=B_alloc, eta=0.5,
        cfg=cfg, max_iter=6, verbose=True,
    )
    print(f"\nSố iteration: {result.iter_count}")
    print(f"Obj history: {[f'{o:.4f}' for o in result.obj_history]}")

    # Đánh giá C&S cuối cùng
    S_final = result.waypoints
    HP_final = result.hover_points
    Psi_c_final = compute_Psi_c(S_final, HP_final, cus, B_alloc, cfg)
    Psi_s_final = compute_Psi_s(HP_final, sts_est, cfg)
    print(f"\nSau tối ưu: Ψc = {Psi_c_final/1e9:.3f} Gbits, Ψs = {Psi_s_final:.4e} m²")
    print(f"Năng lượng tiêu thụ: "
          f"{compute_stage_energy(S_final, start, HP_final.shape[0], cfg)/1e3:.2f} kJ")
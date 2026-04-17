"""
Mô hình hệ thống cho kịch bản ISAC-UAV.

Chứa các hàm tính:
- Công suất bay UAV (eq. 33)
- Thông lượng truyền thông (eq. 7)
- Ma trận FIM và CRB cho sensing (eq. 17-31)
"""

from __future__ import annotations
import numpy as np
from config import SimulationConfig, DEFAULT


# ---------------------------------------------------------------------------
# CÔNG SUẤT BAY UAV (Eq. 33 trong bài báo)
# ---------------------------------------------------------------------------
def uav_propulsion_power(V: np.ndarray | float, cfg: SimulationConfig = DEFAULT) -> np.ndarray | float:
    """
    Công suất động cơ UAV theo vận tốc V (m/s).

    P_uav(V) = P0*(1 + 3V^2/Utip^2)
              + PI*( sqrt(1 + V^4/(4 v0^4)) - V^2/(2 v0^2) )^(1/2)
              + 0.5 * D0 * rho * s * A * V^3
    """
    V = np.asarray(V, dtype=float)
    term1 = cfg.P0 * (1 + 3 * V**2 / cfg.Utip**2)
    inner = np.sqrt(1 + V**4 / (4 * cfg.v0**4)) - V**2 / (2 * cfg.v0**2)
    # Đảm bảo không âm (lỗi số học khi V rất lớn)
    inner = np.maximum(inner, 1e-12)
    term2 = cfg.PI * np.sqrt(inner)
    term3 = 0.5 * cfg.D0 * cfg.rho * cfg.s_rotor * cfg.A * V**3
    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# MÔ HÌNH TRUYỀN THÔNG (Eq. 4-7)
# ---------------------------------------------------------------------------
def comm_rate(way_xy: np.ndarray,
              cu_xy: np.ndarray,
              bandwidth: float,
              cfg: SimulationConfig = DEFAULT) -> float:
    """
    Tốc độ truyền tin (bit/s) từ UAV tại waypoint `way_xy` tới user tại `cu_xy`
    với băng thông `bandwidth` (Hz).

    R = B * log2(1 + Pt * h / σ^2), trong đó h = α0 / d^2, d^2 = H^2 + ||s - u||^2
    """
    d2 = cfg.H**2 + np.sum((way_xy - cu_xy) ** 2)
    h = cfg.alpha0 / d2
    snr = cfg.Pt * h / cfg.noise_power
    return bandwidth * np.log2(1 + snr)


def total_transmitted_data_user(waypoints: np.ndarray,
                                hover_points: np.ndarray,
                                cu_xy: np.ndarray,
                                bandwidth: float,
                                cfg: SimulationConfig = DEFAULT) -> float:
    """
    Tổng dữ liệu truyền được (bit) cho 1 user qua quỹ đạo (eq. 8).
    waypoints: (Nf, 2) - vị trí các waypoints tích luỹ qua các stage
    hover_points: (Nh, 2) - các hover point
    """
    data = 0.0
    for way in waypoints:
        data += cfg.Tf * comm_rate(way, cu_xy, bandwidth, cfg)
    for hp in hover_points:
        data += cfg.Th * comm_rate(hp, cu_xy, bandwidth, cfg)
    return data


# ---------------------------------------------------------------------------
# MÔ HÌNH SENSING - CRB (Eq. 10-31)
# ---------------------------------------------------------------------------
def distance_hp_to_st(hp_xy: np.ndarray,
                      st_xy: np.ndarray,
                      H: float) -> float:
    """Khoảng cách 3D từ HP (trên không, độ cao H) đến ST (mặt đất) - eq. 10."""
    return float(np.sqrt(H**2 + np.sum((hp_xy - st_xy) ** 2)))


def sensing_snr(distance: float, cfg: SimulationConfig = DEFAULT) -> float:
    """SNR echo (eq. 13-14): SNR = Pt * Gp * β0 / (d^4 * σ^2)."""
    return cfg.Pt * cfg.Gp * cfg.beta0 / (distance**4 * cfg.noise_power)


def distance_noise_variance(distance: float, cfg: SimulationConfig = DEFAULT) -> float:
    """
    Variance của đo khoảng cách (eq. 16):
    (σ_τ)^2 = a * σ0^2 / (Pt * Gp * β0 / d^4)
           = a * σ0^2 * d^4 / (Pt * Gp * β0)
    """
    return cfg.a * cfg.noise_power * distance**4 / (cfg.Pt * cfg.Gp * cfg.beta0)


def compute_theta(hover_points: np.ndarray,
                  st_xy: np.ndarray,
                  cfg: SimulationConfig = DEFAULT) -> tuple[float, float, float]:
    """
    Tính Θ_a, Θ_b, Θ_c (eq. 26-28). Các thành phần này được sum qua TẤT CẢ các HP
    tích luỹ (từ stage 1 đến stage j).

    hover_points: (Nh_total, 2) gồm toàn bộ HP từ stage 1 đến hiện tại.
    Trả về Θ_a, Θ_b, Θ_c (scalar) cho 1 ST.
    """
    xk, yk = st_xy[0], st_xy[1]
    xh = hover_points[:, 0]
    yh = hover_points[:, 1]
    dx = xh - xk
    dy = yh - yk
    d2 = cfg.H**2 + dx**2 + dy**2           # d^2 (eq. 10)
    d_s = np.sqrt(d2)                        # d

    coeff = cfg.Pt * cfg.Gp * cfg.beta0 / (cfg.a * cfg.noise_power)

    # Θ_a (eq. 26)
    theta_a = np.sum(coeff * dx**2 / d_s**6 + 8 * dx**2 / d_s**4)
    # Θ_b (eq. 27)
    theta_b = np.sum(coeff * dy**2 / d_s**6 + 8 * dy**2 / d_s**4)
    # Θ_c (eq. 28)
    theta_c = np.sum(coeff * dx * dy / d_s**6 + 8 * dx * dy / d_s**4)

    return float(theta_a), float(theta_b), float(theta_c)


def crb_sum(hover_points: np.ndarray,
            st_xy: np.ndarray,
            cfg: SimulationConfig = DEFAULT) -> float:
    """
    Tính ψ_k^s (eq. 31): tổng CRB(x) + CRB(y) cho ST thứ k.
    Đây chính là chỉ số sensing ta muốn MINIMIZE.
    """
    theta_a, theta_b, theta_c = compute_theta(hover_points, st_xy, cfg)
    denom = theta_a * theta_b - theta_c**2
    if denom <= 0:
        return np.inf
    return (theta_a + theta_b) / denom


def crb_x_y(hover_points: np.ndarray,
            st_xy: np.ndarray,
            cfg: SimulationConfig = DEFAULT) -> tuple[float, float]:
    """CRB của x và y riêng rẽ (eq. 29-30)."""
    theta_a, theta_b, theta_c = compute_theta(hover_points, st_xy, cfg)
    denom = theta_a * theta_b - theta_c**2
    if denom <= 0:
        return np.inf, np.inf
    return theta_b / denom, theta_a / denom


# ---------------------------------------------------------------------------
# Các hàm bound cho C&S (LSE smoothing - eq. 37-38)
# ---------------------------------------------------------------------------
def log_sum_exp_min(vals: np.ndarray, t: float = 1.0) -> float:
    """
    Xấp xỉ min(vals) (smooth từ dưới lên) dạng ỔN ĐỊNH SỐ:
        ≈  (1/-t) log( sum exp(-t * v_i) )
    Dùng trick log-sum-exp: shift bằng min(-t*v_i) = -t*max(v_i):
        result = c - (1/t) log sum exp(-t*(v_i - max v_i))
             với c = max(v_i)
    Kết quả lúc t lớn -> min(vals), tránh overflow.
    """
    vals = np.asarray(vals, dtype=float)
    c = np.min(vals)                              # shift reference
    shifted = -t * (vals - c)                     # <= 0
    return c - (1.0 / t) * np.log(np.sum(np.exp(shifted)))


def log_sum_exp_max(vals: np.ndarray, t: float = 1.0) -> float:
    """Xấp xỉ max(vals) (smooth) ỔN ĐỊNH SỐ:
        result = c + (1/t) log sum exp(t*(v_i - c)),  c = max(v_i).
    """
    vals = np.asarray(vals, dtype=float)
    c = np.max(vals)
    shifted = t * (vals - c)                      # <= 0
    return c + (1.0 / t) * np.log(np.sum(np.exp(shifted)))


if __name__ == "__main__":
    cfg = DEFAULT
    # Kiểm tra công suất UAV
    for V in [0, 5, 10, 15, 20, 25, 30]:
        P = uav_propulsion_power(V)
        print(f"V = {V:3d} m/s -> P_uav = {P:7.2f} W")

    # Kiểm tra tốc độ truyền thông
    way = np.array([750., 750.])
    cu = np.array([1000., 1000.])
    R = comm_rate(way, cu, cfg.B)
    print(f"\nComm rate tại d_ground={np.linalg.norm(way-cu):.1f}m: {R/1e6:.2f} Mbps")

    # Kiểm tra CRB với vài HP xung quanh 1 ST
    st = np.array([1000., 1000.])
    hps = np.array([[900., 900.], [1100., 900.], [1000., 1100.], [1000., 900.]])
    crb = crb_sum(hps, st)
    crb_x, crb_y = crb_x_y(hps, st)
    print(f"\nCRB tổng (x+y) = {crb:.4e} m^2")
    print(f"CRB_x = {crb_x:.4e}, CRB_y = {crb_y:.4e}")

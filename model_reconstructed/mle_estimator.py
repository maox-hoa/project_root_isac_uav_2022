"""
Ước lượng vị trí Sensing Target (ST) qua Maximum Likelihood Estimation.

Đo khoảng cách có dạng:  d_tilde = d_true + w_τ,  w_τ ~ N(0, σ_τ^2).
σ_τ^2 phụ thuộc vào d^4, nên phải dùng log-likelihood đầy đủ (eq. 52).

Hàm log-likelihood:
    log p(d_tilde | x_k, y_k)
    = Σ [ -0.5 * log(2π σ_τ^2)  -  (d_tilde - d_true(x_k,y_k))^2 / (2 σ_τ^2) ]

Dùng hybrid: coarse grid -> refine cục bộ.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from config import SimulationConfig, DEFAULT
from system_model import distance_hp_to_st, distance_noise_variance


def simulate_distance_measurements(hover_points: np.ndarray,
                                   st_xy: np.ndarray,
                                   cfg: SimulationConfig = DEFAULT,
                                   rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sinh đo khoảng cách có nhiễu Gaussian theo eq. 12.
    Trả về vector d_tilde có kích thước Nh (số HP).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = hover_points.shape[0]
    d_tilde = np.zeros(N)
    for i, hp in enumerate(hover_points):
        d_true = distance_hp_to_st(hp, st_xy, cfg.H)
        sigma2 = distance_noise_variance(d_true, cfg)
        sigma = np.sqrt(sigma2)
        d_tilde[i] = d_true + rng.normal(0, sigma)
    return d_tilde


def neg_log_likelihood(xy: np.ndarray,
                       hover_points: np.ndarray,
                       d_tilde: np.ndarray,
                       cfg: SimulationConfig = DEFAULT) -> float:
    """
    Hàm -log p trong eq. 52 (bỏ hằng số không phụ thuộc xy).
    Lưu ý: ta dùng σ_τ^2 đánh giá tại vị trí hiện đang ước lượng (x_k,y_k).
    """
    xk, yk = xy
    dx = hover_points[:, 0] - xk
    dy = hover_points[:, 1] - yk
    d_hat = np.sqrt(cfg.H**2 + dx**2 + dy**2)

    sigma2 = cfg.a * cfg.noise_power * d_hat**4 / (cfg.Pt * cfg.Gp * cfg.beta0)
    residual2 = (d_tilde - d_hat) ** 2

    # -log p  (giữ các phần có phụ thuộc xy)
    nll = 0.5 * np.sum(np.log(2 * np.pi * sigma2) + residual2 / sigma2)
    return float(nll)


def mle_estimate_st(hover_points: np.ndarray,
                    d_tilde: np.ndarray,
                    cfg: SimulationConfig = DEFAULT,
                    grid_step: float = 50.0,
                    refine: bool = True) -> np.ndarray:
    """
    Ước lượng vị trí ST bằng grid search + refine bằng Nelder-Mead.

    Grid phủ từ 0 đến [Lx, Ly] với bước grid_step.
    """
    # --- Grid search thô ---
    xs = np.arange(0, cfg.Lx + 1, grid_step)
    ys = np.arange(0, cfg.Ly + 1, grid_step)
    best_nll = np.inf
    best_xy = np.array([cfg.Lx / 2, cfg.Ly / 2])

    # Vectorize grid search
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    flat_xy = np.stack([X.ravel(), Y.ravel()], axis=1)   # (Ngrid, 2)

    # Vector hóa tính NLL cho mọi điểm grid
    # d_hat shape: (Ngrid, Nh)
    dx = flat_xy[:, 0:1] - hover_points[:, 0][None, :]
    dy = flat_xy[:, 1:2] - hover_points[:, 1][None, :]
    d_hat = np.sqrt(cfg.H**2 + dx**2 + dy**2)
    sigma2 = cfg.a * cfg.noise_power * d_hat**4 / (cfg.Pt * cfg.Gp * cfg.beta0)
    res2 = (d_tilde[None, :] - d_hat) ** 2
    nll_all = 0.5 * np.sum(np.log(2 * np.pi * sigma2) + res2 / sigma2, axis=1)

    idx = int(np.argmin(nll_all))
    best_xy = flat_xy[idx].copy()
    best_nll = float(nll_all[idx])

    # --- Refine bằng Nelder-Mead ---
    if refine:
        try:
            res = minimize(
                neg_log_likelihood,
                best_xy,
                args=(hover_points, d_tilde, cfg),
                method='Nelder-Mead',
                options={'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 500}
            )
            if res.fun < best_nll:
                best_xy = np.clip(res.x, 0, [cfg.Lx, cfg.Ly])
        except Exception:
            pass

    return best_xy


def coarse_initial_estimate(st_true: np.ndarray,
                            cfg: SimulationConfig = DEFAULT,
                            rng: np.random.Generator | None = None,
                            num_points: int = 3,
                            radius: float = 50.0) -> np.ndarray:
    """
    Ước lượng 'coarse' ban đầu (trước khi UAV xuất phát), bằng cách sensing
    từ 3 điểm gần trạm sạc. Theo bài báo: "we acquire a coarse estimate of u_k
    via three sensing points near [x_B, y_B]^T".
    """
    if rng is None:
        rng = np.random.default_rng()

    # Đặt 3 điểm sensing gần trạm sạc (tạo thành tam giác)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    sensing_pts = cfg.base_station + radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )

    # Sinh đo khoảng cách có nhiễu từ các điểm này
    d_tilde = simulate_distance_measurements(sensing_pts, st_true, cfg, rng)

    # Ước lượng ML
    return mle_estimate_st(sensing_pts, d_tilde, cfg, grid_step=50.0)


if __name__ == "__main__":
    cfg = DEFAULT
    rng = np.random.default_rng(42)

    st_true = np.array([1000.0, 1000.0])

    # Test initial coarse estimate
    est0 = coarse_initial_estimate(st_true, cfg, rng)
    print(f"ST thực: {st_true},  ước lượng coarse: {est0},  sai số: "
          f"{np.linalg.norm(est0 - st_true):.1f} m")

    # Test MLE với nhiều HP rải rác
    hover_points = np.array([
        [200., 200.], [400., 400.], [600., 600.],
        [800., 800.], [900., 1100.], [1100., 900.],
        [1200., 1200.], [800., 1200.], [1200., 800.]
    ])
    d_tilde = simulate_distance_measurements(hover_points, st_true, cfg, rng)
    est = mle_estimate_st(hover_points, d_tilde, cfg, grid_step=25.0)
    print(f"\nVới {len(hover_points)} HP: ước lượng = {est},  sai số = "
          f"{np.linalg.norm(est - st_true):.2f} m")

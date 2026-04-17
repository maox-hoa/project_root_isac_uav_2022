"""Test gradient giải tích của f(S_j) so với finite difference."""
import numpy as np
from config import DEFAULT
from trajectory_optimizer import (
    init_hypothetical_trajectory, _extract_hps,
    objective_f, analytical_gradient_f,
    compute_Psi_c, compute_Psi_s,
)

cfg = DEFAULT
rng = np.random.default_rng(7)

# Scenario nhỏ
cus = np.array([[1200., 1200.], [1300., 800.]])
sts_est = np.array([[1000., 1050.], [300., 1050.]])
start = cfg.base_station

Nf = 12  # stage nhỏ để test nhanh
S = init_hypothetical_trajectory(start, cfg.ground_area_corner, Nf, cfg)

B_alloc = np.full(len(cus), cfg.B / len(cus))
HP0 = _extract_hps(S, cfg.mu)

# Psi_prev từ chính trajectory S để có reference không đổi
Psi_c_prev = compute_Psi_c(S, HP0, cus, B_alloc, cfg)
Psi_s_prev = compute_Psi_s(HP0, sts_est, cfg)
print(f"Psi_c_prev = {Psi_c_prev:.4e}, Psi_s_prev = {Psi_s_prev:.4e}")
print()

# Thêm 1 chút nhiễu để tránh điểm singular
S += rng.normal(0, 10, size=S.shape)

# --- Gradient giải tích ---
grad_ana = analytical_gradient_f(
    S, B_alloc, np.zeros((0,2)), np.zeros((0,2)),
    cus, sts_est, Psi_c_prev, Psi_s_prev, 0.5, cfg.mu, cfg
)

# --- Gradient finite diff ---
h = 1.0
grad_fd = np.zeros_like(S)
for i in range(Nf):
    for axis in range(2):
        sp = S.copy(); sm = S.copy()
        sp[i, axis] += h; sm[i, axis] -= h
        fp = objective_f(sp, B_alloc, np.zeros((0,2)), np.zeros((0,2)),
                         cus, sts_est, Psi_c_prev, Psi_s_prev, 0.5, cfg.mu, cfg)
        fm = objective_f(sm, B_alloc, np.zeros((0,2)), np.zeros((0,2)),
                         cus, sts_est, Psi_c_prev, Psi_s_prev, 0.5, cfg.mu, cfg)
        grad_fd[i, axis] = (fp - fm) / (2 * h)

print("Waypoint | Analytical | Finite Diff | Rel. Error")
print("-" * 72)
for i in range(Nf):
    for axis in range(2):
        a = grad_ana[i, axis]
        fd = grad_fd[i, axis]
        rel = abs(a - fd) / (abs(fd) + 1e-12)
        marker = " (HP)" if (i+1) % cfg.mu == 0 else "     "
        print(f"S[{i},{axis}]{marker} | {a:+.3e} | {fd:+.3e} | {rel:.3e}")

# Cosine similarity
cos = np.sum(grad_ana * grad_fd) / (np.linalg.norm(grad_ana) * np.linalg.norm(grad_fd) + 1e-20)
print(f"\nCosine similarity: {cos:.6f}")
print(f"||grad_ana|| = {np.linalg.norm(grad_ana):.4e}")
print(f"||grad_fd||  = {np.linalg.norm(grad_fd):.4e}")

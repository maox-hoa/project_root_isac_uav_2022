"""Tạo summary figure gộp tất cả kết quả."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT

RESULTS_DIR = Path("/home/claude/isac_uav/results")
PLOT_DIR = Path("/home/claude/isac_uav/plots")

cfg = DEFAULT

fig = plt.figure(figsize=(15, 10))

# --- Row 1: Fig. 3 + Fig. 4 ---
# Fig. 3 subplot
with open(RESULTS_DIR / "fig3_convergence.pkl", "rb") as f:
    d3 = pickle.load(f)

ax = fig.add_subplot(2, 3, 1)
obj = d3["obj_history"]
ax.plot(obj, 's-', color='tab:red', linewidth=2, markersize=7)
ax.set_xlabel("Iteration index")
ax.set_ylabel("Objective of P(j)", color='tab:red')
ax.tick_params(axis='y', labelcolor='tab:red')
ax2 = ax.twinx()
ax2.plot(d3["psi_s_per_iter"], 'o-', color='tab:blue', linewidth=2, markersize=7)
ax2.set_ylabel("Ψₛ: CRB sum (m²)", color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax.set_title("(a) Fig. 3 Convergence\n(η=0.5, E_tot=40 kJ)")
ax.grid(alpha=0.3)

# Fig. 4 subplot
with open(RESULTS_DIR / "fig4_mstd_stages.pkl", "rb") as f:
    d4 = pickle.load(f)

ax = fig.add_subplot(2, 3, 2)
crb = d4["crb_mean"]; mse = d4["mse_mean"]
stages = np.arange(1, crb.shape[0]+1)
for k in range(crb.shape[1]):
    color = ['tab:red', 'tab:blue'][k]
    ax.semilogy(stages, crb[:, k], 's-', color=color, linewidth=2, label=f"CRB ST{k+1}")
    ax.semilogy(stages, mse[:, k], 'o--', color=color, alpha=0.7, linewidth=2,
                label=f"MSE ST{k+1}")
ax.set_xlabel("Stage index j")
ax.set_ylabel("Squared error (m²)")
ax.set_title("(b) Fig. 4 MSE vs CRB qua stages\n(η=1, E_tot=60 kJ, MC=5)")
ax.set_xticks(stages)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Fig. 9(a) tradeoff subplot
with open(RESULTS_DIR / "fig9_tradeoff.pkl", "rb") as f:
    d9 = pickle.load(f)

ax = fig.add_subplot(2, 3, 3)
psi_c_i = np.array(d9["isac"]["psi_c"])/1e9
psi_s_i = np.array(d9["isac"]["psi_s"])
psi_c_b = np.array(d9["isac_ba"]["psi_c"])/1e9
psi_s_b = np.array(d9["isac_ba"]["psi_s"])
ax.semilogx(psi_s_i, psi_c_i, 'o-r', linewidth=2, markersize=8, label="ISAC")
ax.semilogx(psi_s_b, psi_c_b, 's-g', linewidth=2, markersize=8, label="ISAC+BA")
for i, eta in enumerate(d9["eta_list"]):
    ax.annotate(f"η={eta}", (psi_s_i[i], psi_c_i[i]),
                textcoords="offset points", xytext=(-30, -5), fontsize=9)
ax.set_xlabel("Ψₛ: CRB sum (m²)")
ax.set_ylabel("Ψc: total data (Gbits)")
ax.set_title("(c) Fig. 9a Tradeoff η (E_tot=40 kJ)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Row 2: Fig. 5 + Fig. 6 trajectories ---
# Fig. 5(a) Sensing
with open(RESULTS_DIR / "fig5_vs_energy.pkl", "rb") as f:
    d5 = pickle.load(f)
E = np.array(d5["Etot_list"])/1e3

ax = fig.add_subplot(2, 3, 4)
ax.semilogy(E, d5["separate"]["psi_s"], '^-k', label="Separate")
ax.semilogy(E, d5["isac"]["psi_s"],     'o-r', label="ISAC")
ax.semilogy(E, d5["isac_ba"]["psi_s"],  's-g', label="ISAC+BA")
ax.semilogy(E, d5["straight"]["psi_s"], 'v-b', label="Straight")
ax.semilogy(E, d5["circle"]["psi_s"],   'd-m', label="Circle")
ax.set_xlabel("Energy supply (kJ)")
ax.set_ylabel("Ψₛ: CRB sum (m²)")
ax.set_title("(d) Fig. 5a Sensing vs E_tot")
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Fig. 5(b) Communication
ax = fig.add_subplot(2, 3, 5)
ax.plot(E, np.array(d5["separate"]["psi_c"])/1e9, '^-k', label="Separate")
ax.plot(E, np.array(d5["isac"]["psi_c"])/1e9,     'o-r', label="ISAC")
ax.plot(E, np.array(d5["isac_ba"]["psi_c"])/1e9,  's-g', label="ISAC+BA")
ax.plot(E, np.array(d5["straight"]["psi_c"])/1e9, 'v-b', label="Straight")
ax.plot(E, np.array(d5["circle"]["psi_c"])/1e9,   'd-m', label="Circle")
ax.set_xlabel("Energy supply (kJ)")
ax.set_ylabel("Ψc: total data (Gbits)")
ax.set_title("(e) Fig. 5b Comm vs E_tot")
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# Fig. 6 trajectory E=60kJ (map)
ax = fig.add_subplot(2, 3, 6)
idx = -1
wp = d5["isac"]["waypoints"][idx]
hp = d5["isac"]["hover_points"][idx]
ax.plot(wp[:, 0], wp[:, 1], '-', color='purple', alpha=0.7, linewidth=1.5)
ax.plot(hp[:, 0], hp[:, 1], 'o', color='green', markersize=7, label="HPs")
ax.plot(d5["sts_true"][:, 0], d5["sts_true"][:, 1], '^',
        color='red', markersize=14, label="STs")
ax.plot(d5["cus"][:, 0], d5["cus"][:, 1], 's',
        color='blue', markersize=12, label="CUs")
ax.plot(cfg.base_station[0], cfg.base_station[1], '*',
        color='black', markersize=16, label="Base")
ax.set_xlim(0, cfg.Lx); ax.set_ylim(0, cfg.Ly)
ax.set_xlabel("L_x (m)"); ax.set_ylabel("L_y (m)")
ax.set_title(f"(f) Fig. 6 Quỹ đạo ISAC (E_tot=60 kJ)")
ax.legend(fontsize=8, loc='lower right')
ax.set_aspect('equal')
ax.grid(alpha=0.3)

fig.suptitle("Tái hiện: Jing et al., \"ISAC from the Sky\" (IEEE TWC 2024)",
             fontsize=14, y=1.00)
fig.tight_layout()
save_path = PLOT_DIR / "summary.png"
fig.savefig(save_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Đã lưu {save_path}")

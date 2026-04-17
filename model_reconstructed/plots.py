"""
Vẽ các biểu đồ tái hiện Fig. 3, 4, 5, 6, 9 của bài báo.
Đọc các file .pkl từ thư mục results/.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # không cần display
import matplotlib.pyplot as plt

from config import DEFAULT

RESULTS_DIR = Path("/home/claude/isac_uav/results")
PLOT_DIR = Path("/home/claude/isac_uav/plots")
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# FIG. 3: Convergence behaviour (2 subplot)
# ---------------------------------------------------------------------------
def plot_fig3(save_path: Path | None = None):
    with open(RESULTS_DIR / "fig3_convergence.pkl", "rb") as f:
        data = pickle.load(f)

    obj = data["obj_history"]           # len = n_iter + 1 (initial + sau mỗi iter)
    psi_c = data["psi_c_per_iter"]
    psi_s = data["psi_s_per_iter"]
    # Align: obj_history có kích thước n_iter+1 (giá trị hypothetical + sau mỗi iter)
    iter_idx = np.arange(len(obj))
    iter_idx_psi = np.arange(len(psi_c))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Sensing performance (CRB) + objective
    ax1 = axs[0]
    l1 = ax1.plot(iter_idx_psi, psi_s, 'o-', color='tab:blue',
                  label="Upper bound CRB (m²)")
    ax1.set_xlabel("Iteration index")
    ax1.set_ylabel("Upper bound of CRB of x + CRB of y (m²)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    l2 = ax2.plot(iter_idx, obj, 's-', color='tab:red', label="Objective of P(j)")
    ax2.set_ylabel("Objective of P(j)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title("(a) Sensing + objective")
    # Gộp legend
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9)

    # (b) Communication + objective
    ax3 = axs[1]
    l3 = ax3.plot(iter_idx_psi, psi_c / 1e9, 'o-', color='tab:green',
                  label="Lower bound total data (Gbits)")
    ax3.set_xlabel("Iteration index")
    ax3.set_ylabel("Lower bound of total transmitted data (Gbits)", color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax4 = ax3.twinx()
    l4 = ax4.plot(iter_idx, obj, 's-', color='tab:red', label="Objective of P(j)")
    ax4.set_ylabel("Objective of P(j)", color='tab:red')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_title("(b) Communication + objective")
    lines = l3 + l4
    ax3.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=9)

    fig.suptitle("Fig. 3: Convergence behaviour of iterative algorithm\n"
                 "(N_stg=60, η=0.5, K=M=2, E_tot=40 kJ)", y=1.02)
    fig.tight_layout()

    if save_path is None:
        save_path = PLOT_DIR / "fig3_convergence.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_fig3] Đã lưu {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# FIG. 4: MSE và CRB qua các stage
# ---------------------------------------------------------------------------
def plot_fig4(save_path: Path | None = None):
    with open(RESULTS_DIR / "fig4_mstd_stages.pkl", "rb") as f:
        data = pickle.load(f)

    crb = data["crb_mean"]    # (num_stages, K)
    mse = data["mse_mean"]    # (num_stages, K)
    num_stages, K = crb.shape
    stage_idx = np.arange(1, num_stages + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for k in range(K):
        ax.semilogy(stage_idx, crb[:, k], 's-', color=colors[k],
                    label=f"CRB+ISAC, ST {k+1}")
        ax.semilogy(stage_idx, mse[:, k], 'o--', color=colors[k],
                    alpha=0.7, label=f"MSE+ISAC, ST {k+1}")

    ax.set_xlabel("Stage index j")
    ax.set_ylabel("Squared estimation error of x + of y (m²)")
    ax.set_title("Fig. 4: Sensing performance based on MSTD\n"
                 "(N_stg=60, η=1, K=2, E_tot=60 kJ)")
    ax.set_xticks(stage_idx)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()

    if save_path is None:
        save_path = PLOT_DIR / "fig4_mstd_stages.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_fig4] Đã lưu {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# FIG. 5: Performance vs energy (so sánh các scheme)
# ---------------------------------------------------------------------------
def plot_fig5(save_path: Path | None = None):
    with open(RESULTS_DIR / "fig5_vs_energy.pkl", "rb") as f:
        data = pickle.load(f)

    E = np.array(data["Etot_list"]) / 1e3   # kJ
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Sensing (CRB) - log scale
    ax = axs[0]
    ax.semilogy(E, data["separate"]["psi_s"], '^-k', label="Separate")
    ax.semilogy(E, data["isac"]["psi_s"],     'o-r', label="ISAC")
    ax.semilogy(E, data["isac_ba"]["psi_s"],  's-g', label="ISAC+BA")
    ax.semilogy(E, data["straight"]["psi_s"], 'v-b', label="Straight")
    ax.semilogy(E, data["circle"]["psi_s"],   'd-m', label="Circle")
    ax.set_xlabel("Energy supply (kJ)")
    ax.set_ylabel("Upper bound of CRB of x + CRB of y (m²)")
    ax.set_title("(a) Sensing")
    ax.legend(fontsize=9)

    # (b) Communication (Ψc)
    ax = axs[1]
    ax.plot(E, np.array(data["separate"]["psi_c"])/1e9, '^-k', label="Separate")
    ax.plot(E, np.array(data["isac"]["psi_c"])/1e9,     'o-r', label="ISAC")
    ax.plot(E, np.array(data["isac_ba"]["psi_c"])/1e9,  's-g', label="ISAC+BA")
    ax.plot(E, np.array(data["straight"]["psi_c"])/1e9, 'v-b', label="Straight")
    ax.plot(E, np.array(data["circle"]["psi_c"])/1e9,   'd-m', label="Circle")
    ax.set_xlabel("Energy supply (kJ)")
    ax.set_ylabel("Lower bound of total transmitted data (Gbits)")
    ax.set_title("(b) Communication")
    ax.legend(fontsize=9)

    fig.suptitle("Fig. 5: Performance vs total energy supply (N_stg=60, η=0.5, K=M=2)",
                 y=1.02)
    fig.tight_layout()

    if save_path is None:
        save_path = PLOT_DIR / "fig5_vs_energy.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_fig5] Đã lưu {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# FIG. 6: Quỹ đạo UAV ở các E_tot khác nhau
# ---------------------------------------------------------------------------
def plot_fig6(save_path: Path | None = None, Etot_indices: list = [0, -1]):
    with open(RESULTS_DIR / "fig5_vs_energy.pkl", "rb") as f:
        data = pickle.load(f)

    cfg = DEFAULT
    E_list = data["Etot_list"]
    cus = data["cus"]
    sts = data["sts_true"]

    n_plots = len(Etot_indices)
    fig, axs = plt.subplots(1, n_plots, figsize=(5.5*n_plots, 5))
    if n_plots == 1:
        axs = [axs]

    for i, idx in enumerate(Etot_indices):
        ax = axs[i]
        wp = data["isac"]["waypoints"][idx]
        hp = data["isac"]["hover_points"][idx]
        E = E_list[idx] / 1e3

        ax.plot(wp[:, 0], wp[:, 1], '-', color='purple', label="Trajectory", alpha=0.7)
        ax.plot(hp[:, 0], hp[:, 1], 'o', color='green',
                markersize=8, label="HPs")
        ax.plot(sts[:, 0], sts[:, 1], '^', color='red',
                markersize=14, label="STs")
        ax.plot(cus[:, 0], cus[:, 1], 's', color='blue',
                markersize=12, label="CUs")
        ax.plot(cfg.base_station[0], cfg.base_station[1], '*',
                color='black', markersize=16, label="Charging base")

        ax.set_xlim(0, cfg.Lx)
        ax.set_ylim(0, cfg.Ly)
        ax.set_xlabel("L_x (m)")
        ax.set_ylabel("L_y (m)")
        ax.set_title(f"(E_tot = {E:.0f} kJ)")
        ax.legend(loc='lower right', fontsize=8)
        ax.set_aspect('equal')

    fig.suptitle("Fig. 6: ISAC-based UAV trajectories under different E_tot "
                 "(N_stg=60, η=0.5)", y=1.00)
    fig.tight_layout()

    if save_path is None:
        save_path = PLOT_DIR / "fig6_trajectories.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_fig6] Đã lưu {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# FIG. 9: tradeoff C&S khi quét η
# ---------------------------------------------------------------------------
def plot_fig9(save_path: Path | None = None):
    with open(RESULTS_DIR / "fig9_tradeoff.pkl", "rb") as f:
        data = pickle.load(f)

    cfg = DEFAULT
    eta_list = data["eta_list"]
    psi_c_isac = np.array(data["isac"]["psi_c"]) / 1e9
    psi_s_isac = np.array(data["isac"]["psi_s"])
    psi_c_ba = np.array(data["isac_ba"]["psi_c"]) / 1e9
    psi_s_ba = np.array(data["isac_ba"]["psi_s"])

    n_plots = 1 + len(data["trajectories"])
    fig, axs = plt.subplots(1, n_plots, figsize=(5.5*n_plots, 5))
    if n_plots == 1:
        axs = [axs]

    # (a) Tradeoff curve
    ax = axs[0]
    ax.plot(psi_s_isac, psi_c_isac, 'o-r', label="ISAC")
    ax.plot(psi_s_ba, psi_c_ba, 's-g', label="ISAC+BA")
    for i, eta in enumerate(eta_list):
        ax.annotate(f"η={eta}", (psi_s_isac[i], psi_c_isac[i]),
                    textcoords="offset points", xytext=(-25, 8), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel("Upper bound of CRB of x + CRB of y (m²)")
    ax.set_ylabel("Lower bound of total transmitted data (Gbits)")
    ax.set_title("(a) Tradeoff between C&S (E_tot=40 kJ)")
    ax.legend(fontsize=9)

    # (b..) Các quỹ đạo cho các η đã lưu
    traj_dict = data["trajectories"]
    for i, (eta, traj) in enumerate(sorted(traj_dict.items())):
        ax = axs[i + 1]
        wp = traj["waypoints"]
        hp = traj["hover_points"]
        ax.plot(wp[:, 0], wp[:, 1], '-', color='purple', alpha=0.7, label="Trajectory")
        ax.plot(hp[:, 0], hp[:, 1], 'o', color='green', markersize=8, label="HPs")
        ax.plot(data["sts_true"][:, 0], data["sts_true"][:, 1], '^',
                color='red', markersize=14, label="STs")
        ax.plot(data["cus"][:, 0], data["cus"][:, 1], 's',
                color='blue', markersize=12, label="CUs")
        ax.plot(cfg.base_station[0], cfg.base_station[1], '*',
                color='black', markersize=16, label="Base")
        ax.set_xlim(0, cfg.Lx); ax.set_ylim(0, cfg.Ly)
        ax.set_xlabel("L_x (m)"); ax.set_ylabel("L_y (m)")
        priority = "sensing-priority" if eta > 0.5 else "comm-priority"
        ax.set_title(f"(b) η = {eta} ({priority})")
        ax.legend(loc='lower right', fontsize=7)
        ax.set_aspect('equal')

    fig.suptitle("Fig. 9: Tradeoff between C&S performances (N_stg=60, M=K=2)", y=1.02)
    fig.tight_layout()

    if save_path is None:
        save_path = PLOT_DIR / "fig9_tradeoff.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_fig9] Đã lưu {save_path}")
    return save_path


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    paths = []
    if which in ("3", "all"):
        try:
            paths.append(plot_fig3())
        except FileNotFoundError as e:
            print(f"[plot_fig3] Bỏ qua - thiếu file: {e}")
    if which in ("4", "all"):
        try:
            paths.append(plot_fig4())
        except FileNotFoundError as e:
            print(f"[plot_fig4] Bỏ qua - thiếu file: {e}")
    if which in ("5", "all"):
        try:
            paths.append(plot_fig5())
        except FileNotFoundError as e:
            print(f"[plot_fig5] Bỏ qua - thiếu file: {e}")
    if which in ("6", "all"):
        try:
            paths.append(plot_fig6())
        except FileNotFoundError as e:
            print(f"[plot_fig6] Bỏ qua - thiếu file: {e}")
    if which in ("9", "all"):
        try:
            paths.append(plot_fig9())
        except FileNotFoundError as e:
            print(f"[plot_fig9] Bỏ qua - thiếu file: {e}")
    print(f"\nHoàn thành {len(paths)} biểu đồ.")

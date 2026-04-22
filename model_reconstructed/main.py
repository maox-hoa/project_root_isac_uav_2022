"""
main.py — Chạy 1 lần: hiển thị quỹ đạo UAV, Rate (Ψc) và CRB (Ψs) qua từng stage.

Cách dùng:
    python main.py                          # dùng tham số mặc định
    python main.py --Etot 60000            # năng lượng 60 kJ
    python main.py --Etot 40000 --eta 0.7  # năng lượng 40 kJ, ưu tiên sensing
    python main.py --help                  # xem tất cả tùy chọn
"""

import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import DEFAULT, SimulationConfig
from mstd import run_mstd
from trajectory_optimizer import compute_Psi_c, compute_Psi_s

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Parsing tham số dòng lệnh
# ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ISAC-UAV: chạy 1 lần, hiển thị quỹ đạo + rate + CRB qua các stage."
    )
    parser.add_argument("--Etot",    type=float, default=40e3,
                        help="Tổng năng lượng UAV (J), mặc định 40000 = 40 kJ")
    parser.add_argument("--eta",     type=float, default=0.5,
                        help="Trọng số sensing/comm [0,1], mặc định 0.5")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Random seed, mặc định 42")
    parser.add_argument("--max_stages",    type=int, default=8,
                        help="Số stage tối đa, mặc định 8")
    parser.add_argument("--max_iter",      type=int, default=10,
                        help="Số iteration tối ưu mỗi stage, mặc định 10")
    parser.add_argument("--ba",   action="store_true",
                        help="Bật bandwidth allocation (mặc định: tắt)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Không lưu plot")
    parser.add_argument("--out_dir", type=str, default="output_main",
                        help="Thư mục lưu kết quả, mặc định 'output_main'")
    # Vị trí CU và ST (tùy chọn, mặc định dùng scenario giấy)
    parser.add_argument("--scenario", type=str, default="paper",
                        choices=["paper", "random"],
                        help="'paper': dùng scenario cố định; 'random': sinh ngẫu nhiên")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# Định nghĩa scenario
# ─────────────────────────────────────────────────────────
def get_scenario(args, cfg: SimulationConfig):
    if args.scenario == "paper":
        cus     = np.array([[1200., 1200.], [1300., 800.]])
        sts_true = np.array([[1000., 1050.], [300., 1050.]])
        print("[Scenario] Dùng scenario cố định từ paper (Fig 6).")
    else:
        rng = np.random.default_rng(args.seed)
        cus = rng.uniform([0.6*cfg.Lx, 0.3*cfg.Ly],
                          [0.95*cfg.Lx, 0.9*cfg.Ly], size=(2, 2))
        sts_true = rng.uniform([0.15*cfg.Lx, 0.3*cfg.Ly],
                               [0.75*cfg.Lx, 0.9*cfg.Ly], size=(2, 2))
        print("[Scenario] Sinh ngẫu nhiên (seed={}).".format(args.seed))
    return cus, sts_true


# ─────────────────────────────────────────────────────────
# In bảng tóm tắt kết quả ra console
# ─────────────────────────────────────────────────────────
def print_summary(result, cus, sts_true, args):
    n = result.num_stages
    print()
    print("=" * 65)
    print(f"  KẾT QUẢ MSTD  —  Etot={args.Etot/1e3:.1f} kJ  |  η={args.eta}  |  Stages={n}")
    print("=" * 65)
    print(f"  {'Stage':>5}  {'Ψc (Gbits)':>12}  {'Ψs CRB (m²)':>14}  "
          f"{'MSE ST1 (m²)':>14}  {'MSE ST2 (m²)':>14}")
    print("  " + "-" * 63)
    for j in range(n):
        pc = result.total_transmitted_data[j] / 1e9
        ps = result.psi_s_history[j]
        mse = result.mse_history[j]
        print(f"  {j+1:>5}  {pc:>12.4f}  {ps:>14.4e}  "
              f"{mse[0]:>14.4e}  {mse[1]:>14.4e}")
    print("  " + "-" * 63)
    print(f"  Tổng năng lượng dùng : {result.total_energy_consumed/1e3:.2f} kJ")
    print(f"  Tổng HP tích luỹ     : {result.all_hover_points.shape[0]}")
    print(f"  Tổng waypoints       : {result.all_waypoints.shape[0]}")
    K = len(sts_true)
    final_ests = result.st_estimates_history[-1]
    print(f"  Ước lượng ST cuối:")
    for k in range(K):
        err = np.linalg.norm(final_ests[k] - sts_true[k])
        print(f"    ST {k+1}: thực={sts_true[k]}, ước lượng={np.round(final_ests[k],1)}, "
              f"sai số={err:.1f} m")
    print("=" * 65)


# ─────────────────────────────────────────────────────────
# Vẽ figure 3 subplot: trajectory, rate, CRB
# ─────────────────────────────────────────────────────────
def plot_results(result, cus, sts_true, args, cfg, out_dir: Path):
    n = result.num_stages
    stages = np.arange(1, n + 1)

    psi_c_gbits = np.array(result.total_transmitted_data) / 1e9
    psi_s       = np.array(result.psi_s_history)
    mse_arr     = np.array(result.mse_history)          # (n_stages, K)
    crb_arr     = np.array(result.crb_history)          # (n_stages, K)

    wp = result.all_waypoints
    hp = result.all_hover_points

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Subplot 1: Quỹ đạo UAV ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # Tô màu theo stage
    colors_stage = plt.cm.plasma(np.linspace(0.1, 0.9, n))
    idx_start = 0
    Nstg = cfg.Nstg
    for j in range(n):
        seg_len = min(Nstg, len(wp) - idx_start)
        seg = wp[idx_start : idx_start + seg_len]
        if len(seg) > 1:
            ax1.plot(seg[:, 0], seg[:, 1], '-',
                     color=colors_stage[j], linewidth=1.5,
                     label=f"Stage {j+1}", alpha=0.85)
        idx_start += seg_len

    # Trạm sạc → waypoint đầu
    ax1.annotate("", xy=wp[0], xytext=cfg.base_station,
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax1.scatter(hp[:, 0], hp[:, 1], s=35, c="lime",
                edgecolors="darkgreen", zorder=5, label="HPs", linewidths=0.8)
    ax1.scatter(sts_true[:, 0], sts_true[:, 1], s=120, marker="^",
                c="red", edgecolors="darkred", zorder=6, label="STs")
    ax1.scatter(cus[:, 0], cus[:, 1], s=100, marker="s",
                c="royalblue", edgecolors="navy", zorder=6, label="CUs")
    ax1.plot(*cfg.base_station, "*", ms=14, c="black", zorder=7, label="Base")

    # Đánh số ST
    for k, st in enumerate(sts_true):
        ax1.annotate(f" ST{k+1}", st, fontsize=8, color="darkred")

    ax1.set_xlim(-50, cfg.Lx + 50)
    ax1.set_ylim(-50, cfg.Ly + 50)
    ax1.set_xlabel("$L_x$ (m)", fontsize=10)
    ax1.set_ylabel("$L_y$ (m)", fontsize=10)
    ax1.set_title(f"Quỹ đạo UAV\n"
                  f"(Etot={args.Etot/1e3:.0f} kJ, η={args.eta})", fontsize=10)
    ax1.legend(fontsize=7, loc="lower right", ncol=2)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.25)

    # ── Subplot 2: Rate (Ψc) qua các stage ──────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(stages, psi_c_gbits, "o-", color="royalblue",
             linewidth=2, markersize=8, markerfacecolor="white",
             markeredgewidth=2, label="ISAC" + (" + BA" if args.ba else ""))
    ax2.fill_between(stages, psi_c_gbits * 0.98, psi_c_gbits * 1.02,
                     alpha=0.12, color="royalblue")

    for j, v in enumerate(psi_c_gbits):
        ax2.annotate(f"{v:.3f}", (stages[j], v),
                     textcoords="offset points", xytext=(4, 5),
                     fontsize=8, color="royalblue")

    ax2.set_xlabel("Stage index j", fontsize=10)
    ax2.set_ylabel("$Ψ_c$ — Tổng data (Gbits)", fontsize=10)
    ax2.set_title("Communication Rate\nqua từng stage", fontsize=10)
    ax2.set_xticks(stages)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # ── Subplot 3: CRB và MSE (Ψs) qua các stage ────────────
    ax3 = fig.add_subplot(gs[2])
    K = len(sts_true)
    colors_st = ["crimson", "steelblue", "forestgreen", "darkorange"]

    for k in range(K):
        c = colors_st[k % len(colors_st)]
        ax3.semilogy(stages, crb_arr[:, k], "s-", color=c, linewidth=2,
                     markersize=7, label=f"CRB ST{k+1}")
        ax3.semilogy(stages, mse_arr[:, k], "o--", color=c, linewidth=1.5,
                     markersize=6, alpha=0.75, label=f"MSE ST{k+1}")

    ax3.set_xlabel("Stage index j", fontsize=10)
    ax3.set_ylabel("Sai số bình phương (m²)", fontsize=10)
    ax3.set_title("CRB & MSE ước lượng ST\nqua từng stage", fontsize=10)
    ax3.set_xticks(stages)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3, which="both")

    # Tiêu đề chung
    fig.suptitle(
        f"ISAC-UAV MSTD  —  Etot = {args.Etot/1e3:.0f} kJ  |  η = {args.eta}  |  "
        f"N_stg = {cfg.Nstg}  |  Stages = {n}",
        fontsize=12, y=1.01
    )
    fig.tight_layout()

    save_path = out_dir / f"main_Etot{int(args.Etot/1e3)}kJ_eta{args.eta:.1f}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Đã lưu → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = DEFAULT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 65)
    print("  ISAC-UAV — Main Run")
    print(f"  Etot    = {args.Etot/1e3:.1f} kJ")
    print(f"  η       = {args.eta}  "
          f"({'sensing-priority' if args.eta > 0.5 else 'comm-priority' if args.eta < 0.5 else 'balanced'})")
    print(f"  BA      = {'ON' if args.ba else 'OFF'}")
    print(f"  Seed    = {args.seed}")
    print(f"  Scenario= {args.scenario}")
    print("=" * 65)

    cus, sts_true = get_scenario(args, cfg)
    print(f"  CUs : {cus.tolist()}")
    print(f"  STs : {sts_true.tolist()}")

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    result = run_mstd(
        cus=cus,
        sts_true=sts_true,
        Etot=args.Etot,
        eta=args.eta,
        cfg=cfg,
        use_bandwidth_alloc=args.ba,
        max_stages=args.max_stages,
        max_iter_per_stage=args.max_iter,
        rng=rng,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[Thời gian chạy] {elapsed:.1f} s")

    print_summary(result, cus, sts_true, args)

    if not args.no_plot:
        plot_results(result, cus, sts_true, args, cfg, out_dir)


if __name__ == "__main__":
    main()
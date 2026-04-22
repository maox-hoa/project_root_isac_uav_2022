"""
monte_carlo.py — Mô phỏng Monte Carlo với Etot tùy chọn.

Chạy nhiều lần lặp (trial) cho từng mức Etot, tính trung bình và độ lệch chuẩn
của Ψc (Rate), Ψs (CRB), MSE, rồi vẽ đồ thị so sánh.

Cách dùng:
    # Mặc định: 3 mức năng lượng, 5 trials mỗi mức
    python monte_carlo.py

    # Chọn năng lượng cụ thể (J)
    python monte_carlo.py --Etot 20000 40000 60000

    # Thêm tùy chọn
    python monte_carlo.py --Etot 30000 50000 --n_trials 10 --eta 0.5 --ba

    # Chạy nhanh để test
    python monte_carlo.py --Etot 40000 --n_trials 3 --max_iter 5

    python monte_carlo.py --help
"""

import argparse
import warnings
import time
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import DEFAULT, SimulationConfig
from mstd import run_mstd
from baselines import (
    straight_trajectory, circle_trajectory,
    evaluate_trajectory, run_separate_scheme,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# Parsing tham số
# ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation: Ψc, Ψs, MSE theo Etot."
    )
    parser.add_argument(
        "--Etot", type=float, nargs="+",
        default=[20e3, 40e3, 60e3],
        help="Danh sách mức năng lượng (J). Ví dụ: --Etot 20000 40000 60000"
    )
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Số lần lặp Monte Carlo mỗi Etot, mặc định 5")
    parser.add_argument("--eta",      type=float, default=0.5,
                        help="Trọng số sensing/comm [0,1], mặc định 0.5")
    parser.add_argument("--ba",       action="store_true",
                        help="Bật bandwidth allocation")
    parser.add_argument("--max_stages", type=int, default=8,
                        help="Số stage tối đa, mặc định 8")
    parser.add_argument("--max_iter",   type=int, default=10,
                        help="Số iteration tối ưu mỗi stage, mặc định 10")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Base seed (seed mỗi trial = base_seed + trial_idx)")
    parser.add_argument("--scenario", type=str, default="paper",
                        choices=["paper", "random"],
                        help="'paper': scenario cố định; 'random': sinh mới mỗi trial")
    parser.add_argument("--baselines", action="store_true",
                        help="Chạy thêm các baseline (Separate, Straight, Circle)")
    parser.add_argument("--out_dir",  type=str, default="output_mc",
                        help="Thư mục lưu kết quả, mặc định 'output_mc'")
    parser.add_argument("--no_plot",  action="store_true",
                        help="Không lưu plot")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# Lấy scenario
# ─────────────────────────────────────────────────────────
def get_scenario(args, cfg: SimulationConfig, trial: int):
    if args.scenario == "paper":
        cus      = np.array([[1200., 1200.], [1300., 800.]])
        sts_true = np.array([[1000., 1050.], [300., 1050.]])
    else:
        rng = np.random.default_rng(args.seed + trial * 1000)
        cus = rng.uniform([0.6*cfg.Lx, 0.3*cfg.Ly],
                          [0.95*cfg.Lx, 0.9*cfg.Ly], size=(2, 2))
        sts_true = rng.uniform([0.15*cfg.Lx, 0.3*cfg.Ly],
                               [0.75*cfg.Lx, 0.9*cfg.Ly], size=(2, 2))
    return cus, sts_true


# ─────────────────────────────────────────────────────────
# Chạy 1 trial ISAC
# ─────────────────────────────────────────────────────────
def run_one_trial(cus, sts_true, Etot, args, cfg, trial_seed):
    rng = np.random.default_rng(trial_seed)
    result = run_mstd(
        cus=cus, sts_true=sts_true, Etot=Etot,
        eta=args.eta, cfg=cfg,
        use_bandwidth_alloc=args.ba,
        max_stages=args.max_stages,
        max_iter_per_stage=args.max_iter,
        rng=rng, verbose=False,
    )
    return {
        "psi_c":    result.total_transmitted_data[-1],
        "psi_s":    result.psi_s_history[-1],
        "mse":      result.mse_history[-1].tolist(),
        "n_stages": result.num_stages,
        "energy":   result.total_energy_consumed,
        # Lịch sử qua stage để vẽ theo stage nếu cần
        "psi_c_history": result.total_transmitted_data,
        "psi_s_history": result.psi_s_history,
        "mse_history":   [m.tolist() for m in result.mse_history],
        "crb_history":   [c.tolist() for c in result.crb_history],
    }


# ─────────────────────────────────────────────────────────
# Chạy 1 trial Baseline (Straight / Circle / Separate)
# ─────────────────────────────────────────────────────────
def run_baselines_one(cus, sts_true, Etot, args, cfg, trial_seed):
    rng = np.random.default_rng(trial_seed)
    results = {}

    # Straight
    wp = straight_trajectory(cfg.base_station, cfg.ground_area_corner, Etot, cfg)
    r  = evaluate_trajectory(wp, cus, sts_true, cfg,
                              rng=np.random.default_rng(trial_seed + 1), num_mc=2)
    results["straight"] = {"psi_c": r.psi_c, "psi_s": r.psi_s}

    # Circle
    wp = circle_trajectory(Etot, cfg)
    r  = evaluate_trajectory(wp, cus, sts_true, cfg,
                              rng=np.random.default_rng(trial_seed + 2), num_mc=2)
    results["circle"] = {"psi_c": r.psi_c, "psi_s": r.psi_s}

    # Separate
    r_sep = run_separate_scheme(
        cus, sts_true, Etot, cfg=cfg,
        rng=np.random.default_rng(trial_seed + 3),
        max_stages=args.max_stages,
        max_iter_per_stage=args.max_iter,
        verbose=False,
    )
    results["separate"] = {"psi_c": r_sep["psi_c"], "psi_s": r_sep["psi_s"]}

    return results


# ─────────────────────────────────────────────────────────
# Tính thống kê mean ± std từ danh sách trial
# ─────────────────────────────────────────────────────────
def compute_stats(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ─────────────────────────────────────────────────────────
# In bảng tổng hợp ra console
# ─────────────────────────────────────────────────────────
def print_mc_table(mc_results, args):
    Etot_list  = mc_results["Etot_list"]
    isac_stats = mc_results["isac_stats"]

    print()
    print("=" * 90)
    scheme_name = "ISAC" + (" + BA" if args.ba else "")
    print(f"  MONTE CARLO — {scheme_name}  |  n_trials = {args.n_trials}  |  η = {args.eta}")
    print("=" * 90)
    print(f"  {'Etot (kJ)':>9}  {'Ψc mean (G)':>12}  {'Ψc std':>8}  "
          f"{'Ψs CRB mean':>12}  {'Ψs CRB std':>11}  "
          f"{'MSE mean (m²)':>14}  {'Stages':>7}")
    print("  " + "-" * 87)
    for i, E in enumerate(Etot_list):
        s = isac_stats[i]
        pc_m, pc_s = s["psi_c"]
        ps_m, ps_s = s["psi_s"]
        mse_m, mse_s = s["mse_mean"]
        ns_m, _    = s["n_stages"]
        print(f"  {E/1e3:>9.0f}  {pc_m/1e9:>12.4f}  {pc_s/1e9:>8.4f}  "
              f"{ps_m:>12.4e}  {ps_s:>11.4e}  "
              f"{mse_m:>14.4e}  {ns_m:>7.1f}")

    if args.baselines and "baselines_stats" in mc_results:
        bl = mc_results["baselines_stats"]
        for scheme in ["separate", "straight", "circle"]:
            print()
            print(f"  --- Baseline: {scheme.upper()} ---")
            for i, E in enumerate(Etot_list):
                s = bl[scheme][i]
                pc_m, pc_s = s["psi_c"]
                ps_m, ps_s = s["psi_s"]
                print(f"  {E/1e3:>9.0f}  {pc_m/1e9:>12.4f}  {pc_s/1e9:>8.4f}  "
                      f"{ps_m:>12.4e}  {ps_s:>11.4e}")
    print("=" * 90)


# ─────────────────────────────────────────────────────────
# Vẽ plot tổng hợp
# ─────────────────────────────────────────────────────────
def plot_mc_results(mc_results, args, out_dir: Path):
    Etot_list  = mc_results["Etot_list"]
    isac_stats = mc_results["isac_stats"]
    E_kJ = np.array(Etot_list) / 1e3

    # ── Dữ liệu ISAC ──────────────────────────────────────
    pc_mean = np.array([s["psi_c"][0] / 1e9 for s in isac_stats])
    pc_std  = np.array([s["psi_c"][1] / 1e9 for s in isac_stats])
    ps_mean = np.array([s["psi_s"][0]        for s in isac_stats])
    ps_std  = np.array([s["psi_s"][1]        for s in isac_stats])
    mse_mean= np.array([s["mse_mean"][0]     for s in isac_stats])
    mse_std = np.array([s["mse_mean"][1]     for s in isac_stats])

    scheme_label = "ISAC" + (" + BA" if args.ba else "")

    has_bl = args.baselines and "baselines_stats" in mc_results
    bl_colors = {"separate": "black", "straight": "royalblue", "circle": "darkorchid"}
    bl_markers = {"separate": "^", "straight": "v", "circle": "D"}

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Monte Carlo ({args.n_trials} trials/Etot)  |  η = {args.eta}  |  "
        f"Scenario: {args.scenario}",
        fontsize=12, y=1.01
    )

    # ── Plot 1: Ψc vs Etot ───────────────────────────────
    ax = axs[0]
    ax.errorbar(E_kJ, pc_mean, yerr=pc_std, fmt="o-r",
                linewidth=2, markersize=8, capsize=5,
                label=scheme_label, markerfacecolor="white", markeredgewidth=2)

    if has_bl:
        for name, bl_s in mc_results["baselines_stats"].items():
            y_m = np.array([s["psi_c"][0] / 1e9 for s in bl_s])
            y_s = np.array([s["psi_c"][1] / 1e9 for s in bl_s])
            ax.errorbar(E_kJ, y_m, yerr=y_s,
                        fmt=bl_markers[name] + "--",
                        color=bl_colors[name], linewidth=1.5,
                        markersize=7, capsize=4, label=name.capitalize())

    ax.set_xlabel("Tổng năng lượng (kJ)", fontsize=10)
    ax.set_ylabel("$Ψ_c$ — Tổng data (Gbits)", fontsize=10)
    ax.set_title("Communication Rate\nvs. Năng lượng", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Plot 2: Ψs (CRB) vs Etot ─────────────────────────
    ax = axs[1]
    ax.errorbar(E_kJ, ps_mean, yerr=ps_std, fmt="o-r",
                linewidth=2, markersize=8, capsize=5,
                label=scheme_label, markerfacecolor="white", markeredgewidth=2)
    ax.set_yscale("log")

    if has_bl:
        for name, bl_s in mc_results["baselines_stats"].items():
            y_m = np.array([s["psi_s"][0] for s in bl_s])
            y_s = np.array([s["psi_s"][1] for s in bl_s])
            ax.errorbar(E_kJ, y_m, yerr=y_s,
                        fmt=bl_markers[name] + "--",
                        color=bl_colors[name], linewidth=1.5,
                        markersize=7, capsize=4, label=name.capitalize())

    ax.set_xlabel("Tổng năng lượng (kJ)", fontsize=10)
    ax.set_ylabel("$Ψ_s$ — CRB (m²)  [log scale]", fontsize=10)
    ax.set_title("Sensing CRB\nvs. Năng lượng", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ── Plot 3: MSE ước lượng ST qua Etot ────────────────
    ax = axs[2]
    ax.errorbar(E_kJ, mse_mean, yerr=mse_std, fmt="s-g",
                linewidth=2, markersize=8, capsize=5,
                label="MSE ST trung bình",
                markerfacecolor="white", markeredgewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tổng năng lượng (kJ)", fontsize=10)
    ax.set_ylabel("MSE (m²)  [log scale]", fontsize=10)
    ax.set_title("MSE Ước lượng vị trí ST\nvs. Năng lượng", fontsize=10)

    # Thêm chú thích error bar
    legend_elems = [
        Line2D([0], [0], color="green", marker="s", linewidth=2,
               markersize=8, markerfacecolor="white",
               label="MSE ST trung bình ± std"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    tag = "ba" if args.ba else "no_ba"
    save_path = out_dir / f"mc_{args.n_trials}trials_eta{args.eta:.1f}_{tag}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Đã lưu → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────
# MAIN Monte Carlo
# ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = DEFAULT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    Etot_list = sorted(args.Etot)

    print("\n" + "=" * 65)
    print("  ISAC-UAV — Monte Carlo Simulation")
    print(f"  Etot levels : {[f'{E/1e3:.0f} kJ' for E in Etot_list]}")
    print(f"  n_trials    : {args.n_trials}")
    print(f"  η           : {args.eta}")
    print(f"  BA          : {'ON' if args.ba else 'OFF'}")
    print(f"  Baselines   : {'ON' if args.baselines else 'OFF'}")
    print(f"  Scenario    : {args.scenario}")
    print(f"  Max stages  : {args.max_stages}")
    print(f"  Max iter    : {args.max_iter}")
    print("=" * 65)

    # ── Cấu trúc lưu trữ kết quả ──────────────────────────
    mc_results = {
        "Etot_list":   Etot_list,
        "args":        vars(args),
        "isac_raw":    {E: [] for E in Etot_list},
        "isac_stats":  [],
    }
    if args.baselines:
        mc_results["baselines_raw"]   = {
            "separate": {E: [] for E in Etot_list},
            "straight": {E: [] for E in Etot_list},
            "circle":   {E: [] for E in Etot_list},
        }
        mc_results["baselines_stats"] = {
            "separate": [], "straight": [], "circle": [],
        }

    total_trials = len(Etot_list) * args.n_trials
    completed    = 0
    t_global     = time.time()

    # ── Vòng lặp chính ────────────────────────────────────
    for E in Etot_list:
        print(f"\n{'─'*60}")
        print(f"  Etot = {E/1e3:.0f} kJ  [{args.n_trials} trials]")
        print(f"{'─'*60}")

        for trial in range(args.n_trials):
            seed_t = args.seed + trial * 97 + int(E / 1000) * 7
            cus, sts_true = get_scenario(args, cfg, trial)

            t_trial = time.time()
            # ISAC
            res = run_one_trial(cus, sts_true, E, args, cfg, seed_t)
            mc_results["isac_raw"][E].append(res)

            # Baselines
            if args.baselines:
                bl = run_baselines_one(cus, sts_true, E, args, cfg, seed_t + 5000)
                for name in ["separate", "straight", "circle"]:
                    mc_results["baselines_raw"][name][E].append(bl[name])

            completed += 1
            elapsed_trial = time.time() - t_trial
            elapsed_total = time.time() - t_global
            eta_remain    = elapsed_total / completed * (total_trials - completed)

            print(f"  Trial {trial+1:>2}/{args.n_trials}  "
                  f"Ψc={res['psi_c']/1e9:.3f} G  "
                  f"Ψs={res['psi_s']:.3e}  "
                  f"MSE={np.mean(res['mse']):.3e}  "
                  f"[{elapsed_trial:.1f}s  ETA {eta_remain:.0f}s]")

        # ── Tính thống kê cho Etot này ─────────────────────
        raw = mc_results["isac_raw"][E]
        stats = {
            "psi_c":    compute_stats([r["psi_c"]  for r in raw]),
            "psi_s":    compute_stats([r["psi_s"]  for r in raw]),
            "mse_mean": compute_stats([np.mean(r["mse"]) for r in raw]),
            "n_stages": compute_stats([r["n_stages"] for r in raw]),
            "energy":   compute_stats([r["energy"]   for r in raw]),
        }
        mc_results["isac_stats"].append(stats)

        if args.baselines:
            for name in ["separate", "straight", "circle"]:
                bl_raw = mc_results["baselines_raw"][name][E]
                mc_results["baselines_stats"][name].append({
                    "psi_c": compute_stats([b["psi_c"] for b in bl_raw]),
                    "psi_s": compute_stats([b["psi_s"] for b in bl_raw]),
                })

        # Tóm tắt Etot này
        s = mc_results["isac_stats"][-1]
        print(f"\n  → Etot {E/1e3:.0f} kJ | "
              f"Ψc = {s['psi_c'][0]/1e9:.4f} ± {s['psi_c'][1]/1e9:.4f} Gbits | "
              f"Ψs = {s['psi_s'][0]:.3e} ± {s['psi_s'][1]:.3e} m²")

    total_time = time.time() - t_global
    print(f"\n[Tổng thời gian] {total_time:.1f} s  "
          f"({total_time/60:.1f} phút)")

    # ── Lưu kết quả ────────────────────────────────────────
    tag = "ba" if args.ba else "no_ba"
    pkl_path = out_dir / f"mc_results_eta{args.eta:.1f}_{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(mc_results, f)
    print(f"[Lưu] Kết quả raw → {pkl_path}")

    # Lưu thống kê dạng JSON (dễ đọc)
    json_data = {
        "Etot_kJ":     [E / 1e3 for E in Etot_list],
        "n_trials":    args.n_trials,
        "eta":         args.eta,
        "scheme":      "ISAC" + (" + BA" if args.ba else ""),
        "isac": [
            {
                "Etot_kJ":        E / 1e3,
                "psi_c_mean_G":   s["psi_c"][0] / 1e9,
                "psi_c_std_G":    s["psi_c"][1] / 1e9,
                "psi_s_mean_m2":  s["psi_s"][0],
                "psi_s_std_m2":   s["psi_s"][1],
                "mse_mean_m2":    s["mse_mean"][0],
                "mse_std_m2":     s["mse_mean"][1],
                "n_stages_mean":  s["n_stages"][0],
            }
            for E, s in zip(Etot_list, mc_results["isac_stats"])
        ],
    }
    json_path = out_dir / f"mc_summary_eta{args.eta:.1f}_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[Lưu] Tóm tắt JSON → {json_path}")

    # ── In bảng + vẽ plot ─────────────────────────────────
    print_mc_table(mc_results, args)

    if not args.no_plot:
        plot_mc_results(mc_results, args, out_dir)


if __name__ == "__main__":
    main()
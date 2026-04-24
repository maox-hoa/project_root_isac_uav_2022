"""
tune_t_lse.py — Tìm t_lse tối ưu cho Ψc, song song hóa qua ProcessPoolExecutor.

Cách dùng:
    python tune_t_lse.py
    python tune_t_lse.py --Etot 60000 --n_trials 10 --n_workers 8
    python tune_t_lse.py --t_list 1 3 5 10 15 20 30 --n_workers 4
"""

import argparse
import warnings
import time
import pickle
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Tìm t_lse tối ưu cho Ψc (parallel).")
    p.add_argument("--Etot", type=float, default=40e3)
    p.add_argument("--eta",  type=float, default=0.5)
    p.add_argument("--t_list", type=float, nargs="+",
                   default=[1, 1.5, 2, 2.5, 3, 3.5, 5, 5.5, 7, 10, 15, 20, 30, 50])
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--max_stages", type=int, default=8)
    p.add_argument("--max_iter",   type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="output_tune")
    p.add_argument("--n_workers", type=int, default=None,

                   help="Số core dùng (mặc định: os.cpu_count()-1)")
    return p.parse_args()


def _worker_init():
    import warnings as _w
    _w.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def run_one(job):
    import numpy as _np
    from config import SimulationConfig
    from mstd import run_mstd

    t_lse, trial, seed, Etot, eta, max_stages, max_iter, cus, sts_true = job
    cfg = SimulationConfig()
    cfg.t_lse = t_lse
    r = run_mstd(
        cus=cus, sts_true=sts_true, Etot=Etot, eta=eta, cfg=cfg,
        use_bandwidth_alloc=True,
        max_stages=max_stages, max_iter_per_stage=max_iter,
        rng=_np.random.default_rng(seed), verbose=False,
    )
    return {
        "t_lse":  t_lse,
        "trial":  trial,
        "psi_c":  float(r.total_transmitted_data[-1]),
        "psi_s":  float(r.psi_s_history[-1]),
        "mse":    float(_np.mean(r.mse_history[-1])),
        "stages": int(r.num_stages),
        "energy": float(r.total_energy_consumed),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    cus      = np.array([[1200., 1200.], [1300., 800.]])
    sts_true = np.array([[1000., 1050.], [300., 1050.]])

    n_workers = args.n_workers or max(1, (os.cpu_count() or 2) - 1)
    total = len(args.t_list) * args.n_trials

    print("=" * 72)
    print(f"  Tune t_lse (parallel)  |  Etot = {args.Etot/1e3:.0f} kJ"
          f"  |  η = {args.eta}  |  BA = ON")
    print(f"  n_trials = {args.n_trials}  |  n_workers = {n_workers}"
          f"  |  total jobs = {total}")
    print(f"  t values: {args.t_list}")
    print("=" * 72)

    jobs = []
    for t_lse in args.t_list:
        for trial in range(args.n_trials):
            seed = args.seed + trial * 97 + int(t_lse * 13)
            jobs.append((t_lse, trial, seed, args.Etot, args.eta,
                         args.max_stages, args.max_iter, cus, sts_true))

    results = {t: [] for t in args.t_list}
    done    = 0
    t_start = time.time()
    n_workers = 5
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init) as ex:
        futures = {ex.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results[r["t_lse"]].append(r)
            done += 1
            elapsed = time.time() - t_start
            eta_s   = elapsed / done * (total - done)
            print(f"  [{done:>3}/{total}] t={r['t_lse']:>5.1f} "
                  f"trial={r['trial']+1:>2}  "
                  f"Ψc={r['psi_c']/1e9:.3f}G  Ψs={r['psi_s']:.3e}  "
                  f"stages={r['stages']}  [ETA {eta_s:.0f}s]")

    print(f"\n  Hoàn thành {total} jobs trong {time.time()-t_start:.1f}s")

    stats = []
    for t_lse in args.t_list:
        vals = results[t_lse]
        pc = np.array([r["psi_c"] for r in vals])
        ps = np.array([r["psi_s"] for r in vals])
        mse = np.array([r["mse"] for r in vals])
        stats.append({
            "t_lse":    t_lse,
            "psi_c_m":  pc.mean(),
            "psi_c_s":  pc.std(),
            "psi_s_m":  ps.mean(),
            "psi_s_s":  ps.std(),
            "mse_m":    mse.mean(),
        })

    print("\n" + "=" * 72)
    print(f"  {'t_lse':>6}  {'Ψc mean (G)':>14}  {'Ψc std':>10}"
          f"  {'Ψs mean':>12}  {'MSE mean':>12}")
    print("  " + "-" * 66)
    best = max(stats, key=lambda s: s["psi_c_m"])
    for s in stats:
        marker = " ← BEST" if s is best else ""
        print(f"  {s['t_lse']:>6.1f}  {s['psi_c_m']/1e9:>14.4f}  "
              f"{s['psi_c_s']/1e9:>10.4f}  {s['psi_s_m']:>12.3e}  "
              f"{s['mse_m']:>12.3e}{marker}")
    print("=" * 72)
    print(f"\n  ★ t_lse TỐI ƯU cho Ψc: {best['t_lse']}  "
          f"→ Ψc = {best['psi_c_m']/1e9:.4f} ± {best['psi_c_s']/1e9:.4f} Gbits")

    with open(out_dir / f"tune_t_lse_E{int(args.Etot/1e3)}.pkl", "wb") as f:
        pickle.dump({"args": vars(args), "stats": stats, "raw": results}, f)

    t_arr   = np.array([s["t_lse"]   for s in stats])
    pc_mean = np.array([s["psi_c_m"] for s in stats]) / 1e9
    pc_std  = np.array([s["psi_c_s"] for s in stats]) / 1e9
    ps_mean = np.array([s["psi_s_m"] for s in stats])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    ax.errorbar(t_arr, pc_mean, yerr=pc_std, fmt="o-", color="royalblue",
                markersize=8, capsize=4, linewidth=2,
                markerfacecolor="white", markeredgewidth=2)
    ax.axvline(best["t_lse"], color="red", ls="--", alpha=0.6,
               label=f"Best t = {best['t_lse']}")
    ax.set_xlabel("t_lse")
    ax.set_ylabel("Ψc mean (Gbits)")
    ax.set_title(f"Ψc vs t_lse  (Etot = {args.Etot/1e3:.0f} kJ, BA = ON)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    ax.plot(t_arr, ps_mean, "s-", color="darkorange", markersize=8,
            linewidth=2, markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("t_lse")
    ax.set_ylabel("Ψs CRB (m²)")
    ax.set_title("Ψs vs t_lse")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / f"tune_t_lse_E{int(args.Etot/1e3)}.png",
                dpi=140, bbox_inches="tight")
    print(f"\n  Plot đã lưu → {out_dir}/")


if __name__ == "__main__":
    main()
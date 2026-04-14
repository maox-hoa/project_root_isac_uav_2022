# run_energy.py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
# ⚠️ Đổi tên module nếu bạn lưu code gốc với tên khác
from experiments import gen_setups, mc_single

def run_task(args):
    """Wrapper để multiprocessing gọi được mc_single"""
    setup, energy = args
    setup['total_energy'] = energy
    return mc_single(setup)

if __name__ == '__main__':
    n_mc = 100
    energy_vec = np.arange(10e3, 36e3, 5e3)
    setups = gen_setups(n_mc)

    # Tạo danh sách công việc: (setup_copy, mức_năng_lượng)
    tasks = [(s.copy(), E) for E in energy_vec for s in setups]
    print(f"🚀 Chạy {len(tasks)} mô phỏng (n_mc={n_mc}) trên {Pool()._processes} nhân...")
##
    # Chạy song song
    with Pool() as pool:
        results = pool.map(run_task, tasks)

    # Gom & tính trung bình theo từng mức năng lượng
    CRB_avg, MSE_avg, Rate_avg = [], [], []
    for i in range(len(energy_vec)):
        chunk = results[i*n_mc : (i+1)*n_mc]
        CRB_avg.append(np.nanmean([r[0] for r in chunk]))
        MSE_avg.append(np.nanmean([r[2] for r in chunk]))
        Rate_avg.append(np.nanmean([r[1] for r in chunk]))

    # Vẽ đồ thị giống hàm gốc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.semilogy(energy_vec/1e3, CRB_avg, 'bo-', label='CRB')
    ax1.semilogy(energy_vec/1e3, MSE_avg, 'rs-', label='MSE')
    ax1.set(xlabel='$E_{tot}$ [KJ]', ylabel='Error [m²]'); ax1.legend(); ax1.grid(True, alpha=.3)
    ax2.plot(energy_vec/1e3, np.array(Rate_avg)/1e6, 'go-')
    ax2.set(xlabel='$E_{tot}$ [KJ]', ylabel='Rate [Mbit/s]'); ax2.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('var_energy_mc100.png', dpi=150)
    plt.show()
    print("✅ Xong! Đã lưu var_energy_mc100.png")
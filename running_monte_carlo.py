import matplotlib.pyplot as plt
from experiments import var_energy, var_eta


def run_energy_experiment():
    print("\n=== Running Energy Variation Experiment ===")
    fig, energy_vec, CRB, MSE, Rate = var_energy(
        n_mc=100,
        save_path="figures/energy_experiment.png"
    )
    plt.show()


def run_eta_experiment():
    print("\n=== Running Eta Variation Experiment ===")
    fig, eta_vec, CRB, MSE, Rate = var_eta(
        n_mc=100,
        save_path="figures/eta_experiment.png"
    )
    plt.show()


if __name__ == "__main__":
    # chọn cái muốn chạy
    run_energy_experiment()
    # run_eta_experiment()
"""
Parameters for ISAC-UAV simulation — TWC version.
Extended from arXiv version: M CUs, K STs, bandwidth allocation.
Matches Table I of Jing et al. (IEEE TWC 2024).
"""
import numpy as np

# ======================== Simulation Parameters (Table I, TWC) ========================
alpha_0 = 10 ** (-50 / 10)           # Comm channel power at ref distance 1m [linear]
N_0 = 10 ** (-170 / 10) * 1e-3       # Noise PSD [W/Hz]
P = 10 ** (20 / 10) * 1e-3           # Transmit power [W] (20 dBm)
B = 10e6                              # Total system bandwidth [Hz] (10 MHz)
H = 200.0                             # UAV altitude [m]
V_str = 25.0                          # Speed for initial trajectory [m/s]
L_x = 1500.0                          # Area x dimension [m]
L_y = 1500.0                          # Area y dimension [m]
T_f = 1.0                             # Flying duration per segment [s]
T_h = 1.5                             # Hovering duration [s]
beta_0 = 10 ** (-49 / 10)            # Sensing channel power at ref 1m [linear]
V_max = 30.0                          # Max flying speed [m/s]
mu = 5                                # Hover every mu waypoints
eta = 0.5                             # C&S tradeoff weight
n_iter = 20                           # Max iterations per stage optimization
a = 200                               # Noise parameter (TWC Table I)
w_star = 0.8                          # Step size for iterative algorithm
N_stg = 60                            # Waypoints per stage (TWC uses 60)
delta_omega = 100                     # Step size search resolution

# Derived parameters
K_stg = N_stg // mu                   # Hover points per stage
G_p = 0.1 * B                         # Signal processing gain
sigma_0 = np.sqrt(B * N_0)            # Noise std (full bandwidth)

# ======================== Energy Parameters (Table I) ========================
P_0 = 80.0         # Blade profile power [W]
P_I = 88.6          # Induced power [W]
U_tip = 120.0       # Tip speed of rotor blade [m/s]
v_0 = 4.03          # Mean rotor induced velocity [m/s]
D_0 = 0.6           # Fuselage drag ratio
rho = 1.225          # Air density [kg/m^3]
s_rotor = 0.05      # Rotor solidity
A_rotor = 0.503     # Rotor disc area [m^2]

# ======================== Optimization Settings ========================
opt_threshold = 1e-20   # Convergence threshold

# ======================== Multi-User / Multi-Target Setup ========================
# M = 2 CUs, K = 2 STs (matching TWC numerical results)
base_station_pos = np.array([0.0, 0.0])

# M communication users — shape (2, M), each column is [x, y]
comm_user_pos = np.array([
    [750.0,  1300.0],    # CU 1
    [1200.0,  800.0],    # CU 2
]).T  # → shape (2, M)

# K sensing targets — TRUE positions, shape (2, K)
sense_target_pos = np.array([
    [400.0,  1100.0],    # ST 1
    [1100.0,  500.0],    # ST 2
]).T  # → shape (2, K)

# K sensing targets — INITIAL ESTIMATES (coarse), shape (2, K)
est_sense_targets = np.array([
    [500.0,  1200.0],    # estimate of ST 1
    [1200.0,  400.0],    # estimate of ST 2
]).T  # → shape (2, K)

# Counts
M = comm_user_pos.shape[1]            # Number of CUs
K = sense_target_pos.shape[1]         # Number of STs

# Bandwidth — initially equal allocation
B_init = np.full(M, B / M)            # (M,) initial bandwidth per CU

# Energy
total_energy = 40e3    # [J] = 40 KJ
E_min = 7e3            # Min energy to start a new stage [J]

# ======================== Precomputed Constants ========================
factor_CRB = (P * G_p * beta_0) / (a * sigma_0 ** 2)

# Per-user noise std (depends on bandwidth, will be recomputed during BA)
def snr_const_user(B_m):
    """SNR constant for a CU with bandwidth B_m."""
    sigma2_m = B_m * N_0
    return (P * alpha_0) / sigma2_m

# Default SNR constant (equal bandwidth)
snr_const = snr_const_user(B / M)

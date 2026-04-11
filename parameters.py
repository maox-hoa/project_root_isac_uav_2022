"""
Parameters for ISAC-UAV simulation.
Matches parameters.m from the MATLAB implementation (arXiv version).
Single CU, single ST, no bandwidth allocation.
"""
import numpy as np

# ======================== Simulation Parameters (Table II) ========================
alpha_0 = 10 ** (-50 / 10)           # Comm channel power at ref distance 1m [linear]
N_0 = 10 ** (-170 / 10) * 1e-3       # Noise PSD [W/Hz]
P = 10 ** (20 / 10) * 1e-3           # Transmit power [W] (20 dBm)
B = 1e6                               # Bandwidth [Hz] (1 MHz)
H = 200.0                             # UAV altitude [m]
V_str = 25.0                          # Speed for initial trajectory [m/s]
L_x = 1500.0                          # Area x dimension [m]
L_y = 1500.0                          # Area y dimension [m]
T_f = 1.5                             # Flying duration per segment [s]
T_h = 1.0                             # Hovering duration [s]
beta_0 = 10 ** (-47 / 10)            # Sensing channel power at ref 1m [linear]
V_max = 30.0                          # Max flying speed [m/s]
mu = 5                                # Hover every mu waypoints
eta = 0.5                             # C&S tradeoff weight (0=comm only, 1=sense only)
n_iter = 20                           # Max iterations per stage optimization
a = 10                                # Noise parameter in eq.(14)
w_star = 0.8                          # Step size for iterative algorithm
N_stg = 25                            # Waypoints per stage

# Derived parameters
K_stg = N_stg // mu                   # Hover points per stage
G_p = 0.1 * B                         # Signal processing gain
sigma_0 = np.sqrt(B * N_0)            # Noise std

# ======================== Energy Parameters (Table I) ========================
P_0 = 80.0        # Blade profile power [W]
P_I = 88.6         # Induced power [W]
U_tip = 120.0      # Tip speed of rotor blade [m/s]
v_0 = 4.03         # Mean rotor induced velocity [m/s]
D_0 = 0.6          # Fuselage drag ratio
rho = 1.225         # Air density [kg/m^3]
s_rotor = 0.05     # Rotor solidity [m^3]
A_rotor = 0.503    # Rotor disc area [m^2]

# ======================== Optimization Settings ========================
opt_threshold = 1e-20   # Convergence threshold

# ======================== Default Setup ========================
base_station_pos = np.array([100.0, 100.0])
comm_user_pos = np.array([1300.0, 1200.0])
est_sense_target = np.array([1200.0, 700.0])
sense_target_pos = np.array([200.0, 1300.0])
total_energy = 35e3    # [J] = 35 KJ
E_min = 7e3            # Min energy to start a new stage [J]

# CRB factor (precomputed constant)
factor_CRB = (P * G_p * beta_0) / (a * sigma_0 ** 2)
# SNR constant for comm
snr_const = (P * alpha_0) / (sigma_0 ** 2)

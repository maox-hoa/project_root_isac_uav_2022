"""
Cấu hình tham số mô phỏng cho kịch bản ISAC-UAV.
Các giá trị tham số lấy từ Bảng I (TABLE I) trong bài báo:
"ISAC from the Sky: UAV Trajectory Design for Joint Communication and Target Localization"
Xiaoye Jing, Fan Liu, Christos Masouros, Yong Zeng
IEEE Transactions on Wireless Communications, 2024
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    # ===== Các tham số công suất bay UAV (Puav trong eq. 33) =====
    P0: float = 80.0        # Công suất blade profile (W)
    PI: float = 88.6        # Công suất induced (W)
    Utip: float = 120.0     # Vận tốc đầu rotor (m/s)
    v0: float = 4.03        # Vận tốc hover trung bình (m/s)
    D0: float = 0.6         # Hệ số lực cản fuselage (drag ratio)
    s_rotor: float = 0.05   # Rotor solidity (m^3 theo bài báo, thực tế là không thứ nguyên)
    rho: float = 1.225      # Mật độ không khí (kg/m^3)
    A: float = 0.503        # Diện tích rotor (m^2)

    # ===== Các tham số kênh =====
    alpha0_dB: float = -50.0   # Công suất kênh tham chiếu (dB) - communication
    beta0_dB: float = -49.0    # Công suất kênh tham chiếu (dB) - sensing
    N0_dBm_Hz: float = -170.0  # Mật độ phổ nhiễu (dBm/Hz)

    # ===== Các tham số truyền dẫn =====
    Pt_dBm: float = 20.0    # Công suất phát (dBm)
    B: float = 10e6         # Băng thông tổng (Hz) = 10 MHz
    Gp_factor: float = 0.1  # Gp = 0.1 * B (signal processing gain)

    # ===== Các tham số UAV và vùng =====
    Vmax: float = 30.0      # Vận tốc tối đa (m/s)
    H: float = 200.0        # Độ cao bay (m)
    Th: float = 1.5         # Thời gian hover tại HP (s)
    Tf: float = 1.0         # Thời lượng mỗi time slot (s)
    Lx: float = 1500.0      # Chiều dài vùng (m)
    Ly: float = 1500.0      # Chiều rộng vùng (m)

    # ===== Các tham số thuật toán =====
    delta_omega_steps: int = 100  # Số bước line search ω (Δω = 1/100)
    mu: int = 5                    # Cứ mỗi μ waypoints đặt 1 HP
    a: float = 200.0               # Hệ số môi trường nhiễu trong CRB (eq. 16)

    # ===== Các tham số MSTD =====
    Nstg: int = 60          # Số time slot trong mỗi stage
    t_lse: float = 1.0      # Hệ số scaling cho log-sum-exp (eq. 37-38)

    # ===== Vị trí trạm sạc =====
    base_station: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )

    # ----- Các tham số dẫn xuất (tính sau khi khởi tạo) -----
    @property
    def Pt(self) -> float:
        """Công suất phát dạng tuyến tính (W)."""
        return 10 ** ((self.Pt_dBm - 30) / 10)

    @property
    def alpha0(self) -> float:
        return 10 ** (self.alpha0_dB / 10)

    @property
    def beta0(self) -> float:
        return 10 ** (self.beta0_dB / 10)

    @property
    def Gp(self) -> float:
        return self.Gp_factor * self.B

    @property
    def noise_power(self) -> float:
        """σ_0^2 = N0 * B (W), N0 từ dBm/Hz chuyển sang W/Hz."""
        N0_W_Hz = 10 ** ((self.N0_dBm_Hz - 30) / 10)
        return N0_W_Hz * self.B

    @property
    def ground_area_corner(self) -> np.ndarray:
        """Góc trên phải của vùng, dùng làm điểm đến của quỹ đạo hypothetical."""
        return np.array([self.Lx, self.Ly])


# Cấu hình mặc định dùng xuyên suốt
DEFAULT = SimulationConfig()


if __name__ == "__main__":
    cfg = DEFAULT
    print(f"Pt (linear) = {cfg.Pt:.4f} W")
    print(f"alpha0 (linear) = {cfg.alpha0:.2e}")
    print(f"beta0 (linear) = {cfg.beta0:.2e}")
    print(f"Gp = {cfg.Gp:.2e} Hz")
    print(f"Noise power σ0^2 = {cfg.noise_power:.4e} W")
    print(f"Puav hovering (V=0) = P0+PI = {cfg.P0 + cfg.PI} W")

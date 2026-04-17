# ISAC-UAV: Tái hiện bài báo "ISAC from the Sky"

Tái hiện bằng Python bài báo của Jing, Liu, Masouros, Zeng (IEEE TWC 2024):
*"ISAC from the Sky: UAV Trajectory Design for Joint Communication and Target Localization"*.

## Cấu trúc mã nguồn

| File | Nội dung |
|------|----------|
| `config.py` | Bảng I - tất cả tham số mô phỏng (P₀, P_I, U_tip, v₀, H, L_x, L_y, …) |
| `system_model.py` | Công suất bay UAV (eq. 33), thông lượng truyền thông (eq. 7), CRB sensing (eq. 17-31) |
| `mle_estimator.py` | Ước lượng ST qua MLE + grid search + Nelder-Mead refine (eq. 52-53) |
| `trajectory_optimizer.py` | Algorithm 1: ascent direction search + Q''(j) qua CVXPY với SCA + DPP caching |
| `mstd.py` | Algorithm 2: Multi-Stage Trajectory Design |
| `baselines.py` | Các scheme so sánh: Straight, Circle, Separate (2 UAV riêng) |
| `simulations.py` | Chạy các kịch bản sinh `.pkl` trong `results/` |
| `plots.py` | Vẽ các figure từ `.pkl` ra `plots/` |

## Các điểm kỹ thuật chính

### 1. CRB sensing
Dựa trên Θ_a, Θ_b, Θ_c (eq. 26-28) và công thức đóng (eq. 31):

`ψ_k^s = (Θ_a + Θ_b) / (Θ_a Θ_b − Θ_c²)`

### 2. Gradient giải tích của f(S_j)
Chain rule đi qua softmin/softmax weights để ra gradient đóng.
Mỗi waypoint có 2 thành phần:
- Comm rate đóng góp: ∂R/∂s (dạng đóng)
- CRB đóng góp (chỉ HP): ∂ψ_k^s/∂hp qua Θ_a, Θ_b, Θ_c

So với finite-difference: **cosine similarity = 0.998**.

### 3. SCA cho ràng buộc năng lượng
Hai ràng buộc phi lồi (eq. 46, 47) được xấp xỉ tuyến tính quanh điểm hiện tại theo công thức Taylor bậc 1. Biến phụ δ và ξ xử lý 1/||v||² trong công suất induced.

### 4. DPP-compliant CVXPY
`_QppCache` class: compile 1 lần, re-use Parameters giữa các stages có cùng Nf. Dùng Clarabel solver.

### 5. MSTD (Algorithm 2)
- Initial: 3 điểm sensing gần trạm sạc → coarse estimate
- Mỗi stage: giải P'_1(j) → "bay" → cập nhật MLE
- Ending stage: tính Nlst từ E_remaining với safety margin 15%
- Khởi tạo trajectory feasible (thử các tốc độ khác nhau)

## Kết quả tái hiện

### Fig. 3 - Convergence
Hội tụ sau 9 iterations. Objective tăng đơn điệu từ -0.38 → +0.14.
Ψc dao động nhẹ rồi ổn định ở 1.69 Gbits. Ψs giảm mạnh ở iter 1 rồi ổn định ở 0.17 m².

### Fig. 4 - MSE & CRB qua stages
CRB giảm ~10× qua 4 stages (0.2 → 0.035 m²). MSE theo sát CRB.

### Fig. 5 - Performance vs E_tot
| E_tot (kJ) | ISAC Ψc (Gbits) | ISAC Ψs (m²) | Separate Ψs | Straight Ψs | Circle Ψs |
|-------|-------|-------|-------|-------|-------|
| 20    | 2.02  | 0.17  | 1.14  | 0.81  | 0.44  |
| 40    | 4.67  | 0.061 | 0.12  | 0.68  | 0.17  |
| 60    | 6.03  | 0.041 | 0.13  | 0.36  | 0.09  |

**ISAC+BA luôn dẫn đầu** cả 2 trục. Straight kém nhất sensing (đúng như bài báo).

### Fig. 6 - Quỹ đạo theo E_tot
- E=20 kJ: UAV bay tập trung khu vực 2 STs (góc trái trên)
- E=60 kJ: UAV phủ rộng cả STs và CUs, hover dày đặc ở cả 2 vùng

### Fig. 9 - Tradeoff η
- η=0.1 (comm-priority): Ψc=5.94 Gbits, Ψs=0.24 m² - UAV bay thẳng tới CUs
- η=0.9 (sens-priority): Ψc=3.49 Gbits, Ψs=0.058 m² - UAV xoắn quanh STs
- CRB biến thiên ~4.2× trong khi Ψc biến thiên ~1.7×, khớp với nhận định "η ảnh hưởng sensing nhiều hơn comm"

## Cách chạy

```bash
# Chạy tất cả mô phỏng
python3 simulations.py all

# Hoặc chỉ 1 figure cụ thể
python3 simulations.py 3   # Fig. 3
python3 simulations.py 4   # Fig. 4
python3 simulations.py 5   # Fig. 5 + 6 (cùng data)
python3 simulations.py 9   # Fig. 9

# Vẽ biểu đồ sau khi có data
python3 plots.py all
```

## Thời gian chạy (Clarabel solver)
- Fig. 3: ~60 s
- Fig. 4: ~65 s (5 MC trials × 4-5 stages × ~3 s/stage)
- Fig. 5: ~85 s (3 E_tot × 5 scheme)
- Fig. 9: ~55 s (3 η)

Tổng ~5 phút trên máy tiêu chuẩn.

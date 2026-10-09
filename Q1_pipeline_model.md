# Mô hình động lực của thiền: sửa sai, khả năng xác định tham số, thiết kế thí nghiệm tối ưu và điều khiển vòng kín

**Pipeline nghiên cứu (chỉ dùng hệ thống hiện có, không tải dữ liệu ngoài)**
Phiên bản 1.3 — cập nhật 09/10/2026 (phần máy hoàn tất)
Loại bài: lý thuyết / tính toán. Dữ liệu = thí nghiệm *in silico* sinh từ mô hình, có seed, tái tạo 100%.
Thay thế tạm thời cho `Q1_pipeline.md` (hướng EEG, hoãn vì dung lượng tải).

**Trạng thái cổng (09/10/2026):** M0 ✅ · M1 ✅ (100% ô khớp, trung vị RMSE/CRB 0.98) · M2 ✅. Toàn bộ 8 hình + 3 bảng đã xuất (`outputs/paper/model/manifest.json`: 0 bỏ qua). 85 test pass. **Phần máy đã xong; còn phần viết.**
Kiểm tra bất kỳ lúc nào: `python main.py gates`. Kết quả số: `outputs/results/model/`; hình/bảng: `outputs/paper/model/`.

---

## Mục lục

1. Luận điểm và định vị
2. Những gì hệ thống đã có
3. Mô hình
4. Gói kết quả 1 — Sửa mô hình gốc (cổng **M0**)
5. Gói kết quả 2 — Khả năng xác định tham số (Bảng 2)
6. Gói kết quả 3 — Thiết kế thí nghiệm tối ưu (cổng **M1**)
7. Gói kết quả 4 — Điều khiển vòng kín (cổng **M2**)
8. Gói kết quả 5 — Phân tích độ nhạy toàn cục
9. Dữ liệu in silico và quy tắc sinh
10. Cổng kiểm tra — định nghĩa, lệnh chạy, trạng thái
11. Cấu trúc mã
12. Lộ trình
13. Khung bài báo và nơi nộp
14. Câu hỏi phản biện
15. Quy tắc liêm chính
16. Checklist

---

## 1. Luận điểm và định vị

> Các mô hình động lực của thiền (DMN, tích hợp thông tin, "chất lượng ý thức") đang được đề xuất mà chưa hỏi: (i) mô hình có nhất quán toán học không, (ii) tham số có xác định được từ các thiết kế thí nghiệm hiện dùng (probe trải nghiệm vài phút một lần, phiên 45–90 phút) không, và (iii) điều khiển vòng kín (neurofeedback) làm được gì khi tham số chỉ biết gần đúng. Bài này trả lời cả ba cho một mô hình cụ thể, và cho ra **khuyến nghị thiết kế thí nghiệm định lượng** mà người làm thực nghiệm dùng được ngay.

Kết quả nào cũng đăng được: mô hình xác định được → kèm cỡ mẫu cần thiết; không xác định được → chỉ ra tham số nào và thiết kế nào cứu được.

**Định vị**: các mô hình tính toán về thiền hiện nay (active inference, "Thoughtseeds" 2025, mô hình mind-wandering dạng chuyển trạng thái) tập trung mô tả hiện tượng. Chưa có bài nào phân tích identifiability hay thiết kế thí nghiệm tối ưu cho dữ liệu experience-sampling khi thiền. Đây là khoảng trống của bài.

**Không tuyên bố**: không nói gì về giá trị tham số của não thật; mọi tham số là giá trị danh nghĩa + dải quét.

### Ba đóng góp

1. **Định lý** (cổng M0): mô hình gốc (eq. 52–54) có QoC\* ≡ 0 với mọi tham số (QoC là bộ lọc thông cao của đạo hàm); nghiệm dừng in trong bài là sai. Đề xuất mô hình không gian trạng thái M1 có trạng thái dừng khác 0, với mô-men dừng dạng đóng.
2. **Identifiability + thiết kế** (Bảng 2, cổng M1): phân tích cấu trúc (Rothenberg trên Fisher chính xác) và thực hành (CRB, profile likelihood, phục hồi tham số) → bản đồ "cần bao nhiêu người, phiên dài bao lâu, probe dày bao nhiêu" để ước lượng từng tham số với sai số ≤ 20%.
3. **Điều khiển** (cổng M2): COM viết lại thành bài toán LQG; chỉ ra khẳng định "bang-bang" của bài gốc chỉ đúng với hàm mục tiêu tuyến tính và ràng buộc hộp; định lượng mức suy giảm hiệu quả khi dùng tham số ước lượng thay vì tham số thật.

---

## 2. Những gì hệ thống đã có

| Thành phần | File | Trạng thái | Dùng cho |
|---|---|---|---|
| Mô hình gốc M0 + nghiệm dừng đúng/sai | `bqi/dmn_ode.py` (`simulate`, `steady_states`, `steady_states_paper`) | Có, có test | M0 |
| Mô hình M1: rời rạc hoá chính xác (Van Loan), mô-men dừng (Lyapunov), `probe_idx` tuỳ ý, trạng thái đầu rút từ phân phối dừng | `bqi/dmn_ode.py` (`StateSpaceDMN`) | Có, có test | M0–M2 |
| Kalman log-likelihood tuần tự (đối chiếu bản ma trận đến 1e-11) + MLE + profile | `eeg/model_fit.py` | Có, có test | M1 |
| **Fisher kỳ vọng chính xác**, CRB, phổ Rothenberg | `model/fisher.py` | Có, có test (khớp Kalman 1e-13; khớp score Monte Carlo 3.000 phiên ≤ 5%) | Bảng 2, M1 |
| P1–P5 | `model/theory.py` | Có, có test | M0 |
| Identifiability cấu trúc + thực hành | `model/identifiability.py` | Có, có test | Bảng 2 |
| Lưới thiết kế CRB + phục hồi MLE song song + Ds-optimal lịch probe | `model/design.py` | Có, có test | M1 |
| LQG (điểm cân bằng, Kalman, LQR), bang-bang, chi phí dừng dạng đóng, độ bền | `model/control.py` | Có, có test | M2 |
| Sobol/Saltelli tất định | `model/sensitivity.py` | Có, có test | Gói 5 |
| Xuất hình/bảng cho bài | `experiments/run_model_paper.py` | Có, có smoke test | Tất cả |
| CLI `python main.py model …`, `gates`; dashboard trang "Bài Q1" (tự lưu mọi khám phá) | `main.py`, `dashboard/views/model.py` | Có | Tái tạo |
| Cổng M0–M2 cưỡng chế trong mã | `eeg/gates.py` | Có | — |

Bộ test: 83 test (`tests/test_model.py` 18 test cho `model/`), tất cả pass (2 test EEG tự bỏ qua khi không có `mne`).

**Không dùng** (xem mục 15): toàn bộ CSV `*_paper_reference`, `dmn_meta_analysis_*`, `ode_params_synthetic_subjects`, `pci_synthetic_subjects`, `kuramoto_*`, `quantum_*`.

---

## 3. Mô hình

**M0 (bài gốc, eq. 52–54)**
```
dA/dt   = −k_inh A + k_spont ξ(t)
dΦ/dt   = k_int (1 − A/A_max) − k_d Φ
dQoC/dt = α dΦ/dt − β dA/dt − γ QoC
```

**M1 (đề xuất, `StateSpaceDMN`)** — tốc độ theo phút:
```
dA   = −k_inh (A − A0) dt + σ_A dW_A
dΦ   = [k_int (1 − A/A_max) − k_d Φ] dt + σ_Φ dW_Φ
QoC  = α Φ − β A + γ0
Quan sát mỗi epoch Δt:   y = [A, Φ] + η,   η ~ N(0, diag(r_A², r_Φ²))
Probe mỗi T_p phút:      rating = QoC + ε,  ε ~ N(0, r_Q²)
```
M1-u (Gói 4): thêm đầu vào điều khiển vào phương trình A: `−k_inh (A − A0) dt − b u dt`, |u| ≤ u_max; ma trận đầu vào rời rạc chính xác B = M⁻¹(F − I)b.

**Giá trị danh nghĩa**: k_inh 0.2, A0 0.6, σ_A 0.1, k_int 0.081, k_d 0.034, σ_Φ 0.05, α 2.34, β 1.87, γ0 0, r_A = r_Φ 0.1, r_Q 0.3. Đây là điểm làm việc, không phải ước lượng; mọi kết luận kiểm lại trên dải quét ở Gói 5. Nghiệm dừng danh nghĩa của M1: A\* = 0.6, Φ\* = 0.953, QoC\* = 1.108, Var_∞(QoC) = 0.477.

---

## 4. Gói kết quả 1 — Sửa mô hình gốc · cổng **M0 ✅ PASS** (09/10/2026)

Lệnh: `python main.py model theory` (vài giây). Kết quả: `outputs/results/model/theory_results.json`, `outputs/gates/M0.json`. Hình/bảng: T1, F1, F2.

| Mệnh đề | Nội dung | Kiểm chứng số (đã chạy) |
|---|---|---|
| P1 | Với E[ξ] = 0: A\* = 0, Φ\* = k_int/k_d, QoC\* = 0 với mọi tham số | Phần dư ODE tại nghiệm sửa = 0 (máy) |
| P2 | Hàm truyền từ (αΦ − βA) sang QoC là `s/(s+γ)` (thông cao) → độ lợi DC = 0 | **Trên ODE thật**, không trên công thức: bước nhảy ξ → mức drive −3.77 nhưng QoC → −8·10⁻⁹ (suy giảm > 10⁸); kích thích sin ở ω = 0.05, 0.2, 1.0 cho tỉ số biên độ khớp \|H(jω)\| với sai số 0.14%, 0, 0 |
| P3 | Eq. 55–57 của bài không thoả phương trình dừng | Phần dư ‖·‖ = 5.28 (dA: −3.4·10⁻³, dQoC: γ·27.8) |
| P4 | M1: mean dừng −M⁻¹u, hiệp phương sai dừng nghiệm Lyapunov MP + PMᵀ + GGᵀ = 0, QoC\* ≠ 0 | Phần dư Lyapunov = 0 (máy); QoC\* = 1.108, Var = 0.477 |
| P5 | Phân phối dừng của QoC trong M1 là Gauss N(QoC\*, hᵀPh) | KS trên 2.000 mẫu **thưa** (mỗi 5 hằng số thời gian, tự tương quan đo được −0.03) **gộp từ 5 chuỗi độc lập** (seed 42–46): KS = 0.020, p = 0.42; sai số mean 0.005, sd 0.007. Lý do gộp nhiều chuỗi: một đường đơn có thể lệch 2–3 SE suốt hàng trăm mẫu thưa (seed 42 đơn lẻ cho z = 3.5), nên cổng không được phụ thuộc vận may của một seed; kiểm trên 12 seed: sd(z) = 0.99, không thiên lệch |

Việc còn lại cho gói này: **viết Appendix A** (chứng minh P1–P5 bằng đại số tuyến tính + Laplace). Không cần máy.

---

## 5. Gói kết quả 2 — Khả năng xác định tham số (Bảng 2, đã có)

Lệnh: `python main.py model identifiability` (~1 phút). Kết quả: `table2_identifiability.csv`, `identifiability_results.json`.

### 5.1 Cấu trúc — kết quả
Định lý Rothenberg (1971) trên Fisher kỳ vọng chính xác của phiên 90 phút, probe/1 phút (γ0 đặt 0.3 để không che đối xứng):

| Cấu hình quan sát | Hạng | Vắng mặt | Hướng không xác định (phải cố định một tham số) |
|---|---|---|---|
| (a) chỉ rating | 6/10 | r_A, r_Φ | 4 hướng; cố định A0, α, σ_A, k_int |
| (b) rating + y_A | 10/11 | r_Φ | {α: +0.71, k_int: −0.62, σ_Φ: −0.34} — co giãn Φ → cΦ (k_int, σ_Φ → c·, α → α/c), **kiểm chứng giải tích** |
| (c) rating + y_Φ | 10/11 | r_A | {γ0: +0.77, β: +0.49, A0: +0.28, k_int: +0.25, σ_A: −0.17} — dịch mức A |
| (d) đủ cả ba | 12/12 | — | không |

Kết luận cho bài: **cần cả hai kênh EEG (y_A và y_Φ) mới ước lượng được đủ 12 tham số**; chỉ có rating thì không tách được α, β khỏi các hằng số động lực.

### 5.2 Thực hành — CRB tương đối (N = 12, 45 phút, probe/2 phút; cấu hình d)
k_inh 0.155 · A0 0.030 · σ_A 0.049 · k_int 0.175 · k_d 0.178 · σ_Φ 0.057 · α 0.037 · β 0.072 · r_A 0.016 · r_Φ 0.014 · r_Q 0.050 (γ0 = 0 nên báo SE = 0.13). Các hằng số tốc độ (k_inh, k_int, k_d) khó đo nhất.

Còn lại: Hình 3 (profile likelihood k_inh, k_int, k_d) — `python main.py model figures --profiles` (~30 phút).

---

## 6. Gói kết quả 3 — Thiết kế thí nghiệm · cổng **M1 ✅ PASS** (09/10/2026)

Lệnh: `python main.py model design --seed 42` (~7 giờ trên máy 4 nhân với `OMP/MKL_NUM_THREADS=1`; đã chạy ở máy 2, kết quả chép về). Đầu ra: `design_grid_crb.csv` (144 ô), `design_grid_mle.csv` (24 ô × 5 lần), `design_results.json`, `gates/M1.json`; Hình 4, 5; Bảng 3.

| Biến thiết kế | Giá trị |
|---|---|
| Số người N | 6, 12, 24, 48 |
| Độ dài phiên | 20, 45, 90 phút |
| Khoảng probe | 0.5, 1, 2, 4 phút |
| Nhiễu EEG r_A = r_Φ | 0.05, 0.1, 0.2 |
| Cấu hình | d (đủ ba kênh) |

**Cách tính**: CRB từ Fisher chính xác **đủ 12 tham số cùng lúc** (MLE ước lượng cả 12 nên cận dưới cũng phải vậy) × N; sai số MLE đo bằng RMSE **cùng thang** với CRB (log cho tham số dương, tương đối cho tham số tuyến tính).

**Kết quả đã có (từ CRB)**:
- Thiết kế kiểu ds001787 (12 người, 45 phút, probe/2 phút, r = 0.1): CRB(k_inh) = 15.5%, A0 3.0%, α 3.7%, β 7.2%. Thử nghiệm MLE trước đó (G3) cho k_inh 23% — khoảng cách này là đối tượng của M1.
- Thiết kế rẻ nhất đạt CRB ≤ 20% (Bảng 3): k_inh cần **24 người × 20 phút**; A0, α, β đạt ngay ở 6 người × 20 phút. → k_inh (tốc độ ức chế DMN) là tham số khó đo nhất.
- Lịch probe: đều là tốt nhất trong các họ đã thử (luỹ thừa p ∈ [0.4, 2.5], theo cặp cách 1–12 epoch); tuyên bố trong bài giới hạn ở "không thua lịch không đều thông dụng".

**Kết quả M1 (Hình 4)**: 24/24 ô đạt tiêu chí; trung vị RMSE/CRB = 0.98 (k_inh 0.97, A0 0.97, α 1.01, β 1.04); không ô nào vượt ×2; 5 tỉ số dưới 0.5 (thấp nhất 0.40) là nhiễu của RMSE từ 5 lần lặp, nằm *dưới* cận. Lo ngại "CRB lạc quan ở N nhỏ" không xảy ra (N = 6: tỉ số 0.9–1.3). Ô kiểu ds001787: RMSE(k_inh) 19.7% so với CRB 15.5% (tỉ số 1.27). ⇒ Bản đồ thiết kế từ CRB (Hình 5, Bảng 3) được phép dùng thay mô phỏng; không cần chạy `--n-starts 6`.

---

## 7. Gói kết quả 4 — Điều khiển vòng kín · cổng **M2 ✅ PASS** (09/10/2026; độ bền đã có)

Lệnh: `python main.py model control` (~1 phút; ~10 phút khi có `design_grid_mle.csv` để tính độ bền). Kết quả: `control_results.json`, `outputs/gates/M2.json`. Hình: F6 (đã có), F7 (chờ M1).

- **LQG** (certainty-equivalence, nguyên lý tách): điểm cân bằng (x_ss, u_ss) giải tích sao cho hᵀx_ss + γ0 = QoC_target; LQR trên độ lệch với chi phí (QoC − target)² + λu², λ = 0.1; Kalman trên y = x + η mỗi epoch; u = u_ss − K(x̂ − x_ss), |u| ≤ 2.
- **Bang-bang**: u = ±u_max theo dấu (target − QoĈ), **cùng bộ ước lượng trạng thái** (so sánh công bằng).
- **Chi phí dừng dạng đóng**: Lyapunov trên trạng thái mở rộng [x, x̂] (không cắt u) — dùng cho Gói 5 và để đối chiếu mô phỏng.

**Kết quả đã có** (mục tiêu 2.0, 270 epoch, 20 trial, cùng nhiễu):

| Chính sách | Chi phí phiên |
|---|---|
| LQG | **0.0375** |
| Bang-bang | 0.555 |
| Không điều khiển | 1.114 |
| LQG dừng: dạng đóng / mô phỏng dài có burn-in | 0.0336 / 0.0346 (lệch 3%) |

u_ss = 0.024 (mục tiêu 2.0 chỉ cao hơn QoC\* = 1.11 nên cần đầu vào trung bình nhỏ).

Lưu ý khi viết: so sánh trên chi phí bậc hai là hàm mục tiêu của chính LQG; bài phải nói rõ bang-bang giải bài toán khác (max QoC(T), ràng buộc hộp) và báo thêm hai chỉ số trung lập đã có trong kết quả: MSE bám mục tiêu và RMS của u.

**Độ bền (Hình 7, đã có)** — LQG (độ lợi **và** bộ lọc) thiết kế trên từng bộ tham số *thực sự ước lượng được* ở M1, chạy trên hệ thật **của chính ô đó** (cùng mức nhiễu EEG r; bản đầu dùng r cố định 0.1 cho mọi ô và gán nhầm 15% mất mát do lệch nhiễu cho sai số ước lượng — đã sửa, có test). Kết quả: mất mát trung vị giảm theo N và độ dài phiên; ô sạch (N ≥ 24, phiên ≥ 45 phút, r = 0.05) 0.2–0.7%; ô kiểu ds001787 2.6% (q90 11%); thiết kế nhỏ (N = 6, 20 phút) có thể mất 38% (q90 55%), N = 12, 20 phút, r = 0.2 có q90 98% — tức thiết kế không đủ để ước lượng thì neurofeedback dựa trên nó cũng không dùng được. Nối thẳng Gói 3 với ứng dụng.

---

## 8. Gói kết quả 5 — Phân tích độ nhạy toàn cục (bản cho bài đã có: n_base 256, QMC, log, bootstrap)

Lệnh: `python main.py model sensitivity --n-base 256` (~30 phút với 3 tiến trình và `OMP/MKL_NUM_THREADS=1`); tính lại không mô phỏng: `--from-saved`.

**Kết quả (Hình 8, ST với CI 95%)**: Var(QoC): α 0.36 [0.30, 0.42], σ_Φ 0.26, A0 0.19 (hội tụ: max|ΔST| 0.013). **CRB(k_inh): A0 0.58 [0.44, 0.73], CV(A) 0.36 [0.27, 0.46], còn chính k_inh chỉ 0.13 [0.10, 0.17]** — đo được tốc độ ức chế DMN hay không do mức nền và độ ồn của A quyết định, không do giá trị k_inh (hội tụ 0.076). Hiệu quả LQG: β 0.29, α 0.29, A0 0.23, không tham số nào thống trị (0.030). QoC\* (thang thô): A0 0.44 [0.23, 0.88], k_d 0.36, k_int 0.32 — thứ tự rõ nhưng CI rộng, hội tụ 0.10; trong bài chỉ tuyên bố "A0, k_d, k_int chi phối; α, β không đáng kể"; muốn CI hẹp hơn chạy `--n-base 512` (~1 giờ).

- Sobol (Saltelli S1 / Jansen ST, NumPy), 8 tham số, dải ×0.25–×4 log-uniform quanh danh nghĩa; **mẫu quasi-ngẫu nhiên Sobol có xáo trộn** (scipy `qmc`, n_base là luỹ thừa 2). Mọi đầu ra **tất định** (dạng đóng hoặc Fisher chính xác): QoC\*, Var_∞(QoC), CRB(k_inh) cho thiết kế tham chiếu, tỉ số chi phí LQG/không điều khiển.
- **Thang log** cho ba đầu ra dương (Var, CRB, hiệu quả): trên thang thô chúng biến thiên hàng chục lần nên vài mẫu cực đoan chiếm hết phương sai — chỉ số bị cắt ở 1.00 và S1 ≪ ST giả tạo (đã thấy ở lần chạy thô đầu tiên: A0→QoC\* = 1.00, cv_A→Var = 1.00). QoC\* giữ thang thô vì có thể âm.
- **Khoảng tin cậy 95% bootstrap** (B = 1.000, lấy lại mẫu theo hàng) cho mọi S1, ST; **kiểm tra hội tụ** bằng chỉ số tính trên nửa đầu mẫu (báo max|ΔST|).
- **Lưu toàn bộ giá trị đánh giá thô** (`sobol_evaluations.csv`): mọi phân tích lại chạy bằng `model sensitivity --from-saved` trong vài giây, không mô phỏng lại.
- Nhiễu của A tham số hoá bằng **hệ số biến thiên dừng** CV_∞(A) = σ_A/(A0√(2k_inh)) ∈ [0.05, 0.4] thay vì σ_A: quét σ_A độc lập ×4 cùng k_inh ×0.25 làm A_DMN < 0 ~30% thời gian (ngoài miền [0, A_max]); với CV ≤ 0.4, P(A < 0) ≤ 0.6% (đã kiểm trên 200 mẫu).

Đầu ra: Hình 8.

---

## 9. Dữ liệu in silico và quy tắc sinh

**Thông số thống nhất (`model/config.py`)** — mọi gói, script xuất hình, dashboard và test cùng đọc: seed 42; epoch 10 s; tham số danh nghĩa = mặc định `StateSpaceDMN` (có kiểm tra tự động); **thiết kế tham chiếu** N = 12, 45 phút, probe/2 phút, r = 0.1, cấu hình d (dùng cho Bảng 2, ô CRB tham chiếu, đầu ra Sobol, chân trời điều khiển 270 epoch); phân tích cấu trúc 90 phút, probe/1 phút, γ0 = 0.3; lưới N {6,12,24,48} × phiên {20,45,90} × probe {0.5,1,2,4} × r {0.05,0.1,0.2}, 24 ô đại diện × 5 lần, 2 điểm xuất phát; điều khiển mục tiêu 2.0, |u| ≤ 2, b = 1, Q = 1, λ = 0.1, 20 trial; Sobol ×0.25–×4, CV(A) ∈ [0.05, 0.4], n_base 64 (bài 256); ngưỡng 20%, M1 tỉ số [0.5, 2] ở ≥ 80% ô, M2 ±20%. Đổi một giá trị ở đây là đổi ở mọi nơi, và phải chạy lại `model all`.

- Mọi dữ liệu sinh bởi `StateSpaceDMN.simulate` với seed = hàm xác định của (ô thiết kế, lần lặp); trạng thái đầu rút từ phân phối dừng (khớp giả định của Kalman và Fisher).
- Kết quả số: `outputs/results/model/` (< 50 MB, không lưu chuỗi thô — tái tạo bằng seed). Khám phá tương tác trên dashboard: `outputs/results/model/explorations/` (tách biệt với `outputs/results/bqi/explorations/` của bài BQI gốc).
- Không dùng bất kỳ CSV nào có sẵn trong `datasets/generated/`.
- Chi phí tính: Fisher chính xác ~2 s/thiết kế → lưới 144 ô ~2 phút; một fit MLE ≈ 100 s (12 người × 270 epoch) → 24 ô × 5 lần, 3 tiến trình ≈ 5–8 giờ.

---

## 10. Cổng kiểm tra — định nghĩa, lệnh chạy, trạng thái

| Cổng | Điều kiện PASS | Lệnh | Trạng thái | Nếu FAIL |
|---|---|---|---|---|
| **M0** | P1–P5 đều đúng: phần dư < 1e-6; P2 suy giảm DC > 10⁴ và sai số biên độ < 1%; P5 KS p > 0.01 trên chuỗi thưa | `model theory` | ✅ PASS 09/10 | Sửa chứng minh trước khi viết |
| **M1** | Ở ≥ 80% trong 24 ô đại diện, ≥ nửa số tham số chính có RMSE/CRB ∈ [0.5, 2] | `model design` | ⏳ dừng ở 7/24 | (1) Chạy lại `--n-starts 6`: nếu RMSE giảm → lỗi tối ưu, dùng kết quả mới. (2) Nếu không đổi → CRB lạc quan ở N nhỏ: giữ kết quả, bản đồ thiết kế báo rõ vùng tin cậy; đây là kết quả khoa học, không phải lỗi |
| **M2** | LQG (tham số thật) rẻ hơn bang-bang và không điều khiển; chi phí dừng dạng đóng khớp mô phỏng có burn-in trong 20% | `model control` | ✅ PASS 09/10 (lệch 3%) | Kiểm Riccati / rời rạc hoá |

Phụ thuộc khai báo trong `eeg/gates.py`: M1 ← M0, M2 ← M1. **Hiện các bước `model …` chưa gọi `gates.require`**, nên chạy được theo thứ tự bất kỳ (vì vậy M2 đã PASS trước M1); ràng buộc thực tế duy nhất là **phần độ bền của Gói 4 chỉ chạy khi có `design_grid_mle.csv`** (kết quả M1). Mỗi cổng ghi `outputs/gates/M*.json` (timestamp UTC + chi tiết số); `python main.py gates` in trạng thái. Trước khi nộp, chạy lại toàn bộ theo đúng thứ tự `model all` để M0 → M1 → M2 có timestamp tăng dần.

---

## 11. Cấu trúc mã

```
bqi/dmn_ode.py            # M0 (ODE), M1 (StateSpaceDMN): discretize, stationary_*, simulate(probe_idx)
eeg/model_fit.py          # kalman_loglik (tuần tự), fit (L-BFGS-B, n_starts), profile_likelihood
eeg/gates.py              # record/require/status cho G0–G3, M0–M2
model/
  config.py               # NGUỒN DUY NHẤT của mọi thông số (seed, epoch, danh nghĩa, thiết kế tham chiếu,
                          # lưới, điều khiển, Sobol, ngưỡng cổng); mỗi run() ghi config_snapshot.json cạnh kết quả
  fisher.py               # Design, session_moments, fisher_exact, crb, identifiability_spectrum
  theory.py               # P1–P5, verify_transfer_on_ode, gate_M0
  identifiability.py      # structural_identifiability (Rothenberg), cramer_rao_bound, identifiability_table
  design.py               # design_grid_crb, design_grid_mle (song song, n_reps/n_starts), doptimal_probe_schedule, gate_M1
  control.py              # state_space_with_control, equilibrium, solve_lqg, simulate_policy,
                          # lqg_stationary_cost, compare_policies, control_efficiency_vs_param_error, gate_M2
  sensitivity.py          # saltelli_sample, _evaluate_model (tất định), sobol_indices, run(n_base)
experiments/run_model_paper.py   # T1–T3, F1–F8 → outputs/paper/model (PNG 300 dpi + PDF, CSV + LaTeX, manifest.json)
dashboard/views/model.py  # trang "Bài Q1": giải thích vs BQI, cổng, chạy bước, 5 tab, thư viện hình, kết quả đã lưu
tests/test_model.py       # 18 test
```

CLI:
```bash
python main.py model theory | identifiability | design | control | sensitivity | figures | all
       [--seed 42] [--fast] [--n-base 256] [--n-reps 5] [--n-starts 2] [--profiles]
python main.py gates
```
Phần `eeg/metrics/`, `eeg/preprocess.py`, `eeg/loaders/`, `preregistration/` thuộc hướng EEG: giữ nguyên, không dùng trong bài này.

---

## 12. Lộ trình (cập nhật)

| Việc | Trạng thái | Lệnh / ghi chú |
|---|---|---|
| Gói 1 + M0 | ✅ xong | — |
| Gói 2, Bảng 2 | ✅ xong | Hình 3 cần `figures --profiles` |
| Gói 3: lưới CRB, Hình 5, Bảng 3 | ✅ xong | — |
| Gói 3: phục hồi MLE, **M1**, Hình 4 | ✅ xong (máy 2, ~7 giờ) | — |
| Gói 4 + M2, Hình 6 | ✅ xong | — |
| Gói 4: độ bền, Hình 7 | ✅ xong | — |
| Gói 5 bản cho bài, Hình 8 | ✅ xong (n_base 256) | tuỳ chọn `--n-base 512` cho QoC\* |
| Xuất lại hình/bảng | ✅ xong (11 tệp, 0 bỏ qua) | — |
| Appendix A (chứng minh P1–P5, đối xứng config a/c) | ⏳ không cần máy | viết tay |
| Bản thảo | ⏳ | sau khi có M1 |
| Chạy lại từ đầu trên máy sạch (`model all --n-base 256`), nộp + preprint | ⏳ | — |

Lưu ý vận hành: đặt `OMP_NUM_THREADS=1` và `MKL_NUM_THREADS=1` trước mọi bước chạy song song (3 worker × 4 luồng MKL trên 4 nhân làm chậm 5–6 lần). Hai máy chạy `sensitivity` và `design` song song được vì không chia sẻ đầu vào; chép 5 file (`design_grid_*.csv`, `design_results.json`, `config_snapshot.json`, `gates/M1.json`) về máy chính rồi chạy `control` → `figures --profiles`.

---

## 13. Khung bài báo và nơi nộp

**Tên (mẫu)**: *Identifiability, optimal experimental design and closed-loop control for a stochastic model of meditation dynamics*

| Phần | Nội dung | Nguồn số liệu |
|---|---|---|
| Introduction | Mô hình động lực của thiền đang tăng; câu hỏi nhất quán – xác định – điều khiển chưa được hỏi; 3 đóng góp | — |
| Model | M0, lỗi của M0 (P1–P3), M1 (P4–P5) | T1, F1, F2 (M0) |
| Identifiability | Rothenberg theo 4 cấu hình; CRB; profile | T2, F3 |
| Experimental design | Bản đồ thiết kế (hình chính); khuyến nghị tối thiểu; CRB có tin được không | F5, T3, F4 (M1) |
| Closed-loop control | LQG vs bang-bang; độ bền theo chất lượng ước lượng | F6, F7 (M2) |
| Sensitivity | Sobol; giới hạn của kết luận | F8 |
| Discussion | Ý nghĩa cho thiết kế nghiên cứu experience-sampling và neurofeedback; giới hạn: tuyến tính, không fit dữ liệu thật, tối ưu lịch probe chỉ trên các họ lịch | — |
| Appendix | Chứng minh P1–P5, đối xứng co giãn, rời rạc hoá Van Loan, Kalman tuần tự, Fisher chính xác | — |

**Tuyên bố nên dùng**: "với probe mỗi 2 phút và phiên 45 phút, cần ≥ 24 người để ước lượng k_inh với sai số ≤ 20% (CRB); A0, α, β đạt mức này ngay với 6 người × 20 phút".
**Tránh**: mọi câu dạng "thiền làm tăng ý thức / Φ"; mọi giá trị tham số như sự thật về não; "lịch probe đều là tối ưu" (chỉ "không thua các lịch không đều đã thử").

**Nơi nộp** (kiểm tra quartile trên Scimago năm hiện hành trước khi chọn): *Cognitive Neurodynamics*, *Journal of Mathematical Biology*, *Mathematical Biosciences*, *Biological Cybernetics*, *Chaos, Solitons & Fractals*.

**Đánh giá thẳng**: bài lý thuyết không có dữ liệu thật có khả năng Q1 thấp hơn bài thực nghiệm. Điểm cứu là Gói 3 (khuyến nghị thiết kế cụ thể, dùng được ngay) và Gói 4 (nối với neurofeedback).

---

## 14. Câu hỏi phản biện

| Câu hỏi | Trả lời |
|---|---|
| Mô hình không fit dữ liệu thật thì có ý nghĩa gì? | Bài trả lời câu hỏi *trước* dữ liệu: thiết kế nào đủ để fit. Đây là bước bắt buộc trước khi chạy thí nghiệm, và là thứ thiếu trong tài liệu |
| Tham số danh nghĩa lấy đâu ra? | Là điểm làm việc; mọi kết luận kiểm lại trên dải ×0.25–×4 (Gói 5) |
| Vì sao mô hình tuyến tính? | Cho phép likelihood chính xác, Fisher chính xác, LQG chính xác; phi tuyến là hướng tiếp theo, nêu trong Limitations |
| QoC đo thế nào? | Bằng rating probe có nhiễu r_Q; Bảng 2 cho thấy chỉ rating thì không tách được α, β |
| CRB có tin được ở N nhỏ không? | Chính là cổng M1: so RMSE MLE với CRB ở 24 ô; báo cáo vùng CRB lạc quan nếu có |
| So LQG với bang-bang có công bằng không? | Cùng bộ ước lượng trạng thái; chi phí bậc hai là của LQG nên báo thêm MSE bám mục tiêu và RMS(u) riêng rẽ; bang-bang là lời giải của bài toán khác (eq. 69–72) |
| Khác các mô hình active inference về thiền? | Chúng mô tả cơ chế; bài này phân tích khả năng kiểm chứng và điều khiển của một lớp mô hình cụ thể |

---

## 15. Quy tắc liêm chính

- Không dùng làm dữ liệu hay bằng chứng: Table 4, 5, 6, 7, 11, 14 và eq. 50, 59 của bài BQI; các CSV trong `datasets/generated/` sinh từ chúng.
- Trích bài BQI chỉ như nguồn của mô hình M0 (phương trình), kèm chỉ ra lỗi ở Gói 1.
- Không trích "47 nghiên cứu, N = 12.847".
- Mọi số trong bài sinh từ `python main.py model all --seed 42 --n-base 256` rồi `run_model_paper.py`; seed ghi trong bài.

---

## 16. Checklist

- [x] P1–P5 kiểm chứng số (cổng M0 PASS)
- [ ] P1–P5 có chứng minh viết (Appendix A)
- [x] Bảng identifiability cho 4 cấu hình quan sát (Bảng 2)
- [x] Profile likelihood (Hình 3)
- [x] Bản đồ thiết kế + khuyến nghị tối thiểu (Hình 5, Bảng 3)
- [x] CRB khớp phục hồi mô phỏng (cổng M1 PASS, Hình 4)
- [x] LQG vs bang-bang (cổng M2 PASS, Hình 6)
- [x] Độ bền theo tham số ước lượng (Hình 7)
- [x] Sobol bản cho bài (n_base 256, QMC, log, bootstrap CI; Hình 8)
- [x] Một lệnh xuất toàn bộ hình/bảng (`model figures`)
- [ ] Mã + kết quả công bố (GitHub + Zenodo DOI)
- [x] Không còn số liệu nào từ bảng của bài BQI trong `model/` và `run_model_paper.py`
- [ ] Limitations: tuyến tính, không dữ liệu thật, QoC là cấu trúc giả định, tối ưu lịch probe chỉ trên họ lịch

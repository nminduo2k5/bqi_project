# Thiền nằm ở đâu trên trục ý thức? Kiểm định các dự đoán định lượng của mô hình BQI trên EEG mở

> **HOÃN (03/10/2026)** — cần tải ~90 GB dữ liệu EEG. Hướng đang làm: `Q1_pipeline_model.md` (chỉ dùng hệ thống hiện có).

**Pipeline nghiên cứu đã chốt và hướng dẫn triển khai**
Phiên bản 1.0 — tháng 10/2026
Mục tiêu: một bài Q1 (journal) chỉ cần laptop + dữ liệu mở, tái dùng tối đa `bqi_project`.

---

## Mục lục

1. Luận điểm và định vị
2. Câu hỏi và giả thuyết (đăng ký trước)
3. Dữ liệu
4. Tiền xử lý thống nhất
5. Bộ chỉ số
6. Thiết kế phân tích
7. Mô hình động lực (phần lý thuyết)
8. Kiểm định công cụ bằng dữ liệu tổng hợp
9. Cổng dừng (kill criteria)
10. Cấu trúc mã nguồn (gắn vào hệ thống hiện tại)
11. Lộ trình 16 tuần
12. Khung bài báo và nơi nộp
13. Câu hỏi phản biện và câu trả lời chuẩn bị sẵn
14. Những chỗ hay hỏng
15. Quy tắc liêm chính với bài BQI gốc
16. Checklist trước khi nộp

---

## 1. Luận điểm và định vị

### 1.1 Luận điểm trung tâm

> Nhiều mô hình lý thuyết (trong đó có BQI) khẳng định thiền là trạng thái ý thức *cao hơn* thức tỉnh: phức tạp hơn, đồng bộ hơn, tích hợp hơn. Khẳng định này chưa từng được kiểm tra bằng **cùng một pipeline** đặt thiền cạnh các tương phản ý thức đã được xác lập (ngủ, gây mê), và chưa tách khỏi nhiễu phổ (1/f, alpha). Chúng tôi dựng **trục mức-ý-thức** từ các tương phản không gây tranh cãi, rồi định vị thiền trên trục đó.

Kết quả nào cũng công bố được:
- Thiền vượt thức tỉnh theo hướng trục → ủng hộ dự đoán BQI.
- Thiền nằm cùng chỗ với thức tỉnh (tương đương, TOST) → bác dự đoán.
- Thiền lệch *vuông góc* với trục → thiền là trạng thái khác loại, không phải "cao hơn". Đây là kết quả thú vị nhất.

### 1.2 Khoảng trống

| Hướng đã có | Còn thiếu |
|---|---|
| LZc giảm theo độ sâu NREM, giảm khi gây mê (Schartner 2015; Aamodt 2021; nhiều bài 2022–2023) | Không đặt thiền vào cùng hệ quy chiếu |
| LZc và độ dốc 1/f cùng thay đổi theo trạng thái (Sci Rep 2023) | Ít bài tách độ phức tạp "độc lập phổ" khỏi phổ, đặc biệt cho thiền |
| Gamma tăng ở thiền giả (Braboszcz 2017, chính dữ liệu ds003969) | Chưa có chỉ số phức tạp/tích hợp và so với ngủ/gây mê |
| Cessation trong thiền làm *tăng* độ phức tạp (Shinozuka 2025, bioRxiv) | Mâu thuẫn với khung "phức tạp = mức ý thức"; cần phân xử trên dữ liệu lớn hơn |

### 1.3 Ba đóng góp tuyên bố

1. **Phương pháp**: trục mức-ý-thức hiệu chuẩn từ ngủ + gây mê, với chỉ số phức tạp đã khử phổ bằng surrogate; mã và dữ liệu dẫn xuất công bố.
2. **Thực nghiệm**: vị trí của thiền (3 truyền thống, 2 bộ dữ liệu độc lập) trên trục, đối chiếu với các dự đoán định lượng đã đăng ký trước.
3. **Mô hình**: sửa và kiểm định mô hình động lực DMN–Φ–QoC (chứng minh dạng gốc có QoC\* ≡ 0; đề xuất dạng không gian trạng thái ngẫu nhiên, fit trên chuỗi probe).

---

## 2. Câu hỏi và giả thuyết (đăng ký trước)

Chốt **trước khi tải dữ liệu thiền**: `python main.py preregister` khoá `preregistration/predictions.md` bằng SHA-256 + thời điểm vào `predictions.lock`; mã tải dữ liệu thiền từ chối chạy nếu chưa có lock hoặc file đã bị sửa sau khi khoá. Có thể đẩy thêm lên OSF khi nộp. Ngưỡng lấy từ bài BQI và tài liệu.

| Mã | Dự đoán | Nguồn | Phép thử chính | Ủng hộ khi |
|---|---|---|---|---|
| H1 | Độ phức tạp (LZc khử phổ) thiền > mind-wandering | BQI Table 4, Remark 6.1 | LMM paired, ds003969 | d > 0, CI 95% loại 0 |
| H2 | Thiền dịch về phía "cao" của trục mức-ý-thức | BQI §6–7 | Chiếu Δ(thiền−MW) lên trục (mục 6.3) | hệ số chiếu > 0, CI loại 0 |
| H3 | Đồng bộ pha theta tăng khi thiền (BQI: r_θ 0.42 → 0.71) | BQI Table 11 | Kuramoto r trên dữ liệu CSD, băng 4–8 Hz | Δr > 0 |
| H4 | Thiền làm "giảm nhiệt độ" (BQI Cor. 12.1) → phổ dốc hơn | BQI Cor. 12.1 | Exponent aperiodic (specparam) | exponent thiền > MW |
| H5 | Algebraic connectivity (Fiedler) tăng (BQI: 0.18 → 0.31) | BQI Table 11 | λ₂ của mạng wPLI theta | Δλ₂ > 0 |

**Mâu thuẫn được đăng ký trước:** H1 (phức tạp ↑) và H4 (nhiễu ↓, phổ dốc ↑) kéo ngược nhau vì phổ dốc thường đi kèm LZc thô thấp. Kết quả của surrogate (mục 5.2) phân xử.

Giả thuyết phụ (khám phá, báo cáo riêng): hiệu ứng tăng theo số năm kinh nghiệm thiền; khác biệt giữa 3 truyền thống.

---

## 3. Dữ liệu

| Vai trò | Bộ dữ liệu | Người | Kênh | Tương phản dùng | Ghi chú |
|---|---|---|---|---|---|
| Trục: gây mê | Chennu et al., Cambridge repository | 20 | 91 | baseline vs mild vs moderate vs recovery | Đã lọc 0.5–45 Hz, epoch 10 s → **không có gamma** |
| Trục: giấc ngủ | ANPHY-Sleep, OSF (10.17605/OSF.IO/R26FH) | 27 trên OSF (bài mô tả 29); phân tích chính 15 người đầu theo ID | 83 | wake vs N2 vs N3 vs REM | Zip ~3 GB/người, 82.5 GB tổng: tải từng người, đọc lười các cửa sổ 30 s đã chấm, lưu 19 kênh rồi xoá file gốc |
| Chính: thiền | ds003969 (Braboszcz/Delorme 2017) | ~64 (3 truyền thống + đối chứng) | 64 (Biosemi) | thiền vs instructed mind-wandering | Có nhóm đối chứng → kiểm tra trait vs state |
| Lặp lại: thiền | ds001787 (Brandmeyer & Delorme 2018) | 24 (12 expert, 12 novice) | 64, 256 Hz | thiền sâu vs MW theo probe | ≥ 30 probe/người → dùng cho mô hình (mục 7) |

Quy tắc:
- Kiểm tra số người, tần số lấy mẫu và dung lượng thực tế khi tải; ghi vào `data/manifest.csv`. Tải **theo từng subject**, không tải cả bộ.
- Băng phân tích chung: **1–40 Hz** (giới hạn bởi Chennu). Gamma 60–110 Hz chỉ phân tích khám phá trên ds003969.
- So sánh giữa bộ dữ liệu chỉ qua **effect size chuẩn hoá trong từng bộ**, không bao giờ so giá trị tuyệt đối.

---

## 4. Tiền xử lý thống nhất

Một hàm duy nhất cho mọi bộ dữ liệu, tham số trong `config.yaml`:

1. Chọn tập kênh: (a) **19 kênh 10-20 chung** cho phân tích xuyên bộ; (b) toàn bộ montage cho phân tích trong bộ.
2. Lọc 1–40 Hz (FIR zero-phase), notch nếu cần, hạ mẫu về 250 Hz.
3. Re-reference trung bình; đánh dấu kênh xấu (RANSAC/z-score), nội suy.
4. ICA loại mắt/cơ (ICLabel ngưỡng 0.8) — Chennu đã sạch thì bỏ qua.
5. Cắt epoch **10 s**, loại epoch biên độ > 150 µV.
6. **Cân bằng số epoch** giữa các điều kiện trong cùng người (lấy mẫu ngẫu nhiên có seed) — LZc nhạy với độ dài.
7. Biến đổi CSD (surface Laplacian) cho mọi chỉ số đồng bộ/kết nối (chống volume conduction).

Mọi bước ghi log vào `derivatives/<dataset>/<sub>/prep_log.json`.

---

## 5. Bộ chỉ số

### 5.1 Danh sách

| Nhóm | Chỉ số | Cách tính | Gắn với |
|---|---|---|---|
| Phức tạp | LZc (Schartner 2015) | Nhị phân hoá biên độ Hilbert theo trung bình kênh, nối kênh, chuẩn hoá bằng chuỗi xáo trộn | H1, H2 |
| Phức tạp | ACE, SCE | Amplitude / synchrony coalition entropy | H2 |
| Phổ | Exponent aperiodic, offset | specparam 1–40 Hz | H4, covariate |
| Phổ | Công suất tương đối delta/theta/alpha/beta | Welch | covariate, so với BQI Table 14 |
| Đồng bộ | Kuramoto r, metastability (SD của r(t)) | Pha Hilbert theo băng, dữ liệu CSD | H3 |
| Mạng | λ₂ của Laplacian chuẩn hoá trên ma trận wPLI theta **đầy đủ** (không ngưỡng); E_glob, σ trên đồ thị ngưỡng | Ngưỡng 10–30% chỉ cho E_glob/σ, so với null bảo toàn bậc (ngưỡng làm đồ thị 19 kênh rời rạc → λ₂ = 0, cổng G0 phát hiện) | H5 |
| Tích hợp | ΦR = ΦAR (whole-minus-sum, Barrett & Seth 2011) + redundancy MMI (Mediano 2021) | Cụm 6–8 kênh, MIB vét cạn bipartition; ΦWMS có thể âm (cổng G0 phát hiện) nên chỉ báo kèm | Khám phá |

### 5.2 Khử phổ (đóng góp phương pháp)

Mỗi chỉ số phức tạp/đồng bộ X được tính trên dữ liệu thật và trên **50 surrogate phase-randomized** (giữ nguyên phổ từng kênh; phiên bản đa kênh giữ cả cross-spectrum):

```
X_spec-indep = (X_real − mean(X_surr)) / sd(X_surr)
```

Báo cáo song song X thô và X_spec-indep cho mọi giả thuyết. Thêm phân tích nhạy: LMM với exponent và alpha power làm covariate.

---

## 6. Thiết kế phân tích

### 6.1 Thống kê trong từng bộ

- LMM: `metric ~ condition + (1 | subject)` (thêm `(1 | session)` cho ds001787).
- Effect size: d_z paired, CI 95% bằng bootstrap theo subject (10.000 lần).
- Theo kênh: cluster-based permutation (5.000 hoán vị).
- Đa so sánh: FDR (Benjamini–Hochberg) trên tập chỉ số của mỗi giả thuyết.
- Kết quả âm: **TOST** với vùng tương đương |d| < 0.2.

### 6.2 Kiểm định pipeline (bắt buộc trước khi chạm dữ liệu thiền)

Pipeline phải tái hiện kết quả đã biết:

| Tương phản | Kết quả kỳ vọng (tài liệu) |
|---|---|
| ANPHY: wake vs N3 | LZc ↓, exponent dốc hơn |
| ANPHY: wake vs REM | LZc REM gần wake |
| Chennu: baseline vs moderate | LZc ↓ (Schartner 2015), alpha frontal tăng |
| Chennu: recovery | Quay lại gần baseline |

### 6.3 Trục mức-ý-thức (phân tích H2)

1. Với mỗi người trong bộ ngủ và gây mê, lấy vector Δ của K chỉ số đã z-score trong bộ: Δ_sleep = wake − N3, Δ_prop = baseline − moderate.
2. Trục **u** = thành phần chính thứ nhất của {Δ_sleep, Δ_prop} gộp (kiểm tra hai tương phản có cùng hướng: cosine > 0.5).
3. Với mỗi người thiền: Δ_med = thiền − MW, chiếu lên **u** → hệ số chiếu `c` (đơn vị: "phần của khoảng wake→N3").
4. Báo cáo cả phần dư vuông góc ‖Δ_med − c·u‖ — đây là phép thử "thiền khác loại".
5. Kiểm định: bootstrap theo subject cho `c` và cho phần dư; so phần dư với phân phối null hoán vị nhãn điều kiện.

Phiên bản chính dùng 19 kênh chung; phiên bản nhạy dùng chỉ số tổng hợp toàn montage.

---

## 7. Mô hình động lực (phần lý thuyết)

### 7.1 Sửa mô hình gốc (eq. 52–54 của bài BQI)

- Vì E[ξ] = 0, nghiệm dừng là A\* = 0 chứ không phải (k_spont/k_inh)√(2πσ²) (eq. 55).
- Đặt dQoC/dt = 0 khi dΦ/dt = dA/dt = 0 cho γ·QoC\* = 0 ⇒ **QoC\* ≡ 0** (eq. 57 sai): QoC chỉ là đáp ứng quá độ.
- Với ξ là nhiễu trắng, A_DMN là quá trình Ornstein–Uhlenbeck: phân phối dừng N(0, k_spont²/(2k_inh)).

### 7.2 Mô hình đề xuất (không gian trạng thái)

```
dA = −k_inh (A − A₀) dt + σ_A dW              # OU có mức nền A₀ > 0
dΦ = [k_int (1 − A/A_max) − k_d Φ] dt
QoC = α Φ − β A + γ₀ + ε                       # đại số, có trạng thái dừng khác 0
Quan sát: y_EEG(t) = H [A, Φ]ᵀ + η            # LZc_spec-indep, exponent
          rating(t_probe) ~ ordinal(QoC(t_probe))
```

- Fit trên ds001787 (mỗi người ≥ 30 probe): Kalman filter / EM, tham số theo nhóm (expert vs novice).
- Phân tích identifiability (profile likelihood) trước khi diễn giải tham số.
- Phép thử: k_inh expert > novice (dự đoán BQI); so sánh mô hình với baseline không động lực (hồi quy rating trên y_EEG tức thời) bằng cross-validation theo probe.

Phần này là **mục tiêu phụ**: nếu không identifiable, bài vẫn đứng bằng mục 6; báo cáo mô hình như phân tích lý thuyết + kết quả âm.

---

## 8. Kiểm định công cụ bằng dữ liệu tổng hợp

Tái dùng các mô phỏng trong `bqi/`, đổi vai thành **kiểm thử có đáp án**:

| Mô phỏng | Phải chứng minh |
|---|---|
| `simulate_tms_eeg` (độ phức tạp đã biết) | LZc tăng đơn điệu theo `complexity_level`; LZc_spec-indep ≈ 0 với surrogate của chính nó |
| Tín hiệu 1/f thuần với exponent khác nhau | LZc thô thay đổi theo exponent, LZc_spec-indep **không** đổi → chứng minh mục 5.2 cần thiết |
| Kuramoto **trên đồ thị** với coupling đã biết | r tăng theo K; λ₂ của mạng wPLI phục hồi đúng thứ tự |
| OU + Φ + QoC (mục 7.2) với tham số đã biết | Parameter recovery: sai số tương đối < 20% với số probe như ds001787 |

Đưa vào Supplementary làm bằng chứng tính hợp lệ của công cụ.

---

## 9. Cổng dừng (kill criteria)

| Cổng | Điều kiện dừng | Hành động |
|---|---|---|
| G0 (tuần 3) | Kiểm định công cụ mục 8 thất bại | Sửa chỉ số; không chạy dữ liệu thật |
| G1 (tuần 5) | Pipeline không tái hiện wake > N3 và baseline > moderate về LZc | Sửa pipeline, **không** chạy dữ liệu thiền |
| G2 (tuần 6) | Δ_sleep và Δ_prop không cùng hướng (cosine < 0.5) | Bỏ trục chung; báo cáo hai trục riêng, H2 thành khám phá |
| G3 (tuần 9) | Parameter recovery mục 8 thất bại | Bỏ phần fit mô hình, giữ 7.1 (sửa toán) |

**Trạng thái 03/10/2026:** ANPHY tạm hoãn (82.5 GB). Chạy nửa Chennu của G1 trước; G1 tổng, G2 và việc khoá đăng ký trước chờ ANPHY.

Mỗi cổng ghi kết quả vào `outputs/gates/G*.json`; bước sau đọc file này và từ chối chạy nếu cổng trước chưa pass (ghi đè chỉ bằng cờ `--force-gate` và được log lại).

---

## 10. Cấu trúc mã nguồn (gắn vào hệ thống hiện tại)

```
bqi_project/
  bqi/                     # giữ: lý thuyết
    dmn_ode.py             # SỬA: OU + nghiệm dừng đúng + mô hình 7.2 + test
    pci.py                 # SỬA: chuẩn hoá Casali 2013; giữ simulate_tms_eeg làm ground truth
    spectral_graph.py      # SỬA: Kuramoto trên đồ thị (eq. 97 đúng như bài)
  eeg/                     # MỚI
    config.yaml            # mọi tham số tiền xử lý / chỉ số / thống kê
    fetch.py               # tải theo subject, kiểm tra checksum, ghi manifest
    loaders/               # chennu.py, anphy.py, ds003969.py, ds001787.py
    preprocess.py          # mục 4
    metrics/               # lzc.py, coalition.py, aperiodic.py, sync.py, network.py, phi_ar.py
    surrogate.py           # phase randomization đơn/đa kênh
    axis.py                # mục 6.3
    stats.py               # LMM, bootstrap, cluster permutation, TOST
    model_fit.py           # mục 7.2: Kalman/EM, profile likelihood
  tests/                   # thêm: test_lzc_ground_truth, test_surrogate_spectrum, test_axis, test_ou_recovery
  experiments/
    validate_tools.py      # mục 8
    run_pipeline.py        # mục 6, theo từng cổng
  dashboard/views/         # thêm trang: QC dữ liệu, chỉ số theo trạng thái, trục mức-ý-thức
  main.py                  # thêm subcommand: fetch-eeg, preprocess, metrics, stats, fit-model
  preregistration/predictions.md
```

Phụ thuộc thêm: `mne`, `mne-bids`, `specparam`, `statsmodels`, `pingouin`, `openneuro-py`.
Bỏ khỏi bài (giữ trong repo như lịch sử): `quantum_decoherence.py`, `com_algorithm.py`, `bqi_algorithm.py`, bảng `STUDIES_REFERENCE`.

Nguyên tắc: mọi số liệu trong bài sinh ra từ `python main.py all-eeg --seed 42` trên dữ liệu đã tải.

---

## 11. Lộ trình 16 tuần

| Tuần | Việc | Đầu ra kiểm tra được |
|---|---|---|
| 1 | Viết `predictions.md` + khoá (`preregister`); sửa `dmn_ode.py`, `pci.py`, `spectral_graph.py` + test | `predictions.lock` có timestamp trước khi tải dữ liệu thiền |
| 2 | `fetch.py`, loader Chennu + ANPHY; tiền xử lý | Manifest; QC từng subject trên dashboard |
| 3 | Chỉ số mục 5.1 + surrogate; `validate_tools.py` | Mục 8 pass |
| 4–5 | Chạy ngủ + gây mê | **Cổng G1** |
| 6 | Trục mức-ý-thức | **Cổng G2**; Figure 2 |
| 7–8 | Loader + tiền xử lý ds003969, ds001787; chạy H1–H5 | Bảng kết quả chính |
| 9–10 | Mô hình 7.2: recovery → fit ds001787 | **Cổng G3**; Figure 5 |
| 11 | Phân tích nhạy (covariate, montage, độ dài epoch, ngưỡng mạng) | Bảng robustness |
| 12–14 | Viết bản thảo | Bản thảo hoàn chỉnh |
| 15 | Tự phản biện theo mục 13–14 + chạy lại toàn bộ từ đầu trên máy sạch | Danh sách sửa; số liệu khớp 100% |
| 16 | Sửa, nộp; preprint cùng ngày | Nộp |

---

## 12. Khung bài báo và nơi nộp

**Tên (mẫu)**: *Where does meditation sit on the axis of conscious level? A spectrally controlled EEG comparison with sleep and anaesthesia*

| Phần | Nội dung bắt buộc |
|---|---|
| Abstract | Khẳng định "thiền = ý thức cao hơn" → trục hiệu chuẩn → khử phổ → vị trí của thiền có số → mã mở |
| Introduction | Khẳng định trong tài liệu (BQI là một ví dụ, kèm entropic brain, Shinozuka 2025); vấn đề nhiễu phổ; 3 đóng góp |
| Methods | Dữ liệu; tiền xử lý; chỉ số; khử phổ; trục; thống kê; đăng ký trước |
| Results 1 | Kiểm định pipeline trên ngủ + gây mê (**Figure 1**) |
| Results 2 | Trục mức-ý-thức (**Figure 2**) |
| Results 3 | Thiền trên trục, thô vs khử phổ (**Figure 3** — hình chính) |
| Results 4 | H3–H5: đồng bộ, phổ, mạng (**Figure 4**) |
| Results 5 | Mô hình động lực: sửa toán + fit (**Figure 5**) |
| Discussion | Dự đoán nào đứng, dự đoán nào đổ; ý nghĩa cho các lý thuyết "phức tạp = mức ý thức" |
| Limitations | Không có TMS (không phải PCI thật); băng 1–40 Hz; khác bộ dữ liệu; thiền ≠ một trạng thái |
| Supplementary | Mục 8; robustness; toàn bộ tham số |

**Tuyên bố nên dùng**: "Sau khi khử phổ, thiền dịch chuyển c = X (CI) khoảng wake→N3 trên trục mức-ý-thức."
**Tránh**: "thiền là ý thức cao hơn" nếu c chỉ dương ở phiên bản thô; "PCI" khi không có TMS (dùng "LZc tự phát").

**Nơi nộp** (kiểm tra quartile Scimago năm hiện hành trước khi chọn):
1. NeuroImage — đích cao nhất, phù hợp bài phương pháp + dữ liệu.
2. Neuroscience of Consciousness — đúng chủ đề nhất.
3. Dự phòng: Cortex, Brain Topography, Cognitive Neurodynamics, Computers in Biology and Medicine.

---

## 13. Câu hỏi phản biện và câu trả lời chuẩn bị sẵn

| Câu hỏi | Trả lời |
|---|---|
| So sánh giữa các bộ dữ liệu khác thiết bị? | Không so giá trị tuyệt đối; chỉ so Δ trong người, z-score trong bộ; trục xây từ hai bộ độc lập và kiểm tra cùng hướng |
| LZc chỉ phản ánh phổ? | Đó là lý do có mục 5.2; báo cáo thô và khử phổ song song; mô phỏng 1/f trong Supplementary |
| Mind-wandering có phải đối chứng tốt? | Là đối chứng trong người, cùng tư thế, cùng phiên; thêm nhóm đối chứng không thiền của ds003969 cho hiệu ứng trait |
| Volume conduction làm giả đồng bộ? | CSD + wPLI; Kuramoto r báo cáo cả trên CSD |
| ΦAR có phải Φ của IIT? | Không; gọi là chỉ số tích hợp lấy cảm hứng IIT, chỉ khám phá |
| Vì sao dựa vào bài BQI chưa qua bình duyệt? | Bài chỉ là nguồn dự đoán định lượng đã nêu trước; mọi kết luận đứng bằng dữ liệu mở và đăng ký trước |
| Đăng ký trước có thật không? | `predictions.lock` (SHA-256 + timestamp) sớm hơn log tải ds003969/ds001787; mã tải bị chặn nếu chưa khoá |

---

## 14. Những chỗ hay hỏng

1. **Chạy dữ liệu thiền trước khi qua cổng G1** → pipeline lỗi được "khớp" theo kết quả mong muốn.
2. **Không cân bằng số epoch** → LZc khác do độ dài, không do trạng thái.
3. **Quên surrogate** → mọi khác biệt LZc có thể chỉ là alpha/1-f.
4. **Đồng bộ trên dữ liệu không CSD** → volume conduction thổi phồng r.
5. **Chọn ngưỡng mạng sau khi xem kết quả** → báo cáo toàn dải 10–30%.
6. **Tải nguyên bộ ANPHY** → hết ổ; chỉ trích đoạn theo giai đoạn ngủ.
7. **Diễn giải tham số mô hình khi không identifiable** → chạy profile likelihood trước.
8. **Trích bảng số liệu của bài BQI** → xem mục 15.

---

## 15. Quy tắc liêm chính với bài BQI gốc

Các quy tắc dưới đây được **cưỡng chế trong mã**, không phụ thuộc quyết định bên ngoài:

- **Không trích, không dùng** Table 4, 6, 7, 11, 14, các hồi quy eq. 50, 59 và con số "47 nghiên cứu, N = 12.847" của bài BQI làm bằng chứng. Các bảng này không truy vết được nguồn (bibliography trống, cột Source là `?`, có mục trỏ tới bài review không có dữ liệu, có PCI cho trải nghiệm cận tử).
- Chỉ dùng bài BQI như **nguồn giả thuyết**: "mô hình dự đoán X", không phải "dữ liệu cho thấy X".
- Nếu bài BQI chưa công khai: mô tả các dự đoán trực tiếp trong Introduction (công thức + ngưỡng), không trích số liệu của bài.

---

## 16. Checklist trước khi nộp

- [ ] `predictions.lock` có trước log tải dữ liệu thiền; hash khớp `predictions.md`
- [ ] Cổng G1, G2, G3 ghi rõ kết quả trong bài (kể cả cổng thất bại)
- [ ] Mọi chỉ số báo cáo cả thô và khử phổ
- [ ] Effect size + CI bootstrap cho mọi giả thuyết; TOST cho kết quả âm
- [ ] Cân bằng epoch, tham số tiền xử lý công bố trong `config.yaml`
- [ ] Mục 8 (kiểm định công cụ) trong Supplementary
- [ ] Phân tích nhạy: montage, độ dài epoch, ngưỡng mạng, covariate phổ
- [ ] Mã + dữ liệu dẫn xuất công bố (GitHub + Zenodo DOI); dữ liệu gốc trích theo DOI của từng bộ
- [ ] Không còn số liệu nào từ các bảng BQI ở mục 15
- [ ] Limitations: không TMS, băng 1–40 Hz, khác bộ dữ liệu, thiền không đồng nhất
- [ ] Preprint cùng ngày nộp

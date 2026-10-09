# BQI — Python Implementation of "The Brain as a Query Interface"

This project is a runnable Python implementation of the algorithms,
mathematical models, and quantitative results described in the paper
**"The Brain as a Query Interface"** (Vu, Phenikaa University), covering:

- Bayesian Predictive Coding (Sec. 3.1) and the BQI Convergence Theorem
- Attention-based SNIS search (Sec. 3.3, Definition 3.4)
- **Algorithm 1** — BQI three-phase belief update (Sec. 4.5)
- **Algorithm 2** — COM: Consciousness Optimisation Model (Sec. 8.2)
- Integrated Information Theory / Φ computation (Sec. 5.1)
- Perturbational Complexity Index via Lempel-Ziv complexity (Sec. 5.2)
- DMN / Φ / Quality-of-Consciousness ODE system (Sec. 7.2)
- Random-effects meta-analysis of 16 meditation/DMN studies (Sec. 7.3)
- Classic vs. modern (dense) Hopfield network capacity (Sec. 9.2)
- Graph-theoretic brain network model + Kuramoto synchrony (Sec. 4.4, 9.4)
- Open quantum system entropy dynamics / Orch-OR collapse time (Sec. 6)

Every module is a literal translation of the paper's equations/pseudocode
into runnable code, together with **new synthetic datasets** (not copied
from the paper) generated from first principles so the algorithms can be
exercised and tested end-to-end.

> **Note on scope.** The source paper is a theoretical/speculative synthesis
> across neuroscience, IIT, quantum theory, and deep learning. Several of its
> published tables are themselves illustrative meta-analytic summaries rather
> than primary datasets. This project reproduces the *algorithms* faithfully
> and generates *independent* synthetic data whose qualitative behaviour
> (state orderings, direction of effects, convergence guarantees) matches the
> paper's claims — it is not a claim that the numeric magnitudes are
> empirically validated.

---

## 🎯 Bài Q1 — Pipeline mô hình (đọc phần này trước)

Repo có **hai phần**. Phần tái hiện bài BQI (các mục bên dưới) chỉ là minh hoạ thuật toán.
**Phần nghiên cứu mới** nằm trong `model/`, được viết để nộp tạp chí Q1, và **không dùng bất kỳ
bảng số liệu nào của bài gốc**: mọi kết quả là thí nghiệm *in silico* sinh từ mô hình, có seed, tái tạo 100%.

### Câu hỏi

Bài BQI đề xuất hệ ODE DMN–Φ–QoC (eq. 52–54) cho động lực của thiền. Pipeline hỏi ba câu kiểm chứng được bằng toán và mô phỏng:

| # | Câu hỏi | Gói kết quả | Cổng |
|---|---|---|---|
| 1 | Mô hình có **nhất quán** không? | `theory` — mệnh đề P1–P5; mô hình gốc có QoC\* ≡ 0, đề xuất M1 có mức dừng ≠ 0 | **M0** |
| 2 | Tham số có **đo được** từ thiết kế experience-sampling thông dụng không? | `identifiability` (Rothenberg trên Fisher chính xác) + `design` (bản đồ CRB, phục hồi MLE) | **M1** |
| 3 | **Điều khiển vòng kín** (neurofeedback) làm được gì khi tham số chỉ biết gần đúng? | `control` (LQG vs bang-bang, độ bền) | **M2** |
| + | Kết luận có đứng vững trên dải tham số rộng không? | `sensitivity` (Sobol, QMC, bootstrap) | — |

### Trạng thái (09/10/2026): phần máy đã xong

```
python main.py gates
M0  PASS   Mệnh đề P1–P5: mô hình gốc M0 có lỗi toán học, M1 nhất quán
M1  PASS   CRB khớp sai số phục hồi MLE ở 24/24 ô (trung vị RMSE/CRB = 0.98)
M2  PASS   LQG thắng bang-bang và không điều khiển; chi phí dừng khớp dạng đóng (3 %)
```
85 test pass (`python -m pytest tests -q`).

### Kết quả nằm ở đâu

| Thứ cần xem | Đường dẫn |
|---|---|
| **Tóm tắt 1 trang cho người đọc** | [`outputs/paper/model/TOM_TAT_GUI_THAY.md`](outputs/paper/model/TOM_TAT_GUI_THAY.md) |
| **8 hình + 3 bảng cho bài** (PNG 300 dpi + PDF, CSV + LaTeX) | [`outputs/paper/model/`](outputs/paper/model/) — `fig1`…`fig8`, `table1`…`table3`, `manifest.json` |
| Số liệu thô của từng gói (JSON/CSV) + bản chụp cấu hình | [`outputs/results/model/`](outputs/results/model/) — `config_snapshot.json` ghi mọi thông số đã dùng |
| Kết quả các cổng kiểm tra (timestamp + chi tiết số) | [`outputs/gates/M0.json`](outputs/gates/), `M1.json`, `M2.json` |
| Pipeline đầy đủ: phương pháp, thông số, cổng, khung bài báo | [`Q1_pipeline_model.md`](Q1_pipeline_model.md) |
| Mọi thông số của mọi thực nghiệm (một nguồn duy nhất) | [`model/config.py`](model/config.py) |
| Khám phá tương tác | Dashboard → trang **Bài Q1 › Pipeline mô hình** (`python main.py dashboard`) |

### Kết quả chính

| Hình | Thông điệp |
|---|---|
| F1, F2, T1 | Mô hình gốc: QoC là bộ lọc thông cao của dΦ/dt, dA/dt → QoC\* ≡ 0; nghiệm dừng in trong bài (eq. 55–57) không thoả ODE. M1 sửa lại: mức dừng 1.11, phân phối Gauss khớp dạng đóng |
| T2, F3 | Cần cả hai kênh EEG (A và Φ) mới ước lượng đủ 12 tham số; chỉ có rating thì hạng 6/10 |
| F4 | Cận Cramér–Rao dự đoán sai số thật trong hệ số 2 ở 24/24 thiết kế → được phép dùng CRB thay mô phỏng |
| **F5, T3** (hình chính) | **k_inh** (tốc độ ức chế DMN) là tham số khó đo nhất: cần **24 người × 20 phút** để sai số ≤ 20 %; A0, α, β đạt ngay với 6 người × 20 phút |
| F6, F7 | LQG rẻ hơn bang-bang 14× và không điều khiển 38×; thiết kế thí nghiệm đủ tốt → neurofeedback mất < 3 % hiệu quả, thiết kế nhỏ mất tới 38–98 % |
| F8 | Khả năng đo k_inh do mức nền A0 (S_T 0.58) và độ ồn của A (0.36) quyết định, không do chính k_inh (0.13) |

### Tái tạo toàn bộ

```bash
pip install -r requirements.txt
set OMP_NUM_THREADS=1              # Windows; Linux/macOS: export ...
set MKL_NUM_THREADS=1              # (3 tiến trình × 4 luồng MKL trên 4 nhân làm chậm 5–6 lần)
python main.py model all --seed 42 --n-base 256 --profiles     # ~8 giờ trên máy 4 nhân
python main.py gates
```
Từng bước: `python main.py model theory | identifiability | design | control | sensitivity | figures`
(`--fast` để thử nhanh; `design` là bước lâu nhất, ~7 giờ; kết quả giống hệt khi chạy trên máy khác với cùng seed).

### Việc còn lại (không cần máy)

Appendix A (chứng minh P1–P5, đối xứng không xác định), bản thảo theo khung `Q1_pipeline_model.md` §13, đưa mã + `outputs/` lên GitHub/Zenodo khi nộp.

---

## Project structure

```
bqi_project/
├── main.py                        CLI trung tâm: generate-data / run-experiments / test / all / dashboard / model / gates
├── Q1_pipeline_model.md           ★ Pipeline bài Q1: phương pháp, thông số, cổng, khung bài
├── model/                         ★ Bài Q1 — 5 gói kết quả + cổng M0–M2
│   ├── config.py                  Một nguồn duy nhất cho mọi thông số thực nghiệm
│   ├── fisher.py                  Fisher kỳ vọng chính xác, CRB, phổ Rothenberg
│   ├── theory.py                  Gói 1: P1–P5, cổng M0
│   ├── identifiability.py         Gói 2: xác định tham số (Bảng 2)
│   ├── design.py                  Gói 3: bản đồ thiết kế, phục hồi MLE, cổng M1
│   ├── control.py                 Gói 4: LQG vs bang-bang, độ bền, cổng M2
│   └── sensitivity.py             Gói 5: Sobol (QMC, log, bootstrap)
├── bqi/                          Core algorithm package (tái hiện bài BQI)
│   ├── predictive_coding.py      Sec 3.1 — Bayesian Predictive Coding
│   ├── attention.py               Sec 3.3 — SNIS attention (Definition 3.4)
│   ├── bqi_algorithm.py           Sec 4.5 — Algorithm 1 (BQI belief update)
│   ├── com_algorithm.py           Sec 8.2 — Algorithm 2 (COM)
│   ├── iit_phi.py                 Sec 5.1 — Integrated Information Φ
│   ├── pci.py                     Sec 5.2 — Perturbational Complexity Index
│   ├── dmn_ode.py                 Sec 7.2 — DMN/Φ/QoC ODE system
│   ├── meta_analysis.py           Sec 7.3 — Random-effects meta-analysis
│   ├── hopfield.py                Sec 9.2 — Classic/modern Hopfield capacity
│   ├── spectral_graph.py          Sec 4.4/9.4 — Graph theory + Kuramoto
│   └── quantum_decoherence.py     Sec 6 — Open quantum system / Orch-OR
├── datasets/
│   ├── generate_datasets.py       Generates all synthetic datasets (CSV)
│   └── generated/                 Output CSVs (created on first run)
├── experiments/
│   ├── run_all.py                 Runs every algorithm, saves figures + JSON summary (phần BQI)
│   └── run_model_paper.py         ★ Xuất 8 hình + 3 bảng của bài Q1 → outputs/paper/model
├── eeg/, preregistration/         Hướng EEG thực nghiệm (Q1_pipeline.md) — HOÃN, không dùng trong bài
├── tests/                         pytest (85 test; test_model.py cho bài Q1)
├── dashboard/                     Giao diện trực quan Streamlit (python main.py dashboard)
│   ├── app.py                     Điểm vào: điều hướng giữa các trang
│   ├── core.py                    Dùng chung: đăng ký module, màu, chạy CLI/pytest, scorecard
│   └── views/                     overview · lab · data · experiments · tests · model (★ Bài Q1)
├── outputs/
│   ├── paper/model/               ★ KẾT QUẢ BÀI Q1: fig1–fig8, table1–table3, TOM_TAT_GUI_THAY.md
│   ├── results/model/             ★ số liệu thô 5 gói + config_snapshot.json
│   ├── gates/                     ★ M0.json, M1.json, M2.json (PASS/FAIL + timestamp + chi tiết)
│   ├── figures/                   PNG figures reproducing the paper's plots (phần BQI)
│   └── results/                   summary.json with all numeric results (phần BQI)
├── requirements.txt
└── README.md
```

## Quickstart (dùng `main.py` — khuyến nghị)

`main.py` là điểm khởi chạy trung tâm, gộp cả 3 việc: sinh dữ liệu, chạy
thực nghiệm, và chạy test — tất cả điều khiển qua tham số dòng lệnh.

```bash
pip install -r requirements.txt

# Xem hướng dẫn / các lựa chọn hiện có
python main.py info

# Chạy TOÀN BỘ pipeline: sinh dữ liệu -> chạy thực nghiệm -> chạy test
python main.py all --seed 42

# Chỉ sinh dữ liệu tổng hợp (tất cả dataset), với seed tuỳ chỉnh
python main.py generate-data --seed 123

# Chỉ sinh một số dataset cụ thể + tuỳ chỉnh kích thước mẫu
python main.py generate-data --only pci,ode --n-subjects-per-state 60 --n-ode-subjects 1000

# Chạy toàn bộ 10 thực nghiệm/thuật toán, lưu hình PNG + summary.json
python main.py run-experiments --seed 7

# Chỉ chạy một vài thực nghiệm cụ thể
python main.py run-experiments --only bqi,com,hopfield --print-summary

# Chạy bộ test (pytest) — toàn bộ, hoặc lọc theo file/từ khoá
python main.py test
python main.py test --path tests/test_hopfield.py -k "modern" -v

# Xem danh sách dataset đã sinh ra, và xem nhanh nội dung + thống kê 1 dataset
python main.py list-data
python main.py inspect-data --name pci_synthetic_subjects.csv --group-by state

# Mở dashboard trực quan trên trình duyệt (http://localhost:8501)
python main.py dashboard

# Bài Q1 — pipeline mô hình (Q1_pipeline_model.md): 5 gói kết quả, cổng M0–M2
#   M0 (theory)  : mô hình gốc sai nghiệm dừng, M1 nhất quán         -> outputs/gates/M0.json
#   M1 (design)  : CRB có dự đoán được sai số MLE thật (24 ô, nhiều giờ) -> outputs/gates/M1.json
#   M2 (control) : LQG thắng bang-bang / không điều khiển             -> outputs/gates/M2.json
python main.py model theory | identifiability | design | control | sensitivity | figures | all \
       [--seed 42] [--fast] [--n-base 256] [--n-reps 5] [--n-starts 2] [--profiles]
python main.py gates          # trạng thái M0–M2; kết quả số: outputs/results/model; hình: outputs/paper/model
```

### Bảng tham số chính

| Subcommand | Tham số | Ý nghĩa |
|---|---|---|
| `generate-data` | `--seed` | Seed ngẫu nhiên toàn cục (mặc định 42) |
| | `--only` | Danh sách dataset cách nhau bởi dấu phẩy: `pci,ode,meta,kuramoto,quantum,reference_tables` |
| | `--out-dir` | Thư mục ghi CSV (mặc định `datasets/generated`) |
| | `--n-subjects-per-state`, `--n-ode-subjects`, `--n-bootstrap`, `--n-kuramoto-trials`, `--n-quantum-trials` | Kích thước từng dataset |
| `run-experiments` | `--seed` | Seed ngẫu nhiên toàn cục |
| | `--only` | Danh sách thực nghiệm: `bpc,bqi,com,phi,pci,ode,meta,hopfield,spectral,quantum` |
| | `--fig-dir`, `--res-dir` | Thư mục lưu hình / summary.json |
| | `--no-figures` | Chỉ ghi JSON, không lưu PNG (chạy nhanh hơn) |
| `test` | `--path` | File/thư mục test cụ thể (mặc định `tests/`) |
| | `-k/--keyword` | Lọc test theo từ khoá (giống `pytest -k`) |
| | `-x/--stop-on-fail` | Dừng ngay khi có test đầu tiên fail |
| `inspect-data` | `--name` (bắt buộc) | Tên file CSV trong thư mục dữ liệu |
| | `--group-by` | Cột để nhóm khi tính trung bình (vd `state`) |
| `all` | (gộp tham số của `generate-data` + `run-experiments`) | `--skip-tests` để bỏ qua bước test |
| `dashboard` | `--port` | Cổng HTTP của dashboard (mặc định 8501) |
| | `--headless` | Không tự mở trình duyệt (vd khi chạy trên server) |

### Cách chạy thủ công (không qua `main.py`)

Các script gốc vẫn chạy độc lập được nếu muốn:

```bash
python datasets/generate_datasets.py --seed 42 --only pci,ode
python experiments/run_all.py --seed 42 --only bqi,com
pytest tests/ -v
```

## Dashboard (Streamlit)

Giao diện web để chạy toàn bộ pipeline và xem kết quả mà không cần gõ lệnh.
Dashboard gọi lại đúng CLI `main.py` / `pytest` ở trên, nên kết quả giống hệt khi chạy bằng dòng lệnh.

### Cài đặt

`requirements.txt` đã gồm `streamlit>=1.50` và `plotly`. Nếu môi trường đã có Streamlit cũ hơn
(vd Anaconda đi kèm 1.45), cần nâng cấp vì dashboard dùng API mới:

```bash
pip install -U "streamlit>=1.50" plotly
pip install pytest-cov        # tuỳ chọn: bật đo coverage ở trang Kiểm thử
```

### Khởi động

```bash
python main.py dashboard                      # mở http://localhost:8501
python main.py dashboard --port 8600          # đổi cổng
python main.py dashboard --headless           # không tự mở trình duyệt

# hoặc gọi Streamlit trực tiếp (chạy trong thư mục dashboard/ để nhận .streamlit/config.toml)
cd dashboard && streamlit run app.py
```

`python main.py dashboard` tự mở trình duyệt. Khi gọi `streamlit run` trực tiếp, `config.toml` đặt
`headless = true` nên trình duyệt không tự mở — hãy truy cập http://localhost:8501.

Dừng dashboard bằng `Ctrl+C` trong terminal.

### Các trang

| Trang | Chức năng |
|---|---|
| **Tổng quan** | Số liệu dự án, sơ đồ kiến trúc, scorecard đối chiếu các khẳng định của bài báo (tính trực tiếp từ `bqi/`), bảng module |
| **Phòng thí nghiệm** | Chỉnh tham số và chạy trực tiếp từng thuật toán trong 11 module, biểu đồ tương tác |
| **Datasets** | Sinh lại dữ liệu (= `generate-data`), xem log trực tiếp; biểu đồ, bảng, thống kê và tải về từng file CSV |
| **Thực nghiệm** | Chạy toàn bộ hoặc một phần thực nghiệm (= `run-experiments`), xem kết quả chính, so sánh PCI với bài báo, thư viện hình, `summary.json` |
| **Kiểm thử** | Chạy pytest theo file / từ khoá, xem test lỗi kèm chi tiết, test chậm nhất, coverage, lịch sử các lần chạy |
| **Bài Q1 · Pipeline mô hình** | Giải thích pipeline Q1 so với BQI gốc, trạng thái cổng M0–M2, chạy từng bước `model …`, kết quả 5 gói (nghiệm dừng & Bode, Rothenberg + CRB, bản đồ thiết kế, LQG vs bang-bang tương tác, Sobol) |

Ghi chú:

- Khi chạy **một phần** thực nghiệm từ dashboard, kết quả mới được gộp vào `outputs/results/summary.json`
  (CLI `run-experiments --only` thì ghi đè cả file).
- Lịch sử chạy test được lưu ở `outputs/results/test_history.json` (giữ 200 lần gần nhất).
- Trên Windows, nếu terminal báo `UnicodeEncodeError` khi in tiếng Việt, đặt biến môi trường
  `PYTHONIOENCODING=utf-8` trước khi chạy.


## Module-by-module summary

| Module | Paper section | What it implements |
|---|---|---|
| `predictive_coding.py` | 3.1 | Single-level & hierarchical variational free energy, gradient flow, exponential-convergence bound |
| `attention.py` | 3.3 | Scaled dot-product SNIS attention Σ(Q,K,V), multi-head attention |
| `bqi_algorithm.py` | 4.5 | **Algorithm 1**: Phase 1 (prediction errors) → Phase 2 (free-energy minimisation) → Phase 3 (SNIS query & decode) |
| `com_algorithm.py` | 8.2 | **Algorithm 2**: Phase 1 (DMN suppression via closed-loop BCI) → Phase 2 (Φ↑ via free-energy descent) → Phase 3 (SNIS query expansion → QoC) |
| `iit_phi.py` | 5.1 | Brute-force φ(M,P) over all bipartitions via a causal-cut TPM + Earth Mover's Distance; NP-hardness partition counter |
| `pci.py` | 5.2 | LZ76 Lempel-Ziv complexity, PCI = K(s)/L(s), synthetic TMS-EEG generator calibrated per consciousness state |
| `dmn_ode.py` | 7.2 | 3-state ODE system (scipy `solve_ivp`), closed-form steady states, control-vs-meditator simulation |
| `meta_analysis.py` | 7.3 | DerSimonian-Laird random-effects pooling, I² heterogeneity, the 16 published studies as reference data, bootstrap resampling |
| `hopfield.py` | 9.2 | Classic Hebbian Hopfield (linear capacity ~0.14n), modern softmax Hopfield (~e^{n/2}), Transformer-attention equivalence |
| `spectral_graph.py` | 4.4, 9.4 | Watts-Strogatz brain graph, Fiedler value, small-world index σ, global efficiency, Kuramoto synchrony simulation |
| `quantum_decoherence.py` | 6 | Von Neumann entropy S(ρ) under Lindblad dephasing, Orch-OR collapse timescale τ = ℏ/E_G |

## Datasets generated

Running `datasets/generate_datasets.py` produces, in `datasets/generated/`:

- `pci_synthetic_subjects.csv` — per-subject PCI values (9 states × 40 subjects) from simulated TMS-EEG + real LZ-complexity computation
- `ode_params_synthetic_subjects.csv` — 500 synthetic subjects' ODE parameter estimates sampled around the published CIs
- `dmn_meta_analysis_bootstrap.csv` — 5000 bootstrap-resampled pooled effect sizes
- `kuramoto_synthetic_trials.csv` — Kuramoto order-parameter trials across 6 states × 3 frequency bands
- `quantum_decoherence_synthetic.csv` — entropy trajectories across Hilbert-space dimensions × decoherence rates
- `reasoning_taxonomy_reference.csv`, `psychiatric_disorders_reference.csv` — structured reference tables
- `*_paper_reference.csv` — the paper's own published table values, for side-by-side comparison

## Validating against the paper

`experiments/run_all.py` prints/saves, for each algorithm, a sanity check
against the paper's qualitative claims, e.g.:

- Hierarchical BPC prediction-error energy → 0 (belief converges)
- BQI Algorithm 1's free energy F decreases monotonically to a minimum
- COM Algorithm 2: DMN activity ↓, Φ ↑, QoC → target
- Simulated PCI: Coma < Propofol > NREM < wakefulness ≈ meditation ≈ NDE (ordering, not magnitude)
- Meta-analysis: pooled Cohen's d ≈ 1.5–1.7 (paper reports 1.70), I² in the 30–45% range (paper reports 31%)
- Modern Hopfield / Transformer attention retrieval error stays ~0 far beyond classic Hopfield's collapse point
- Meditation-transformed graph has higher global efficiency than baseline
- Entropy S(ρ) rises monotonically toward ln(dim) under decoherence, faster for larger γ

## Caveats

- `iit_phi.py` is exponential in system size (`2^n` partitions) and is only
  intended for small demonstration systems (n ≤ ~8), exactly as the paper's
  own NP-hardness proposition predicts.
- All "biological" parameter values (precisions, coupling strengths, decay
  rates) are either taken directly from the paper's tables or chosen to
  produce paper-consistent qualitative behaviour; they are not fit to real
  neuroimaging data.

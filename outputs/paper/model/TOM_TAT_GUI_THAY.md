# Tóm tắt kết quả — Pipeline bài mô hình (hướng Q1)

Sinh viên: Messiuuu · Ngày: 09/10/2026 · Mã nguồn: thư mục `bqi_project` (nhánh `model/`)

## 1. Câu hỏi nghiên cứu

Bài "The Brain as a Query Interface" đề xuất hệ ODE DMN–Φ–QoC (eq. 52–54) cho động lực của thiền. Thay vì tái hiện số liệu của bài, pipeline này hỏi ba câu kiểm chứng được hoàn toàn bằng toán và mô phỏng:

1. Mô hình có **nhất quán** không?
2. Tham số có **đo được** từ các thiết kế thí nghiệm experience-sampling thông dụng (probe vài phút một lần, phiên 20–90 phút, 6–48 người) không?
3. **Điều khiển vòng kín** (neurofeedback) làm được gì khi tham số chỉ biết gần đúng?

Mọi kết quả là thí nghiệm *in silico* có seed, tái tạo 100% bằng một lệnh. Không dùng bất kỳ bảng số liệu nào của bài gốc.

## 2. Kết quả chính (8 hình, 3 bảng trong cùng thư mục)

| | Kết quả | Hình/Bảng |
|---|---|---|
| **Sửa mô hình** | Mô hình gốc có QoC\* ≡ 0 với mọi tham số (QoC là bộ lọc thông cao của dΦ/dt, dA/dt; nghiệm dừng eq. 55–57 trong bài không thoả ODE). Đề xuất M1: A_DMN là quá trình Ornstein–Uhlenbeck quanh mức nền, QoC = αΦ − βA + γ0 → mức dừng 1.11, phân phối Gauss khớp dạng đóng. | T1, F1, F2 |
| **Khả năng xác định** (Rothenberg trên Fisher chính xác) | Cần cả hai kênh EEG (A và Φ) mới ước lượng đủ 12 tham số; chỉ có kênh A thì α dính với k_int, σ_Φ (đối xứng co giãn Φ, kiểm chứng giải tích); chỉ có rating thì hạng 6/10. | T2, F3 |
| **Cận Cramér–Rao có tin được không** | 24 ô thiết kế đại diện × 5 lần phục hồi MLE: sai số thật / cận = 0.98 (trung vị), 24/24 ô trong hệ số 2. → Bản đồ thiết kế tính bằng CRB là đáng tin. | F4 |
| **Thiết kế thí nghiệm** (hình chính) | k_inh (tốc độ ức chế DMN) là tham số khó đo nhất: cần **24 người × 20 phút** để sai số ≤ 20%; A0, α, β đạt ngay với 6 người × 20 phút. Thiết kế kiểu ds001787 (12 người, 45 phút, probe/2 phút): sai số k_inh 19.7%. Lịch probe đều không thua các lịch không đều đã thử. | F5, T3 |
| **Điều khiển** | LQG (Kalman + LQR, điểm cân bằng giải tích) rẻ hơn bang-bang 14 lần và không điều khiển 38 lần; chi phí dừng dạng đóng khớp mô phỏng 3%. Thiết kế thí nghiệm đủ tốt → neurofeedback mất < 3% hiệu quả; thiết kế nhỏ (6 người, 20 phút) mất tới 38–98%. | F6, F7 |
| **Độ nhạy toàn cục** (Sobol, QMC, bootstrap) | Khả năng đo k_inh do mức nền A0 (S_T = 0.58) và độ ồn CV(A) (0.36) quyết định, không do chính k_inh (0.13). | F8 |

## 3. Kiểm soát chất lượng

- Ba cổng kiểm tra cưỡng chế trong mã, đều PASS: M0 (mệnh đề P1–P5 kiểm số trên ODE thật), M1 (CRB khớp sai số MLE), M2 (LQG thắng và khớp dạng đóng). Kết quả ghi timestamp trong `outputs/gates/`.
- 85 test tự động pass. Fisher chính xác đối chiếu với Kalman (10⁻¹³) và Monte Carlo (≤ 5%).
- Mọi thông số nằm trong một file `model/config.py`; mỗi kết quả kèm `config_snapshot.json`.
- Tái tạo toàn bộ: `python main.py model all --seed 42 --n-base 256 --profiles` (~8 giờ máy 4 nhân).

## 4. Việc còn lại

- Appendix A: chứng minh viết tay P1–P5 và các đối xứng không xác định.
- Bản thảo (khung đã có trong `Q1_pipeline_model.md` §13).
- Đưa mã + kết quả lên GitHub/Zenodo khi nộp.

## 5. Xin ý kiến thầy

1. Tên tạp chí: dự kiến *Cognitive Neurodynamics*, *Journal of Mathematical Biology*, *Mathematical Biosciences*, *Biological Cybernetics* — thầy thấy nơi nào hợp nhất?
2. Cách trích bài gốc: chỉ trích phương trình (52–54) và chỉ ra lỗi nghiệm dừng; không dùng các bảng số liệu của bài. Thầy có đồng ý cách định vị này không?
3. Có nên mở rộng sang mô hình phi tuyến ở bài sau, hay giữ tuyến tính để bài này gọn?

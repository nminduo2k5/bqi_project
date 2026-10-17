# Đăng ký trước: fit M1 (dạng độ lệch) trên ds001787

**Phiên bản để khoá.** Soạn 17/10/2026 sau cổng E0 và E1, **trước khi tải bất kỳ file EEG (BDF) nào**.
Khoá bằng `python main.py realdata preregister --yes` (ghi SHA-256 + thời điểm vào
`preregistration/predictions_realdata.lock`); `python main.py realdata fetch` từ chối tải BDF khi chưa có lock
hoặc file này đã bị sửa. Mọi thay đổi sau khi khoá phải báo cáo là "sửa đổi sau đăng ký".

**Đã xem trước khi khoá (khai báo trung thực):** toàn bộ metadata của ds001787 (participants.tsv,
40 file events.tsv, 40 file *_eeg.json), ba file trong `code/` kể cả 42 file hành vi `*_info.txt`.
**Chưa xem:** bất kỳ mẫu EEG nào.

## 1. Dữ liệu và cổng đã qua

| Mục | Kết quả (E0, E1 ngày 17/10/2026) |
|---|---|
| Phiên / probe | 40 phiên (22 expert, 18 novice), 983 probe, 1656 phút-người |
| Giải mã rating (E0) | 950/983 probe khớp file hành vi trong 0,5 s (96,6 %); Q1 và Q2 biết được ở 96,6 % probe; mã trigger khớp phím ở 99,4 % |
| Giả thuyết "phím 0 không có trigger" | **Xác nhận**: 0/741 lần bấm phím 0 có trigger trong events.tsv |
| Ngoại lệ | sub-013 ses-01: file hành vi không khớp phiên EEG (3/25 probe), chỉ 4 probe giải mã được từ events; sub-021: 33/37 |
| Hạng Fisher trên lịch probe thật (E1) | 10/10 với 10 tham số; 11/11 khi thêm r_AΦ; trị riêng tương đối nhỏ nhất 0,20 |
| CRB dự đoán tại tham số danh nghĩa, gộp 40 phiên | k_inh 9,1 %, κ 11,1 %, k_d 14,1 %, α 2,0 %, β 4,0 %, σ_A 3,0 %, σ_Φ 3,6 %, r_A 1,1 %, r_Φ 0,9 %, r_Q 2,7 % |
| CRB theo nhóm | expert (22 phiên): k_inh 12,4 %, k_d 19,1 %, α 2,7 %, β 5,3 %; novice (18): k_inh 13,5 %, k_d 21,0 %, α 3,0 %, β 5,9 % |
| MDE hiệu log k_inh expert − novice (lực 80 %, α = 0,05) | 0,51 đơn vị log, tức hệ số 1,67 |

Tham số danh nghĩa dùng cho CRB (dạng độ lệch, sau chuẩn hoá đặc trưng): k_inh 0,20; κ 0,081; k_d 0,034 /phút;
σ_A 0,10; σ_Φ 0,05; α 2,34; β 1,87; r_A = r_Φ 0,10; r_Q 0,30. SNR danh nghĩa của kênh DMN: sd∞(A)/r_A = 1,58.

## 2. Giả thuyết và phép thử (khoá)

| Mã | Dự đoán | Phép thử | Ủng hộ khi |
|---|---|---|---|
| E-H1 | Nếu SNR đặc trưng DMN đạt mức danh nghĩa, sai số tương đối của k_inh gộp ≤ 20 % | SE từ Fisher tại MLE; RMSE bootstrap 200 lần trên đúng thiết kế thật | SE ≤ 0,20 và RMSE/SE ∈ [0,5; 2]. Nếu SNR fit được thấp hơn danh nghĩa: so SE với CRB tính lại tại SNR đó, báo cả hai |
| E-H2 | M1 dự đoán rating ngoài mẫu tốt hơn mô hình rỗng | Leave-one-session-out, log predictive density Gauss tại probe; chế độ (a) chỉ EEG so với N2 (hồi quy tức thời), chế độ (b) EEG + rating quá khứ so với N1 (M1 chỉ rating) và N3 (trung bình phiên) | ΔLPD > 0 với CI 95 % bootstrap theo phiên (2000 lần) loại 0, ở cả (a) và (b) |
| E-H3 | α > 0 (proxy tích hợp liên hệ dương với rating) | CI 95 % của α (SE Fisher) | CI nằm trên 0 |
| E-H4 | Dấu của β, hai phía | CI 95 % của β | Báo dấu; β > 0 nghĩa là alpha sau đóng vai DMN như BQI; β < 0 như Brandmeyer |
| E-H5 | k_inh expert > novice | Hiệu log k_inh giữa hai fit theo nhóm; kiểm định tỉ số hợp lý | Hiệu > 0 và CI loại 0; nếu CI chứa 0, báo kèm MDE 0,51 log |

**Câu hỏi ước lượng (không có giả thuyết):** 1/k_inh gần 5 phút (danh nghĩa của bài) hay 23 phút (giá trị in trong bài BQI)? Báo CI.

**Khám phá, báo riêng, không dùng để kết luận:** Q3 (buồn ngủ) như covariate; xu hướng trong phiên; khác biệt giữa các phiên của cùng người.

## 3. Kế hoạch phân tích (khoá)

1. **Tiền xử lý** (`eeg/realdata_features.py`, tham số trong `model/config.py::REALDATA`): 64 kênh, lọc 1–40 Hz + notch 50 Hz, kênh xấu z > 4 nội suy, tham chiếu trung bình, ICA (ICLabel ≥ 0,8 nếu cài được; nếu không: heuristic mắt |r| > 0,8 với Fp1/Fp2, cơ > 60 % công suất trên 20 Hz, ghi rõ phương pháp vào log).
2. **Lưới epoch**: lưới đều 10 s từ đầu bản ghi; epoch **đánh dấu thiếu** (không xoá) khi chồng lên cửa sổ probe [mốc 128, phản hồi cuối + 3 s] (15 s nếu không có phản hồi) hoặc biên độ đỉnh-đỉnh > 150 µV. Rating là quan sát tại **đúng thời điểm mốc 128**, bộ lọc Kalman dự đoán trạng thái tới thời điểm đó (bước thời gian thay đổi).
3. **Đặc trưng chính**: y_A = log công suất alpha 8–12 Hz vùng sau (Pz, POz, P3, P4, PO3, PO4, O1, O2, Oz); y_Φ = LZc khử phổ (19 kênh 10-20, 20 surrogate đa kênh). **Ba phương án thay thế** (chỉ dùng trong phân tích nhạy): alpha tuần hoàn (trừ thành phần aperiodic tại 10 Hz), log theta 4–7 Hz đường giữa trán (AFz, Fz, FCz, Cz), −exponent aperiodic 1–40 Hz.
4. **Rating** r = Q1 − Q2 (thang −3…3). **Chuẩn hoá**: trừ trung bình theo phiên cho y_A, y_Φ, r; chia y_A, y_Φ cho một SD gộp trong-phiên toàn bộ dữ liệu (một hằng số mỗi đặc trưng).
5. **Giữ phiên**: bỏ phiên có > 40 % epoch thiếu hoặc < 8 probe có rating. Báo số phiên bị bỏ theo nhóm.
6. **Mô hình**: M1 dạng độ lệch, 10 tham số (k_inh, κ, k_d, σ_A, σ_Φ, α, β, r_A, r_Φ, r_Q); κ, α, β thang tuyến tính dấu tự do, còn lại thang log. Fit gộp 40 phiên với 6 điểm xuất phát (danh nghĩa, mô-men, 4 điểm nhiễu, seed 42); theo nhóm; theo người + meta-analysis REML trên log k_inh.
7. **Ba con số sai số** cho k_inh, k_d, α, β: CRB dự đoán (bảng trên), SE Fisher tại MLE, RMSE bootstrap 200 lần. Profile likelihood 9 điểm ±0,8 log cho k_inh, κ, k_d.
8. **Kiểm tra mô hình**: Ljung–Box trễ 1–30 trên innovation y_A, y_Φ (trung vị p > 0,01 → E3); tương quan innovation y_A–y_Φ (nếu |ρ| > 0,2 thì phân tích chính dùng R đầy đủ với r_AΦ, đã kiểm tham số này xác định được ở E1); LOSO như E-H2; dự đoán hậu nghiệm cho rating thứ bậc.
9. **Năm phân tích nhạy** (báo tất cả): ba đặc trưng thay thế (mỗi lần thay một kênh), z-score theo phiên, chỉ phiên ≤ 20 % epoch thiếu, chỉ 20 probe đầu mỗi phiên. Kết luận chính chỉ phát biểu nếu dấu α, β và khoảng 1/k_inh giữ nguyên ở ≥ 4/5 phân tích.
10. **Tiêu chí dừng**: K1 rating không giải mã được (không xảy ra: E0 PASS); K2 tự tương quan trễ 1 của y_A < 0,3 (trung vị) → dừng fit, báo SNR đo được; K3 > 50 % phiên bị loại → dừng; K4 chưa qua E2 sau 6 tuần → nộp bài model như hiện tại.

## 4. Sửa đổi so với Q1_pipeline.md v2.0 (ghi trước khi khoá)

- Lưới epoch neo theo probe (§4 mục 6 của pipeline) được thay bằng lưới đều từ đầu bản ghi + quan sát rating tại đúng mốc 128 với bộ lọc Kalman bước thay đổi. Lý do: giữ Δ = 10 s cho kênh EEG mà không cần xoá hay dịch epoch, và rating được gán vào đúng thời điểm người tập đánh giá thay vì vào một epoch.
- File hành vi được gán cho phiên EEG bằng **căn chỉnh thời điểm probe** (chọn gán có số probe khớp lớn nhất), không theo hậu tố _2/_3 của tên file; nhờ đó sub-014 khớp 24/24 (file sub14_2_info.txt) thay vì 3/24.
- Với probe không có file hành vi khớp, chỉ probe có đủ 3 phản hồi trong events.tsv được giải mã; các probe còn lại không có rating (sub-013: 21 probe).

## 5. Những gì không được làm sau khi khoá

Không đổi đặc trưng chính, cửa sổ che, tiêu chí giữ phiên, số điểm xuất phát, số lần bootstrap, mô hình rỗng, danh sách phân tích nhạy; không báo A0, Φ* hay bất kỳ mức tuyệt đối nào trên dữ liệu thật; không trích bảng số liệu của bài BQI.

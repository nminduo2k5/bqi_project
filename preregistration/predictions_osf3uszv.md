# Đăng ký trước: fit M1 (dạng độ lệch) trên bộ dữ liệu thứ hai, OSF 3uszv

**Phiên bản để khoá.** Soạn 03/11/2026 sau cổng F0 và F1, **trước khi đọc bất kỳ mẫu EEG nào ngoài phần khai báo dưới đây**.
Khoá bằng `python main.py osf3uszv preregister --yes` (ghi SHA-256 + thời điểm vào
`preregistration/predictions_osf3uszv.lock`); `python main.py osf3uszv preprocess` từ chối chạy khi chưa có lock
hoặc file này đã bị sửa. Mọi thay đổi sau khi khoá phải báo cáo là "sửa đổi sau đăng ký".

**Bộ dữ liệu.** OSF 3uszv, thành phần v8msp "RAW EEG DATA" (Rodriguez-Larios, Bracho Montes de Oca & Alaerts 2021,
*NeuroImage* 245:118669; CC0): 29 người thiền lâu năm (`m_`, "expert") và 29 người chưa từng thiền (`c_`, "novice"),
mỗi người một phiên thiền tập trung hơi thở khoảng 60 phút bị ngắt bởi chuông để tự báo cáo ("task"), EEG Nexus-32,
21 kênh (19 kênh 10-20 + A1, A2), 512 Hz. Đây là phòng thí nghiệm, thiết bị và quy trình **độc lập** với ds001787.

**Đã xem trước khi khoá (khai báo trung thực):**
- 8 script MATLAB của tác giả (cách đọc file, thứ tự mã trigger, cách họ tiền xử lý), `name_coding_text_files.txt`, `EEG21locs.mat`.
- **Cột trigger `Events`** của 58 file task (giải mã probe, cổng F0) — tương đương events.tsv của ds001787, không chứa EEG.
- **Một đoạn 2 phút của m_task_2** (dòng 3 000 000 đến 3 245 760, 21 kênh) để xác định tần số lấy mẫu: mỗi mẫu EEG được ghi lặp
  đúng 4 lần (2048 dòng/giây, EEG thật 512 Hz) và đỉnh alpha chẩm nằm ở 10 Hz chỉ khi hiểu như vậy; độ lệch chuẩn thô của O1 ≈ 194 µV
  (chưa lọc). Ngoài ra 400 byte đầu của cùng file (một dòng số liệu) khi kiểm tra định dạng nén.
- Toàn bộ 174 file (task, med, rest; 15,35 GB) đã được tải về **trước** khi khoá vì người dùng yêu cầu tải hết một lần;
  manifest ghi thời điểm tải từng file. Không file nào được giải nén hay mở ngoài hai mục trên.

**Chưa xem:** bất kỳ mẫu EEG nào khác; mọi đặc trưng; mọi tương quan EEG–rating.

## 1. Dữ liệu và cổng đã qua

| Mục | Kết quả (F0, F1 ngày 03/11/2026) |
|---|---|
| Phiên / probe | 58 phiên (29 expert, 29 novice), 2258 lần chuông, 3304 phút-người (55,1 giờ) |
| Độ dài phiên | trung vị 57,0 phút (34,3–64,6); chuông đầu ở phút 6,7 (trung vị); chuông cách nhau trung vị 77 s |
| Probe mỗi phiên | trung vị 40 (ít nhất 22, c_n11); không phiên nào dưới 8 probe hợp lệ |
| Giải mã rating (F0) | 2233/2258 probe có câu trả lời ∈ {1,2,3} và độ chắc chắn ∈ 1–7 (98,9 %); 99,0 % probe có đủ 3 mã; 61 mã đến muộn (> 60 s sau chuông) bị bỏ; 8 probe có mã trả lời lạ (4–7) → không có rating |
| Thang đo quan sát được | độ chắc chắn 1–7, buồn ngủ 1–7, mức tập trung 1–7, mức cuốn theo suy nghĩ 1–7 |
| Câu trả lời | "tập trung hơi thở" 1440 (64 %), "đang nghĩ chuyện khác" 528, "bị phân tán bởi thứ khác" 277; expert 73 % hơi thở, novice 55 % |
| Rating r (mục 3.4) | phân bố hai cực: 88 % probe ở |r| ≥ 5; SD trong phiên trung vị 5,55; trung bình expert 2,95, novice 0,42; không có mức 0 |
| Hạng Fisher trên lịch probe thật (F1) | 10/10 với 10 tham số (trị riêng tương đối nhỏ nhất 0,20); 11/11 khi thêm r_AΦ |
| CRB dự đoán tại tham số danh nghĩa, gộp 58 phiên | k_inh 6,7 %, κ 7,9 %, k_d 10,4 %, α 1,3 %, β 2,7 %, σ_A 2,3 %, σ_Φ 2,7 %, r_A 0,8 %, r_Φ 0,7 %, r_Q 1,8 % |
| CRB theo nhóm (29 phiên mỗi nhóm) | k_inh 9,4 %, k_d 14,7 %, α 1,9 %, β 3,8 % (cả hai nhóm) |
| MDE hiệu log k_inh expert − novice (lực 80 %, α = 0,05) | **0,37** đơn vị log, tức hệ số 1,45 (ds001787: 0,51) |

Tham số danh nghĩa dùng cho CRB: **giống hệt ds001787** (`model/config.py::REALDATA.nominal_dev`): k_inh 0,20; κ 0,081; k_d 0,034 /phút;
σ_A 0,10; σ_Φ 0,05; α 2,34; β 1,87; r_A = r_Φ 0,10; r_Q 0,30. SNR danh nghĩa kênh DMN 1,58. Lưu ý: thang rating ở đây rộng hơn
(−7…7 thay vì −3…3) nên α, β và r_Q trên dữ liệu thật sẽ có đơn vị khác ds001787; để so sánh giữa hai bộ, α và β được báo thêm
dưới dạng chia cho SD trong phiên của rating (mục 3.7).

## 2. Giả thuyết và phép thử (khoá)

Năm giả thuyết F-H1…F-H5 lặp lại đúng E-H1…E-H5 của ds001787; ba mục R1–R3 là **kiểm tra lặp lại** kết quả đã thu được trên
ds001787 (`outputs/paper/realdata/summary.json`, 25/10/2026), viết ra trước khi xem EEG của bộ này.

| Mã | Dự đoán | Phép thử | Ủng hộ khi |
|---|---|---|---|
| F-H1 | Sai số tương đối của k_inh gộp ≤ 20 % nếu SNR đạt mức danh nghĩa | SE Fisher tại MLE; RMSE bootstrap 200 lần trên đúng thiết kế thật | SE ≤ 0,20 và RMSE/SE ∈ [0,5; 2]; nếu SNR fit được thấp hơn danh nghĩa: so SE với CRB tính lại tại SNR đó, báo cả hai |
| F-H2 | M1 dự đoán rating ngoài mẫu tốt hơn mô hình rỗng | LOSO, log predictive density Gauss tại probe; (a) chỉ EEG so với N2; (b) EEG + rating quá khứ so với N1 và N3 | ΔLPD > 0 với CI 95 % bootstrap theo phiên (2000 lần) loại 0, ở cả (a) và (b). *Trên ds001787 giả thuyết này KHÔNG đạt* ((a) −0,010, (b) so N1 −0,031); dự đoán giữ nguyên theo lý thuyết |
| F-H3 | α > 0 | CI 95 % của α (SE Fisher) | CI nằm trên 0 (ds001787: α = 13,1, CI 4,9–21,3) |
| F-H4 | Dấu của β, hai phía | CI 95 % của β | Báo dấu (ds001787: β ≈ 0, CI chứa 0) |
| F-H5a | k_inh expert > novice (lý thuyết gốc) | Hiệu log k_inh giữa hai fit theo nhóm (SE Fisher mỗi nhóm); kiểm định tỉ số hợp lý 1 bậc tự do | Hiệu > 0 và CI loại 0 |
| F-H5b | k_inh expert < novice (chiều quan sát được trên ds001787: −1,19 log, CI −1,56…−0,81) | như trên | Hiệu < 0 và CI loại 0. Chỉ một trong F-H5a/F-H5b có thể đạt; nếu CI chứa 0, báo kèm MDE 0,37 |
| R1 | Hằng số thời gian 1/k_inh lặp lại | Khoảng profile 95 % của 1/k_inh | Khoảng profile chồng lên khoảng của ds001787 [0,49; 1,17] phút; báo thêm tỉ số hai ước lượng điểm (ds001787: 0,76 phút) |
| R2 | Dấu hiệu sai đặc tả lặp lại | Tỉ số SE thông tin quan sát / SE Fisher cho k_inh, σ_A, r_A trên dữ liệu thật và trên dữ liệu mô phỏng từ θ̂ | Lặp lại khi tỉ số > 1,5 trên dữ liệu thật và ≈ 1 trên mô phỏng (ds001787: 2,30/2,48/2,14 so với 1,05–1,07) |
| R3 | 1/k_d | Ước lượng điểm và CI profile | Câu hỏi ước lượng, không có giả thuyết (ds001787: 5,0 phút, CI 3,1–8,7) |

**Khám phá, báo riêng, không dùng để kết luận:** buồn ngủ (mã thứ ba) như covariate, như Q3 của ds001787; mức tập trung / mức cuốn
theo suy nghĩ (mã thứ tư) thay cho độ chắc chắn trong định nghĩa rating; xu hướng trong phiên; fit chung hai bộ dữ liệu với k_inh
riêng từng bộ.

## 3. Kế hoạch phân tích (khoá)

1. **Đọc dữ liệu** (`eeg/osf_features.py`): 21 cột EEG, lấy một dòng trong mỗi nhóm 4 dòng lặp (pha phát hiện tự động trong 10 s
   đầu), đơn vị µV → V, bỏ các dòng NaN ở cuối file; montage `standard_1020` (T3/T4/T5/T6 → T7/T8/P7/P8); **hạ mẫu 512 → 256 Hz**
   để LZc được tính ở cùng tần số với ds001787.
2. **Tiền xử lý**: đúng mã của ds001787 (`eeg/realdata_features.py`, tham số `REALDATA`): lọc 1–40 Hz + notch 50 Hz, kênh xấu z > 4 nội suy,
   tham chiếu trung bình, ICA Picard mở rộng trên bản 1–100 Hz, ICLabel ≥ 0,8 (mắt, cơ); số thành phần = min(30, 21 − kênh xấu − 1).
3. **Lưới epoch**: lưới đều 10 s từ đầu bản ghi; epoch **đánh dấu thiếu** khi chồng lên cửa sổ probe [chuông, mã trả lời cuối + 3 s]
   (chuông + 15 s nếu không có mã) hoặc theo quy tắc biên độ. **Quy tắc biên độ chính = thích nghi theo phiên**
   (Q1_pipeline.md v2.2: đỉnh-đỉnh trên các kênh đặc trưng > max(150 µV, trung vị phiên + 5 MAD)); quy tắc cố định 150 µV trên mọi kênh
   là phân tích nhạy (mục 9). Lý do đổi so với ds001787, quyết định trước khi xem EEG: trên ds001787 quy tắc cố định loại 17/40 phiên
   vì khác biệt gain giữa phiên, không vì nhiễu. Rating là quan sát tại **đúng thời điểm chuông**, bộ lọc Kalman bước thay đổi.
4. **Rating** r = +độ chắc chắn nếu trả lời "tập trung hơi thở" (mã 1), −độ chắc chắn nếu "đang nghĩ chuyện khác" (2) hoặc "bị phân tán"
   (3); thang −7…7, không có 0. Probe không hợp lệ (trả lời ∉ {1,2,3} hoặc độ chắc chắn ∉ 1–7, hoặc thiếu mã) → không có rating.
   Các mã trước chuông đầu tiên là probe tập dượt, bỏ. **Chuẩn hoá**: trừ trung bình theo phiên cho y_A, y_Φ, r; chia y_A, y_Φ cho một SD gộp
   trong-phiên (một hằng số mỗi đặc trưng).
5. **Đặc trưng chính**: y_A = log công suất alpha 8–12 Hz vùng sau trên các kênh có mặt **P3, Pz, P4, O1, O2**; y_Φ = LZc khử phổ trên 19 kênh
   10-20 (đủ cả 19), 20 surrogate đa kênh. Ba phương án thay thế: alpha tuần hoàn, log theta 4–7 Hz trán giữa trên **Fz, Cz**, −exponent 1–40 Hz.
6. **Giữ phiên**: bỏ phiên có > 40 % epoch thiếu hoặc < 8 probe có rating. Báo số phiên bị bỏ theo nhóm.
7. **Mô hình và fit**: M1 dạng độ lệch, 10 tham số, cùng thang và cùng điểm xuất phát như ds001787 (6 điểm, seed 42); fit gộp 58 phiên;
   theo nhóm; theo người (mỗi người một phiên) + meta-analysis REML trên log k_inh. Báo α/SD_r và β/SD_r (SD_r = SD trong phiên gộp của rating)
   để so với ds001787.
8. **Ba con số sai số** cho k_inh, k_d, α, β: CRB dự đoán (bảng trên), SE Fisher tại MLE, RMSE bootstrap 200 lần; thêm SE thông tin quan sát
   (Hessian số) và CRB tại SNR fit được. Profile likelihood 9 điểm ±0,8 log cho k_inh, κ, k_d.
9. **Kiểm tra mô hình**: Ljung–Box trễ 1–30 trên innovation y_A, y_Φ (trung vị p > 0,01 → F3); tương quan innovation; LOSO như F-H2;
   dự đoán hậu nghiệm cho rating thứ bậc trên 15 mức −7…7 (mức 0 không tồn tại trong dữ liệu nhưng mô hình liên tục sinh ra được: báo, không
   dùng để đánh giá).
10. **Bảy phân tích nhạy** (báo tất cả): ba đặc trưng thay thế (mỗi lần thay một kênh), z-score theo phiên, chỉ phiên ≤ 20 % epoch thiếu,
    chỉ 20 probe đầu mỗi phiên, quy tắc biên độ cố định 150 µV. Kết luận chính chỉ phát biểu nếu dấu α, β và khoảng 1/k_inh giữ nguyên
    ở ≥ 80 % các phân tích (≥ 6/7).
11. **Cổng**: F2 = RMSE bootstrap / SE Fisher ∈ [0,5; 2] cho ≥ 3/4 tham số chính và profile đơn mốt; F3 = phần dư trắng + LOSO + ≥ 5 phân tích nhạy.
12. **Tiêu chí dừng**: K1 rating không giải mã được (không xảy ra: F0 PASS); K2 tự tương quan trễ 1 của y_A < 0,3 (trung vị) → dừng fit, báo
    SNR đo được; K3 > 50 % phiên bị loại → dừng; K4 không qua F2 → báo cáo như kết quả âm.

## 4. Khác biệt so với ds001787 (ghi trước khi khoá)

- Dữ liệu một phiên mỗi người (ds001787: 1–3 phiên) → meta-analysis theo người có 58 đơn vị, không có biến thiên giữa các phiên của cùng người.
- Rating hai cực và rộng hơn; dấu do câu trả lời quyết định, độ lớn do độ chắc chắn. Không có mức 0.
- 21 kênh thay vì 64: ít kênh sau hơn cho y_A (5 thay vì 9), 2 kênh trán giữa cho theta (thay vì 4); ICA tối đa 20 thành phần.
- Quy tắc biên độ chính là quy tắc thích nghi (mục 3.3).
- Probe đến dày hơn (77 s so với ~120 s) và phiên dài hơn (57 so với ~41 phút): CRB dự đoán tốt hơn (k_inh 6,7 % so với 9,1 %).
- Các file EEG đã nằm trên máy trước khi khoá (khai báo ở đầu); chỉ cột trigger và một đoạn 2 phút được đọc.

## 5. Những gì không được làm sau khi khoá

Không đổi định nghĩa rating, đặc trưng chính, cửa sổ che, quy tắc biên độ chính, tiêu chí giữ phiên, số điểm xuất phát, số lần bootstrap,
mô hình rỗng, danh sách phân tích nhạy, hai chiều của F-H5; không báo A0, Φ* hay mức tuyệt đối nào trên dữ liệu thật; không dùng các
file med/rest trong bài này (chúng được tải cùng lúc nhưng không thuộc kế hoạch phân tích).

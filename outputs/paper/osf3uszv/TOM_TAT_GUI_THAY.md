# Tóm tắt kết quả — Bộ dữ liệu thứ hai OSF 3uszv (mục 4.8 của bài)

Sinh viên: Messiuuu · Ngày: 09/11/2026 · Mã nguồn: `bqi_project` (lệnh `python main.py osf3uszv …`)

## 1. Làm gì

Theo yêu cầu của thầy (02/11/2026), lặp lại toàn bộ phân tích của ds001787 (mục 4.7) trên một bộ dữ liệu độc lập của phòng thí nghiệm khác: EEG thiền của Rodriguez-Larios và cộng sự 2021 (*NeuroImage*; OSF 3uszv, CC0). 29 người thiền trên ba năm và 29 người chưa từng thiền, mỗi người một phiên khoảng một giờ thiền hơi thở, chuông mỗi 77 s, 40 lần hỏi mỗi phiên (58 phiên, 2 258 lần hỏi, 55 giờ EEG, 21 kênh ở 512 Hz). Cùng mô hình M1, cùng mã, cổng F0–F3 riêng, bản đăng ký trước riêng khoá bằng SHA-256 lúc 03/11/2026 09:21 UTC (file đã tải trước khi khoá, nhưng chỉ cột trigger được đọc; có khai báo trong file).

Hai câu hỏi như ds001787, cộng ba kiểm tra lặp lại cố định trước: R1 hằng số thời gian DMN có trùng khoảng 0,49–1,17 phút của ds001787 không; R2 dấu hiệu sai đặc tả của kênh alpha có lặp lại không; R3 giá trị của 1/k_d.

## 2. Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Dữ liệu giữ lại | 48/58 phiên (25 người lâu năm, 23 người mới; 1 838 lần hỏi có điểm). 10 phiên loại vì hơn 40 % epoch thiếu. Riêng cửa sổ trả lời đã chiếm 34 % mỗi phiên. |
| Cổng kiểm tra | F0 (giải mã 98,9 %), F1, F2, F3 đều PASS. |
| Sai số của k_inh | CRB dự đoán trước khi khoá 6,7 %; đạt được: Fisher 8,5 %, Hessian 12,2 %, profile 11,8 %, bootstrap 8,8 %. Sai số Fisher vượt cận dự đoán 28 % (ds001787: 8 %), vẫn nằm gọn trong hệ số hai mà cận đã được kiểm chứng trên mô phỏng, trên phòng thí nghiệm khác, máy khác, thang điểm khác. |
| F-H1 (SE k_inh ≤ 20 %) | Đạt theo cả hai tiêu chí (Fisher và profile). |
| Hằng số thời gian | 1/k_inh = 7,1 phút (5,7–9,1), gấp chín lần 0,76 phút của ds001787; 1/k_d = 0,63 phút, so với 5,0 phút. Hai hằng số đổi chỗ cho nhau. R1 không lặp lại. |
| Sai đặc tả kênh alpha | Tỉ số SE Hessian/Fisher 1,4 (ds001787: 2,3), dưới ngưỡng 1,5 đã đăng ký; R2 không lặp lại, nhưng dấu hiệu vẫn còn. |
| F-H2 (dự đoán ngoài mẫu) | Không đạt: ΔLPD ≈ 0 so với mọi mô hình rỗng. Điểm tự chấm của bộ này không có phụ thuộc chuỗi và hai cực (88 % có \|r\| ≥ 5), nên không có gì để dự đoán và mô hình Gauss không tái tạo được phân bố. |
| F-H3 (α > 0) | Không đạt: α = 0,79, CI −0,24…1,82 chứa 0 (ds001787: đạt). |
| F-H4 (dấu β) | Không xác định được (β = 0,02). |
| F-H5 (khác biệt nhóm về k_inh) | Không đạt cả hai chiều: hiệu log +0,14 (CI −0,20…0,47, p = 0,10), dưới mức phát hiện được 0,37. Chiều ngược −1,19 của ds001787 không lặp lại. Hai nhóm khác nhau ở các tham số khác. |
| Dị biệt giữa người | τ = 1,49 log, I² = 0,92 (ds001787: 1,19 và 0,92): fit gộp cùng lắm mô tả một người trung bình. |
| Quy tắc độ bền 80 % | Đạt với dấu α (5/6), không đạt với β (1/6) và 1/k_inh (4/6). Biến thể ngoài đăng ký che đoạn trước lần hỏi đầu cho 1/k_inh = 16 phút. |

## 3. Cái gì lặp lại, cái gì không

Lặp lại trên cả hai bộ: (i) con số sai số theo Fisher chính xác; (ii) EEG không cải thiện dự đoán điểm tự chấm; (iii) dị biệt giữa người rất lớn.

Không lặp lại: (iv) α > 0; (v) các hằng số tốc độ và chiều khác biệt nhóm.

Cách đọc đề xuất trong bài: fit Ornstein–Uhlenbeck trả về thang thời gian trội của đặc trưng được đưa vào. Trên ds001787, LZc là nhiễu nên k_d đo độ trôi 5 phút của điểm tự chấm; trên OSF, LZc có tín hiệu khoảng 40 s ghim Φ, điểm không trôi, còn alpha trôi theo phút trong phiên dài một giờ. Thang thời gian đó là tính chất của bản ghi và của đặc trưng, chưa phải của DMN. Tham số chỉ có nghĩa khi ánh xạ từ đặc trưng EEG sang biến trạng thái được hiệu chuẩn.

## 4. Đã cập nhật trong bản thảo (`template_latex/`)

- Mục 4.8 mới, Hình 11 (một phiên ví dụ), Bảng so sánh hai bộ dữ liệu cạnh nhau, đoạn "Điều gì lặp lại".
- Phụ lục H.2: đăng ký trước và khai báo, năm sai lệch so với bản đăng ký, tiền xử lý, Hình 12 (profile), năm bảng (sai số, LOSO, độ nhạy, PPC, giả thuyết).
- Tóm tắt, đóng góp, thảo luận, hạn chế, kết luận, mục dữ liệu, phụ lục tái tạo.
- Bản tiếng Anh 50 trang và tiếng Việt 49 trang, biên dịch sạch.

## 5. Đề xuất về bài

Đủ để thành paper, nhưng là paper về phương pháp, và nên tách:

1. Bài mô hình: chứng minh QoC dừng ≡ 0, mô hình M1, khả năng xác định, bản đồ thiết kế CRB, điều khiển LQG. Chắc nhất, nộp được sau khi có preprint mô hình gốc để trích dẫn.
2. Bài kiểm định đăng ký trước trên hai bộ EEG: CRB đúng trên cả hai, EEG không dự đoán được điểm tự chấm, hằng số thời gian không lặp lại. Kết quả âm tính nhưng có đăng ký trước và hai phòng thí nghiệm.
3. (Nếu làm tiếp) bài mạch: LQG trên FPGA, cần tổng hợp thật.

## 6. Xin ý kiến thầy

1. Cách đọc "hằng số thời gian là của đặc trưng, không phải của DMN" ở mục 3 có ổn không?
2. Tách thành hai hay ba bài như mục 5, hay giữ một bài?
3. Tạp chí nhắm tới cho bài 1 và bài 2 (em nghĩ PLOS Computational Biology và Journal of Neuroscience Methods).

Số liệu: `outputs/paper/osf3uszv/summary.json`, `outputs/paper/comparison/comparison.json`. Bản đăng ký: `preregistration/predictions_osf3uszv.md`.

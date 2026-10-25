# Tóm tắt kết quả — Phần dữ liệu thật ds001787 (mục 4.7 của bài)

Sinh viên: Messiuuu · Ngày: 25/10/2026 · Mã nguồn: `bqi_project` (lệnh `python main.py realdata …`)

## 1. Làm gì

Fit mô hình M1 (dạng độ lệch, 10 tham số) lên bộ EEG thiền mở ds001787 (24 người, 40 phiên, 983 lần hỏi tự chấm) để kiểm tra hai dự đoán của bài:

1. Bản đồ thiết kế có dự đoán đúng sai số ước lượng trên một thiết kế thật không.
2. Mô hình có dự đoán điểm tự chấm ngoài mẫu tốt hơn các mô hình rỗng không.

Mọi phân tích theo bản đăng ký trước đã khoá bằng SHA-256 lúc 19/10/2026 13:48 UTC, trước khi tải file EEG đầu tiên (13:49 UTC). Metadata và file hành vi được xem trước khi khoá, EEG thì không.

## 2. Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Dữ liệu giữ lại | 22/40 phiên (12 người thiền lâu năm, 10 người mới; 16 người; 554 lần hỏi có điểm). Quy tắc 150 µV đã khoá loại 45 % số phiên, dưới ngưỡng dừng 50 %, nhưng không ngẫu nhiên (nhiễu cơ thái dương). |
| Cổng kiểm tra | E0, E1, E2, E3 đều PASS, thời điểm ghi tăng dần. |
| Hằng số thời gian DMN | 0,76 phút (CI profile 0,49–1,17), không phải 5 phút (danh nghĩa) hay 23 phút (bài BQI). |
| Sai số của k_inh | SE Fisher 9,9 %, nằm trong khoảng 9,1–13,3 % dự đoán trước, nên bản đồ thiết kế đúng theo Fisher. Nhưng SE theo profile và theo Hessian là 22 %, gấp đôi: kênh alpha bị sai đặc tả. Trên dữ liệu mô phỏng từ chính mô hình, độ lệch này biến mất. |
| E-H1 (SE k_inh ≤ 20 %) | Đạt theo tiêu chí đã khoá (SE Fisher), không đạt nếu dùng SE profile. |
| E-H2 (dự đoán ngoài mẫu) | Không đạt: thêm EEG không giúp dự đoán điểm tự chấm. |
| E-H3 (α > 0) | Đạt về dấu, nhưng LZc gần như chỉ là nhiễu, nên α chủ yếu phản ánh chính điểm tự chấm. |
| E-H4 (dấu β) | Không xác định được. |
| E-H5 (k_inh người lâu năm > người mới) | Không đạt. Fit riêng cho hiệu ngược chiều (−1,19 log), nhưng mất ý nghĩa khi hai nhóm dùng chung tham số nhiễu (p = 0,068). Hai nhóm còn lệch về giới và tuổi. |
| Quy tắc độ bền 4/5 | Đạt với α, không đạt với β và 1/k_inh, nên bài trình bày kết quả theo lối mô tả. |

## 3. Kết luận để viết

Kết quả vững nhất là dao động công suất alpha vùng sau có thang thời gian dưới một phút. Thành phần chậm khoảng 5 phút của điểm tự chấm được mô hình nắm bắt, nhưng cả alpha lẫn LZc đều không mang thông tin dự đoán về nó. Đây là kết quả âm cho liên hệ EEG–QoC, và bài báo cáo trung thực như vậy. Bài học phương pháp: con số của bản đồ thiết kế nên đọc như cận dưới khi đại diện EEG chưa chắc tuân theo mô hình.

## 4. Đã cập nhật trong bản thảo (`template_latex/`)

- Mục 4.7 mới, Hình 9 (một phiên ví dụ) và Hình 10 (profile trên dữ liệu thật), Bảng 9 (ba con số sai số) và Bảng 10 (dự đoán bỏ-một-phiên).
- Phụ lục H: các sai lệch so với bản đăng ký trước, chi tiết tiền xử lý, phân tích độ nhạy, kiểm tra dự đoán hậu nghiệm, bảng các giả thuyết.
- Tóm tắt, đóng góp, thảo luận, hạn chế, kết luận, mục dữ liệu và mã (thêm ds001787 và DOI), phụ lục tái tạo.
- Bản tiếng Anh (41 trang) và tiếng Việt (40 trang) đều biên dịch sạch, không cảnh báo.

## 5. Xin ý kiến thầy

1. Giữ phần dữ liệu thật làm mục 4.7 như kế hoạch, hay chuyển phần lớn sang phụ lục để bài model gọn hơn?
2. Cách trình bày các kết quả âm (E-H2, E-H5) và sai đặc tả của kênh alpha đã đủ thận trọng chưa?
3. Kết quả này có làm thầy đổi ý về tạp chí dự kiến không?

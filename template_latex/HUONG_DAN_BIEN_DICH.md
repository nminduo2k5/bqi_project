# Hướng dẫn biên dịch hai bản thảo LaTeX

Thư mục `template_latex/` chứa hai bản thảo của cùng một bài:

| File | Ngôn ngữ | Ghi chú |
|---|---|---|
| `paper_reference.tex` | Tiếng Anh | Bản để nộp tạp chí |
| `paper_reference_vi.tex` | Tiếng Việt | Cùng công thức, số liệu, hình, bảng và danh mục tài liệu |
| `sync_bib.py` | — | Chép danh mục tài liệu từ bản Anh sang bản Việt |
| `bqi_original_unified_backup.tex` | Tiếng Anh | Bài BQI gốc (bản template cũ), không biên dịch |

Hai bản dùng chung 8 hình và số liệu trong `../outputs/paper/model/` (do `python main.py model figures --profiles` sinh ra). **Không chép hình vào thư mục này**; file .tex đọc thẳng qua `\graphicspath{{../outputs/paper/model/}}`.

## 1. Yêu cầu

- MiKTeX (Windows) hoặc TeX Live 2022 trở lên. Máy hiện tại đã có MiKTeX tại `C:\Users\HP\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`.
- Lần biên dịch đầu MiKTeX có thể hỏi cài thêm gói (tcolorbox, pgfplots, physics, cleveref, vntex…). Chọn **Install** hoặc bật *"Always install missing packages on-the-fly"* trong MiKTeX Console.
- Bản tiếng Việt cần bộ font T5 (gói `vntex`, MiKTeX tự cài khi gặp `\usepackage[T5]{fontenc}`). Không cần XeLaTeX; cả hai bản dùng **pdflatex**.
- Thư mục `../outputs/paper/model/` phải có `fig1_highpass_qoc.pdf` … `fig8_sobol.pdf`. Nếu thiếu, chạy trước ở thư mục gốc dự án:
  ```
  python main.py model figures --profiles
  ```

## 2. Biên dịch

Mở terminal **trong thư mục `template_latex/`** (đường dẫn hình là tương đối). Mỗi bản chạy pdflatex **hai lần** để tham chiếu chéo (số hình, số bảng, trích dẫn) ổn định.

**Bản tiếng Anh**
```
pdflatex -interaction=nonstopmode paper_reference.tex
pdflatex -interaction=nonstopmode paper_reference.tex
```
→ `paper_reference.pdf` (~32 trang).

**Bản tiếng Việt**
```
python sync_bib.py
pdflatex -interaction=nonstopmode paper_reference_vi.tex
pdflatex -interaction=nonstopmode paper_reference_vi.tex
```
→ `paper_reference_vi.pdf` (~31 trang).

`sync_bib.py` chép nguyên khối `\begin{thebibliography} … \end{thebibliography}` của bản Anh vào giữa hai dấu `%%BIBLIOGRAPHY-BEGIN` / `%%BIBLIOGRAPHY-END` của bản Việt. Chỉ sửa tài liệu tham khảo ở **bản Anh** rồi chạy lại script; không sửa tay phần đó trong bản Việt.

**Cả hai bản một lệnh** (PowerShell, trong `template_latex/`):
```powershell
pdflatex -interaction=nonstopmode paper_reference.tex; pdflatex -interaction=nonstopmode paper_reference.tex
python sync_bib.py
pdflatex -interaction=nonstopmode paper_reference_vi.tex; pdflatex -interaction=nonstopmode paper_reference_vi.tex
```
Hoặc dùng `latexmk -pdf paper_reference.tex` và `latexmk -pdf paper_reference_vi.tex` (latexmk tự chạy đủ số lần).

## 3. Kiểm tra sau khi biên dịch

- Dòng cuối log phải có `Output written on … .pdf`.
- Không có dòng bắt đầu bằng `!` (lỗi). Cảnh báo `Overfull \hbox` vài pt là bình thường.
- Tìm `undefined` trong file `.log`: nếu có *Citation undefined* hay *Reference undefined* thì chạy pdflatex thêm một lần.
- Đối chiếu số liệu với `../outputs/results/model/` nếu đã chạy lại pipeline (xem mục 5).

## 4. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách sửa |
|---|---|
| `! LaTeX Error: File 'fig5_design_map.pdf' not found` | Chưa chạy `python main.py model figures --profiles`, hoặc đang biên dịch từ thư mục khác `template_latex/` |
| `! LaTeX Error: Command \dd already defined` | Gói `physics` đã định nghĩa `\dd`; trong hai file đã dùng `\renewcommand{\dd}`. Đừng đổi lại thành `\newcommand` |
| `Font T5/pplx/m/n/12 not found` (bản Việt) | Thiếu gói `vntex`; cài qua MiKTeX Console (Packages → tìm `vntex`) rồi chạy lại |
| Chữ Việt mất dấu trong PDF | Nguồn phải lưu UTF-8; file đã có `\usepackage[utf8]{inputenc}` và `[T5]{fontenc}`. Kiểm tra editor không lưu thành ANSI |
| `--seed` hiện thành gạch dài `–seed` | Trong `\ttfamily` phải viết `-{}-seed`; hai file đã làm vậy |
| Cột `p{…}` trong `tabular` báo `Undefined control sequence` | Xung đột với bộ gói bảng của template; dùng cột `l` và rút gọn chữ |
| MiKTeX treo ở "Installing package" | Mạng chậm; đợi hoặc cài gói trước bằng MiKTeX Console |

## 5. Khi chạy lại pipeline và số liệu thay đổi

Các bảng trong hai file .tex là **gõ tay** từ `../outputs/results/model/` và `../outputs/paper/model/table*.tex` (để có thêm cột và ghi chú). Nếu chạy lại `python main.py model all` với thiết lập khác, cần cập nhật:

- Bảng 1–3 theo `table1_steady_states.tex`, `table2_identifiability.csv`, `table3_min_designs.tex`
- Bảng chính sách điều khiển và bảng độ bền theo `control_results.json`
- Bảng Sobol theo `sobol_indices.csv`, dòng hội tụ theo `sensitivity_results.json`
- Các con số trong lời văn: cổng M1 (`gates/M1.json`), P1–P5 (`theory_results.json`), profile (`profiles.csv`)

Script `scratchpad/check_tables.py` dùng trong phiên làm việc 10/10/2026 so từng ô bảng của bản Anh với các file trên; có thể dùng lại làm mẫu.

## 6. Những chỗ còn để trống (TODO) trong cả hai bản

- E-mail cơ quan của tác giả liên hệ ở trang bìa (`[institutional e-mail]` / `[e-mail cơ quan]`).
- Tài liệu `\bibitem[Vu(2025)]{BQI}`: thay "Unpublished manuscript" bằng trích dẫn preprint có DOI khi đã nộp.
- Mã số đề tài trong Lời cảm ơn nếu có.

"""Chép nguyên văn khối \\begin{thebibliography} ... \\end{thebibliography}
từ paper_reference.tex (bản tiếng Anh) vào paper_reference_vi.tex (bản tiếng Việt),
giữa hai dấu %%BIBLIOGRAPHY-BEGIN / %%BIBLIOGRAPHY-END, để hai bản luôn dùng
cùng một danh mục tài liệu. Chạy:  python sync_bib.py  (trong thư mục template_latex).
"""
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
en = open(os.path.join(here, "paper_reference.tex"), encoding="utf-8").read()
vi_path = os.path.join(here, "paper_reference_vi.tex")
vi = open(vi_path, encoding="utf-8").read()

m = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", en, flags=re.S)
if not m:
    raise SystemExit("Không tìm thấy thebibliography trong paper_reference.tex")
bib = m.group(0)

new = re.sub(r"%%BIBLIOGRAPHY-BEGIN.*?%%BIBLIOGRAPHY-END",
             lambda _: "%%BIBLIOGRAPHY-BEGIN\n" + bib + "\n%%BIBLIOGRAPHY-END", vi, flags=re.S)
if new == vi and "%%BIBLIOGRAPHY-BEGIN" not in vi:
    raise SystemExit("Không tìm thấy dấu %%BIBLIOGRAPHY-BEGIN/END trong paper_reference_vi.tex")
open(vi_path, "w", encoding="utf-8").write(new)
print(f"Copied {bib.count(chr(92) + 'bibitem')} bibitem entries into paper_reference_vi.tex")

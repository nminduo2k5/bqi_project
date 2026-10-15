# Neural design: vòng kín M1 (Kalman + LQG) dưới dạng mạch neural dấu chấm tĩnh, mô hình hoá bằng VHDL

Gói này hiện thực hoá **bộ điều khiển vòng kín** của bài Q1 (Gói 4, `model/control.py`) thành một
mạch số nhỏ để chạy neurofeedback thời gian thực trên FPGA. Mô hình M1 tuyến tính nên toàn bộ
vòng kín là các phép **tổng có trọng số + độ lệch**, tức mỗi phép tính là một **nơ-ron tuyến tính**
với "hàm kích hoạt" là làm tròn + bão hoà. Mạch có 3 lớp, 6 nơ-ron, chạy một lần mỗi epoch 10 s.

```
          y_A, y_Phi (EEG, mỗi epoch)                    
                 │                                        
   ┌─────────────▼──────────────┐    lớp 2  HIỆU CHỈNH   x̂ = (I−L)·x_pred + L·y      (2 nơ-ron, 4 đầu vào)
   │  ┌──────────────────────┐  │                         
   │  │ lớp 1  DỰ ĐOÁN       │  │    x_pred = F·x̂ + B·u + c                     (2 nơ-ron, 3 đầu vào)
   │  └──────────▲───────────┘  │                         
   │             │ x̂, u (trạng thái mạch, nhớ giữa hai epoch)
   │  ┌──────────┴───────────┐  │                         
   │  │ lớp 3  ĐIỀU KHIỂN    │──┼──► u = clip(−K·x̂ + (u_ss + K·x_ss), ±u_max)    (1 nơ-ron)
   │  │        ĐỌC RA        │──┼──► QoC^ = h·x̂ + γ0                             (1 nơ-ron, giám sát)
   │  └──────────────────────┘  │                         
   └────────────────────────────┘                         
```

Thứ tự tính trong một epoch đúng như `simulate_policy` của Gói 4: nhận `y_{t+1}` → dự đoán từ
`(x̂_t, u_t)` → hiệu chỉnh → `u_{t+1}`. Trạng thái reset là kỳ vọng dừng `μ` của M1 và
`u_0 = control(μ)`.

## Thành phần

| File | Vai trò |
|---|---|
| `model/hardware.py` | Bản tham chiếu **bit-exact** bằng Python: định dạng dấu chấm tĩnh, nơ-ron, mạch `NeuralCircuit`, vòng kín với đối tượng (não) ở dấu chấm động, quét số bit, sinh gói hằng số VHDL + vectơ kiểm tra |
| `hw/vhdl/neuron_pkg.vhd` | Hàm `neuron` (MAC chính xác → cộng ½ LSB → dịch số học → bão hoà), `clip`, `saturate` |
| `hw/vhdl/dmn_lqg_core.vhd` | Mạch RTL: FSM 3 trạng thái (một lớp mỗi chu kỳ), thanh ghi `x̂`, `u`, `QoC^`, tín hiệu `start`/`done` |
| `hw/vhdl/tb_dmn_lqg_core.vhd` | Testbench đọc `gen/vectors.txt`, so **từng bit** 4 đầu ra ở mỗi epoch với Python |
| `hw/vhdl/gen/lqg_params_pkg.vhd` | **Sinh tự động**: W, FRAC và mọi trọng số/độ lệch đã lượng tử hoá |
| `hw/vhdl/gen/vectors.txt` | **Sinh tự động**: 270 epoch của một phiên vòng kín (seed 42) |
| `hw/run_ghdl.ps1`, `hw/run_ghdl.sh` | Chạy mô phỏng với GHDL |
| `tests/test_hardware.py` | 5 test: làm tròn/bão hoà, đường float trùng `simulate_policy` tới 1e-12, dấu chấm tĩnh ≈ float, mạch tất định và vectơ tự nhất quán, file sinh ra hợp lệ |

## Định dạng số và ngữ nghĩa nơ-ron (Python và VHDL phải giống hệt)

- Số có dấu bù hai, `W = FRAC + 4 + 1` bit (4 bit phần nguyên, dải ±16; đủ cho K = 4,6 và độ lệch lớp 3 = 3,99).
- `q = round(v · 2^FRAC)` cho hằng số và cho đầu vào `y`; đầu vào bão hoà về W bit.
- Nơ-ron: `acc = (bias << FRAC) + Σ wᵢ·xᵢ` trên bộ tích luỹ `2W+3` bit (không tràn), rồi
  `y = sat_W((acc + 2^(FRAC−1)) >>ₐ FRAC)` (làm tròn nửa lên, dịch số học). Python dùng `>>` trên int
  (dịch sàn), VHDL dùng `shift_right` trên `signed` (số học): cùng kết quả.
- Lớp 3 cắt thêm ở `±u_max` (`clip`).

## Kết quả quét số bit (`python main.py model hardware`, 20 phiên × 270 epoch, seed 42)

Đối tượng ở dấu chấm động, chỉ bộ điều khiển bị lượng tử hoá; chi phí = MSE bám mục tiêu + 0,1·u².

| FRAC | W (bit) | chi phí dấu chấm tĩnh | chi phí float | tỉ số | max \|u_fixed − u_float\| |
|---|---|---|---|---|---|
| 6  | 11 | 0,05507 | 0,03942 | 1,397 | 1,5·10⁻¹ |
| **8**  | **13** | **0,03972** | 0,03942 | **1,0076** | 3,7·10⁻² |
| 10 | 15 | 0,03945 | 0,03942 | 1,0007 | 8,3·10⁻³ |
| 12 | 17 | 0,03943 | 0,03942 | 1,0004 | 2,3·10⁻³ |
| 14 | 19 | 0,03941 | 0,03942 | 0,9998 | 5,1·10⁻⁴ |
| 16 | 21 | 0,03942 | 0,03942 | 1,0001 | 1,7·10⁻⁴ |
| 18 | 23 | 0,03942 | 0,03942 | 1,0000 | 2,9·10⁻⁵ |

Quy tắc chọn: FRAC nhỏ nhất có chi phí vượt float không quá 1 % → **Q4.8, W = 13 bit** (thua float 0,76 %).
Dưới 8 bit phân số vòng kín suy giảm nhanh (FRAC = 6: +40 %); từ 10 bit trở lên không còn khác biệt đo được.
Hình: `outputs/results/model/hw_quantisation.png`. Số liệu: `outputs/results/model/hardware_results.json`.

Ý nghĩa cho FPGA: một phép nhân 13×13 bit vừa một khối DSP; mạch cần tối đa 8 phép nhân đồng thời
(lớp 2), 3 chu kỳ xung mỗi epoch; với epoch 10 s thì bất kỳ xung nhịp nào cũng dư thừa, nên có thể
dùng chung một bộ nhân duy nhất nếu muốn tiết kiệm tài nguyên (chưa làm, xem "Việc tiếp theo").

## Chạy

```
set OMP_NUM_THREADS=1
python main.py model hardware            # quét số bit, sinh gen/lqg_params_pkg.vhd + gen/vectors.txt (~20 s)
python -m pytest tests/test_hardware.py  # 5 test Python
```

Mô phỏng VHDL bằng **GHDL 6.0.0** (bản `Windows (x86-64, UCRT64, standalone) - mcode backend`
từ https://github.com/ghdl/ghdl/releases; trên máy này giải nén ở `C:\ghdl-mcode-6.0.0-ucrt64`, thêm
`C:\ghdl-mcode-6.0.0-ucrt64\bin` vào PATH):

```
powershell -ExecutionPolicy Bypass -File hw\run_ghdl.ps1
```

**Kết quả 15/10/2026 (GHDL 6.0.0, VHDL-2008):**

```
tb_dmn_lqg_core: 270 vectors, W=13 FRAC=8
PASS: 270 epochs bit-exact with model/hardware.py
```

Kiểm tra ngược: sửa sai 1 LSB của `u` ở epoch 100 trong file vectơ thì testbench báo
`MISMATCH epoch 100 ... u=-3 ... expected u=-2` và `FAIL: 1 of 270 epochs differ`, tức phép so sánh có hiệu lực.

### Xem dạng sóng (giao diện)

GHDL chỉ chạy dòng lệnh. Để nhìn mạch chạy theo từng nhịp đồng hồ:

1. `powershell -ExecutionPolicy Bypass -File hw\run_ghdl.ps1 -Wave` → tạo `hw\vhdl\work\tb_dmn_lqg_core.vcd` và `.ghw`.
2. Tải **Surfer** bản Windows dựng sẵn (một file `.exe`, không cần cài):
   https://gitlab.com/api/v4/projects/42073614/jobs/artifacts/main/raw/surfer_win.zip?job=windows_build
   (link chính thức trên surfer-project.org; **không** clone repo `.git`, đó là mã nguồn).
   Có thể dùng GTKWave thay thế, nhưng phải tải đúng bản `gtkwave-3.3.100-bin-win64.zip`; các gói `gtkwave-gtk3-3.3.1xx` chỉ là mã nguồn.
3. Kéo thả file `.vcd` vào Surfer, mở `tb_dmn_lqg_core` → `dut`, thêm `clk`, `start`, `state`, `xhat`, `u_r`, `qoc_r`, `done`;
   đặt định dạng số là signed decimal (giá trị thật = số nguyên / 256 với Q4.8).

Mỗi epoch thấy `start` lên một nhịp, `state` đi IDLE → PREDICT → CORRECT → CONTROL, `done` bật và `u_r`, `xhat` đổi giá trị.

### Giới hạn của bộ vectơ hiện tại

trong phiên seed 42, `u` nằm trong [−106, 233] so với giới hạn ±512 và
không có giá trị nào chạm bão hoà W bit, nên nhánh `clip`/`saturate` của VHDL **chưa được kiểm thử**
(xem "Việc tiếp theo", mục 6).

Ba lỗi đã sửa khi chạy GHDL lần đầu (ghi lại để tránh lặp lại):
1. Comment nhiều dòng trong gói sinh tự động (ma trận F in xuống dòng, mất `--`) → `model/hardware.py` in mỗi ma trận trên một dòng; test kiểm mọi dòng của gói.
2. VHDL không phân biệt hoa thường: tham số `w` của hàm `neuron` che hằng số độ rộng `W` → đổi tên thành `wt`.
3. PowerShell tách `-gVEC_FILE=gen/vectors.txt` ở dấu chấm → đặt tham số trong ngoặc kép trong `run_ghdl.ps1`.

## Quan hệ với bài Q1 và với BQI gốc

- BQI gốc đặt bài toán điều khiển là COM (Thuật toán 2, dòng 5: `u ← K(QoC_target − QoC)`) và
  nghiệm bang-bang (pt. 69–75). Bài Q1 viết lại thành LQG. Mạch này là **hiện thực phần cứng của LQG đó**:
  cùng ma trận F, B, c, L, K, x_ss, u_ss (in trong phần chú thích của `lqg_params_pkg.vhd`).
- Hình 7 của bài Q1 đo mất mát do *sai số ước lượng tham số*; bảng trên đo mất mát do *lượng tử hoá*.
  Hai nguồn sai số này cộng vào nhau trong một hệ neurofeedback thật.
- Không có tuyên bố nào về não thật; mạch điều khiển một mô hình.

## Việc tiếp theo (đề xuất cho bài thứ hai)

1. ~~Cài GHDL, chạy testbench → xác nhận bit-exact~~ (xong 15/10/2026).
2. Tổng hợp thử trên Vivado/Quartus (phiên bản miễn phí) để có số LUT/FF/DSP và tần số, thêm vào bảng.
3. Phiên bản dùng chung một bộ nhân (18 chu kỳ/epoch) để so tài nguyên.
4. Mô hình đối tượng (M1 + nhiễu LFSR) cũng bằng VHDL để mô phỏng vòng kín hoàn toàn trên FPGA.
5. Độ bền kết hợp: bộ điều khiển thiết kế trên tham số ước lượng (24 ô của Hình 7) **và** lượng tử hoá Q4.8.
6. Kiểm tra khi QoC bão hoà: `clip` và `saturate` đã có, nhưng cần vectơ kiểm tra có đầu vào cực đoan
   (y ngoài dải, u chạm ±u_max) để kiểm thử nhánh bão hoà của VHDL.
7. **Thay phương trình DMN bằng mạng neural học từ dữ liệu** (cách hiểu thứ hai của "neural design", chưa làm, chờ thầy xác nhận).
   Các nơ-ron hiện tại là tuyến tính với trọng số tính sẵn từ phương trình, không học. Không có dữ liệu thật thì mạng neural chỉ học lại M1;
   cách làm có ý nghĩa: sinh dữ liệu từ một "não giả" phi tuyến (chuyển trạng thái, rating thứ bậc), fit cả M1 và một neural ODE/RNN nhỏ,
   so độ chính xác dự đoán với khả năng xác định tham số, rồi đưa mạng học được sang VHDL bằng khung mạch này.

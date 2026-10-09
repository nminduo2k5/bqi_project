#!/usr/bin/env python3
"""
main.py — Điểm khởi chạy trung tâm (CLI) cho dự án BQI.

Tổng hợp 3 việc: sinh dữ liệu tổng hợp (datasets), chạy thực nghiệm cho toàn
bộ thuật toán (experiments), và chạy bộ kiểm thử (tests) — tất cả điều khiển
được qua tham số dòng lệnh, không cần sửa code.

VÍ DỤ SỬ DỤNG
--------------
  # Chạy toàn bộ pipeline (sinh dữ liệu -> thực nghiệm -> test) với seed mặc định
  python main.py all

  # Chỉ sinh dữ liệu tổng hợp (tất cả), với seed tuỳ chỉnh
  python main.py generate-data --seed 123

  # Chỉ sinh một số dataset cụ thể, với kích thước tuỳ chỉnh
  python main.py generate-data --only pci,ode --n-subjects-per-state 60

  # Chạy toàn bộ thực nghiệm (10 thuật toán) và lưu hình + summary.json
  python main.py run-experiments --seed 7

  # Chỉ chạy một số thực nghiệm
  python main.py run-experiments --only bqi,com,hopfield

  # Chạy bộ test (pytest), có thể chỉ định 1 file/module test
  python main.py test
  python main.py test --path tests/test_hopfield.py -k "modern"

  # Xem danh sách dataset đã sinh ra và đọc nhanh 1 dataset
  python main.py list-data
  python main.py inspect-data --name pci_synthetic_subjects.csv

  # Xem thông tin cấu hình / các subcommand có sẵn
  python main.py info

  # Mở dashboard trực quan (Streamlit)
  python main.py dashboard --port 8501

  # Bài mô hình (Q1_pipeline_model.md): 5 gói kết quả, cổng M0–M2, kết quả vào outputs/results/model
  python main.py model theory              # P1–P5, cổng M0 (vài giây)
  python main.py model identifiability     # Rothenberg + CRB chính xác, Bảng 2 (~1 phút)
  python main.py model design              # lưới CRB + phục hồi MLE 24 ô, cổng M1 (nhiều giờ; --fast để thử)
  python main.py model control             # LQG vs bang-bang, cổng M2, độ bền theo design_grid_mle.csv
  python main.py model sensitivity --n-base 256   # Sobol/Saltelli cho bài (~1.5 giờ; mặc định 64 ~20 phút)
  python main.py model figures             # xuất hình PNG/PDF + bảng CSV/LaTeX vào outputs/paper/model
  python main.py model all --seed 42       # 5 bước + xuất hình: tái tạo toàn bộ bài

  # Pipeline Q1 (Q1_pipeline.md): kiểm định công cụ, cổng, đăng ký trước
  python main.py validate-tools            # cổng G0 (dữ liệu tổng hợp)
  python main.py validate-tools --g3       # + cổng G3 (phục hồi tham số mô hình 7.2, chậm)
  python main.py gates                     # trạng thái các cổng + khoá đăng ký trước
  python main.py preregister               # khoá preregistration/predictions.md (không đảo ngược)
  python main.py eeg chennu                # tiền xử lý + đặc trưng, bộ tham chiếu propofol
  python main.py eeg anphy --max-subjects 15   # tải từng người -> epoch theo giai đoạn ngủ -> đặc trưng
  python main.py eeg g1 | axis | status    # cổng G1, trục mức-ý-thức (G2), trạng thái
"""
from __future__ import annotations
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "generated")
DEFAULT_FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
DEFAULT_RES_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")


# =============================================================================
# Subcommand: generate-data
# =============================================================================
def cmd_generate_data(args):
    from datasets.generate_datasets import generate_all, ALL_DATASETS

    only = args.only.split(",") if args.only else None
    if only:
        valid = [o for o in only if o in ALL_DATASETS]
        invalid = set(only) - set(ALL_DATASETS)
        if invalid:
            print(f"[CẢNH BÁO] Bỏ qua dataset không hợp lệ: {sorted(invalid)}. "
                  f"Các lựa chọn hợp lệ: {ALL_DATASETS}")
        only = valid or None

    written = generate_all(
        seed=args.seed,
        out_dir=args.out_dir,
        only=only,
        n_subjects_per_state=args.n_subjects_per_state,
        n_ode_subjects=args.n_ode_subjects,
        n_bootstrap=args.n_bootstrap,
        n_kuramoto_trials=args.n_kuramoto_trials,
        n_quantum_trials=args.n_quantum_trials,
        verbose=not args.quiet,
    )
    print(f"\n✔ Đã sinh {sum(len(v) for v in written.values())} file dữ liệu.")


# =============================================================================
# Subcommand: run-experiments
# =============================================================================
def cmd_run_experiments(args):
    from experiments.run_all import run_all, ALL_EXPERIMENTS

    only = args.only.split(",") if args.only else None
    if only:
        valid = [o for o in only if o in ALL_EXPERIMENTS]
        invalid = set(only) - set(ALL_EXPERIMENTS)
        if invalid:
            print(f"[CẢNH BÁO] Bỏ qua thực nghiệm không hợp lệ: {sorted(invalid)}. "
                  f"Các lựa chọn hợp lệ: {ALL_EXPERIMENTS}")
        only = valid or None

    summary = run_all(
        seed=args.seed,
        fig_dir=args.fig_dir,
        res_dir=args.res_dir,
        only=only,
        save_figures=not args.no_figures,
        verbose=not args.quiet,
    )
    print(f"\n✔ Hoàn tất {len(summary)} thực nghiệm.")
    if args.print_summary:
        print(json.dumps(summary, indent=2, default=str))


# =============================================================================
# Subcommand: test
# =============================================================================
def cmd_test(args):
    import pytest

    pytest_args = [args.path or os.path.join(PROJECT_ROOT, "tests")]
    if args.keyword:
        pytest_args += ["-k", args.keyword]
    if args.verbose_pytest:
        pytest_args += ["-v"]
    else:
        pytest_args += ["-q"]
    if args.stop_on_fail:
        pytest_args += ["-x"]
    if args.markers:
        pytest_args += ["-m", args.markers]

    print(f"Chạy pytest với tham số: {pytest_args}\n")
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


# =============================================================================
# Subcommand: list-data
# =============================================================================
def cmd_list_data(args):
    data_dir = args.out_dir or DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"Chưa có dữ liệu nào trong {data_dir}. Hãy chạy: python main.py generate-data")
        return
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        print(f"Thư mục {data_dir} rỗng. Hãy chạy: python main.py generate-data")
        return
    import pandas as pd
    print(f"Dữ liệu hiện có trong {data_dir}:\n")
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            n_rows = sum(1 for _ in open(path)) - 1
        except Exception:
            n_rows = "?"
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f:<45} {n_rows:>6} dòng   {size_kb:>7.1f} KB")


# =============================================================================
# Subcommand: inspect-data
# =============================================================================
def cmd_inspect_data(args):
    import pandas as pd
    data_dir = args.out_dir or DEFAULT_DATA_DIR
    path = os.path.join(data_dir, args.name)
    if not os.path.isfile(path):
        print(f"[LỖI] Không tìm thấy file: {path}")
        print("Chạy 'python main.py list-data' để xem các file hiện có, "
              "hoặc 'python main.py generate-data' để sinh dữ liệu trước.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"=== {args.name} ===")
    print(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột\n")
    print("Các cột:", list(df.columns), "\n")
    print("5 dòng đầu:")
    print(df.head(5).to_string(index=False))
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print("\nThống kê mô tả (cột số):")
        print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]].to_string())
    if args.group_by and args.group_by in df.columns and len(numeric_cols) > 0:
        print(f"\nTrung bình theo nhóm '{args.group_by}':")
        print(df.groupby(args.group_by)[list(numeric_cols)].mean().to_string())


# =============================================================================
# Subcommand: all  (pipeline đầy đủ: generate-data -> run-experiments -> test)
# =============================================================================
def cmd_all(args):
    print("=" * 70)
    print("BƯỚC 1/3 — Sinh dữ liệu tổng hợp (generate-data)")
    print("=" * 70)
    cmd_generate_data(args)

    print("\n" + "=" * 70)
    print("BƯỚC 2/3 — Chạy toàn bộ thực nghiệm (run-experiments)")
    print("=" * 70)
    cmd_run_experiments(args)

    if not args.skip_tests:
        print("\n" + "=" * 70)
        print("BƯỚC 3/3 — Chạy bộ kiểm thử (pytest)")
        print("=" * 70)
        import pytest
        exit_code = pytest.main([os.path.join(PROJECT_ROOT, "tests"), "-q"])
        if exit_code != 0:
            print("\n⚠ Một số test thất bại — xem log phía trên.")
            sys.exit(exit_code)
    print("\n✔ Pipeline hoàn tất: dữ liệu, thực nghiệm và test đều đã chạy xong.")


# =============================================================================
# Subcommand: info
# =============================================================================
def cmd_info(args):
    from datasets.generate_datasets import ALL_DATASETS
    from experiments.run_all import ALL_EXPERIMENTS
    print(__doc__)
    print("Các dataset có thể sinh (--only cho generate-data):", ", ".join(ALL_DATASETS))
    print("Các thực nghiệm có thể chạy (--only cho run-experiments):", ", ".join(ALL_EXPERIMENTS))
    print("\nThư mục dữ liệu mặc định:  ", DEFAULT_DATA_DIR)
    print("Thư mục hình ảnh mặc định: ", DEFAULT_FIG_DIR)
    print("Thư mục kết quả mặc định: ", DEFAULT_RES_DIR)


# =============================================================================
# Subcommands: Q1 pipeline (validate-tools, gates, preregister)
# =============================================================================
def cmd_validate_tools(args):
    from experiments.validate_tools import main as validate_main
    argv = (["--g3"] if args.g3 else []) + (["--force-gate"] if args.force_gate else [])
    sys.exit(validate_main(argv))


def cmd_eeg(args):
    from experiments.run_pipeline import main as pipeline_main
    argv = [args.step] + (["--max-subjects", str(args.max_subjects)] if args.max_subjects else [])         + (["--force-gate"] if args.force_gate else [])
    sys.exit(pipeline_main(argv))


def cmd_gates(args):
    from eeg import gates
    for g, desc in gates.GATES.items():
        st = gates.status(g)
        mark = "—" if st is None else ("PASS" if st["passed"] else "FAIL")
        when = "" if st is None else f"  ({st['time_utc']})"
        print(f"{g}  {mark:4s}  {desc}{when}")
    if os.path.isfile(gates.LOCK):
        info = gates.read_lock()
        try:
            gates.require_preregistration()
            ok = "hợp lệ"
        except gates.GateError as e:
            ok = f"KHÔNG hợp lệ: {e}"
        print(f"\nĐăng ký trước: khoá lúc {info['locked_utc']}, sha256 {info['sha256'][:12]}…, {ok}")
    else:
        print("\nĐăng ký trước: chưa khoá (python main.py preregister)")


def cmd_preregister(args):
    from eeg import gates
    if not args.yes:
        print("Khoá predictions.md là không đảo ngược: mọi sửa đổi sau đó phải báo cáo là 'deviation'.\n"
              "Chỉ khoá sau khi qua cổng G1 và đã chốt eeg/config.yaml. Chạy lại với --yes để khoá.")
        sys.exit(1)
    try:
        gates.require("G1", force=args.force_gate)
    except gates.GateError as e:
        print(f"[DỪNG] {e}")
        sys.exit(1)
    info = gates.lock_predictions()
    print(f"Đã khoá {info['file']}\n  sha256: {info['sha256']}\n  lúc:    {info['locked_utc']}")


# =============================================================================
# Subcommand: model  (Q1_pipeline_model.md — 5 gói kết quả, cổng M0–M2)
# =============================================================================
def cmd_model(args):
    import importlib
    steps = ["theory", "identifiability", "design", "control", "sensitivity", "figures"]
    chosen = steps if args.step == "all" else [args.step]
    out_dir = args.out_dir or os.path.join(DEFAULT_RES_DIR, "model")
    for step in chosen:
        if step == "figures":
            from experiments.run_model_paper import run as figures_run
            print("=" * 70 + "\nmodel.figures\n" + "=" * 70, flush=True)
            figures_run(seed=args.seed, profiles=args.profiles)
            continue
        print("=" * 70 + f"\nmodel.{step}\n" + "=" * 70, flush=True)
        mod = importlib.import_module(f"model.{step}")
        kw = {"seed": args.seed, "output_dir": out_dir}
        if step in ("design", "control", "sensitivity"):
            kw["fast"] = args.fast
        if step == "design":
            kw.update(n_reps=args.n_reps, n_starts=args.n_starts)
        if step == "sensitivity":
            kw["n_base"] = args.n_base
        if step == "control":
            kw["mle_csv"] = os.path.join(out_dir, "design_grid_mle.csv")
        mod.run(**kw)
    print(f"\n✔ Kết quả trong {out_dir}; trạng thái cổng: python main.py gates")


# =============================================================================
# Subcommand: dashboard
# =============================================================================
def cmd_dashboard(args):
    import subprocess
    app = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(args.port),
           "--server.headless", "true" if args.headless else "false"]
    print("Khởi động dashboard:", " ".join(cmd))
    try:
        sys.exit(subprocess.call(cmd, cwd=os.path.join(PROJECT_ROOT, "dashboard")))
    except KeyboardInterrupt:
        pass


# =============================================================================
# Argument parser
# =============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="BQI Project — CLI điều phối sinh dữ liệu / chạy thực nghiệm / kiểm thử.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate-data ---
    p_gen = sub.add_parser("generate-data", help="Sinh các dataset tổng hợp (CSV) cho các thuật toán.")
    p_gen.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên toàn cục (mặc định: 42)")
    p_gen.add_argument("--only", type=str, default=None,
                        help="Danh sách dataset cách nhau bởi dấu phẩy, vd: pci,ode,meta,kuramoto,quantum,reference_tables")
    p_gen.add_argument("--out-dir", type=str, default=None, help="Thư mục ghi CSV (mặc định: datasets/generated)")
    p_gen.add_argument("--n-subjects-per-state", type=int, default=40, help="PCI: số subject / trạng thái ý thức")
    p_gen.add_argument("--n-ode-subjects", type=int, default=500, help="ODE: số subject tổng hợp")
    p_gen.add_argument("--n-bootstrap", type=int, default=5000, help="Meta-analysis: số lần bootstrap resample")
    p_gen.add_argument("--n-kuramoto-trials", type=int, default=15, help="Kuramoto: số trial / (trạng thái x băng tần)")
    p_gen.add_argument("--n-quantum-trials", type=int, default=10, help="Quantum: số trial / (dim x gamma)")
    p_gen.add_argument("--quiet", action="store_true", help="Tắt log chi tiết")
    p_gen.set_defaults(func=cmd_generate_data)

    # --- run-experiments ---
    p_run = sub.add_parser("run-experiments", help="Chạy toàn bộ (hoặc một phần) thuật toán, lưu hình + summary.json.")
    p_run.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên toàn cục (mặc định: 42)")
    p_run.add_argument("--only", type=str, default=None,
                        help="Danh sách thực nghiệm cách nhau bởi dấu phẩy, vd: bpc,bqi,com,phi,pci,ode,meta,hopfield,spectral,quantum")
    p_run.add_argument("--fig-dir", type=str, default=None, help="Thư mục lưu hình (mặc định: outputs/figures)")
    p_run.add_argument("--res-dir", type=str, default=None, help="Thư mục lưu summary.json (mặc định: outputs/results)")
    p_run.add_argument("--no-figures", action="store_true", help="Không lưu hình PNG, chỉ ghi summary.json")
    p_run.add_argument("--print-summary", action="store_true", help="In summary.json ra màn hình sau khi chạy")
    p_run.add_argument("--quiet", action="store_true", help="Tắt log chi tiết")
    p_run.set_defaults(func=cmd_run_experiments)

    # --- test ---
    p_test = sub.add_parser("test", help="Chạy bộ kiểm thử pytest.")
    p_test.add_argument("--path", type=str, default=None, help="Đường dẫn file/thư mục test cụ thể (mặc định: tests/)")
    p_test.add_argument("-k", "--keyword", type=str, default=None, help="Lọc test theo từ khoá (pytest -k)")
    p_test.add_argument("-m", "--markers", type=str, default=None, help="Lọc test theo marker (pytest -m)")
    p_test.add_argument("-x", "--stop-on-fail", action="store_true", help="Dừng ngay khi có test đầu tiên fail")
    p_test.add_argument("-v", "--verbose-pytest", action="store_true", help="In chi tiết từng test (pytest -v)")
    p_test.set_defaults(func=cmd_test)

    # --- list-data ---
    p_list = sub.add_parser("list-data", help="Liệt kê các dataset CSV đã sinh ra.")
    p_list.add_argument("--out-dir", type=str, default=None, help="Thư mục dữ liệu (mặc định: datasets/generated)")
    p_list.set_defaults(func=cmd_list_data)

    # --- inspect-data ---
    p_inspect = sub.add_parser("inspect-data", help="Xem nhanh nội dung + thống kê mô tả của một dataset CSV.")
    p_inspect.add_argument("--name", type=str, required=True, help="Tên file CSV, vd: pci_synthetic_subjects.csv")
    p_inspect.add_argument("--out-dir", type=str, default=None, help="Thư mục dữ liệu (mặc định: datasets/generated)")
    p_inspect.add_argument("--group-by", type=str, default=None, help="Tên cột để nhóm khi tính trung bình, vd: state")
    p_inspect.set_defaults(func=cmd_inspect_data)

    # --- all ---
    p_all = sub.add_parser("all", help="Chạy toàn bộ pipeline: generate-data -> run-experiments -> test.")
    p_all.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên toàn cục (mặc định: 42)")
    p_all.add_argument("--only", type=str, default=None,
                        help="(Áp dụng cho cả generate-data và run-experiments nếu tên trùng khớp)")
    p_all.add_argument("--out-dir", type=str, default=None, help="Thư mục ghi CSV")
    p_all.add_argument("--fig-dir", type=str, default=None, help="Thư mục lưu hình")
    p_all.add_argument("--res-dir", type=str, default=None, help="Thư mục lưu summary.json")
    p_all.add_argument("--n-subjects-per-state", type=int, default=40)
    p_all.add_argument("--n-ode-subjects", type=int, default=500)
    p_all.add_argument("--n-bootstrap", type=int, default=5000)
    p_all.add_argument("--n-kuramoto-trials", type=int, default=15)
    p_all.add_argument("--n-quantum-trials", type=int, default=10)
    p_all.add_argument("--no-figures", action="store_true")
    p_all.add_argument("--print-summary", action="store_true")
    p_all.add_argument("--skip-tests", action="store_true", help="Bỏ qua bước chạy test")
    p_all.add_argument("--quiet", action="store_true")
    p_all.set_defaults(func=cmd_all)

    # --- info ---
    p_info = sub.add_parser("info", help="Hiển thị hướng dẫn sử dụng và các lựa chọn hiện có.")
    p_info.set_defaults(func=cmd_info)

    # --- Q1 pipeline ---
    p_val = sub.add_parser("validate-tools", help="Kiểm định công cụ trên dữ liệu tổng hợp (cổng G0, tuỳ chọn G3).")
    p_val.add_argument("--g3", action="store_true", help="Chạy thêm phục hồi tham số mô hình 7.2 (cổng G3, chậm)")
    p_val.add_argument("--force-gate", action="store_true", help="Bỏ qua cổng chưa pass (bị ghi log)")
    p_val.set_defaults(func=cmd_validate_tools)

    p_eeg = sub.add_parser("eeg", help="Các bước pipeline EEG: chennu | anphy | g1 | axis | status.")
    p_eeg.add_argument("step", choices=["chennu", "anphy", "g1", "axis", "status"])
    p_eeg.add_argument("--max-subjects", type=int, default=None)
    p_eeg.add_argument("--force-gate", action="store_true")
    p_eeg.set_defaults(func=cmd_eeg)

    p_gates = sub.add_parser("gates", help="Trạng thái các cổng G0–G3 và khoá đăng ký trước.")
    p_gates.set_defaults(func=cmd_gates)

    p_pre = sub.add_parser("preregister", help="Khoá preregistration/predictions.md (SHA-256 + thời điểm).")
    p_pre.add_argument("--yes", action="store_true", help="Xác nhận khoá")
    p_pre.add_argument("--force-gate", action="store_true", help="Khoá dù G1 chưa pass (bị ghi log)")
    p_pre.set_defaults(func=cmd_preregister)

    # --- dashboard ---
    p_model = sub.add_parser("model", help="Bài mô hình (Q1_pipeline_model.md): theory | identifiability | "
                                           "design | control | sensitivity | all.")
    p_model.add_argument("step", choices=["theory", "identifiability", "design", "control", "sensitivity",
                                          "figures", "all"])
    p_model.add_argument("--profiles", action="store_true", help="figures: tính thêm profile likelihood (chậm)")
    p_model.add_argument("--n-base", type=int, default=None,
                         help="sensitivity: số mẫu gốc Saltelli (mặc định 64; bài: >= 256)")
    p_model.add_argument("--n-reps", type=int, default=None, help="design: số lần phục hồi mỗi ô (mặc định 5)")
    p_model.add_argument("--n-starts", type=int, default=2,
                         help="design: số điểm xuất phát L-BFGS-B mỗi fit (mặc định 2; 4–6 nếu M1 fail ở N nhỏ)")
    p_model.add_argument("--seed", type=int, default=42)
    p_model.add_argument("--fast", action="store_true", help="Ít lần lặp (thử nhanh, không dùng cho bài)")
    p_model.add_argument("--out-dir", type=str, default=None, help="Mặc định: outputs/results/model")
    p_model.set_defaults(func=cmd_model)

    p_dash = sub.add_parser("dashboard", help="Mở dashboard Streamlit (cần: pip install streamlit plotly).")
    p_dash.add_argument("--port", type=int, default=8501, help="Cổng HTTP (mặc định: 8501)")
    p_dash.add_argument("--headless", action="store_true", help="Không tự mở trình duyệt")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

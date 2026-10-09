"""
Kill-criteria gates (Q1_pipeline.md Sec 9) and the preregistration lock
(Sec 2), enforced in code: later steps refuse to run until earlier gates
have passed, and meditation data cannot be fetched before the predictions
file is locked.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_DIR = os.path.join(ROOT, "outputs", "gates")
PREREG_DIR = os.path.join(ROOT, "preregistration")
PREDICTIONS = os.path.join(PREREG_DIR, "predictions.md")
LOCK = os.path.join(PREREG_DIR, "predictions.lock")

GATES = {
    "G0": "Ki\u1ec3m \u0111\u1ecbnh c\xf4ng c\u1ee5 tr\xean d\u1eef li\u1ec7u t\u1ed5ng h\u1ee3p (m\u1ee5c 8)",
    "G1": "Pipeline t\u00e1i hi\u1ec7n wake > N3 v\u00e0 baseline > moderate (LZc)",
    "G2": "Hai t\u01b0\u01a1ng ph\u1ea3n tham chi\u1ebfu c\u00f9ng h\u01b0\u1edbng (cosine > 0.5)",
    "G3": "Parameter recovery c\u1ee7a m\xf4 h\xecnh 7.2",
    # Model paper gates (Q1_pipeline_model.md)
    "M0": "M\u1ec7nh \u0111\u1ec1 P1-P5: m\xf4 h\xecnh g\u1ed1c M0 c\xf3 l\u1ed7i to\u00e1n h\u1ecdc, M1 nh\u1ea5t qu\u00e1n",
    "M1": "Fisher/CRB kh\u1edbp sai s\u1ed1 ph\u1ee5c h\u1ed3i t\u1ea1i >= 80% \xf4 \u0111\u1ea1i di\u1ec7n",
    "M2": "LQG tr\xean tham s\u1ed1 th\u1eadt th\u1eafng bang-bang v\u00e0 kh\xf4ng \u0111i\u1ec1u khi\u1ec3n",
}
REQUIRES = {"G1": ["G0"], "G2": ["G1"], "G3": ["G0"], "M1": ["M0"], "M2": ["M1"]}


class GateError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(gate: str, passed: bool, details: dict) -> str:
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, f"{gate}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gate": gate, "description": GATES.get(gate, ""), "passed": bool(passed),
                   "time_utc": _now(), "details": details}, f, indent=2, ensure_ascii=False, default=str)
    return path


def status(gate: str) -> dict | None:
    try:
        with open(os.path.join(GATE_DIR, f"{gate}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def require(*gates: str, force: bool = False) -> None:
    """Raise GateError unless every gate has passed. `force` overrides but is logged."""
    missing = [g for g in gates if not (status(g) or {}).get("passed")]
    if not missing:
        return
    if force:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "overrides.log"), "a", encoding="utf-8") as f:
            f.write(f"{_now()}\tforced past {','.join(missing)}\n")
        return
    raise GateError(f"Cong chua pass: {', '.join(f'{g} ({GATES[g]})' for g in missing)}. "
                    "Chay buoc tuong ung truoc, hoac dung --force-gate (se bi ghi log).")


# ---- preregistration lock ----------------------------------------------------
def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def lock_predictions() -> dict:
    if not os.path.isfile(PREDICTIONS):
        raise GateError(f"Khong tim thay {PREDICTIONS}")
    if os.path.isfile(LOCK):
        cur = read_lock()
        if cur["sha256"] == _sha256(PREDICTIONS):
            return cur
        raise GateError("predictions.md da bi sua sau khi khoa. Khong duoc khoa lai; "
                        "ghi thay doi thanh muc 'Sua doi sau dang ky' trong bai.")
    info = {"file": "preregistration/predictions.md", "sha256": _sha256(PREDICTIONS), "locked_utc": _now()}
    with open(LOCK, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def read_lock() -> dict:
    with open(LOCK, encoding="utf-8") as f:
        return json.load(f)


def require_preregistration() -> dict:
    """Used by the fetcher before any meditation dataset is downloaded."""
    if not os.path.isfile(LOCK):
        raise GateError("Chua khoa dang ky truoc. Chay: python main.py preregister")
    info = read_lock()
    if info["sha256"] != _sha256(PREDICTIONS):
        raise GateError("predictions.md khac ban da khoa (hash khong khop).")
    return info
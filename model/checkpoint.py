"""
Resumable JSON checkpoints for the long real-data steps (fit, boot, check).

One file per step in <results>/checkpoints/<step>.json holding a fingerprint
of the inputs (feature data, seed, settings) and a dict of finished work
items. A file whose fingerprint does not match the current inputs is set
aside (renamed *.stale) instead of being reused. The step deletes its file
once the final results JSON is written, so a later run starts fresh.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np


def _feed(h, p) -> None:
    if isinstance(p, np.ndarray):
        h.update(f"{p.dtype}{p.shape}".encode())
        h.update(np.ascontiguousarray(p).tobytes())
    elif isinstance(p, dict):
        h.update(b"{")
        for k in sorted(p, key=str):
            h.update(str(k).encode())
            _feed(h, p[k])
        h.update(b"}")
    elif isinstance(p, (list, tuple)):
        h.update(b"[")
        for x in p:
            _feed(h, x)
        h.update(b"]")
    else:
        h.update(repr(p).encode())


def fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        _feed(h, p)
    return h.hexdigest()[:16]


def sessions_key(sessions: list[dict]) -> list:
    """The data that determines a fit: observation times, features, ratings."""
    return [{k: np.asarray(s.get(k, []), float) for k in ("t", "y", "t_probe", "rating")} for s in sessions]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


class Checkpoint:
    def __init__(self, path: str, fp: str, verbose: bool = True):
        self.path, self.fp, self.data = path, fp, {}
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            obj = {}
        if obj.get("fingerprint") == fp:
            self.data = obj.get("items", {})
            if verbose and self.data:
                print(f"  checkpoint: nap {len(self.data)} muc da xong tu {path}", flush=True)
        else:
            os.replace(path, path + ".stale")
            if verbose:
                print(f"  checkpoint cu khong khop du lieu/cau hinh, de sang {path}.stale", flush=True)

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fp, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "items": self.data},
                      f, default=_json_default)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)

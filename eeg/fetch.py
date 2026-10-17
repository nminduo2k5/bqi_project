"""
Download ds001787 from OpenNeuro's public S3 bucket, subject by subject, with
MD5 verification against the git-annex keys of the metadata clone.

The BDF files (6.1 GB) are refused until `preregistration/predictions_realdata.md`
is locked (Q1_pipeline.md v2.0 Sec 3); the small files in code/ are always allowed.
Every downloaded file is appended to data/ds001787/manifest.csv.
"""
from __future__ import annotations

import csv
import hashlib
import os
import time
import urllib.request

from eeg import gates
from eeg.loaders import ds001787 as DS

CODE_FILES = ("code/MW_Current_TextFileBIDS.zip", "code/run_mw_experiment6.m", "code/README.txt")


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_add(rel: str, size: int, md5: str) -> None:
    path = os.path.join(DS.data_dir(), "manifest.csv")
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "size", "md5", "downloaded_utc"])
        w.writerow([rel, size, md5, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])


def _download(url: str, dest: str, expected_size: int | None = None, verbose: bool = True) -> None:
    """Streaming download with resume (HTTP Range)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        total = (int(r.headers.get("Content-Length", 0)) + have) or expected_size or 0
        done, t0, last = have, time.time(), 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if verbose and time.time() - last > 2:
                last = time.time()
                rate = (done - have) / max(time.time() - t0, 1e-9) / 1e6
                pct = f"{100 * done / total:5.1f}%" if total else ""
                print(f"\r    {os.path.basename(dest)} {done / 1e6:8.1f} MB {pct} {rate:5.1f} MB/s", end="", flush=True)
    if verbose:
        print()
    os.replace(part, dest)


def plan(subjects=None) -> list[dict]:
    """Files to fetch: BDF of the selected subjects (all if None)."""
    items = []
    for s in DS.sessions():
        if subjects and s["sub"] not in subjects and s["sub"].replace("sub-", "") not in subjects:
            continue
        items.append({"rel": s["bdf_rel"], "dest": s["bdf_local"], "size": s["bdf_size"], "md5": s["bdf_md5"]})
    return items


def fetch_code(verbose: bool = True) -> list[str]:
    out = []
    for rel in CODE_FILES:
        dest = os.path.join(DS.data_dir(), *rel.split("/"))
        if not os.path.isfile(dest):
            _download(f"{DS.cfg().s3_base}/{rel}", dest, verbose=verbose)
            _manifest_add(rel, os.path.getsize(dest), _md5(dest))
        out.append(dest)
    DS.behaviour_files()     # unzip
    return out


def fetch(subjects=None, dry_run: bool = False, force_gate: bool = False, verbose: bool = True) -> dict:
    """Download the BDF files (after the preregistration lock) and verify MD5."""
    items = plan(subjects)
    total = sum(i["size"] or 0 for i in items) / 1e9
    if verbose:
        print(f"{len(items)} BDF files, {total:.2f} GB -> {DS.data_dir()}")
    if dry_run:
        for i in items:
            print(f"  {i['rel']}  {(i['size'] or 0) / 1e6:7.1f} MB  {'have' if os.path.isfile(i['dest']) else 'missing'}")
        return {"n_files": len(items), "gb": total, "dry_run": True}
    try:
        gates.require_preregistration(gates.PREDICTIONS_REALDATA, gates.LOCK_REALDATA)
    except gates.GateError as e:
        if not force_gate:
            raise
        gates.require("E1", force=True)   # logs the override
        print(f"[CANH BAO] {e} -- tiep tuc vi --force-gate (da ghi log)")
    done, bad = [], []
    for i in items:
        if os.path.isfile(i["dest"]) and (i["md5"] is None or _md5(i["dest"]) == i["md5"]):
            done.append(i["rel"])
            continue
        _download(f"{DS.cfg().s3_base}/{i['rel']}", i["dest"], i["size"], verbose)
        md5 = _md5(i["dest"])
        if i["md5"] and md5 != i["md5"]:
            bad.append(i["rel"])
            os.replace(i["dest"], i["dest"] + ".badmd5")
            continue
        _manifest_add(i["rel"], os.path.getsize(i["dest"]), md5)
        done.append(i["rel"])
    if verbose:
        print(f"ok {len(done)}, md5 mismatch {len(bad)}")
    return {"n_files": len(items), "gb": total, "ok": done, "bad": bad}

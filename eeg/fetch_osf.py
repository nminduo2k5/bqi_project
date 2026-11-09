"""
Download the experience-sampling EEG of Rodriguez-Larios et al. (2021, NeuroImage)
from OSF (project 3uszv, component v8msp "RAW EEG DATA", CC0), with MD5
verification against the hashes OSF reports.

By default only the interrupted-meditation files are fetched (m_task_*,
c_task_*: 29 meditators + 29 controls, about 12.1 GB), plus two small files of
the parent project (name coding, 21-channel locations); the uninterrupted
meditation (med, 2.2 GB) and rest (1.1 GB) files are fetched on request, after
the task files. The MATLAB scripts are not needed. The archives are not
unpacked here: reading them waits for the preregistration lock of this dataset.
Every downloaded file is appended to data/osf3uszv/manifest.csv.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
import urllib.request

from eeg.fetch import _download, _md5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "osf3uszv")
RAW_NODE = "v8msp"
PROJECT_NODE = "3uszv"
DOC_FILES = ("name_coding_text_files.txt", "matlab variables/EEG21locs.mat")
RAW = re.compile(r"^([mc])_(task|med|rest)_(n?)(\d+)\.txt\.cpgz$")     # controls are numbered 2..30 and n1..n12
CONDITIONS = ("task", "med", "rest")


def _get(url: str, retries: int = 5) -> dict:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError:                         # OSF API is sometimes slow: timeouts, resets
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


def _walk(node: str, path: str = "", url: str | None = None):
    """(path, size, md5, download url) of every file stored in an OSF node."""
    url = url or f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"
    while url:
        d = _get(url)
        for f in d["data"]:
            a = f["attributes"]
            if a["kind"] == "folder":
                yield from _walk(node, path + a["name"] + "/", f["relationships"]["files"]["links"]["related"]["href"])
            else:
                yield path + a["name"], a.get("size") or 0, (a.get("extra") or {}).get("hashes", {}).get("md5"), \
                    f["links"]["download"]
        url = d["links"].get("next")


def _manifest_add(rel: str, size: int, md5: str) -> None:
    path = os.path.join(DATA_DIR, "manifest.csv")
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "size", "md5", "downloaded_utc"])
        w.writerow([rel, size, md5, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])


def plan(groups=("m", "c"), conditions=("task",)) -> list[dict]:
    """Files to fetch: recordings of the selected groups (m meditators, c controls) and conditions + docs."""
    items = []
    for path, size, md5, url in _walk(RAW_NODE):
        m = RAW.match(os.path.basename(path))
        if m and m.group(1) in groups and m.group(2) in conditions:
            items.append({"rel": "raw/" + os.path.basename(path), "url": url, "size": size, "md5": md5,
                          "key": (CONDITIONS.index(m.group(2)), int(m.group(4)), m.group(3), m.group(1))})
    items.sort(key=lambda i: i["key"])          # task first; m and c interleaved so a partial download stays balanced
    for path, size, md5, url in _walk(PROJECT_NODE):
        if path in DOC_FILES:
            items.insert(0, {"rel": "docs/" + os.path.basename(path), "url": url, "size": size, "md5": md5})
    for i in items:
        i["dest"] = os.path.join(DATA_DIR, *i["rel"].split("/"))
    return items


def fetch(groups=("m", "c"), conditions=("task",), dry_run: bool = False, verbose: bool = True,
          retries: int = 3) -> dict:
    """Download (with resume) and verify MD5; files already present and valid are skipped."""
    items = plan(groups, conditions)
    total = sum(i["size"] for i in items) / 1e9
    need = sum(i["size"] for i in items if not os.path.isfile(i["dest"])) / 1e9
    if verbose:
        print(f"{len(items)} files, {total:.2f} GB (missing {need:.2f} GB) -> {DATA_DIR}")
    if dry_run:
        for i in items:
            print(f"  {i['rel']:28s} {i['size'] / 1e6:7.1f} MB  {'have' if os.path.isfile(i['dest']) else 'missing'}")
        return {"n_files": len(items), "gb": total, "dry_run": True}
    os.makedirs(DATA_DIR, exist_ok=True)
    free = shutil.disk_usage(DATA_DIR).free / 1e9
    if free < need + 1.0:
        raise OSError(f"Not enough disk space: need {need:.1f} GB + 1 GB margin, free {free:.1f} GB")
    done, bad, failed = [], [], []
    for k, i in enumerate(items, 1):
        if os.path.isfile(i["dest"]) and (i["md5"] is None or _md5(i["dest"]) == i["md5"]):
            done.append(i["rel"])
            continue
        if verbose:
            print(f"[{k}/{len(items)}] {i['rel']} ({i['size'] / 1e6:.0f} MB, {time.strftime('%H:%M')})")
        for attempt in range(1, retries + 1):
            try:
                _download(i["url"], i["dest"], i["size"], verbose)
                break
            except OSError as e:                # network drop: the .part file is kept and resumed
                print(f"\n    error ({e}); retry {attempt}/{retries} in 30 s")
                time.sleep(30)
        else:
            failed.append(i["rel"])
            continue
        md5 = _md5(i["dest"])
        if i["md5"] and md5 != i["md5"]:
            bad.append(i["rel"])
            os.replace(i["dest"], i["dest"] + ".badmd5")
            continue
        _manifest_add(i["rel"], os.path.getsize(i["dest"]), md5)
        done.append(i["rel"])
    if verbose:
        print(f"ok {len(done)}/{len(items)}, md5 mismatch {len(bad)}, failed {len(failed)}"
              + (" -- run the same command again to resume" if bad or failed else ""))
    return {"n_files": len(items), "gb": total, "ok": done, "bad": bad, "failed": failed}

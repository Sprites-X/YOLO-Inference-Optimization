#!/usr/bin/env python3
"""Fetch COCO train2017 images as the pool for INT8 calibration.

    data/train_pool/  2000 train2017 images -> 500 sampled from it to calibrate

**Why not download all of train2017.zip** — train2017 is 118,287 images / ~19GB, and
the machine this was built on had 13GB free. So only the images named in
train_pool_manifest.txt are fetched, one at a time from images.cocodataset.org
(~330MB). Calibration needs a few hundred images, not a whole split, so a pool this
size is plenty.

**Why the list lives in a manifest instead of being sampled at runtime** — sampling at
runtime would give every machine a different pool, and calibration caches could no
longer be compared across machines. The manifest is committed (34KB), so a clone gets
the same pool without downloading 241MB of annotations to regenerate it.

The manifest was built by sorting all 118,287 train2017 filenames, running
random.Random(1337).shuffle(), and taking the first 2000. 2000 rather than 500 because
if INT8 costs too much mAP, one fix is raising the calibration set to 1000 — this
leaves room to try that.

**This set must not overlap data/val500** — val500 comes from val2017 and this pool
from train2017, different COCO splits, so they cannot overlap by definition. The
script checks anyway at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "http://images.cocodataset.org/train2017"
MANIFEST = "train_pool_manifest.txt"

# Fingerprint of the manifest, to catch it being edited or truncated by accident.
EXPECTED_MANIFEST = "6c787a8293f8ea2223b451a45e3c10d137716496f5b782c59b38a5428049b7e6"

# 16 at a time is enough for ~2000 small files without hammering COCO into throttling.
WORKERS = 16
RETRIES = 3


def manifest_hash(names: list[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode() + b"\n")
    return h.hexdigest()


def fetch(name: str, dest: Path) -> tuple[str, str | None]:
    """Download one image. Returns (name, error message or None).

    Writes to .part and renames on completion. Interrupted mid-download, a partial file
    would otherwise sit there under the real name, the next run's exists() check would
    skip it, and you would end up with a corrupt JPEG that cv2.imread returns None for
    — quietly dropping that image from calibration.
    """
    out = dest / name
    if out.exists() and out.stat().st_size > 0:
        return name, None

    tmp = out.with_suffix(".part")
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=30) as r:
                data = r.read()
            if not data:
                raise OSError("got 0 bytes")
            tmp.write_bytes(data)
            tmp.rename(out)
            return name, None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == RETRIES - 1:
                tmp.unlink(missing_ok=True)
                return name, str(e)
    return name, "unreachable"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default="data/train_pool")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="fetch only the first N images of the manifest (for testing); 0 = all")
    args = ap.parse_args()

    names = [l.strip() for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    digest = manifest_hash(names)
    if digest != EXPECTED_MANIFEST:
        print(f"[pool] warning: manifest sha256 {digest[:16]}… does not match the recorded "
              f"value ({EXPECTED_MANIFEST[:16]}…) — this pool will differ from other machines",
              file=sys.stderr)
    if args.limit:
        names = names[:args.limit]

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (dest / n).exists()]
    print(f"[pool] manifest {len(names)} images | already have {len(names) - len(todo)} | "
          f"to fetch {len(todo)}")

    failed = []
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (name, err) in enumerate(ex.map(lambda n: fetch(n, dest), todo), 1):
                if err:
                    failed.append((name, err))
                if i % 100 == 0 or i == len(todo):
                    print(f"\r[pool] {i}/{len(todo)}  failed {len(failed)}", end="", flush=True)
        print()

    have = sorted(p.name for p in dest.iterdir() if p.suffix.lower() == ".jpg")
    if failed:
        for name, err in failed[:5]:
            print(f"[pool] failed {name}: {err}", file=sys.stderr)
        raise SystemExit(f"[pool] {len(failed)} images did not download — rerun, existing files are skipped")

    # Double-check the calibration pool does not overlap the set used to score mAP,
    # even though they are already different COCO splits — if someone ever points --out
    # at data/val500, this is where it surfaces.
    val500 = Path("data/val500")
    if val500.is_dir():
        overlap = {p.name for p in val500.iterdir()} & set(have)
        if overlap:
            raise SystemExit(f"[pool] overlaps val500 by {len(overlap)} images — leakage")
        print("[pool] no overlap with data/val500 — safe to calibrate without leakage")

    size_mb = sum(p.stat().st_size for p in dest.iterdir()) / 1024 / 1024
    print(f"[pool] {dest}: {len(have)} images ({size_mb:.0f} MB)")
    print(f"[pool] sample for calib: ls {dest}/*.jpg | shuf -n 500 | xargs -I{{}} cp {{}} data/calib/")


if __name__ == "__main__":
    main()

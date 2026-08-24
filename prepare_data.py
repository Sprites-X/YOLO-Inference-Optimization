#!/usr/bin/env python3
"""Pick 500 images from COCO val2017 as the measurement set.

    data/val500/  500 images -> latency (benchmark.py) and mAP (evaluate.py)

INT8 calibration images do not come from here. They live in data/train_pool/, pulled
from COCO train2017 by fetch_train_pool.py. Keeping them in separate splits stops us
calibrating on the very images we are about to score: INT8 would have its dynamic
ranges tuned to the test set and report an mAP that is too good, with nothing to
flag it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Do not change these once numbers have been recorded. Change them and val500 becomes
# a different set of images, so earlier results no longer compare to later ones.
SEED = 1337
VAL_NUM = 500

# Fingerprint of the val500 the defaults produce, from the official val2017 set of
# 5000 images. A mismatch means --src is not the complete val2017, and the resulting
# mAP cannot be compared against the README.
EXPECTED_VAL500 = "faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7"


def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink where possible, fall back to copying.

    A hardlink is a real file in every way that matters to cv2.imread and Path.rglob,
    unlike a symlink that breaks when the folder moves, and it costs no extra disk on
    top of the data/val2017 already there. Hardlinks cannot cross partitions (EXDEV),
    hence the fallback.
    """
    if dst.exists():
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def manifest_hash(names: list[str]) -> str:
    """Hash the filenames, not the image contents.

    It answers one question — did we pick the same set of images? — which is what the
    seed controls. The pixels come from COCO's official zip, so there is nothing to
    re-verify there.
    """
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode() + b"\n")
    return h.hexdigest()


def pick(files: list[Path], seed: int, num: int) -> list[Path]:
    """Seeded shuffle, then take num images.

    Always sorted() before shuffling: iterdir() returns filesystem inode order, which
    differs per machine, so shuffling from that would give a different set on every
    machine even with the same seed.

    Shuffled rather than the first 500 by name, because COCO image_ids are not randomly
    ordered to begin with, so slicing off the front could skew the set.
    """
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:num])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="the unzipped val2017 folder")
    ap.add_argument("--out", default="data", help="destination folder (default: data)")
    ap.add_argument("--num", type=int, default=VAL_NUM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if len(files) < args.num:
        raise SystemExit(f"found {len(files)} images in {src} — fewer than --num {args.num}")
    print(f"[data] found {len(files)} images in {src}")

    chosen = pick(files, args.seed, args.num)
    dest = out / "val500"
    dest.mkdir(parents=True, exist_ok=True)
    modes = {"link": 0, "copy": 0, "skip": 0}
    for p in chosen:
        # Keep filenames exactly as they are. evaluate.image_id_from_name() turns
        # 000000000139.jpg into image_id 139 to match instances_val2017.json. Rename
        # them and COCOeval matches nothing and returns mAP 0.000 without an error.
        modes[link_or_copy(p, dest / p.name)] += 1
    print(f"[data] {dest}: {len(chosen)} images {modes}")

    digest = manifest_hash([p.name for p in chosen])
    split_path = out / "split.json"
    split_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Store the folder name, not an absolute path: a full path is tied to the home
        # directory of whoever ran it, so this file would differ between two machines
        # that picked exactly the same images.
        "source": src.name,
        "seed": args.seed,
        "total": len(files),
        "val500": {"count": len(chosen), "manifest_sha256": digest,
                   "files": [p.name for p in chosen]},
    }, indent=2))
    print(f"[data] wrote {split_path}")

    # Only comparable to the known value when the defaults are used. If the seed or
    # num changed, a different hash is intentional, not a mistake.
    if (args.seed, args.num) != (SEED, VAL_NUM):
        print(f"[data] val500 {digest[:16]}… (non-default seed/num, so not compared to the README)")
        return
    if digest != EXPECTED_VAL500:
        raise SystemExit(
            f"[data] val500 {digest[:16]}… ✗ does not match the README\n"
            "--src is probably not the full 5000-image val2017 — the resulting mAP "
            "cannot be compared against the README table")
    print(f"[data] val500 {digest[:16]}… ✓ matches the README")


if __name__ == "__main__":
    main()

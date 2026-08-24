#!/usr/bin/env python3
"""ดึงภาพ COCO train2017 มาเป็นบ่อสำหรับ INT8 calibration (Phase 3)

    data/train_pool/  2000 ภาพจาก train2017 -> Phase 3 สุ่มออกมา 500 ภาพไปใช้จริง

**ทำไมไม่โหลด train2017.zip ทั้งก้อน** — train2017 คือ 118,287 ภาพ / ~19GB ส่วนเครื่อง
ที่ทำโปรเจกต์นี้เหลือดิสก์ 13GB จึงดึงเฉพาะภาพที่อยู่ใน train_pool_manifest.txt
ทีละไฟล์จาก images.cocodataset.org แทน (~330MB) guide ไม่ได้บังคับว่า train_pool
ต้องเป็น train2017 ครบชุด — Phase 3 ใช้แค่ `shuf -n 500` จากบ่อนี้อยู่แล้ว

**ทำไมรายชื่ออยู่ในไฟล์ manifest ไม่ใช่สุ่มตอนรัน** — ถ้าสุ่มตอนรัน แต่ละเครื่องจะได้
คนละบ่อ แล้ว calibration cache เทียบกันข้ามเครื่องไม่ได้ manifest ถูก commit ขึ้น repo
(34KB) คนที่ clone จึงได้บ่อเดียวกันโดยไม่ต้องโหลด annotations 241MB มา generate เอง

รายชื่อใน manifest มาจาก: เรียงชื่อไฟล์ทั้ง 118,287 ภาพของ train2017 แล้ว
random.Random(1337).shuffle() แล้วตัด 2000 ตัวแรก (2000 ไม่ใช่ 500 เพราะ guide บอกว่า
ถ้า mAP ตกเกิน 3-4% ให้ลองเพิ่มภาพ calibration เป็น 1000 — เผื่อไว้ให้ลองได้)

**ภาพชุดนี้ต้องไม่ทับกับ data/val500** — val500 มาจาก val2017 ส่วนบ่อนี้มาจาก train2017
เป็นคนละ split ของ COCO จึงไม่ทับกันโดยนิยาม (สคริปต์ยังตรวจซ้ำตอนจบ)
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

# fingerprint ของ manifest ใช้จับกรณีไฟล์ถูกแก้/ตัดทอนโดยไม่ตั้งใจ
EXPECTED_MANIFEST = "6c787a8293f8ea2223b451a45e3c10d137716496f5b782c59b38a5428049b7e6"

# 16 ต่อครั้งพอสำหรับ ~2000 ไฟล์เล็ก และไม่ยิงถี่จนโดน throttle จากฝั่ง COCO
WORKERS = 16
RETRIES = 3


def manifest_hash(names: list[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode() + b"\n")
    return h.hexdigest()


def fetch(name: str, dest: Path) -> tuple[str, str | None]:
    """โหลดภาพเดียว คืน (ชื่อ, ข้อความ error ถ้าพลาด)

    เขียนลง .part ก่อนแล้วค่อย rename เพราะถ้าโดนขัดจังหวะกลางคัน ไฟล์ที่โหลดไม่ครบ
    จะค้างอยู่ด้วยชื่อจริง แล้วรอบหน้า exists() มองว่ามีแล้วเลยข้าม — ได้ JPEG เสียๆ
    ที่ cv2.imread คืน None แล้วภาพนั้นหายจาก calibration แบบเงียบๆ
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
                raise OSError("ได้ข้อมูล 0 ไบต์")
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
                    help="โหลดแค่ N ภาพแรกของ manifest (ไว้ทดสอบ) 0 = ทั้งหมด")
    args = ap.parse_args()

    names = [l.strip() for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    digest = manifest_hash(names)
    if digest != EXPECTED_MANIFEST:
        print(f"[pool] เตือน: manifest sha256 {digest[:16]}… ไม่ตรงกับที่บันทึกไว้ "
              f"({EXPECTED_MANIFEST[:16]}…) — บ่อนี้จะไม่ตรงกับของเครื่องอื่น",
              file=sys.stderr)
    if args.limit:
        names = names[:args.limit]

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (dest / n).exists()]
    print(f"[pool] manifest {len(names)} ภาพ | มีอยู่แล้ว {len(names) - len(todo)} | "
          f"ต้องโหลด {len(todo)}")

    failed = []
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (name, err) in enumerate(ex.map(lambda n: fetch(n, dest), todo), 1):
                if err:
                    failed.append((name, err))
                if i % 100 == 0 or i == len(todo):
                    print(f"\r[pool] {i}/{len(todo)}  พลาด {len(failed)}", end="", flush=True)
        print()

    have = sorted(p.name for p in dest.iterdir() if p.suffix.lower() == ".jpg")
    if failed:
        for name, err in failed[:5]:
            print(f"[pool] พลาด {name}: {err}", file=sys.stderr)
        raise SystemExit(f"[pool] โหลดไม่ครบ {len(failed)} ภาพ — รันซ้ำได้ ของเดิมจะถูกข้าม")

    # ตรวจซ้ำว่าบ่อ calibration ไม่ทับกับชุดที่ใช้วัด mAP แม้จะเป็นคนละ split
    # ของ COCO อยู่แล้ว — ถ้าวันหนึ่งมีคนชี้ --out ไปที่ data/val500 จะได้รู้ตรงนี้
    val500 = Path("data/val500")
    if val500.is_dir():
        overlap = {p.name for p in val500.iterdir()} & set(have)
        if overlap:
            raise SystemExit(f"[pool] ทับกับ val500 {len(overlap)} ภาพ — leakage")
        print("[pool] ไม่ทับกับ data/val500 — Phase 3 calibrate ได้โดยไม่ leak")

    size_mb = sum(p.stat().st_size for p in dest.iterdir()) / 1024 / 1024
    print(f"[pool] {dest}: {len(have)} ภาพ ({size_mb:.0f} MB)")
    print(f"[pool] Phase 3: ls {dest}/*.jpg | shuf -n 500 | xargs -I{{}} cp {{}} data/calib/")


if __name__ == "__main__":
    main()

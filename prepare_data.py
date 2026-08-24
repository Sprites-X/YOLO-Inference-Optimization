#!/usr/bin/env python3
"""แบ่ง COCO val2017 (5000 ภาพ) ออกเป็นสองส่วนที่ไม่ทับกัน

    data/val500/     500 ภาพ  -> วัด latency (benchmark.py) และ mAP (evaluate.py)
    data/calib_pool/ 4500 ภาพ -> บ่อสำหรับสุ่มทำ INT8 calibration ใน Phase 3

ต้องแบ่งตั้งแต่ Phase 1 ไม่ใช่รอถึง Phase 3 เพราะถ้าเลือก val500 ไปก่อนโดยไม่กัน
ส่วนที่เหลือไว้ พอถึง Phase 3 จะไม่มีภาพที่ calibrate ได้โดยไม่ทับกับ val set
(calibrate ด้วยภาพที่กำลังจะเอาไปให้คะแนน = INT8 ถูกจูน dynamic range ให้พอดีกับ
ข้อสอบ แล้ว mAP ออกมาดีเกินจริงโดยไม่มีอะไรฟ้อง)

pool มาจากส่วนที่เหลือของ val2017 ไม่ใช่ COCO train2017 ตามที่ guide เขียนไว้
(`data/train_pool`) เพราะ train2017 คือ 118k ภาพ / 18GB ส่วนเครื่องนี้เหลือดิสก์ 13GB
เงื่อนไขที่ calibration ต้องการจริงคือ "distribution เดียวกัน + ไม่ทับกับ val set"
ซึ่งผ่านทั้งสองข้อ แต่ไม่ใช่ train split จริง — ต้องเขียนกำกับใน Limitations
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

# ห้ามเปลี่ยนหลังเริ่มเก็บตัวเลขแล้ว — เปลี่ยนเมื่อไร val500 กลายเป็นภาพคนละชุด
# แล้วผลที่วัดไว้ก่อนหน้าเทียบกับผลใหม่ไม่ได้
SEED = 1337
VAL_NUM = 500

# fingerprint ของ split ที่ค่า default ให้ ใช้ยืนยันว่าเครื่องอื่นแบ่งได้ชุดเดียวกัน
# วัดจาก val2017 ชุดทางการ (5000 ภาพ) ด้วย SEED/VAL_NUM ข้างบน
# ถ้าไม่ตรง แปลว่า source ไม่ใช่ val2017 ครบชุด — ตัวเลข mAP จะเทียบกับ README ไม่ได้
EXPECTED = {
    "val500": "faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7",
    "calib_pool": "aaca64bcf21426cf8c4bc92b614a226042812b541d1880d011443334e366dff0",
}


def link_or_copy(src: Path, dst: Path) -> str:
    """hardlink ก่อน ถ้าไม่ได้ค่อย copy

    val500 + calib_pool รวมกันคือ val2017 ทั้งชุด ถ้า copy จะกินดิสก์เพิ่มอีก ~800MB
    โดยไม่ได้อะไร — hardlink เป็นไฟล์จริงทุกประการสำหรับ cv2.imread และ Path.rglob
    ไม่ใช่ symlink ที่พังเวลาย้ายโฟลเดอร์ ข้ามพาร์ทิชันไม่ได้ (EXDEV) จึงมี fallback
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
    """hash จากรายชื่อไฟล์ ไม่ใช่จากเนื้อภาพ

    ตอบคำถามเดียวคือ "เลือกภาพชุดเดียวกันไหม" ซึ่งเป็นสิ่งที่ seed ควบคุม
    ส่วนเนื้อภาพมาจาก zip ทางการของ COCO อยู่แล้วไม่ต้องตรวจซ้ำ
    """
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode() + b"\n")
    return h.hexdigest()


def split_files(files: list[Path], seed: int, val_num: int) -> tuple[list[Path], list[Path]]:
    """seeded shuffle แล้วตัดเป็นสองส่วน

    ต้อง sorted() ก่อน shuffle เสมอ — iterdir() คืนลำดับตาม inode ของ filesystem
    ซึ่งต่างกันทุกเครื่อง ถ้า shuffle จากลำดับนั้น seed เดียวกันจะได้คนละชุด

    shuffle ไม่ใช่ตัด 500 ตัวแรกที่เรียงตามชื่อ เพราะ image_id ของ COCO ไม่ได้
    สุ่มเรียงมาแต่แรก การตัดหัวจึงอาจได้ภาพที่เอียงไปทางใดทางหนึ่ง
    """
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:val_num]), sorted(shuffled[val_num:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="โฟลเดอร์ val2017 ที่แตก zip แล้ว")
    ap.add_argument("--out", default="data", help="โฟลเดอร์ปลายทาง (default: data)")
    ap.add_argument("--val-num", type=int, default=VAL_NUM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if len(files) < args.val_num:
        raise SystemExit(f"เจอ {len(files)} รูปใน {src} — น้อยกว่า --val-num {args.val_num}")
    print(f"[data] เจอ {len(files)} รูปใน {src}")

    val_files, pool_files = split_files(files, args.seed, args.val_num)
    groups = {"val500": val_files, "calib_pool": pool_files}

    # ตรวจก่อนเขียน ไม่ใช่หลังเขียน — ถ้าทับกันแล้วเขียนไฟล์ไปแล้ว จะเหลือ
    # โฟลเดอร์ที่ leakage อยู่บนดิสก์รอให้เผลอใช้
    overlap = {p.name for p in val_files} & {p.name for p in pool_files}
    if overlap:
        raise SystemExit(f"val500 กับ calib_pool ทับกัน {len(overlap)} รูป — leakage")

    manifest = {}
    for name, group in groups.items():
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        modes = {"link": 0, "copy": 0, "skip": 0}
        for p in group:
            # ชื่อไฟล์เดิมเป๊ะ ห้ามเปลี่ยน — evaluate.image_id_from_name() แปลง
            # 000000000139.jpg -> image_id 139 เพื่อจับคู่กับ instances_val2017.json
            # ถ้าเปลี่ยนชื่อ COCOeval จับคู่ไม่ได้แล้วคืน mAP 0.000 โดยไม่มี error
            modes[link_or_copy(p, dest / p.name)] += 1
        manifest[name] = manifest_hash([p.name for p in group])
        print(f"[data] {dest}: {len(group)} รูป {modes}")

    split_path = out / "split.json"
    split_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        # เก็บชื่อโฟลเดอร์ ไม่ใช่ absolute path — path เต็มผูกกับ home ของเครื่องที่รัน
        # ทำให้ diff ของไฟล์นี้ระหว่างสองเครื่องต่างกันทั้งที่ split เหมือนกันเป๊ะ
        "source": src.name,
        "seed": args.seed,
        "total": len(files),
        "val500": {"count": len(val_files), "manifest_sha256": manifest["val500"],
                   "files": [p.name for p in val_files]},
        # ไม่เก็บรายชื่อ 4500 ตัว ไฟล์จะบวมโดยไม่ได้ใช้ — hash พอสำหรับยืนยันว่าเป็น
        # ชุดเดียวกัน ส่วนภาพที่ Phase 3 หยิบไปจริงถูกบันทึกใน calib cache meta
        "calib_pool": {"count": len(pool_files), "manifest_sha256": manifest["calib_pool"]},
    }, indent=2))
    print(f"[data] เขียน {split_path}")

    # เทียบกับค่าที่รู้คำตอบได้เฉพาะตอนใช้ default — ถ้าผู้ใช้เปลี่ยน seed/val-num
    # hash ย่อมต่างโดยตั้งใจ ไม่ใช่ความผิดพลาด
    default_run = (args.seed, args.val_num) == (SEED, VAL_NUM)
    for name in groups:
        mark = ""
        if default_run:
            mark = " ✓ ตรงกับ README" if manifest[name] == EXPECTED[name] else " ✗ ไม่ตรงกับ README"
        print(f"[data] {name:11s} {manifest[name][:16]}…{mark}")

    if default_run and any(manifest[n] != EXPECTED[n] for n in groups):
        raise SystemExit(
            "manifest ไม่ตรงกับที่ README บันทึกไว้ — --src น่าจะไม่ใช่ val2017 ครบ 5000 ภาพ\n"
            "ตัวเลข mAP ที่ได้จะเทียบกับตารางใน README ไม่ได้")

    print("[data] val500 กับ calib_pool ไม่ทับกัน — Phase 3 calibrate ได้โดยไม่ leak")


if __name__ == "__main__":
    main()

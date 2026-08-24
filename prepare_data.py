#!/usr/bin/env python3
"""เลือก 500 ภาพจาก COCO val2017 มาเป็นชุดวัดผล

    data/val500/  500 ภาพ -> วัด latency (benchmark.py) และ mAP (evaluate.py)

ภาพสำหรับ INT8 calibration ไม่ได้มาจากที่นี่ — อยู่ใน data/train_pool/ ซึ่งดึงมาจาก
COCO train2017 ด้วย fetch_train_pool.py แยกคนละ split กันเพื่อไม่ให้
calibrate ด้วยภาพที่กำลังจะเอาไปให้คะแนน (INT8 จะถูกจูน dynamic range ให้พอดีกับข้อสอบ
แล้ว mAP ออกมาดีเกินจริงโดยไม่มีอะไรฟ้อง)
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

# fingerprint ของ val500 ที่ค่า default ให้ วัดจาก val2017 ชุดทางการ (5000 ภาพ)
# ถ้าไม่ตรง แปลว่า --src ไม่ใช่ val2017 ครบชุด — ตัวเลข mAP จะเทียบกับ README ไม่ได้
EXPECTED_VAL500 = "faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7"


def link_or_copy(src: Path, dst: Path) -> str:
    """hardlink ก่อน ถ้าไม่ได้ค่อย copy

    hardlink เป็นไฟล์จริงทุกประการสำหรับ cv2.imread และ Path.rglob ไม่ใช่ symlink
    ที่พังเวลาย้ายโฟลเดอร์ และไม่กินดิสก์เพิ่มจาก data/val2017 ที่มีอยู่แล้ว
    ข้ามพาร์ทิชันไม่ได้ (EXDEV) จึงมี fallback
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


def pick(files: list[Path], seed: int, num: int) -> list[Path]:
    """seeded shuffle แล้วตัดมา num ภาพ

    ต้อง sorted() ก่อน shuffle เสมอ — iterdir() คืนลำดับตาม inode ของ filesystem
    ซึ่งต่างกันทุกเครื่อง ถ้า shuffle จากลำดับนั้น seed เดียวกันจะได้คนละชุด

    shuffle ไม่ใช่ตัด 500 ตัวแรกที่เรียงตามชื่อ เพราะ image_id ของ COCO ไม่ได้
    สุ่มเรียงมาแต่แรก การตัดหัวจึงอาจได้ภาพที่เอียงไปทางใดทางหนึ่ง
    """
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:num])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="โฟลเดอร์ val2017 ที่แตก zip แล้ว")
    ap.add_argument("--out", default="data", help="โฟลเดอร์ปลายทาง (default: data)")
    ap.add_argument("--num", type=int, default=VAL_NUM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if len(files) < args.num:
        raise SystemExit(f"เจอ {len(files)} รูปใน {src} — น้อยกว่า --num {args.num}")
    print(f"[data] เจอ {len(files)} รูปใน {src}")

    chosen = pick(files, args.seed, args.num)
    dest = out / "val500"
    dest.mkdir(parents=True, exist_ok=True)
    modes = {"link": 0, "copy": 0, "skip": 0}
    for p in chosen:
        # ชื่อไฟล์เดิมเป๊ะ ห้ามเปลี่ยน — evaluate.image_id_from_name() แปลง
        # 000000000139.jpg -> image_id 139 เพื่อจับคู่กับ instances_val2017.json
        # ถ้าเปลี่ยนชื่อ COCOeval จับคู่ไม่ได้แล้วคืน mAP 0.000 โดยไม่มี error
        modes[link_or_copy(p, dest / p.name)] += 1
    print(f"[data] {dest}: {len(chosen)} รูป {modes}")

    digest = manifest_hash([p.name for p in chosen])
    split_path = out / "split.json"
    split_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        # เก็บชื่อโฟลเดอร์ ไม่ใช่ absolute path — path เต็มผูกกับ home ของเครื่องที่รัน
        # ทำให้ diff ของไฟล์นี้ระหว่างสองเครื่องต่างกันทั้งที่เลือกภาพชุดเดียวกันเป๊ะ
        "source": src.name,
        "seed": args.seed,
        "total": len(files),
        "val500": {"count": len(chosen), "manifest_sha256": digest,
                   "files": [p.name for p in chosen]},
    }, indent=2))
    print(f"[data] เขียน {split_path}")

    # เทียบกับค่าที่รู้คำตอบได้เฉพาะตอนใช้ default — ถ้าผู้ใช้เปลี่ยน seed/num
    # hash ย่อมต่างโดยตั้งใจ ไม่ใช่ความผิดพลาด
    if (args.seed, args.num) != (SEED, VAL_NUM):
        print(f"[data] val500 {digest[:16]}… (seed/num ไม่ใช่ค่า default จึงไม่เทียบกับ README)")
        return
    if digest != EXPECTED_VAL500:
        raise SystemExit(
            f"[data] val500 {digest[:16]}… ✗ ไม่ตรงกับ README\n"
            "--src น่าจะไม่ใช่ val2017 ครบ 5000 ภาพ — ตัวเลข mAP ที่ได้จะเทียบกับ"
            "ตารางใน README ไม่ได้")
    print(f"[data] val500 {digest[:16]}… ✓ ตรงกับ README")


if __name__ == "__main__":
    main()

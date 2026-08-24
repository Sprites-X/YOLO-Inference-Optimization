from __future__ import annotations

import argparse
import json
import contextlib
import io
from pathlib import Path

import cv2
import numpy as np

from benchmark import ONNXRunner, PyTorchRunner, TensorRTRunner
from common import COCO80_TO_91, VAL_CONF, VAL_IOU, VAL_MAX_DET, postprocess, preprocess

IMG_EXT = {".jpg", ".jpeg", ".png"}


def image_id_from_name(p: Path) -> int:
    """COCO val2017 ตั้งชื่อไฟล์เป็น 000000xxxxxx.jpg -> image_id คือเลขนั้น

    เลขนี้ต้องตรงกับ image_id ใน instances_val2017.json เป๊ะ ถ้าไม่ตรง COCOeval
    จะไม่จับคู่ detection กับ ground truth เลย แล้วคืน mAP 0.000 โดยไม่มี error
    """
    try:
        return int(p.stem.lstrip("0") or "0")
    except ValueError:
        raise SystemExit(f"ชื่อไฟล์ไม่ใช่รูปแบบ COCO: {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True, help="instances_val2017.json")
    ap.add_argument("--runtime", required=True, choices=["pytorch", "onnx", "tensorrt"])
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--per-class", action="store_true",
                    help="พิมพ์ AP รายคลาส — ใช้ตอบว่า INT8 ทำคลาสไหนพัง")
    ap.add_argument("--out", default="accuracy.jsonl")
    args = ap.parse_args()

    # import ตรงนี้ไม่ใช่ข้างบน เพราะ pycocotools เป็น optional dependency
    # (verify_env.py ให้ WARN ไม่ใช่ FAIL) — benchmark.py ยังรันได้โดยไม่มีมัน
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # sorted() ไม่ใช่แค่ความเป็นระเบียบ — --limit ตัดจากหัว ถ้าลำดับไม่คงที่
    # ชุดภาพที่ใช้วัดจะเปลี่ยนไปทุกครั้ง แล้ว mAP ของแต่ละ runtime เทียบกันไม่ได้
    files = sorted(p for p in Path(args.images).rglob("*")
                   if p.suffix.lower() in IMG_EXT)[:args.limit]
    if not files:
        raise SystemExit(f"ไม่เจอรูปใน {args.images}")

    if args.runtime == "pytorch":
        runner = PyTorchRunner(args.model, args.device, args.half)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    elif args.runtime == "onnx":
        runner = ONNXRunner(args.model, args.device)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    else:
        # batch=1 เสมอ: mAP ไม่ขึ้นกับ batch size อยู่แล้ว และ batch 1 ทำให้จับคู่
        # ผลกับ image_id ตรงไปตรงมา ไม่ต้องตามว่าภาพไหนอยู่ตำแหน่งไหนในก้อน
        runner = TensorRTRunner(args.model, batch=1)
        device_label = "GPU"

    print(f"[eval] {runner.name} {runner.precision} {device_label} บน {len(files)} รูป")
    print(f"[eval] conf={VAL_CONF} iou={VAL_IOU} max_det={VAL_MAX_DET} (ตรงกับ ultralytics val)")

    detections = []
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        x, r, pad = preprocess(img)
        raw = runner.infer(x)
        # sync ก่อนอ่านเสมอ แล้ว np.array() เพื่อ copy ออกมา — TensorRTRunner.infer
        # คืน host_out buffer ตัวเดิมทุกครั้ง ถ้าไม่ copy ผลจะถูกทับในรอบถัดไป
        runner.sync()
        raw = np.array(raw)

        # ใช้ VAL_* ไม่ใช่ DEPLOY_* — conf 0.001 คือสิ่งที่ COCO AP ต้องการ
        # (ดูเหตุผลเต็มที่หัวไฟล์ common.py)
        boxes, scores, cls = postprocess(raw, (r, pad), VAL_CONF, VAL_IOU, VAL_MAX_DET)

        # clip เข้ากรอบภาพเดิม: กล่องที่ทะลุขอบทำให้พื้นที่ในสูตร IoU บวมเกินจริง
        # แล้ว AP ตกทั้งที่ตำแหน่งถูก — postprocess ไม่ clip ให้เพราะ benchmark ไม่ต้องใช้
        h, w = img.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)

        img_id = image_id_from_name(p)
        for b, s, c in zip(boxes, scores, cls):
            detections.append({
                "image_id": img_id,
                "category_id": COCO80_TO_91[int(c)],
                # COCO ต้องการ [x, y, width, height] ไม่ใช่ xyxy — ส่ง xyxy ไปตรงๆ
                # จะไม่มี error แต่กล่องจะผิดรูปหมดแล้ว mAP ออกมาเกือบ 0
                "bbox": [round(float(b[0]), 2), round(float(b[1]), 2),
                         round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)],
                "score": round(float(s), 5),
            })
        if (i + 1) % 50 == 0:
            print(f"\r[eval] {i+1}/{len(files)}", end="", flush=True)
    print()

    if not detections:
        raise SystemExit("ไม่มี detection เลย — เช็ก postprocess หรือ export ก่อน")

    # pycocotools พิมพ์ progress ของมันเองเยอะมาก กลบผลจริง — กลืนเฉพาะช่วง
    # โหลด/evaluate/accumulate ส่วน summarize() อยู่นอก with เพราะตารางที่มันพิมพ์
    # คือผลลัพธ์ที่เราต้องการเห็น
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        # จำกัดให้เหลือเฉพาะภาพที่รันจริง — ถ้าไม่ตั้ง COCOeval จะวัดครบทั้ง 5000 ภาพ
        # ของ val2017 แล้วนับ 4500 ภาพที่เราไม่ได้รันเป็น miss ทั้งหมด mAP จะเหลือ ~1/10
        #
        # NOTE: ภาพที่ cv2.imread คืน None ถูก continue ข้ามไปแล้ว แต่ยังอยู่ในลิสต์นี้
        # เลยถูกนับเป็น miss จริงๆ — ยังไม่เคยเจอเคสนั้น ถ้าเจอ mAP จะต่ำแบบอธิบายไม่ได้
        ev.params.imgIds = [image_id_from_name(p) for p in files]
        ev.evaluate()
        ev.accumulate()
    ev.summarize()

    stats = ev.stats
    # คีย์ runtime/precision/device ต้องสะกดตรงกับที่ benchmark.py เขียนลง results.jsonl
    # เพราะ make_report.key_of() ใช้สามตัวนี้เป็น join key ระหว่างสองไฟล์
    # ถ้าไม่ตรง คอลัมน์ mAP ในตารางจะเป็น "—" ทั้งแถวโดยไม่มีอะไรฟ้อง
    record = {
        "runtime": runner.name, "precision": runner.precision, "device": device_label,
        "model": args.model, "num_images": len(files), "num_detections": len(detections),
        "mAP50_95": round(float(stats[0]), 4),
        "mAP50": round(float(stats[1]), 4),
        "mAP75": round(float(stats[2]), 4),
        "mAP_small": round(float(stats[3]), 4),
        "mAP_medium": round(float(stats[4]), 4),
        "mAP_large": round(float(stats[5]), 4),
    }

    if args.per_class:
        # precision: [T, R, K, A, M] — K คือคลาส, A=0 (all areas), M=2 (maxDet 100)
        # A=0 เพราะอยากได้ภาพรวมทุกขนาดวัตถุ (ขนาดแยกดูจาก mAP_small/medium/large แล้ว)
        # M=2 เพราะ params.maxDets = [1, 10, 100] — index 2 คือ 100 ซึ่งเป็นตัวที่
        # COCO ใช้รายงาน AP มาตรฐาน (ไม่ใช่ 300 ที่เราตั้งไว้ใน NMS — 300 คือเพดาน
        # ก่อนส่งเข้า eval ส่วน 100 คือจำนวนที่ COCO ยอมนับ)
        prec = ev.eval["precision"]
        per_class = {}
        cat_ids = coco_gt.getCatIds()
        for k, cid in enumerate(cat_ids):
            pr = prec[:, :, k, 0, 2]
            # pycocotools ใส่ -1 ให้ช่องที่คลาสนั้นไม่มี GT ในชุดที่ประเมิน
            # ถ้าเอามาเฉลี่ยด้วยจะได้ AP ติดลบ — ต้องกรองทิ้งก่อน
            pr = pr[pr > -1]
            if pr.size:
                per_class[coco_gt.loadCats(cid)[0]["name"]] = round(float(pr.mean()), 4)
        record["per_class_AP"] = per_class
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
        print("\n  คลาสที่ AP ต่ำสุด 10 อันดับ (เทียบข้าม precision เพื่อดูว่า INT8 พังตรงไหน):")
        for n, v in worst:
            print(f"    {n:<20} {v:.4f}")

    with open(args.out, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n  mAP50-95 = {record['mAP50_95']:.4f} | mAP50 = {record['mAP50']:.4f}")
    print(f"  small {record['mAP_small']:.3f} / medium {record['mAP_medium']:.3f} "
          f"/ large {record['mAP_large']:.3f}")
    print(f"  -> ต่อท้ายลง {args.out}")

    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()

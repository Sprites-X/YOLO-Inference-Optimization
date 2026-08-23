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
    """COCO val2017 ตั้งชื่อไฟล์เป็น 000000xxxxxx.jpg -> image_id คือเลขนั้น"""
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

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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
        runner.sync()
        raw = np.array(raw)

        boxes, scores, cls = postprocess(raw, (r, pad), VAL_CONF, VAL_IOU, VAL_MAX_DET)

        h, w = img.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)

        img_id = image_id_from_name(p)
        for b, s, c in zip(boxes, scores, cls):
            detections.append({
                "image_id": img_id,
                "category_id": COCO80_TO_91[int(c)],
                "bbox": [round(float(b[0]), 2), round(float(b[1]), 2),
                         round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)],
                "score": round(float(s), 5),
            })
        if (i + 1) % 50 == 0:
            print(f"\r[eval] {i+1}/{len(files)}", end="", flush=True)
    print()

    if not detections:
        raise SystemExit("ไม่มี detection เลย — เช็ก postprocess หรือ export ก่อน")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = [image_id_from_name(p) for p in files]
        ev.evaluate()
        ev.accumulate()
    ev.summarize()

    stats = ev.stats
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
        prec = ev.eval["precision"]
        per_class = {}
        cat_ids = coco_gt.getCatIds()
        for k, cid in enumerate(cat_ids):
            pr = prec[:, :, k, 0, 2]
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
    print("\n  [note] วัตถุเล็ก (mAP_small) มักตกหนักสุดตอน INT8 — "
          "ถ้า small ตกเยอะกว่า large ชัดเจน เขียนอธิบายตรงนี้ใน Analysis")

    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()

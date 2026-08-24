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
    """COCO val2017 names files 000000xxxxxx.jpg -> the image_id is that number.

    It has to match the image_id in instances_val2017.json exactly. If it does not,
    COCOeval pairs no detections with ground truth at all and returns mAP 0.000
    without raising anything.
    """
    try:
        return int(p.stem.lstrip("0") or "0")
    except ValueError:
        raise SystemExit(f"filename is not in COCO format: {p.name}")


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
                    help="print per-class AP — shows which classes INT8 breaks")
    ap.add_argument("--out", default="accuracy.jsonl")
    args = ap.parse_args()

    # Imported here rather than at the top because pycocotools is an optional
    # dependency (verify_env.py warns rather than fails) — benchmark.py runs fine
    # without it.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # sorted() is not just tidiness — --limit slices from the front, so an unstable
    # order would change which images get measured on every run, and the runtimes'
    # mAP numbers would stop being comparable.
    files = sorted(p for p in Path(args.images).rglob("*")
                   if p.suffix.lower() in IMG_EXT)[:args.limit]
    if not files:
        raise SystemExit(f"no images found in {args.images}")

    if args.runtime == "pytorch":
        runner = PyTorchRunner(args.model, args.device, args.half)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    elif args.runtime == "onnx":
        runner = ONNXRunner(args.model, args.device)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    else:
        # Always batch=1: mAP does not depend on batch size anyway, and batch 1 keeps
        # each result paired with its image_id directly, with no bookkeeping about
        # which image sat where in the batch.
        runner = TensorRTRunner(args.model, batch=1)
        device_label = "GPU"

    print(f"[eval] {runner.name} {runner.precision} {device_label} on {len(files)} images")
    print(f"[eval] conf={VAL_CONF} iou={VAL_IOU} max_det={VAL_MAX_DET} (matches ultralytics val)")

    detections = []
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        x, r, pad = preprocess(img)
        raw = runner.infer(x)
        # Always sync before reading, then np.array() to copy it out —
        # TensorRTRunner.infer hands back the same host_out buffer every call, so
        # without a copy this result is overwritten on the next iteration.
        runner.sync()
        raw = np.array(raw)

        # VAL_* here, not DEPLOY_* — conf 0.001 is what COCO AP needs
        # (full reasoning at the top of common.py).
        boxes, scores, cls = postprocess(raw, (r, pad), VAL_CONF, VAL_IOU, VAL_MAX_DET)

        # Clip back inside the original frame: boxes running past the edge inflate the
        # area term in IoU, so AP drops even when the position was right. postprocess
        # does not clip because benchmark has no use for it.
        h, w = img.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)

        img_id = image_id_from_name(p)
        for b, s, c in zip(boxes, scores, cls):
            detections.append({
                "image_id": img_id,
                "category_id": COCO80_TO_91[int(c)],
                # COCO wants [x, y, width, height], not xyxy. Pass xyxy straight
                # through and nothing errors, you just get malformed boxes and an
                # mAP near 0.
                "bbox": [round(float(b[0]), 2), round(float(b[1]), 2),
                         round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)],
                "score": round(float(s), 5),
            })
        if (i + 1) % 50 == 0:
            print(f"\r[eval] {i+1}/{len(files)}", end="", flush=True)
    print()

    if not detections:
        raise SystemExit("no detections at all — check postprocess or the export first")

    # pycocotools prints plenty of its own progress and buries the real output, so
    # swallow just the load/evaluate/accumulate phase. summarize() stays outside the
    # with block because the table it prints is the result we came for.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        # Restrict to the images actually run. Without this, COCOeval scores all 5000
        # val2017 images and counts the 4500 we skipped as misses, leaving roughly a
        # tenth of the real mAP.
        #
        # NOTE: images where cv2.imread returned None were skipped by the continue
        # above but are still in this list, so they do get counted as real misses.
        # Never hit that case yet; if it happens, mAP drops for no visible reason.
        ev.params.imgIds = [image_id_from_name(p) for p in files]
        ev.evaluate()
        ev.accumulate()
    ev.summarize()

    stats = ev.stats
    # runtime/precision/device have to be spelled exactly as benchmark.py writes them
    # into results.jsonl, because make_report.key_of() joins the two files on those
    # three. Any mismatch and the mAP column reads "—" for the whole row, silently.
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
        # precision: [T, R, K, A, M] — K is class, A=0 (all areas), M=2 (maxDet 100).
        # A=0 because we want the picture across all object sizes (per-size numbers
        # already come from mAP_small/medium/large). M=2 because params.maxDets is
        # [1, 10, 100] and index 2 is 100, which is what COCO reports standard AP at.
        # That is not the 300 set in NMS — 300 is the cap before eval, 100 is how many
        # COCO agrees to count.
        prec = ev.eval["precision"]
        per_class = {}
        cat_ids = coco_gt.getCatIds()
        for k, cid in enumerate(cat_ids):
            pr = prec[:, :, k, 0, 2]
            # pycocotools writes -1 where a class has no GT in the evaluated set.
            # Average those in and the AP comes out negative, so filter them first.
            pr = pr[pr > -1]
            if pr.size:
                per_class[coco_gt.loadCats(cid)[0]["name"]] = round(float(pr.mean()), 4)
        record["per_class_AP"] = per_class
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
        print("\n  10 lowest-AP classes (compare across precisions to see where INT8 breaks):")
        for n, v in worst:
            print(f"    {n:<20} {v:.4f}")

    with open(args.out, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n  mAP50-95 = {record['mAP50_95']:.4f} | mAP50 = {record['mAP50']:.4f}")
    print(f"  small {record['mAP_small']:.3f} / medium {record['mAP_medium']:.3f} "
          f"/ large {record['mAP_large']:.3f}")
    print(f"  -> appended to {args.out}")

    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()

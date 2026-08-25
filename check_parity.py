from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from benchmark import IMG_EXT, ONNXRunner, PyTorchRunner, TensorRTRunner
from common import DEPLOY_CONF, DEPLOY_IOU, _xywh2xyxy, postprocess, preprocess

# The gate between export and measurement. An export that quietly changed the model
# — wrong opset, a layer TRT fused differently, an input laid out the way the runtime
# did not expect — still produces perfectly plausible latency numbers, so a speed
# table can never catch it. Comparing detections against the PyTorch model the export
# came from does catch it, before any number goes in the table.
#
# Runs before benchmark.py and evaluate.py in run_all.sh: a failure here means the
# rows about to be measured are different models, and comparing them says nothing.

# Two gates on geometry, because two different things can go wrong.
#
# MIN_PAIR_IOU is per detection, and scale-free on purpose: the same pixel delta means
# different things at different box sizes. 5 px on a 600 px box is nothing; 5 px on a
# 30 px box is a different detection. Measured against the FP16 engine over 497 matched
# pairs, mean absolute delta runs 0.17 px for boxes under 100 px and 0.46 px for boxes
# over 500 px, while mean relative delta runs the other way, 0.38% against 0.08%. So
# neither a pixel budget nor a percentage works as a per-box rule — they fail at
# opposite ends. (Absolute delta does climb with box size, but weakly: r = 0.17. Not
# the clean proportionality it first looked like, which is why the rule is not built
# on it.)
#
# The outliers are not precision at all. NMS is a discrete choice: on
# 000000097585.jpg the two best candidates for one object score 0.75133 and 0.74966 in
# FP32, FP16 rounds both to exactly 0.75049, the tie-break then keeps the other one,
# and the surviving box moves 6.6 px. Nothing about the export is wrong — the same
# object is found either way — so the gate asks whether both runtimes found the same
# object, which is what IoU asks.
MIN_PAIR_IOU = 0.90

# IoU stops being a fair question on a very small box. 1.14 px of disagreement on a
# 28 px box is IoU 0.9093 — which is the worst pair ONNX FP32 produces against PyTorch
# FP32 over the 500, two runtimes that are numerically all but identical. A slightly
# smaller box would put a correct export under 0.90. So a pair also passes if the boxes
# are within this many pixels outright: below that, the two ways of being the same box
# are "they overlap well" or "they are a pixel apart", and small boxes can only satisfy
# the second. Checked against the perturbation table below — the floor does not blunt
# any of it.
BOX_ABS_FLOOR_PX = 2.0

# MEAN_BOX_REL_TOL is a percentage too, but averaged over every pair rather than
# applied to each one, which is what makes it usable where a per-box percentage is not.
# It is the systematic gate, and catches what MIN_PAIR_IOU alone would miss: an export
# that shifts or rescales every box slightly (a letterbox off by a few px, the wrong
# pad colour) can stay above 0.90 IoU on every large box while being wrong everywhere.
# Precision noise and the odd NMS swap move a handful of boxes; a broken export moves
# all of them, so the mean separates the two.
# Measured over the full 500: ONNX 0.014%, TensorRT FP16 0.256%. 0.5% is only ~2x the
# FP16 figure, which is deliberate — it is tight enough to be worth something. INT8
# will drift further and may need this raised; set it from a measurement then, not
# from a guess now.
#
# That this gate earns its place was checked rather than assumed, by perturbing a
# working engine's output and watching which gate fires:
#
#   perturbation      verdict   mean drift   pairs failing per-box   unmatched
#   none                 pass       0.240%                       0           0
#   x shifted 1 px       fail       1.338%                       0           0
#   x shifted 3 px       fail       3.806%                     184           0
#   x shifted 10 px      fail       8.508%                     298         153
#   width x1.02          fail       0.864%                       0           0
#   width x1.10          fail       3.819%                      11           0
#
# The two rows that matter are the ones where nothing per-box fires: a 1 px shift and a
# 2% widening are caught by the mean alone. Without it both would pass. That is the
# whole reason there are two gates rather than one.
MEAN_BOX_REL_TOL = 0.005
SCORE_TOL = 0.02

# A detection scoring within this of the conf threshold may legitimately appear on one
# runtime and not the other: FP16 moves scores by ~0.007, so a box at 0.2510 survives
# on one side and falls under 0.25 on the other. That is the threshold being stepped
# over, not a detection being lost, and it can only happen to boxes already sitting on
# the line. An unmatched detection scoring clear of the band is a real disagreement
# and still fails.
THRESHOLD_BAND = 0.05

# Below this IoU two boxes are not the same detection, so a delta between them is
# meaningless — they are counted as unmatched instead.
MATCH_IOU = 0.5

# How closely a kept box has to match one of the other runtime's pre-NMS candidates to
# count as the same box rather than a different one.
#
# This is what makes MIN_PAIR_IOU survivable. Comparing the box one runtime kept
# against the box the other kept is ill-posed wherever an object has near-duplicate
# candidates: NMS keeps exactly one, the scores deciding it can differ in the fourth
# decimal, and the two survivors can then sit 20% of a box apart while both models
# produced both boxes. Over the full 500 images four pairs land between 0.79 and 0.88
# IoU for that reason alone — on 000000394206.jpg PyTorch keeps a box scoring 0.46719
# and TensorRT one scoring 0.46680, and PyTorch's own candidates contain TensorRT's box
# at 0.9979 IoU scoring 0.46665.
#
# So a pair that fails MIN_PAIR_IOU is checked against the candidates each side
# produced before NMS ran. If both kept boxes appear on both sides, the models agree
# and only the tie-break differs. If they do not, the export really did move a box.
# Measured on those four: 0.9878 to 0.9979, so 0.95 separates them from a real miss.
CANDIDATE_IOU = 0.95


def build_runner(spec: str):
    """'pytorch:yolov8n.pt:cuda' | 'onnx:yolov8n.onnx:cpu' | 'tensorrt:x_fp16.engine'"""
    parts = spec.split(":")
    kind, model = parts[0], parts[1]
    device = parts[2] if len(parts) > 2 else "cuda"
    if kind == "pytorch":
        return PyTorchRunner(model, device), f"PyTorch/{device}"
    if kind == "onnx":
        return ONNXRunner(model, device), f"ONNX/{Path(model).stem}/{device}"
    if kind == "tensorrt":
        # batch=1 to match how evaluate.py runs the engine; parity is per image.
        return TensorRTRunner(model, batch=1), f"TensorRT/{Path(model).stem}"
    raise SystemExit(f"unknown runtime in --ref/--cmp: {spec}")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def match(ref, cmp):
    """Greedy IoU pairing within a class. Returns (pairs, unmatched_ref, unmatched_cmp).

    Paired by IoU rather than by position in the list, because NMS returns detections
    in score order and two boxes scoring 0.501 and 0.502 can swap places between
    runtimes. Compared by index that swap reads as two large box errors; it is really
    two correct boxes in a different order.
    """
    rb, rs, rc = ref
    cb, cs, cc = cmp
    ious = iou_matrix(rb, cb)
    # Never pair across classes: the same box under a different label is a real
    # disagreement, not a small numeric one.
    ious[rc[:, None] != cc[None, :]] = 0.0

    pairs, used_r, used_c = [], set(), set()
    for i, j in sorted(np.ndindex(ious.shape), key=lambda ij: -ious[ij]):
        if ious[i, j] < MATCH_IOU:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    return (pairs,
            [i for i in range(len(rb)) if i not in used_r],
            [j for j in range(len(cb)) if j not in used_c])


def class_candidates(raw: np.ndarray, cls_id: int, conf: float, meta):
    """Pre-NMS boxes of one class above conf, mapped back to original image coordinates.

    Mirrors common.postprocess up to the point NMS runs, so the boxes are comparable to
    the detections it returns. Only built for a pair that already failed MIN_PAIR_IOU,
    which is rare enough that the cost does not matter.
    """
    r = raw[0] if raw.ndim == 3 else raw
    if r.shape[0] < r.shape[1]:
        r = r.T
    cls_scores = r[:, 4:]
    ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), ids]
    m = (ids == cls_id) & (scores > conf)
    if not m.any():
        return np.zeros((0, 4), np.float32)

    boxes = _xywh2xyxy(r[m][:, :4]).astype(np.float32)
    scale, (padx, pady) = meta
    boxes[:, [0, 2]] -= padx
    boxes[:, [1, 3]] -= pady
    boxes /= scale
    return boxes


def is_nms_pick(ref_raw, cmp_raw, cls_id, ref_box, cmp_box, ref_score, cmp_score,
                conf, meta) -> bool:
    """True when both runtimes produced both boxes and NMS merely kept different ones."""
    # Near-equal scores are what let the tie-break go either way in the first place; a
    # real difference in confidence means these are not interchangeable candidates.
    if abs(float(ref_score) - float(cmp_score)) > SCORE_TOL:
        return False
    ref_cand = class_candidates(ref_raw, cls_id, conf, meta)
    cmp_cand = class_candidates(cmp_raw, cls_id, conf, meta)
    if len(ref_cand) == 0 or len(cmp_cand) == 0:
        return False
    # Each side has to have produced the box the other side kept.
    return (float(iou_matrix(cmp_box[None, :], ref_cand).max()) >= CANDIDATE_IOU
            and float(iou_matrix(ref_box[None, :], cmp_cand).max()) >= CANDIDATE_IOU)


def compare(runner, label, imgs, ref_raw, ref_dets, files, conf, iou, min_iou):
    worst = {"raw": 0.0, "raw_img": "", "box_frac": 0.0, "box_px": 0.0,
             "box_img": "", "box_side": 0.0, "score": 0.0, "min_pair_iou": 1.0}
    frac_sum, n_pairs = 0.0, 0
    near_thresh = []      # detections excused for sitting on the conf threshold
    real_unmatched = []   # detections one runtime found and the other did not
    nms_pick = []         # same box on both sides, different NMS survivor
    low_iou = []
    shape_fail = None

    for k, im in enumerate(imgs):
        # Preprocessed here rather than held for every image at once: at 500 images
        # that array alone is 2.3 GB. preprocess is deterministic (verified byte-equal
        # across repeated calls), so every runtime still sees the identical input.
        x, scale, pad = preprocess(im)
        meta = (scale, pad)
        raw = runner.infer(x)
        runner.sync()
        raw = np.array(raw)   # TensorRTRunner hands back the same host buffer each call

        if raw.shape != ref_raw[k].shape:
            shape_fail = f"output shape {raw.shape} != reference {ref_raw[k].shape}"
            break

        # Raw tensor delta, before conf/NMS can hide a small difference — or turn one
        # into a whole missing detection. Reported, never gated: the max runs over all
        # 8400 anchors including those predicting nothing, whose boxes are
        # unconstrained garbage, so a few px of disagreement there means nothing.
        d = float(np.abs(raw.astype(np.float32) - ref_raw[k]).max())
        if d > worst["raw"]:
            worst["raw"], worst["raw_img"] = d, files[k].name

        dets = postprocess(raw, meta, conf, iou)
        rb, rs, rc = ref_dets[k]
        cb, cs, cc = dets
        pairs, un_r, un_c = match(ref_dets[k], dets)

        for i, j in pairs:
            px = float(np.abs(rb[i] - cb[j]).max())
            side = max(float(rb[i][2] - rb[i][0]), float(rb[i][3] - rb[i][1]), 1.0)
            frac = px / side
            if frac > worst["box_frac"]:
                worst.update(box_frac=frac, box_px=px, box_side=side,
                             box_img=files[k].name)
            frac_sum += frac
            n_pairs += 1
            worst["score"] = max(worst["score"], float(abs(rs[i] - cs[j])))
            pair_iou = float(iou_matrix(rb[i:i + 1], cb[j:j + 1])[0, 0])
            worst["min_pair_iou"] = min(worst["min_pair_iou"], pair_iou)
            if pair_iou < min_iou and px > BOX_ABS_FLOOR_PX:
                where = (f"{files[k].name}: cls {int(rc[i])} IoU {pair_iou:.4f}, "
                         f"off {px:.2f} px on a {side:.0f} px long side "
                         f"({100 * frac:.2f}%)")
                if is_nms_pick(ref_raw[k], raw, int(rc[i]), rb[i], cb[j],
                               rs[i], cs[j], conf, meta):
                    nms_pick.append(where + " — both sides produced both boxes")
                else:
                    low_iou.append(where)

        for i in un_r:
            entry = f"{files[k].name}: cls {int(rc[i])} score {rs[i]:.4f} in reference only"
            (near_thresh if rs[i] < conf + THRESHOLD_BAND else real_unmatched).append(entry)
        for j in un_c:
            entry = f"{files[k].name}: cls {int(cc[j])} score {cs[j]:.4f} in {label} only"
            (near_thresh if cs[j] < conf + THRESHOLD_BAND else real_unmatched).append(entry)

    worst["mean_frac"] = frac_sum / n_pairs if n_pairs else 0.0
    worst["n_pairs"] = n_pairs
    return worst, near_thresh, nms_pick, real_unmatched, low_iou, shape_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/val500")
    ap.add_argument("--n", type=int, default=20, help="images to compare")
    ap.add_argument("--ref", default="pytorch:yolov8n.pt:cuda",
                    help="the runtime everything else is compared against")
    ap.add_argument("--cmp", action="append", default=None)
    ap.add_argument("--min-iou", type=float, default=MIN_PAIR_IOU,
                    help="lowest IoU allowed between a detection and its counterpart")
    ap.add_argument("--conf", type=float, default=DEPLOY_CONF)
    ap.add_argument("--iou", type=float, default=DEPLOY_IOU)
    args = ap.parse_args()

    cmp_specs = args.cmp or ["onnx:yolov8n.onnx:cuda", "onnx:yolov8n.onnx:cpu",
                             "tensorrt:yolov8n_fp16.engine"]

    files = sorted(p for p in Path(args.images).rglob("*")
                   if p.suffix.lower() in IMG_EXT)[:args.n]
    if not files:
        raise SystemExit(f"no images found in {args.images}")
    # Drop unreadable images from both lists together. Filtering only imgs would shift
    # every later index, and each disagreement would then be reported against the wrong
    # filename — the one thing this script exists to tell you.
    kept = [(p, im) for p, im in ((p, cv2.imread(str(p))) for p in files) if im is not None]
    files = [p for p, _ in kept]
    imgs = [im for _, im in kept]

    print(f"[parity] {len(imgs)} images, conf={args.conf} iou={args.iou}")
    print(f"[parity] gates: matched-pair IoU >= {args.min_iou}, mean box drift "
          f"<= {100 * MEAN_BOX_REL_TOL:.1f}% of box size, score |Δ| <= {SCORE_TOL}")

    ref_runner, ref_label = build_runner(args.ref)
    print(f"[parity] reference: {ref_label}")

    # The reference raw output is kept for every image — the comparison needs it, and
    # so does the candidate check when a pair disagrees. The preprocessed inputs are
    # not kept; see the note in compare().
    ref_raw, ref_dets = [], []
    for im in imgs:
        x, scale, pad = preprocess(im)
        raw = ref_runner.infer(x)
        ref_runner.sync()
        raw = np.array(raw)
        ref_raw.append(raw)
        ref_dets.append(postprocess(raw, (scale, pad), args.conf, args.iou))
    print(f"[parity] reference found {sum(len(d[0]) for d in ref_dets)} detections "
          f"over {len(imgs)} images\n")

    failures = []
    for spec in cmp_specs:
        runner, label = build_runner(spec)
        worst, near_thresh, nms_pick, real_unmatched, low_iou, shape_fail = compare(
            runner, label, imgs, ref_raw, ref_dets, files, args.conf, args.iou,
            args.min_iou)

        ok = (not shape_fail and not real_unmatched and not low_iou
              and worst["mean_frac"] <= MEAN_BOX_REL_TOL
              and worst["score"] <= SCORE_TOL)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if shape_fail:
            print(f"        {shape_fail}")
        else:
            print(f"        matched pairs    {worst['n_pairs']}, worst IoU "
                  f"{worst['min_pair_iou']:.4f}")
            print(f"        box drift        mean {100 * worst['mean_frac']:.3f}% of box "
                  f"size, worst {worst['box_px']:.2f} px on a {worst['box_side']:.0f} px "
                  f"long side ({100 * worst['box_frac']:.2f}%)"
                  + (f"  ({worst['box_img']})" if worst["box_img"] else ""))
            print(f"        score max|Δ|     {worst['score']:.5f}")
            print(f"        raw max|Δ|       {worst['raw']:.4f} "
                  f"(over all 8400 anchors, not gated)")
            print(f"        on the threshold {len(near_thresh)} (allowed)")
            print(f"        NMS pick differs {len(nms_pick)} (allowed)")
            print(f"        disagreements    {len(real_unmatched)}")
        for e in near_thresh + nms_pick:
            print(f"          ~ {e}")
        for e in real_unmatched + low_iou:
            print(f"          ! {e}")

        if not ok:
            failures.append(f"{label}: " + (shape_fail or
                            f"{len(real_unmatched)} disagreements, {len(low_iou)} boxes "
                            f"below IoU {args.min_iou}, mean drift "
                            f"{100 * worst['mean_frac']:.3f}%"))
        if hasattr(runner, "close"):
            runner.close()
        print()

    if hasattr(ref_runner, "close"):
        ref_runner.close()

    if failures:
        print("PARITY FAILED — the exports do not agree with the reference model:")
        for f in failures:
            print(f"  - {f}")
        print("\nMeasuring now would compare different models, not different runtimes.")
        sys.exit(1)
    print("parity ok — every runtime returns the same detections, safe to measure")


if __name__ == "__main__":
    main()

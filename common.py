from __future__ import annotations

import numpy as np
import cv2

IMG_SIZE = 640

# ultralytics' grey — has to be exactly 114, not 0 or 128. The model saw borders
# this colour during training, so changing it moves mAP with nothing to warn you.
# Checked on cv2 5.0.0: letterbox() matches ultralytics
# LetterBox(auto=False, scaleup=False) byte for byte on all 8 ratios tried.
PAD_VALUE = 114

# Never swap these two sets.
#
# VAL_* is for measuring mAP. conf 0.001 looks absurdly low, but COCO AP is the
# area under the PR curve and needs that low-scoring tail to carry the curve all
# the way out. At 0.25 mAP drops several points because recall never reaches the
# end, and nothing tells you.
VAL_CONF = 0.001
VAL_IOU = 0.7
VAL_MAX_DET = 300

# DEPLOY_* is what you would actually ship, and what latency is measured with,
# since postprocess counts toward post_ms. Timing with conf 0.001 means NMS
# filtering thousands of boxes every frame — not a load any real system carries,
# so the number inflates for nothing.
#
# So the mAP and latency columns come from different thresholds, on purpose.
DEPLOY_CONF = 0.25
DEPLOY_IOU = 0.45


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------
def letterbox(img: np.ndarray, new_shape: int = IMG_SIZE, scaleup: bool = False):
    # scaleup=False to match ultralytics val: images already under 640 stay put.
    # Upscaling adds no detail, it just drifts from the val run we compare to.
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # The ±0.1 is not arbitrary. dw/dh are half the missing width/height, so they
    # can land on .5. round(x-0.1) and round(x+0.1) split an odd pad into n and
    # n+1, which sums back to exactly what was missing. int() on both loses a
    # pixel and TRT then complains the shape misses the profile; round() on both
    # gains one. Copied straight from ultralytics so bbox mapping lines up.
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return img, r, (left, top)


def preprocess(bgr: np.ndarray, size: int = IMG_SIZE):
    """BGR uint8 HWC  ->  RGB float32 NCHW [0,1] (contiguous)

    ascontiguousarray is not just being defensive. TensorRT copies raw from
    arr.ctypes.data for nbytes (benchmark.py:203) and ignores strides, while
    transpose() hands back a view whose strides are out of order. Feed that in
    directly and the GPU gets garbage — infer still succeeds, no error, and you
    are left with a mysteriously low mAP to chase.
    """
    img, r, pad = letterbox(bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(img[None]), r, pad


def preprocess_batch(bgr_list: list[np.ndarray], size: int = IMG_SIZE):
    outs, metas = [], []
    for bgr in bgr_list:
        x, r, pad = preprocess(bgr, size)
        outs.append(x)
        metas.append((r, pad))
    return np.ascontiguousarray(np.concatenate(outs, axis=0)), metas


# --------------------------------------------------------------------------
# postprocess
# --------------------------------------------------------------------------
def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_w, half_h = x[:, 2] / 2, x[:, 3] / 2
    y[:, 0] = x[:, 0] - half_w
    y[:, 1] = x[:, 1] - half_h
    y[:, 2] = x[:, 0] + half_w
    y[:, 3] = x[:, 1] + half_h
    return y


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    # Hand-rolled in numpy instead of torchvision.ops.nms / cv2.dnn.NMSBoxes
    # because all three runtimes have to use the same one. If PyTorch used
    # torchvision and TRT used something else, the mAP column would compare two
    # NMS implementations, not two runtimes. NMS also counts toward the reported
    # post_ms, so every row has to carry the same work.
    #
    # NOTE: O(n²) in a Python loop. At mAP thresholds (conf 0.001) that leaves
    # thousands of boxes per image, which makes evaluate.py slower than it should
    # be. Have not actually timed how much.
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(
    raw: np.ndarray,
    meta,
    conf_thres: float = DEPLOY_CONF,
    iou_thres: float = DEPLOY_IOU,
    max_det: int = 300,
):
    # Every runtime emits a different layout (torch head / ONNX export / TRT
    # engine), so work it out from the shape instead of hardcoding: the short
    # axis is channels (84 = 4 box + 80 class), the long one anchors (8400 at
    # input 640). Verified on (1,84,8400) (1,8400,84) (84,8400) (8400,84).
    #
    # NOTE: this breaks if anchors < channels, which needs an input under ~200px.
    # Not this project, but come back here if you ever use an imgsz that small.
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[1]:     
        raw = raw.T

    boxes_xywh = raw[:, :4]
    cls_scores = raw[:, 4:]

    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    m = scores > conf_thres
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = _xywh2xyxy(boxes_xywh[m])
    scores, class_ids = scores[m], class_ids[m]

    # class-offset NMS: shift each class into its own coordinate band first, so
    # boxes from different classes that genuinely overlap (a person holding a
    # bag) do not suppress each other. 7680 = 640×12 — boxes are still in
    # letterbox coordinates (640 max), so that is plenty of separation. Lets us
    # run NMS once instead of looping over 80 classes.
    offset = class_ids.astype(np.float32) * 7680.0
    keep = nms_numpy(boxes + offset[:, None], scores, iou_thres)[:max_det]

    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # Undo the letterbox. Order matters: subtract pad first, then divide by r
    # (letterbox resizes, then pads). Swap them and boxes shift by the pad size.
    #
    # No edge clipping here because benchmark does not need it — evaluate.py:74
    # clips before anything goes to COCO.
    #
    # NOTE: this runs in whatever dtype raw arrived in. Measured: float32 is off
    # by 0.000 px, float16 by 1.0 px (1080p image, r=0.333), because float16 only
    # resolves to 0.5 up around 640. You reach this through build_engine.py
    # --fp16-head, which set_output_type's the last layers to float16 — cast to
    # float32 before here if you actually use that mode.
    r, (padx, pady) = meta
    boxes[:, [0, 2]] -= padx
    boxes[:, [1, 3]] -= pady
    boxes /= r
    return boxes.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)


# --------------------------------------------------------------------------
# COCO helpers
# --------------------------------------------------------------------------
# The model emits contiguous class indices 0-79, but instances_val2017.json uses
# category_id 1-90 with gaps (12, 26, 29, 30, 45, 66, 68, 69, 71, 83 unused).
# Hand COCOeval the raw index and every class maps to the wrong one, giving an
# mAP near 0 without a single line of error — pycocotools never checks whether
# the category_id you passed makes any sense.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

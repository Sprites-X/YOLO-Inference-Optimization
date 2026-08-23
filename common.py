from __future__ import annotations

import numpy as np
import cv2

IMG_SIZE = 640
PAD_VALUE = 114

VAL_CONF = 0.001
VAL_IOU = 0.7
VAL_MAX_DET = 300

DEPLOY_CONF = 0.25
DEPLOY_IOU = 0.45


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------
def letterbox(img: np.ndarray, new_shape: int = IMG_SIZE, scaleup: bool = False):
    
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return img, r, (left, top)


def preprocess(bgr: np.ndarray, size: int = IMG_SIZE):
    """BGR uint8 HWC  ->  RGB float32 NCHW [0,1] (contiguous)"""
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

    
    offset = class_ids.astype(np.float32) * 7680.0
    keep = nms_numpy(boxes + offset[:, None], scores, iou_thres)[:max_det]

    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    
    r, (padx, pady) = meta
    boxes[:, [0, 2]] -= padx
    boxes[:, [1, 3]] -= pady
    boxes /= r
    return boxes.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)


# --------------------------------------------------------------------------
# COCO helpers
# --------------------------------------------------------------------------
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

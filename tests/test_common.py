"""The shared preprocess/postprocess every runtime goes through.

These are the functions the comparison rests on: all three runtimes call the same ones,
so a fault here moves every row of the table together and no amount of cross-runtime
agreement would reveal it. check_parity.py compares runtimes against each other and
would stay green through all of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from common import (COCO80_TO_91, DEPLOY_CONF, IMG_SIZE, PAD_VALUE, VAL_CONF,
                    _xywh2xyxy, letterbox, nms_numpy, postprocess, preprocess,
                    preprocess_batch)

N_CLASSES = 80
# Comfortably more anchors than the 84 channels. postprocess infers the layout from
# which axis is longer, so a tensor with fewer anchors than channels would be read
# transposed — the documented limit, reached only at an input under ~200px.
N_ANCHORS = 200


def bgr(h: int, w: int) -> np.ndarray:
    """A deterministic BGR frame. Content is arbitrary; only the geometry is under test."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def raw_with_boxes(boxes_xyxy, cls_ids, scores) -> np.ndarray:
    """A (1, 84, N_ANCHORS) head output whose leading anchors hold exactly these boxes.

    Every other anchor scores 0 across all classes, so it falls to the conf filter and
    the assertions below see only what was put in.
    """
    raw = np.zeros((1, 4 + N_CLASSES, N_ANCHORS), np.float32)
    for i, ((x1, y1, x2, y2), c, s) in enumerate(zip(boxes_xyxy, cls_ids, scores)):
        raw[0, 0, i] = (x1 + x2) / 2
        raw[0, 1, i] = (y1 + y2) / 2
        raw[0, 2, i] = x2 - x1
        raw[0, 3, i] = y2 - y1
        raw[0, 4 + c, i] = s
    return raw


# --------------------------------------------------------------------------
# letterbox
# --------------------------------------------------------------------------
@pytest.mark.parametrize("h,w", [
    (1080, 1920), (480, 640), (640, 640), (1000, 500), (333, 777),
    # Ratios that land the pad on a half pixel, which is what the ±0.1 in letterbox
    # exists for: int() on both sides loses a pixel and TRT then rejects the shape
    # for missing its profile, round() on both gains one.
    (641, 640), (640, 641), (427, 640), (500, 375),
])
def test_letterbox_output_is_exactly_square(h, w):
    out, r, (left, top) = letterbox(bgr(h, w))
    assert out.shape[:2] == (IMG_SIZE, IMG_SIZE)
    # The split pad has to add back to what was missing, not one more or one less.
    assert left + round(w * r) + (IMG_SIZE - left - round(w * r)) == IMG_SIZE
    assert top + round(h * r) + (IMG_SIZE - top - round(h * r)) == IMG_SIZE


def test_letterbox_does_not_upscale():
    # scaleup=False matches how ultralytics runs val. Upscaling adds no detail and only
    # moves us away from the run being compared against.
    out, r, (left, top) = letterbox(bgr(100, 200))
    assert r == 1.0
    assert out[top:top + 100, left:left + 200].shape == (100, 200, 3)


def test_letterbox_pads_with_the_training_grey():
    # 114 spelled out rather than compared against PAD_VALUE, which would only assert
    # that letterbox uses whatever the constant says. The value is the thing under
    # test: the model saw 114 borders during training, and any other grey moves mAP
    # with nothing to warn you.
    assert PAD_VALUE == 114
    out, _, _ = letterbox(bgr(200, 400))
    assert (out[0, 0] == 114).all()
    assert (out[-1, -1] == 114).all()


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------
def test_preprocess_contract():
    x, r, pad = preprocess(bgr(1080, 1920))
    assert x.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    # Load-bearing, not defensive. TensorRT copies nbytes straight from arr.ctypes.data
    # and ignores strides, and transpose() returns a view whose strides are out of
    # order — feeding that in succeeds, produces garbage, and reports nothing.
    assert x.flags["C_CONTIGUOUS"]


def test_preprocess_converts_bgr_to_rgb():
    img = np.zeros((640, 640, 3), np.uint8)
    img[:, :, 0] = 255                      # channel 0 of BGR is blue
    x, _, _ = preprocess(img)
    # Blue has to come out at channel 2, where RGB keeps it. Skip the conversion and
    # the model is fed red where the image was blue, which costs mAP and looks like
    # nothing in particular.
    assert x[0, 2].max() == pytest.approx(1.0)
    assert x[0, 0].max() == pytest.approx(0.0)


def test_preprocess_is_deterministic():
    # check_parity.py re-preprocesses per runtime rather than holding 2.3 GB of inputs,
    # which is only sound because repeated calls are byte-identical.
    img = bgr(720, 1280)
    a, _, _ = preprocess(img)
    b, _, _ = preprocess(img)
    assert np.array_equal(a, b)


def test_preprocess_batch_concatenates():
    imgs = [bgr(480, 640), bgr(1080, 1920), bgr(300, 300)]
    x, metas = preprocess_batch(imgs)
    assert x.shape == (3, 3, IMG_SIZE, IMG_SIZE)
    assert x.flags["C_CONTIGUOUS"]
    assert len(metas) == 3
    # Different aspect ratios have to keep their own scale and pad, or the boxes of
    # every image but the first come back mapped through the wrong geometry.
    assert metas[0] != metas[1]


# --------------------------------------------------------------------------
# postprocess
# --------------------------------------------------------------------------
def test_xywh2xyxy():
    got = _xywh2xyxy(np.array([[10.0, 20.0, 4.0, 6.0]], np.float32))
    assert got == pytest.approx(np.array([[8.0, 17.0, 12.0, 23.0]]))


def test_postprocess_inverts_the_letterbox():
    """A box put in at letterbox coordinates comes back at original-image coordinates.

    This is the whole geometric chain in one assertion. Subtracting the pad after
    dividing by the scale instead of before shifts every box by the pad size, which
    looks like a plausible detection and costs mAP with nothing to point at.
    """
    img = bgr(1080, 1920)
    _, r, (padx, pady) = preprocess(img)

    want = np.array([[100.0, 200.0, 500.0, 600.0]], np.float32)
    lb = want * r + np.array([padx, pady, padx, pady], np.float32)

    boxes, scores, cls = postprocess(raw_with_boxes(lb, [0], [0.9]), (r, (padx, pady)))
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx(want[0], abs=0.01)
    assert scores[0] == pytest.approx(0.9)
    assert cls[0] == 0


@pytest.mark.parametrize("shape", ["(1,84,N)", "(1,N,84)", "(84,N)", "(N,84)"])
def test_postprocess_reads_every_head_layout(shape):
    # torch's head, the ONNX export and the TRT engine each emit a different layout, so
    # postprocess works it out from which axis is longer instead of hardcoding one.
    meta = (0.5, (0, 20))
    base = raw_with_boxes([[10.0, 20.0, 110.0, 140.0]], [3], [0.8])   # (1, 84, N)
    variants = {
        "(1,84,N)": base,
        "(1,N,84)": base.transpose(0, 2, 1).copy(),
        "(84,N)": base[0],
        "(N,84)": base[0].T.copy(),
    }
    boxes, scores, cls = postprocess(variants[shape], meta)
    ref = postprocess(base, meta)
    assert np.array_equal(boxes, ref[0])
    assert np.array_equal(scores, ref[1])
    assert np.array_equal(cls, ref[2])


def test_postprocess_returns_empty_arrays_below_threshold():
    # Shapes and dtypes still have to be right when nothing was found: evaluate.py
    # iterates the result unconditionally.
    boxes, scores, cls = postprocess(
        raw_with_boxes([[10.0, 10.0, 50.0, 50.0]], [0], [DEPLOY_CONF - 0.01]),
        (1.0, (0, 0)))
    assert boxes.shape == (0, 4) and scores.shape == (0,) and cls.shape == (0,)
    assert (boxes.dtype, scores.dtype, cls.dtype) == (np.float32, np.float32, np.int32)


def test_postprocess_keeps_overlapping_boxes_of_different_classes():
    # The class-offset trick: a person and the bag they are holding genuinely overlap,
    # and neither should suppress the other. Same class at the same place still should.
    box = [10.0, 10.0, 100.0, 100.0]
    two_classes = postprocess(raw_with_boxes([box, box], [0, 1], [0.9, 0.8]), (1.0, (0, 0)))
    one_class = postprocess(raw_with_boxes([box, box], [0, 0], [0.9, 0.8]), (1.0, (0, 0)))
    assert len(two_classes[0]) == 2
    assert len(one_class[0]) == 1


def test_postprocess_caps_at_max_det():
    boxes = [[i * 100.0, 0.0, i * 100.0 + 50.0, 50.0] for i in range(6)]
    got, _, _ = postprocess(raw_with_boxes(boxes, [0] * 6, [0.9] * 6),
                            (1.0, (0, 0)), max_det=3)
    assert len(got) == 3


# --------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------
def test_nms_empty():
    assert nms_numpy(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.45) == []


def test_nms_suppresses_duplicates_and_keeps_the_best():
    boxes = np.array([[0, 0, 100, 100], [2, 2, 102, 102]], np.float32)
    scores = np.array([0.6, 0.9], np.float32)
    assert nms_numpy(boxes, scores, 0.45) == [1]        # index of the higher score


def test_nms_keeps_disjoint_boxes_in_score_order():
    boxes = np.array([[0, 0, 10, 10], [500, 500, 510, 510], [200, 200, 210, 210]], np.float32)
    scores = np.array([0.3, 0.9, 0.6], np.float32)
    assert nms_numpy(boxes, scores, 0.45) == [1, 2, 0]


def test_nms_threshold_is_the_boundary():
    # Two 10x10 boxes overlapping on exactly half their union.
    boxes = np.array([[0, 0, 10, 10], [0, 5, 10, 15]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    iou = 50 / 150
    assert nms_numpy(boxes, scores, iou + 0.01) == [0, 1]    # above threshold, both kept
    assert nms_numpy(boxes, scores, iou - 0.01) == [0]       # below, the weaker goes


# --------------------------------------------------------------------------
# COCO category mapping
# --------------------------------------------------------------------------
def test_coco80_to_91_is_the_real_mapping():
    # Hand COCOeval a contiguous 0-79 index and every class maps to the wrong one, for
    # an mAP near zero and not one line of error — pycocotools never checks the id.
    assert len(COCO80_TO_91) == 80
    assert len(set(COCO80_TO_91)) == 80
    assert COCO80_TO_91 == sorted(COCO80_TO_91)
    assert min(COCO80_TO_91) == 1 and max(COCO80_TO_91) == 90
    # The ten ids val2017 leaves unused.
    assert set(range(1, 91)) - set(COCO80_TO_91) == {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}


def test_val_and_deploy_thresholds_are_not_interchangeable():
    # Swapping them is the documented hazard: COCO AP at 0.25 loses several points
    # because the PR curve never reaches the end, and latency at 0.001 pays for NMS
    # over thousands of boxes no real system would carry.
    assert VAL_CONF < DEPLOY_CONF

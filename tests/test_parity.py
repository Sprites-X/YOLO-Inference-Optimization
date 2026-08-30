"""check_parity.py — the matching that decides whether two runtimes agree.

The gate's verdict is only as good as its pairing. Comparing detections by their
position in the list reads a score-order swap as two large box errors, which would
fail a correct export; pairing across classes reads a genuine relabelling as a small
numeric difference, which would pass a broken one.
"""
from __future__ import annotations

import numpy as np
import pytest

from check_parity import MATCH_IOU, class_candidates, iou_matrix, match

N_CLASSES = 80
N_ANCHORS = 200


def dets(boxes, scores, classes):
    return (np.array(boxes, np.float32),
            np.array(scores, np.float32),
            np.array(classes, np.int32))


def raw_with_boxes(boxes_xyxy, cls_ids, scores):
    """A (1, 84, N_ANCHORS) head output holding exactly these boxes, pre-NMS."""
    raw = np.zeros((1, 4 + N_CLASSES, N_ANCHORS), np.float32)
    for i, ((x1, y1, x2, y2), c, s) in enumerate(zip(boxes_xyxy, cls_ids, scores)):
        raw[0, 0, i] = (x1 + x2) / 2
        raw[0, 1, i] = (y1 + y2) / 2
        raw[0, 2, i] = x2 - x1
        raw[0, 3, i] = y2 - y1
        raw[0, 4 + c, i] = s
    return raw


# --------------------------------------------------------------------------
# iou_matrix
# --------------------------------------------------------------------------
def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 100, 100]], np.float32)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[500, 500, 600, 600]], np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_known_overlap():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 5, 10, 15]], np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-6)


def test_iou_empty_sides_keep_their_shape():
    a = np.zeros((0, 4), np.float32)
    b = np.array([[0, 0, 10, 10]], np.float32)
    assert iou_matrix(a, b).shape == (0, 1)
    assert iou_matrix(b, a).shape == (1, 0)


# --------------------------------------------------------------------------
# match
# --------------------------------------------------------------------------
def test_match_never_pairs_across_classes():
    # The same box under a different label is a real disagreement, not a small numeric
    # one, and has to reach the report as an unmatched detection on both sides.
    ref = dets([[0, 0, 100, 100]], [0.9], [0])
    cmp = dets([[0, 0, 100, 100]], [0.9], [1])
    pairs, un_r, un_c = match(ref, cmp)
    assert pairs == []
    assert un_r == [0] and un_c == [0]


def test_match_pairs_by_iou_not_by_list_position():
    """NMS returns detections in score order, and 0.501 against 0.502 can swap.

    Compared by index that swap reads as two boxes moving several hundred pixels; it is
    really two correct boxes in a different order.
    """
    a, b = [0, 0, 100, 100], [300, 300, 400, 400]
    ref = dets([a, b], [0.501, 0.502], [0, 0])
    cmp = dets([b, a], [0.502, 0.501], [0, 0])
    pairs, un_r, un_c = match(ref, cmp)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert un_r == [] and un_c == []


def test_match_leaves_poorly_overlapping_boxes_unmatched():
    # Below MATCH_IOU the two are not the same detection, so a delta between them would
    # be meaningless — they belong in the unmatched columns instead.
    ref = dets([[0, 0, 100, 100]], [0.9], [0])
    cmp = dets([[0, 0, 100, 30]], [0.9], [0])       # IoU 0.30
    pairs, un_r, un_c = match(ref, cmp)
    assert pairs == []
    assert un_r == [0] and un_c == [0]


def test_match_is_greedy_best_first():
    # One reference box against two candidates that both clear MATCH_IOU. The better
    # overlap takes the pair and the other is reported, rather than quietly absorbed.
    ref = dets([[0, 0, 100, 100]], [0.9], [0])
    cmp = dets([[0, 0, 60, 100],        # IoU 0.60
                [2, 2, 102, 102]],      # IoU 0.92
               [0.8, 0.85], [0, 0])
    pairs, un_r, un_c = match(ref, cmp)
    assert pairs == [(0, 1)]
    assert un_r == [] and un_c == [0]


def test_match_handles_empty_sides():
    empty = dets(np.zeros((0, 4)), [], [])
    one = dets([[0, 0, 10, 10]], [0.9], [0])
    assert match(empty, one) == ([], [], [0])
    assert match(one, empty) == ([], [0], [])


def test_match_iou_floor_is_below_the_pair_gate():
    # A pair has to be matchable before MIN_PAIR_IOU can judge it. Raise MATCH_IOU above
    # that gate and failing pairs would silently become unmatched detections instead.
    from check_parity import MIN_PAIR_IOU
    assert MATCH_IOU < MIN_PAIR_IOU


# --------------------------------------------------------------------------
# class_candidates
# --------------------------------------------------------------------------
def test_class_candidates_returns_one_class_above_conf_in_image_coordinates():
    # What rescues a pair that failed MIN_PAIR_IOU: if both runtimes produced both
    # boxes before NMS, only the tie-break differed and the export is fine.
    meta = (0.5, (10, 20))
    raw = raw_with_boxes(
        [[30, 40, 130, 140], [200, 200, 300, 300], [400, 400, 500, 500]],
        [0, 0, 1],
        [0.9, 0.1, 0.9])
    got = class_candidates(raw, cls_id=0, conf=0.25, meta=meta)
    assert got.shape == (1, 4)
    # (30,40,130,140) less the (10,20) pad, over the 0.5 scale.
    assert got[0] == pytest.approx([40.0, 40.0, 240.0, 240.0], abs=0.01)


def test_class_candidates_empty_when_nothing_qualifies():
    raw = raw_with_boxes([[30, 40, 130, 140]], [5], [0.9])
    assert class_candidates(raw, cls_id=0, conf=0.25, meta=(1.0, (0, 0))).shape == (0, 4)

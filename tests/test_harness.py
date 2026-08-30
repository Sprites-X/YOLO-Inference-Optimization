"""Statistics, filename parsing and the seeded set selection.

Small pure functions, each guarding a failure that reports nothing when it happens:
a percentile that reports a latency no iteration ever had, an image_id that matches no
ground truth, a val500 that is a different 500 images than the README's.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import fetch_train_pool
import prepare_data
from benchmark import pct, summarize
from evaluate import image_id_from_name


# --------------------------------------------------------------------------
# percentiles
# --------------------------------------------------------------------------
def test_pct_returns_a_sample_that_actually_happened():
    # Nearest-rank, not np.percentile. Interpolating hands back the average of two
    # iterations, and the point of p99 is to name one genuinely slow iteration.
    rng = random.Random(0)
    vals = [rng.uniform(1.0, 5.0) for _ in range(97)]
    for q in (0, 25, 50, 90, 99, 100):
        assert pct(vals, q) in vals


def test_pct_does_not_interpolate():
    # np.percentile([1, 2], 50) is 1.5, which is neither of the two measurements.
    assert pct([1.0, 2.0], 50) in (1.0, 2.0)


def test_pct_endpoints():
    vals = [float(i) for i in range(1, 101)]
    assert pct(vals, 0) == 1.0
    assert pct(vals, 100) == 100.0
    assert pct(vals, 99) == 99.0


def test_pct_single_sample():
    assert pct([7.5], 99) == 7.5


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def test_summarize_reports_per_image_not_per_batch():
    # Without dividing by batch the b1 and b8 rows carry different units and the table
    # compares one frame against eight.
    s = summarize([8.0] * 10, batch=8)
    assert s["mean_ms"] == pytest.approx(1.0)
    assert s["p50_ms"] == pytest.approx(1.0)
    assert s["fps"] == pytest.approx(1000.0)


def test_summarize_fps_is_the_reciprocal_of_the_mean():
    s = summarize([2.0, 4.0, 6.0], batch=1)
    assert s["mean_ms"] == pytest.approx(4.0)
    assert s["fps"] == pytest.approx(250.0)
    assert s["min_ms"] == 2.0 and s["max_ms"] == 6.0


# --------------------------------------------------------------------------
# COCO filenames
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,want", [
    ("000000000139.jpg", 139),
    ("000000581781.jpg", 581781),
    ("000000000000.jpg", 0),       # the lstrip("0") or "0" branch
])
def test_image_id_from_name(name, want):
    # Has to match instances_val2017.json exactly. Miss and COCOeval pairs nothing at
    # all, returning mAP 0.000 without raising.
    assert image_id_from_name(Path(name)) == want


def test_image_id_rejects_a_non_coco_filename():
    with pytest.raises(SystemExit):
        image_id_from_name(Path("frame_a.jpg"))


# --------------------------------------------------------------------------
# set selection
# --------------------------------------------------------------------------
def test_pick_is_deterministic_for_a_seed():
    files = [Path(f"{i:012d}.jpg") for i in range(1000)]
    assert prepare_data.pick(files, 1337, 500) == prepare_data.pick(files, 1337, 500)


def test_pick_ignores_the_order_it_was_handed():
    """The sorted() before the shuffle is what makes the seed mean anything.

    iterdir() returns filesystem order, which differs per machine, so shuffling
    straight from it would give every machine a different val500 under the same seed.
    """
    files = [Path(f"{i:012d}.jpg") for i in range(1000)]
    shuffled = files[:]
    random.Random(99).shuffle(shuffled)
    assert prepare_data.pick(files, 1337, 500) == prepare_data.pick(shuffled, 1337, 500)


def test_pick_changes_with_the_seed():
    files = [Path(f"{i:012d}.jpg") for i in range(1000)]
    assert prepare_data.pick(files, 1337, 500) != prepare_data.pick(files, 1338, 500)


def test_manifest_hash_ignores_order_and_notices_content():
    names = [f"{i:012d}.jpg" for i in range(50)]
    shuffled = names[:]
    random.Random(3).shuffle(shuffled)
    assert prepare_data.manifest_hash(names) == prepare_data.manifest_hash(shuffled)
    assert prepare_data.manifest_hash(names) != prepare_data.manifest_hash(names[:-1])


def test_committed_manifest_still_matches_its_recorded_hash():
    # The pool a clone calibrates INT8 on. Edited or truncated, the INT8 rows would be
    # calibrated on a different image set than the README's while claiming its numbers.
    names = Path(fetch_train_pool.MANIFEST).read_text().split()
    assert len(names) == 2000
    assert fetch_train_pool.manifest_hash(names) == fetch_train_pool.EXPECTED_MANIFEST


def test_the_two_image_sets_cannot_overlap():
    # Calibrating on the images being scored tunes the dynamic ranges to the test set
    # and reports an INT8 mAP that is too good, with nothing to flag it.
    pool = set(Path(fetch_train_pool.MANIFEST).read_text().split())
    split = Path("data/split.json")
    if not split.exists():
        pytest.skip("data/split.json not built yet — run prepare_data.py")
    import json
    val500 = set(json.loads(split.read_text())["val500"]["files"])
    assert pool & val500 == set()


# --------------------------------------------------------------------------
# calibration cache fingerprint
# --------------------------------------------------------------------------
def test_calib_fingerprint_changes_with_every_input():
    """A generic cache name once let a second model reuse the first model's ranges.

    TensorRT printed "skipping calibration" and built an engine whose mAP was broken,
    with no error anywhere. The fingerprint is what makes that collision impossible.
    """
    pytest.importorskip("tensorrt")
    from build_engine import _calib_fingerprint

    base = ("a.onnx", "data/train_pool", 640, 1000, 8)
    fp = _calib_fingerprint(*base)
    assert fp == _calib_fingerprint(*base)
    for i, changed in enumerate([("b.onnx",), ("data/other",), (320,), (500,), (4,)]):
        args = list(base)
        args[i] = changed[0]
        assert _calib_fingerprint(*args) != fp

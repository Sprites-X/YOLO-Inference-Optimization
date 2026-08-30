"""make_report.py — the join between results.jsonl and accuracy.jsonl.

Both files are append-only, and every mistake this module can make is a wrong number
printed beside a plausible one: an mAP from a different image set, a speedup measured
against the wrong baseline, or the same config listed twice with no way to tell the
runs apart.
"""
from __future__ import annotations

import json

import pytest

from make_report import key_of, label_of, load_jsonl, newest_per_config, row_key


def result(runtime="TensorRT", precision="FP16", device="GPU", batch=1, tag="",
           ts="2026-08-30T06:00:00+00:00", fps=100.0):
    return {"timestamp": ts, "runtime": runtime, "precision": precision,
            "device": device, "batch": batch, "tag": tag, "fps": fps}


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------
def test_accuracy_key_ignores_batch():
    # mAP does not depend on batch size, so the b1 and b8 rows share one accuracy row
    # rather than leaving the b8 row's mAP column empty.
    assert key_of(result(batch=1)) == key_of(result(batch=8))


def test_row_key_separates_the_batch_sweep_from_the_static_engine():
    # These two collide on every column the accuracy join looks at while being
    # different engines with different numbers — the sweep's dynamic-shape build is
    # 12% slower at batch 1 than the static one.
    static = result(batch=1, tag="")
    sweep = result(batch=1, tag="batch-sweep")
    assert key_of(static) == key_of(sweep)          # same accuracy row, correctly
    assert row_key(static) != row_key(sweep)        # separate table rows, correctly


def test_row_key_separates_batches():
    assert row_key(result(batch=1)) != row_key(result(batch=8))


def test_label_of_shows_what_the_table_would_otherwise_hide():
    assert "batch-sweep" in label_of(result(tag="batch-sweep"))
    assert "b8" in label_of(result(batch=8))
    assert "b1" not in label_of(result(batch=1))    # the common case stays unlabelled


# --------------------------------------------------------------------------
# dedupe
# --------------------------------------------------------------------------
def test_no_duplicates_passes_through_untouched():
    rows = [result(precision="FP16"), result(precision="INT8"), result(device="CPU")]
    kept, collapsed = newest_per_config(rows)
    assert kept == rows
    assert collapsed == []


def test_a_second_sweep_collapses_to_the_newer_run():
    """The state a run without --fresh leaves behind: every config measured twice.

    Printing both is not a comparison, it is the same table twice with nothing saying
    which run a row came from.
    """
    old = result(ts="2026-08-30T06:51:50+00:00", fps=699.4)
    new = result(ts="2026-08-30T09:42:54+00:00", fps=698.4)
    kept, collapsed = newest_per_config([old, new])
    assert kept == [new]
    assert collapsed == [(row_key(old), 2)]


def test_newest_wins_even_when_the_file_is_out_of_order():
    # Two runs' files concatenated the wrong way round still has to resolve to the
    # measurement actually taken last.
    new = result(ts="2026-08-30T09:00:00+00:00", fps=1.0)
    old = result(ts="2026-08-30T06:00:00+00:00", fps=2.0)
    kept, _ = newest_per_config([new, old])
    assert kept == [new]


def test_rows_without_a_timestamp_fall_back_to_file_order():
    first = {"runtime": "PyTorch", "precision": "FP32", "device": "GPU", "fps": 1.0}
    second = {"runtime": "PyTorch", "precision": "FP32", "device": "GPU", "fps": 2.0}
    kept, collapsed = newest_per_config([first, second])
    assert kept == [second]
    assert collapsed == [(row_key(first), 2)]


def test_the_batch_sweep_survives_deduping():
    # The regression worth guarding: a dedupe keyed only on runtime/precision/device
    # would keep one of these four and drop the rest, silently deleting the whole sweep.
    rows = [result(batch=1, tag=""), result(batch=1, tag="batch-sweep"),
            result(batch=4, tag="batch-sweep"), result(batch=8, tag="batch-sweep")]
    kept, collapsed = newest_per_config(rows)
    assert kept == rows
    assert collapsed == []


def test_dedupe_keeps_one_row_per_config_across_a_full_doubled_sweep():
    configs = [("PyTorch", "FP32", "CPU", 1, ""), ("PyTorch", "FP32", "GPU", 1, ""),
               ("ONNX Runtime", "FP32", "CPU", 1, ""), ("ONNX Runtime", "FP32", "GPU", 1, ""),
               ("TensorRT", "FP16", "GPU", 1, ""), ("TensorRT", "INT8", "GPU", 1, ""),
               ("TensorRT", "INT8+FP16head", "GPU", 1, ""),
               ("TensorRT", "FP16", "GPU", 1, "batch-sweep"),
               ("TensorRT", "FP16", "GPU", 4, "batch-sweep"),
               ("TensorRT", "FP16", "GPU", 8, "batch-sweep")]
    first = [result(*c, ts="2026-08-30T06:00:00+00:00", fps=1.0) for c in configs]
    second = [result(*c, ts="2026-08-30T09:00:00+00:00", fps=2.0) for c in configs]
    kept, collapsed = newest_per_config(first + second)
    assert len(kept) == 10
    assert len(collapsed) == 10
    assert all(r["fps"] == 2.0 for r in kept)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def test_load_jsonl_missing_file_is_not_an_error(tmp_path):
    # accuracy.jsonl often does not exist yet straight after a benchmark run, and the
    # timing table is still worth printing with the mAP column blank.
    assert load_jsonl(str(tmp_path / "nope.jsonl")) == []


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    assert load_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_outdir_is_created_if_missing(tmp_path):
    # --outdir is how a run writes its table outside the repo. Without the mkdir the
    # failure lands after the table is built and printed, so the work looks done right
    # up until nothing is on disk.
    import subprocess
    import sys

    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({
        "timestamp": "2026-08-30T06:00:00+00:00", "runtime": "TensorRT",
        "precision": "FP16", "device": "GPU", "batch": 1, "tag": "",
        "latency_ms_per_image": {"mean": 1.43, "std_across_repeats": 0.0,
                                 "p50": 1.43, "p99": 1.49},
        "fps": 699.4, "end_to_end_ms": 3.62, "model_size_mb": 7.9, "vram_mb": 189,
    }) + "\n")
    out = tmp_path / "does" / "not" / "exist"
    r = subprocess.run(
        [sys.executable, "make_report.py", "--results", str(results),
         "--accuracy", str(tmp_path / "none.jsonl"), "--outdir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "report_table.md").exists()

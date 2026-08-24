# YOLO Inference Optimization Benchmark

Benchmarking YOLOv8n across PyTorch / ONNX Runtime /
TensorRT FP16 / TensorRT INT8 on RTX 5060 (Blackwell, sm_120).

**Status:** Phase 1 complete — PyTorch baseline measured on
500 COCO val2017 images. Export / TensorRT / INT8 still to come.

## Baseline (YOLOv8n, 640x640, batch 1, 500 images)

| Runtime | Device | Precision | p50 (ms) | p99 (ms) | FPS | mAP50-95 |
|---|---|---|---|---|---|---|
| PyTorch | RTX 5060 | FP32 | 3.76 | 4.54 | 261.9 | 0.4008 |
| PyTorch | Ryzen 5 7500F | FP32 | 21.56 | 26.76 | 45.4 | — |

Inference-only, CUDA-synchronised, mean of 3 runs of 300 iterations
(CPU: 100) after 50 warm-up iterations. GPU is 5.8x the CPU baseline.
mAP is measured at conf 0.001 / IoU 0.7 as COCO AP requires; the latency
rows use deploy thresholds (0.25 / 0.45) — see `common.py` for why these
differ on purpose.

## Verified environment
Driver 595.84 / PyTorch 2.11.0+cu128 / ONNX Runtime 1.23.2 /
TensorRT 10.16.1.11 / Python 3.10.12.
Full detail in `env_report.json`, exact packages in `requirements.lock.txt`.

## What's here
- `verify_env.py` — checks the things that fail silently:
  ONNX Runtime falling back to CPU, PyTorch without sm_120 kernels
- `benchmark.py` — warm-up, CUDA sync, stage-separated timing, p50/p99 over 3 runs
- `build_engine.py` — TensorRT 10 builder with a real INT8 calibrator
- `prepare_data.py` — deterministic 500-image measurement set from val2017
- `fetch_train_pool.py` — pulls the train2017 calibration pool for Phase 3

## Install

`requirements.lock.txt` pins versions but not the non-PyPI indexes.
Install torch and TensorRT first:

    pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    pip install --extra-index-url https://pypi.nvidia.com \
        tensorrt-cu12==10.16.1.11 tensorrt-cu12-libs==10.16.1.11 \
        tensorrt-cu12-bindings==10.16.1.11
    pip install -r requirements.lock.txt

Then run `python verify_env.py` — it must report zero failures.

## Data

Measurement set — 500 images from val2017:

    curl -O http://images.cocodataset.org/zips/val2017.zip
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip -d data/
    unzip -j annotations_trainval2017.zip annotations/instances_val2017.json -d annotations/
    python prepare_data.py --src data/val2017

Calibration pool for Phase 3 — 2000 images from train2017 (~340 MB):

    python fetch_train_pool.py

Calibration images come from train2017 and the measurement set from val2017, so
they are different COCO splits and cannot overlap. That separation is the point:
calibrating INT8 on the images used to score mAP tunes the dynamic ranges to the
test set and reports a number that is too good, with nothing to flag it. Both
scripts check for overlap anyway.

`fetch_train_pool.py` downloads the images named in `train_pool_manifest.txt`
one at a time rather than pulling `train2017.zip`, which is ~19 GB against the
13 GB free on this machine. The guide does not require the full split — Phase 3
only does `shuf -n 500` over the pool. The manifest is committed (34 KB) so a
clone gets the same pool without re-deriving it from the 241 MB annotations
archive; it is the first 2000 of all 118,287 train2017 filenames after a seeded
shuffle. 2000 rather than 500 leaves room for the guide's suggestion to retry
with 1000 images if INT8 mAP drops more than 3-4%.

Both selections are seeded (1337) and self-verifying — each script recomputes
its manifest hash and exits non-zero on a mismatch, so pointing `--src` at the
wrong directory fails immediately instead of silently benchmarking other images:

| set | source split | manifest sha256 |
|---|---|---|
| `data/val500` | val2017 | `faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7` |
| `data/train_pool` | train2017 | `6c787a8293f8ea2223b451a45e3c10d137716496f5b782c59b38a5428049b7e6` |

val500 images are hardlinked from `data/val2017`, so they cost no extra disk.

One caveat on the mAP column: it is measured over 500 images, so it sits a
little above the 0.373 Ultralytics reports over all 5000. The number is for
comparing runtimes on identical images, not for restating the published figure.

Results and analysis to follow.

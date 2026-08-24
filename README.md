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
- `prepare_data.py` — deterministic val500 / calibration-pool split (seeded,
  disjoint by construction so Phase 3 calibration cannot leak into the mAP set)

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

    curl -O http://images.cocodataset.org/zips/val2017.zip
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip -d data/
    unzip -j annotations_trainval2017.zip annotations/instances_val2017.json -d annotations/
    python prepare_data.py --src data/val2017

Splits val2017's 5000 images into `data/val500` (measurement) and
`data/calib_pool` (4500, held back for Phase 3 INT8 calibration). The two are
disjoint by construction and the script refuses to write if they ever overlap:
calibrating on the images used to score mAP would inflate the INT8 result with
nothing to flag it.

The split is a seeded shuffle (seed 1337), so a clone gets the same 500 images.
`prepare_data.py` checks this itself and exits non-zero on a mismatch:

| set | manifest sha256 |
|---|---|
| `val500` | `faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7` |
| `calib_pool` | `aaca64bcf21426cf8c4bc92b614a226042812b541d1880d011443334e366dff0` |

Two caveats worth stating plainly. The pool is val2017's remainder, not COCO
train2017 as the guide assumes — calibration needs the same distribution with no
overlap with the val set, which holds, but it is not the train split. And
mAP here is over 500 images, so it sits a little above the 0.373 Ultralytics
reports over all 5000; the point of the number is comparing runtimes on
identical images, not restating the published figure.

Images are hardlinked, so the split costs no extra disk.

Results and analysis to follow.

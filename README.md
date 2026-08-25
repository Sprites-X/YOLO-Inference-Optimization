# YOLO Inference Optimization Benchmark

Benchmarking YOLOv8n across PyTorch / ONNX Runtime /
TensorRT FP16 / TensorRT INT8 on RTX 5060 (Blackwell, sm_120).

**Status:** PyTorch baseline measured on 500 COCO val2017 images. ONNX (static and
dynamic) and TensorRT FP16 exported and checked against that baseline. INT8 and the
latency comparison still to come.

## Baseline (YOLOv8n, 640x640, batch 1)

| Runtime | Device | Precision | Images | p50 (ms) | p99 (ms) | FPS | mAP50-95 |
|---|---|---|---|---|---|---|---|
| PyTorch | RTX 5060 | FP32 | 500 | 3.76 | 4.54 | 261.9 | 0.4008 |
| PyTorch | Ryzen 5 7500F | FP32 | 500 | 21.56 | 26.76 | 45.4 | — |
| PyTorch | RTX 5060 | FP32 | 5000 (full val2017) | — | — | — | 0.3651 |

Latency is inference-only, CUDA-synchronised, mean of 3 runs of 300 iterations
(CPU: 100) after 50 warm-up iterations, over the 500-image set. GPU is 5.8x the
CPU baseline. The 5000-image row is accuracy only — a full-val2017 reference
point, not a separate latency measurement.

mAP is measured at conf 0.001 / IoU 0.7 as COCO AP requires; the latency rows use
deploy thresholds (0.25 / 0.45) — see `common.py` for why these differ on
purpose.

## Export parity

Every export is checked against the PyTorch model it came from before any speed is
measured, because an export that quietly changed the model still produces perfectly
plausible latency numbers. `check_parity.py` runs the same images through each runtime
and compares detections, and `run_all.sh` stops the run if they disagree.

| Runtime | Precision | mAP50-95 | mAP50 | vs baseline |
|---|---|---|---|---|
| PyTorch (baseline) | FP32 | 0.4008 | 0.5476 | — |
| ONNX Runtime GPU | FP32 | 0.4008 | 0.5475 | 0.0000 |
| ONNX Runtime CPU | FP32 | 0.4008 | 0.5476 | 0.0000 |
| TensorRT | FP16 | 0.4003 | 0.5480 | -0.0005 |

Same 500 images, same `common.py` pre/postprocess, conf 0.001 / IoU 0.7 as COCO AP
requires.

**The gate is not a pixel tolerance.** Two things make FP16 detections differ from FP32
without anything being wrong with the export:

- *Box error scales with box size.* yolov8 decodes a box by summing 16 DFL bins scaled
  by the stride of the level it came from, so the ~3 decimal digits FP16 carries cost
  more on a large box than a small one — 5.4 px on a 608 px box against 1.1 px on a
  232 px box, the same small relative error. A flat pixel budget either fails the first
  or waves the second through.
- *NMS is a discrete choice.* On `000000097585.jpg` the two best candidates for one
  object score 0.75133 and 0.74966 in FP32. FP16 rounds both to exactly 0.75049, the
  tie-break then keeps the other one, and the surviving box moves 6.6 px. The same
  object is found either way.

So the check asks whether both runtimes found the same objects, not whether they agree
bit for bit: every detection must pair with one of the same class at IoU >= 0.90, mean
box drift must stay under 0.5% of box size — that is what catches a systematic shift,
which per-box IoU on its own would miss — and scores must agree within 0.02.
Detections within 0.05 of the confidence threshold may appear on one side only: at conf
0.25, a box scoring 0.2510 on one runtime and 0.2498 on the other is the threshold
being stepped over, not a detection being lost. Two of those turn up across 100 images,
and the check prints them rather than hiding them.

## Verified environment
Driver 595.84 / PyTorch 2.11.0+cu128 / ONNX Runtime 1.23.2 /
TensorRT 10.16.1.11 / Python 3.10.12.
Full detail in `env_report.json`, exact packages in `requirements.lock.txt`.

ONNX Runtime's CUDA provider links libraries that pip installs under
`site-packages/nvidia/*/lib`, where the dynamic loader does not look, so `benchmark.py`
dlopens them before opening a session. Without that, ORT found CUDA only when torch had
been imported first — which `verify_env.py` did, so it reported PASS while
`evaluate.py --runtime onnx --device cuda` could not get a GPU session at all. That
check now runs in a subprocess with no torch in it.

## What's here

Run `./run_all.sh` to do the whole pipeline end to end. The individual steps, in
the order it runs them:

| | |
|---|---|
| `verify_env.py` | Gate before measuring anything. Checks what fails silently: ONNX Runtime falling back to CPU, PyTorch without sm_120 kernels. Writes `env_report.json`. |
| `prepare_data.py` | Deterministic 500-image measurement set from val2017. |
| `fetch_train_pool.py` | Pulls the train2017 pool used for INT8 calibration. |
| `build_engine.py` | TensorRT 10 builder, with a real INT8 calibrator and a fingerprinted calibration cache. |
| `check_parity.py` | The gate between export and measurement. Same images through every runtime, detections compared against PyTorch. |
| `benchmark.py` | Latency: warm-up, CUDA sync, stage-separated timing, p50/p99 over 3 runs. Appends to `results.jsonl`. |
| `evaluate.py` | Accuracy: COCO mAP via pycocotools. Appends to `accuracy.jsonl`. |
| `make_report.py` | Joins those two files into `report_table.md` and the figures. |
| `common.py` | Shared preprocess and postprocess. Every runtime calls the same one — that is what makes the comparison a comparison. |

`train_pool_manifest.txt` and `env_report.json` are committed so a clone can
reproduce the same image pool and see what the numbers were measured on.

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

INT8 calibration pool — 2000 images from train2017 (~315 MB):

    python fetch_train_pool.py

Calibration images come from train2017 and the measurement set from val2017, so
they are different COCO splits and cannot overlap. That separation is the point:
calibrating INT8 on the images used to score mAP tunes the dynamic ranges to the
test set and reports a number that is too good, with nothing to flag it. Both
scripts check for overlap anyway.

`fetch_train_pool.py` downloads the images named in `train_pool_manifest.txt`
one at a time rather than pulling `train2017.zip`, which is ~19 GB against the
13 GB free on this machine. Calibration needs a few hundred images rather than a
whole split, so a pool this size is plenty. The manifest is committed (34 KB) so a
clone gets the same pool without re-deriving it from the 241 MB annotations
archive; it is the first 2000 of all 118,287 train2017 filenames after a seeded
shuffle. 2000 rather than 500 leaves room to retry with 1000 images if INT8 turns
out to cost too much mAP.

Both selections are seeded (1337) and self-verifying — each script recomputes
its manifest hash and exits non-zero on a mismatch, so pointing `--src` at the
wrong directory fails immediately instead of silently benchmarking other images:

| set | source split | manifest sha256 |
|---|---|---|
| `data/val500` | val2017 | `faabd1586d3313cc6cdac1db9b7a570c4dd0ef980e8fde83cdd31ac8a846e9f7` |
| `data/train_pool` | train2017 | `6c787a8293f8ea2223b451a45e3c10d137716496f5b782c59b38a5428049b7e6` |

val500 images are hardlinked from `data/val2017`, so they cost no extra disk.

## What the mAP numbers do and do not mean

**Against the published figure.** Running `evaluate.py` over all 5000 val2017
images gives mAP50-95 **0.3651**, against the **0.373** Ultralytics reports for
these same weights. Running `yolo val model=yolov8n.pt data=coco.yaml` on this
machine reproduces the published side at 0.374 (pycocotools) / 0.368 (Ultralytics'
own metric), so the roughly 0.9-point gap is real and belongs to the postprocess
here, not to the weights or the environment.

**Where the gap comes from.** One difference in postprocess accounts for it:
`yolo val` runs NMS with `multi_label=True`, so a single box can emit one
detection per class scoring above the threshold. `common.postprocess` takes
`argmax` over the class scores instead — one class per box. At conf 0.001 that
changes how far the low-scoring tail carries the PR curve, which is exactly where
COCO AP is won and lost. Everything else lines up: both letterbox to a square
640x640 with `scaleup=False`, and both use conf 0.001 / IoU 0.7 / max_det 300.
Rectangular inference is *not* a factor — it is off by default in `yolo val`.

This is a deliberate trade. A single simple postprocess shared by all three
runtimes is worth more here than matching a published number, because the
comparison being made is between runtimes, and any NMS difference between them
would show up as an accuracy difference that has nothing to do with the runtime.
Adopting multi-label NMS would close most of the gap while making the shared
postprocess more complex, and would not improve a single number this project
actually reports.

**On the 500-image subset.** The subset scores 0.4008 against 0.3651 on the full
set — a 0.036 spread, which is large. mAP over 500 images is noisy, so the subset
figure is only ever used to compare runtimes against each other on identical
images. It is not comparable to any published number, including the one above.

Results and analysis to follow.

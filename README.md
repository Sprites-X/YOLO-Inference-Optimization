# YOLO Inference Optimization Benchmark

Benchmarking YOLOv8n across PyTorch / ONNX Runtime /
TensorRT FP16 / TensorRT INT8 on RTX 5060 (Blackwell, sm_120).

**Status:** Phase 0 complete — environment verified,
harness written. Measurements not yet run.

## Verified environment
Driver 595.84 / PyTorch 2.11.0+cu128 / ONNX Runtime 1.23.2 /
TensorRT 10.16.1.11 / Python 3.10.12.
Full detail in `env_report.json`, exact packages in `requirements.lock.txt`.

## What's here
- `verify_env.py` — checks the things that fail silently:
  ONNX Runtime falling back to CPU, PyTorch without sm_120 kernels
- `benchmark.py` — warm-up, CUDA sync, stage-separated timing, p50/p99 over 3 runs
- `build_engine.py` — TensorRT 10 builder with a real INT8 calibrator

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


Results and analysis to follow.

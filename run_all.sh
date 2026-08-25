#!/usr/bin/env bash
# Needs bash, not sh: `set -o pipefail` is not in POSIX sh.
set -euo pipefail

IMAGES=data/val500
# Point straight at the pool rather than copying a sample into data/calib first.
# ImageCalibrator already does random.Random(0).shuffle(files)[:--calib-num], so
# --calib-num 500 draws a deterministic 500 out of the 2000 without duplicating
# them on disk. data/train_pool is train2017 and $IMAGES is val2017, so the calib
# set still cannot overlap the images being scored.
CALIB=data/train_pool
ANN=annotations/instances_val2017.json
MODEL=yolov8n.pt

echo "=== environment ==="
python verify_env.py || { echo "fix the failures before going further"; exit 1; }

echo "=== export ==="
# Dynamic first, then static. `yolo export` always writes yolov8n.onnx, so doing it the
# other way round leaves the dynamic graph sitting in the static file's name and the
# static export gone — every later step then silently uses the wrong one.
yolo export model=$MODEL format=onnx opset=17 dynamic=True simplify=True
mv yolov8n.onnx yolov8n_dyn.onnx
yolo export model=$MODEL format=onnx opset=17 dynamic=False simplify=True
python build_engine.py --onnx yolov8n.onnx --precision fp16

echo "=== INT8 calibration ==="
python build_engine.py --onnx yolov8n.onnx --precision int8 \
    --calib-dir $CALIB --calib-num 500

# Gate: do the exports still detect what the PyTorch model detects? A wrong export
# produces perfectly reasonable-looking latency numbers, so this has to run before any
# measurement, not after. INT8 is left out on purpose — quantization is meant to move
# boxes, so it is judged on mAP below rather than against FP32 geometry.
echo "=== export parity ==="
python check_parity.py --images $IMAGES --n 100 \
    --cmp onnx:yolov8n.onnx:cuda \
    --cmp onnx:yolov8n_dyn.onnx:cuda \
    --cmp onnx:yolov8n.onnx:cpu \
    --cmp tensorrt:yolov8n_fp16.engine \
    || { echo "exports disagree with the baseline — fix that before measuring"; exit 1; }

echo "=== benchmark ==="
python benchmark.py --images $IMAGES --runtime pytorch  --model $MODEL         --device cpu --iters 100
python benchmark.py --images $IMAGES --runtime pytorch  --model $MODEL         --device cuda
python benchmark.py --images $IMAGES --runtime onnx     --model yolov8n.onnx   --device cpu --iters 100
python benchmark.py --images $IMAGES --runtime onnx     --model yolov8n.onnx   --device cuda
python benchmark.py --images $IMAGES --runtime tensorrt --model yolov8n_fp16.engine
python benchmark.py --images $IMAGES --runtime tensorrt --model yolov8n_int8.engine

echo "=== accuracy ==="
python evaluate.py --images $IMAGES --ann $ANN --runtime pytorch  --model $MODEL
python evaluate.py --images $IMAGES --ann $ANN --runtime onnx     --model yolov8n.onnx
python evaluate.py --images $IMAGES --ann $ANN --runtime tensorrt --model yolov8n_fp16.engine
python evaluate.py --images $IMAGES --ann $ANN --runtime tensorrt --model yolov8n_int8.engine --per-class

echo "=== report ==="
python make_report.py
echo "done: report_table.md, fig_latency.png, fig_tradeoff.png"

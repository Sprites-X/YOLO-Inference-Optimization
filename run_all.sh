set -euo pipefail

IMAGES=data/val500
CALIB=data/calib
ANN=annotations/instances_val2017.json
MODEL=yolov8n.pt

echo "=== environment ==="
python verify_env.py || { echo "fix the failures before going further"; exit 1; }

echo "=== export ==="
yolo export model=$MODEL format=onnx opset=17 dynamic=False simplify=True
python build_engine.py --onnx yolov8n.onnx --precision fp16

echo "=== INT8 calibration ==="
python build_engine.py --onnx yolov8n.onnx --precision int8 \
    --calib-dir $CALIB --calib-num 500

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

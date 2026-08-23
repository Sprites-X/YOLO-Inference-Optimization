from __future__ import annotations

import argparse
import ctypes
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

sys.path.insert(0, str(Path(__file__).parent))
from common import IMG_SIZE, preprocess  

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _check(err):
    if isinstance(err, tuple):
        err, *rest = err
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")
        return rest[0] if len(rest) == 1 else rest
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


# --------------------------------------------------------------------------
class ImageCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calib_dir: str, cache_path: str, num_images: int = 500,
                 batch_size: int = 8, size: int = IMG_SIZE, seed: int = 0):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.size = size

        files = sorted(p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"ไม่เจอรูปใน {calib_dir}")
        random.Random(seed).shuffle(files) 
        self.files = files[:num_images]
        self.index = 0

        nbytes = batch_size * 3 * size * size * 4
        self.d_input = _check(cudart.cudaMalloc(nbytes))
        self.nbytes = nbytes
        print(f"[calib] ใช้ {len(self.files)} รูป, batch {batch_size} "
              f"-> {(len(self.files) + batch_size - 1) // batch_size} รอบ")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.files):
            return None
        import cv2

        chunk = self.files[self.index:self.index + self.batch_size]
        arrs = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            arrs.append(preprocess(img, self.size)[0])
        if not arrs:
            self.index += self.batch_size
            return self.get_batch(names)

        while len(arrs) < self.batch_size:
            arrs.append(arrs[0])

        batch = np.ascontiguousarray(np.concatenate(arrs, axis=0), dtype=np.float32)
        _check(cudart.cudaMemcpy(
            self.d_input, batch.ctypes.data, batch.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ))

        self.index += self.batch_size
        done = min(self.index, len(self.files))
        print(f"\r[calib] {done}/{len(self.files)}", end="", flush=True)
        if done >= len(self.files):
            print()
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            print(f"[calib] เจอ cache เดิม {self.cache_path} — ข้าม calibration")
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        print(f"[calib] เขียน cache -> {self.cache_path}")

    def free(self):
        if getattr(self, "d_input", None):
            cudart.cudaFree(self.d_input)
            self.d_input = None


# --------------------------------------------------------------------------
def build(args):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[onnx] {parser.get_error(i)}", file=sys.stderr)
            raise RuntimeError("parse ONNX ไม่ผ่าน")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE, args.workspace << 30)

    inp = network.get_input(0)
    print(f"[net] input '{inp.name}' shape={inp.shape}")

    if inp.shape[0] == -1:
        profile = builder.create_optimization_profile()
        c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
        if h == -1 or w == -1:
            h = w = args.size
        profile.set_shape(inp.name,
                          (args.min_batch, c, h, w),
                          (args.opt_batch, c, h, w),
                          (args.max_batch, c, h, w))
        config.add_optimization_profile(profile)
        print(f"[net] dynamic batch min={args.min_batch} opt={args.opt_batch} "
              f"max={args.max_batch}")
    elif args.max_batch > 1:
        print("[net] ONNX เป็น static batch — --max-batch ไม่มีผล "
              "(export ใหม่ด้วย dynamic=True ถ้าจะทดสอบ batch)")

    calibrator = None
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[warn] platform ไม่มี fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    elif args.precision == "int8":
        if not builder.platform_has_fast_int8:
            print("[warn] platform ไม่มี fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        
        config.set_flag(trt.BuilderFlag.FP16)

        if not args.calib_dir:
            raise SystemExit(
                "INT8 ต้องมี --calib-dir\n"
                "ถ้าไม่ให้รูป TensorRT จะเดา dynamic range เอง แล้ว mAP จะพังโดยไม่มี error"
            )
        calibrator = ImageCalibrator(
            args.calib_dir, args.calib_cache, args.calib_num,
            args.calib_batch, args.size,
        )
        config.int8_calibrator = calibrator

        if args.fp16_head:
            _force_fp16_output_layers(network, config, n=args.fp16_head)

    out = args.engine or str(
        Path(args.onnx).with_suffix("").name + f"_{args.precision}.engine"
    )

    print(f"[build] เริ่ม build {args.precision} — ครั้งแรกใช้เวลาหลายนาที ปกติ ไม่ใช่บั๊ก")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0

    if calibrator:
        calibrator.free()
    if serialized is None:
        raise RuntimeError("build ไม่สำเร็จ")

    with open(out, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"[build] เสร็จใน {dt:.1f}s -> {out} ({size_mb:.2f} MB)")
    print("[note] engine ผูกกับ GPU รุ่นนี้ + TensorRT เวอร์ชันนี้ ย้ายเครื่องต้อง build ใหม่")


def _force_fp16_output_layers(network, config, n: int = 10):

    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    total = network.num_layers
    count = 0
    for i in range(max(0, total - n), total):
        layer = network.get_layer(i)
        if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT,
                          trt.LayerType.CONCATENATION, trt.LayerType.GATHER):
            continue
        layer.precision = trt.float16
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float16)
        count += 1
    print(f"[mixed] บังคับ {count} layer ท้าย ({total} ทั้งหมด) เป็น FP16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--workspace", type=int, default=4, help="GB")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--calib-cache", default="calibration.cache")
    ap.add_argument("--calib-num", type=int, default=500)
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--fp16-head", type=int, default=0,
                    help="จำนวน layer ท้ายที่บังคับเป็น FP16 (ลองใช้เมื่อ INT8 mAP ตกมาก)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

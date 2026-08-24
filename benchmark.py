from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common import (DEPLOY_CONF, DEPLOY_IOU, IMG_SIZE, postprocess,
                    preprocess_batch)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ==========================================================================
# GPU state
# ==========================================================================
def gpu_state() -> dict:
    # เก็บอุณหภูมิ/คล็อกไว้คู่กับทุกรอบ เพื่อให้ตอบได้ว่ารอบที่ช้ากว่าเพื่อนเป็นเพราะ
    # throttle หรือเป็นความแกว่งของการวัดเอง — ถ้าไม่เก็บไว้ ย้อนกลับไปดูไม่ได้
    #
    # NOTE: split(",") ทั้งก้อน stdout เลยรองรับ GPU ใบเดียว ถ้ามีสองใบ nvidia-smi
    # คืนสองบรรทัด แล้ว float() พังที่ out[3] ('512\n38') → โดน except กลืน → คืน {}
    # แล้ว telemetry หายทั้งหมดแบบเงียบๆ ตอนนี้เครื่องมีใบเดียวเลยยังไม่โผล่
    # TODO: ถ้าย้ายไปเครื่องหลาย GPU ต้อง splitlines() ก่อนแล้วเลือกด้วย CUDA_VISIBLE_DEVICES
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().split(",")
        return {"temp_c": float(out[0]), "sm_clock_mhz": float(out[1]),
                "power_w": float(out[2]), "mem_used_mb": float(out[3])}
    except Exception:
        return {}


# ==========================================================================
# Runners 
# ==========================================================================
# ทั้งสามคลาสต้องมี infer/sync/peak_vram_mb/model_size_mb เหมือนกัน เพื่อให้
# run_once จับเวลาด้วยโค้ดชุดเดียว — ความต่างของ runtime ต้องอยู่ในคลาส ไม่ใช่ในตัวจับเวลา
class PyTorchRunner:
    name = "PyTorch"

    def __init__(self, model_path: str, device: str, half: bool = False):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = device
        self.half = half and device == "cuda"
        self.precision = "FP16" if self.half else "FP32"

        yolo = YOLO(model_path)
        # ใช้ yolo.model ตรงๆ ไม่ใช่ yolo.predict() เพราะ predict ห่อ pre/postprocess
        # ของ ultralytics ไว้ด้วย ซึ่งจะทำให้เทียบกับ ONNX/TRT ไม่ได้ — ต้องให้ทั้งสาม
        # runtime ใช้ common.preprocess/postprocess ตัวเดียวกัน
        self.model = yolo.model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # reset ก่อนวัด เพราะ YOLO() ตอนโหลดจองหน่วยความจำชั่วคราวไว้ ถ้าไม่ reset
        # peak จะติดยอดตอนโหลดมาด้วย ไม่ใช่ยอดตอน inference
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        # non_blocking=False ตั้งใจ — H2D ต้องอยู่ในช่วงที่จับเวลา เท่ากับที่ TRT
        # และ ORT ต้องจ่าย ไม่งั้นแถว PyTorch จะได้เปรียบเพราะซ่อน copy ไว้นอกนาฬิกา
        t = self.torch.from_numpy(batch).to(self.device, non_blocking=False)
        if self.half:
            t = t.half()
        with self.torch.inference_mode():
            out = self.model(t)
        # Detect head ตอน eval คืน (y, x) — y คือผลที่ decode แล้ว (B, 84, 8400)
        # ส่วน x เป็น feature map ดิบของแต่ละ scale ที่ postprocess ไม่ได้ใช้
        out = out[0] if isinstance(out, (list, tuple)) else out
        # .float() ก่อนออก: ถ้าปล่อย half ไป postprocess จะไป de-letterbox ด้วย float16
        # แล้วกล่องคลาด ~1 px (ดู NOTE ใน common.postprocess)
        return out.float().cpu().numpy()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def peak_vram_mb(self):
        # ยอดนี้นับเฉพาะ tensor ที่ allocator ของ torch จอง ไม่รวม CUDA context
        # (~300-600MB) จึงเทียบกับตัวเลขของ TensorRT ที่มาจาก nvidia-smi ไม่ได้ตรงๆ
        # ดู NOTE ที่ TensorRTRunner.peak_vram_mb
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1024 / 1024
        return None

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024


class ONNXRunner:
    name = "ONNX Runtime"
    # ฮาร์ดโค้ดไว้เพราะตอนนี้ export เป็น FP32 อย่างเดียว
    # TODO: ถ้าเพิ่มแถว ONNX FP16 ต้องอ่าน dtype จาก graph จริง
    # ไม่งั้นตารางจะติดป้าย FP32 ให้ผลที่วัดจากโมเดล FP16
    precision = "FP32"

    def __init__(self, model_path: str, device: str):
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(model_path, opts, providers=providers)
        used = self.sess.get_providers()

        # ด่านนี้คือเหตุผลหลักที่ verify_env.py มีอยู่
        # ORT ไม่ถือว่าการตกไป CPU เป็น error — มันแค่เงียบแล้วรันช้าลง 20-50 เท่า
        # ถ้าปล่อยผ่าน แถว "ONNX Runtime GPU" จะเป็นตัวเลข CPU ที่ติดป้าย GPU
        # ซึ่งเป็นความผิดพลาดชนิดที่มองไม่ออกจากตาราง ต้อง raise ตรงนี้เท่านั้น
        if device == "cuda" and "CUDAExecutionProvider" not in used:
            raise RuntimeError(
                f"ขอ CUDA แต่ ORT ใช้ {used} — ถ้าปล่อยผ่าน แถวนี้ในตารางจะเป็นตัวเลข CPU "
                f"ที่ติดป้ายว่า GPU"
            )
        self.providers = used
        self.iname = self.sess.get_inputs()[0].name
        self.device = device

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.iname: batch})[0]

    def sync(self):
        pass  # sess.run() บล็อกจนเสร็จอยู่แล้ว มีเมธอดนี้ไว้ให้ run_once เรียกได้เหมือนกันทุก runtime

    def peak_vram_mb(self):
        return None

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024


class TensorRTRunner:
    name = "TensorRT"

    def __init__(self, engine_path: str, batch: int = 1, size: int = IMG_SIZE):
        import tensorrt as trt
        # cuda.cudart ถูก deprecate แล้ว ย้ายไป cuda.bindings.runtime
        # เก็บ fallback ไว้เผื่อเครื่องที่ยังเป็น cuda-python รุ่นเก่า (เครื่องนี้ 12.9.7)
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart

        self.trt, self.cudart = trt, cudart
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"โหลด engine ไม่ได้: {engine_path}\n"
                f"engine ผูกกับ GPU + TensorRT เวอร์ชันที่ build — ถ้าเปลี่ยนอย่างใดอย่างหนึ่งต้อง build ใหม่"
            )
        self.ctx = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        self._ck(err)

        # TRT 10 API: วน num_io_tensors แล้วอ้างด้วยชื่อ ไม่ใช่ bindings[] แบบ TRT 8
        # (verify_env.py เช็กว่าเป็น TRT >= 10 ก่อนถึงตรงนี้)
        self.inputs, self.outputs, self.ptrs = [], [], {}
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            is_in = self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
            shape = list(self.engine.get_tensor_shape(nm))
            if is_in:
                if shape[0] == -1:
                    shape[0] = batch
                    self.ctx.set_input_shape(nm, tuple(shape))
                elif batch != shape[0]:
                    raise RuntimeError(
                        f"engine เป็น static batch {shape[0]} แต่ขอ batch {batch} "
                        f"— build engine ใหม่ด้วย --min/opt/max-batch"
                    )
                self.inputs.append(nm)
            else:
                self.outputs.append(nm)

        self.precision = self._detect_precision(engine_path)
        self.batch = batch
        self.host_out = {}
        self.dev_mem = []

        # จองครั้งเดียวตอน init แล้วใช้ซ้ำทุก iteration — ถ้าจอง/คืนทุกรอบ cudaMalloc
        # จะกลายเป็นส่วนหนึ่งของ latency ที่วัดได้ ซึ่งไม่ใช่สิ่งที่ระบบจริงทำ
        # get_tensor_shape ต้องอ่านจาก ctx ไม่ใช่ engine เพราะ dynamic batch เพิ่งถูกตั้งข้างบน
        for nm in self.inputs + self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(nm))
            dtype = trt.nptype(self.engine.get_tensor_dtype(nm))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, ptr = cudart.cudaMalloc(nbytes)
            self._ck(err)
            self.ptrs[nm] = (ptr, nbytes, shape, dtype)
            self.dev_mem.append(ptr)
            self.ctx.set_tensor_address(nm, int(ptr))
            if nm in self.outputs:
                self.host_out[nm] = np.empty(shape, dtype=dtype)

        # TODO: เก็บไว้แล้วแต่ยังไม่ได้ใช้ — ตั้งใจจะเอาไปลบออกจาก peak_vram_mb
        # เพื่อแยกหน่วยความจำของ engine ออกจากของ process อื่น แต่ยังไม่ได้ทำ
        base = gpu_state().get("mem_used_mb")
        self._vram_after_load = base

    def _ck(self, err):
        # cuda-python คืน (error, value) เป็น tuple ไม่ใช่ค่าเดี่ยว และคืนต่างกันตามฟังก์ชัน
        #   cudaMalloc / cudaStreamCreate     -> (err, value)   ผู้เรียกแกะเองก่อนส่งมา
        #   cudaMemcpyAsync / StreamSynchronize -> (err,)       ส่งทั้ง tuple มาตรงนี้
        # ถ้าเทียบ tuple กับ cudaSuccess ตรงๆ จะไม่มีวันเท่ากัน แล้ว raise ทุกครั้ง
        # แม้ตอนสำเร็จ — อาการคือ RuntimeError: CUDA error: (<cudaError_t.cudaSuccess: 0>,)
        # ซึ่งอ่านแล้วสับสนเพราะข้างในบอกว่า success
        # (build_engine._check() แกะถูกอยู่แล้ว ยกตรรกะเดียวกันมาให้ตรงกัน)
        if isinstance(err, tuple):
            err, *rest = err
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA error: {err}")
            return rest[0] if len(rest) == 1 else rest
        if err != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")

    def _detect_precision(self, path: str) -> str:
        # engine ไม่มี field เดียวที่บอก precision ได้ เพราะ TRT ผสม layer หลาย
        # precision ในตัวเดียวกันอยู่แล้ว (INT8 engine มี layer FP16/FP32 ปนเสมอ)
        # เลยอ่านจากชื่อไฟล์ที่ build_engine.py ตั้งให้ (_fp16.engine / _int8.engine)
        # เช็ก int8 ก่อน fp16 เพราะ INT8 engine เปิดแฟล็ก FP16 ไว้ด้วย
        p = Path(path).name.lower()
        for tag, label in (("int8", "INT8"), ("fp16", "FP16"),
                           ("half", "FP16"), ("fp32", "FP32")):
            if tag in p:
                return label
        return "unknown"

    def infer(self, batch: np.ndarray) -> np.ndarray:
        # ทั้งสามขั้น (H2D → execute → D2H) อยู่บน stream เดียวกัน จึงเรียงตามลำดับ
        # ให้เอง ไม่ต้อง sync คั่น — ผู้เรียกต้อง sync() ก่อนอ่านผล (run_once ทำที่ t3)
        cudart = self.cudart
        nm_in = self.inputs[0]
        ptr, nbytes, _, dtype = self.ptrs[nm_in]
        arr = np.ascontiguousarray(batch, dtype=dtype)

        # NOTE: arr เป็นตัวแปร local และเป็น pageable memory ส่วน cudaMemcpyAsync
        # คืนทันทีโดยยังไม่ copy เสร็จ ถ้า ascontiguousarray สร้างสำเนาใหม่จริง
        # (dtype ไม่ตรง) สำเนานั้นอาจถูก free ก่อน DMA จบ — สาย TRT ปกติใช้ pinned
        # memory ด้วย cudaHostAlloc เพื่อกันเคสนี้
        # TODO: ยังไม่เจออาการจริง แต่ยังไม่ได้ทดสอบแบบกดดันด้วย batch ใหญ่
        self._ck(cudart.cudaMemcpyAsync(
            ptr, arr.ctypes.data, arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))

        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 ล้มเหลว")

        nm_out = self.outputs[0]
        optr, onbytes, _, _ = self.ptrs[nm_out]
        # คืน buffer ตัวเดิมทุกครั้ง ไม่ได้ copy ใหม่ — ตั้งใจ เพื่อไม่ให้ malloc
        # โผล่ในเวลาที่วัด แต่แปลว่าผลรอบก่อนถูกทับทันทีที่เรียก infer รอบใหม่
        # ใครจะเก็บไว้ใช้ทีหลังต้อง copy เอง (evaluate.py:69 ทำด้วย np.array)
        host = self.host_out[nm_out]
        self._ck(cudart.cudaMemcpyAsync(
            host.ctypes.data, optr, onbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        return host

    def sync(self):
        self._ck(self.cudart.cudaStreamSynchronize(self.stream))

    def peak_vram_mb(self):
        # NOTE: ตัวเลขนี้เทียบกับแถว PyTorch ไม่ได้
        # อันนี้คือ nvidia-smi memory.used = ทั้งการ์ด ทุก process รวม CUDA context
        # ส่วน PyTorch รายงาน max_memory_allocated() = เฉพาะ tensor ไม่รวม context
        # และ ONNX คืน None ไปเลย ทั้งสามอย่างอยู่คอลัมน์ VRAM เดียวกันในตาราง
        # TODO: ถ้าจะให้คอลัมน์นี้มีความหมาย ต้องวัดฐาน (mem_used ตอนยังไม่โหลดอะไร)
        # แล้วลบออกให้ทุก runtime เหมือนกัน หรือไม่ก็ตัดคอลัมน์นี้ทิ้ง
        cur = gpu_state().get("mem_used_mb")
        return cur if cur is not None else None

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024

    def close(self):
        for p in self.dev_mem:
            self.cudart.cudaFree(p)
        self.dev_mem = []
        self.cudart.cudaStreamDestroy(self.stream)


# ==========================================================================
# harness
# ==========================================================================
def load_images(d: str, limit: int) -> list[np.ndarray]:
    files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)[:limit]
    if not files:
        raise FileNotFoundError(f"ไม่เจอรูปใน {d}")
    imgs = [cv2.imread(str(p)) for p in files]
    imgs = [i for i in imgs if i is not None]
    print(f"[data] โหลด {len(imgs)} รูปเข้า RAM (ตัด disk I/O ออกจากการวัด)")
    return imgs


def pct(vals: list[float], q: float) -> float:
    # nearest-rank ไม่ใช่ np.percentile — percentile ใช้ interpolation แล้วคืนค่า
    # ที่ไม่เคยเกิดขึ้นจริง ซึ่งใช้ไม่ได้กับ p99 latency ที่เราต้องการชี้ไปที่
    # "รอบที่ช้าจริงรอบหนึ่ง" ไม่ใช่ค่าเฉลี่ยระหว่างสองรอบ
    s = sorted(vals)
    k = min(int(round(q / 100 * (len(s) - 1))), len(s) - 1)
    return s[k]


def run_once(runner, imgs, batch, warmup, iters, do_post):
    """หนึ่งรอบการวัด คืน dict ของ latency ทุก stage (มิลลิวินาที)"""
    n = len(imgs)
    # วนซ้ำภาพชุดเดิมแบบ wrap-around เพื่อให้ทุก iteration ได้ภาพต่างกัน
    # ถ้ายิงภาพเดิมซ้ำๆ cache จะอุ่นผิดปกติแล้วตัวเลขดีเกินจริง
    batches = [imgs[i % n:i % n + batch] if i % n + batch <= n
               else (imgs[i % n:] + imgs[:batch - (n - i % n)])
               for i in range(0, n)]

    # warmup ไม่ใช่พิธีกรรม — รอบแรกๆ รวมเวลา autotune kernel ของ cuDNN/TRT,
    # การโหลด kernel เข้า GPU และ clock ที่ยังไม่ขึ้นจาก idle
    # ถ้าไม่ทิ้งรอบพวกนี้ mean จะเพี้ยนและ p99 จะกลายเป็นเวลา warmup
    for i in range(warmup):
        x, _ = preprocess_batch(batches[i % len(batches)])
        runner.infer(x)
    runner.sync()

    pre_ms, inf_ms, post_ms = [], [], []

    for i in range(iters):
        chunk = batches[i % len(batches)]

        t0 = time.perf_counter()
        x, metas = preprocess_batch(chunk)
        t1 = time.perf_counter()

        runner.sync()                 # ปิดบัญชีงาน GPU ที่ค้างจากรอบก่อน ไม่ให้ไหลมาลง inf_ms
        t2 = time.perf_counter()
        raw = runner.infer(x)
        runner.sync()                 # infer เป็น async ถ้าไม่ sync จะวัดได้แค่เวลา enqueue
        t3 = time.perf_counter()

        if do_post:
            for b in range(len(metas)):
                postprocess(raw[b] if raw.ndim == 3 else raw,
                            metas[b], DEPLOY_CONF, DEPLOY_IOU)
        t4 = time.perf_counter()

        pre_ms.append((t1 - t0) * 1e3)
        inf_ms.append((t3 - t2) * 1e3)
        post_ms.append((t4 - t3) * 1e3)

    return {"pre": pre_ms, "inf": inf_ms, "post": post_ms}


def summarize(inf: list[float], batch: int) -> dict:
    # หารด้วย batch ทุกตัวเพื่อให้แถว batch 1 กับ batch 8 อยู่หน่วยเดียวกัน
    # (ms ต่อภาพ) ไม่งั้นเทียบกันในตารางไม่ได้
    per_img = [v / batch for v in inf]
    return {
        "mean_ms": statistics.mean(per_img),
        "std_ms": statistics.pstdev(per_img),
        "p50_ms": pct(per_img, 50),
        "p90_ms": pct(per_img, 90),
        "p99_ms": pct(per_img, 99),
        "min_ms": min(per_img),
        "max_ms": max(per_img),
        "fps": 1000.0 / statistics.mean(per_img),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--runtime", required=True, choices=["pytorch", "onnx", "tensorrt"])
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--half", action="store_true", help="PyTorch FP16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    # 3 รอบไม่ใช่เพื่อความแม่นของ mean แต่เพื่อให้เห็น std ระหว่างรอบ
    # ถ้า std ระหว่างรอบใหญ่กว่าส่วนต่างที่กำลังจะเคลม แปลว่าเคลมนั้นยังไม่มีน้ำหนัก
    ap.add_argument("--repeats", type=int, default=3, help="จำนวนรอบ รายงาน mean±std ระหว่างรอบ")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--tag", default="", help="ป้ายกำกับเพิ่มในผลลัพธ์")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    imgs = load_images(args.images, args.limit)

    if args.runtime == "pytorch":
        runner = PyTorchRunner(args.model, args.device, args.half)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    elif args.runtime == "onnx":
        runner = ONNXRunner(args.model, args.device)
        device_label = "GPU" if args.device == "cuda" else "CPU"
        print(f"[onnx] providers ที่ใช้จริง: {runner.providers}")
    else:
        runner = TensorRTRunner(args.model, args.batch)
        device_label = "GPU"

    state_before = gpu_state()
    print(f"[gpu] ก่อนวัด: {state_before}")
    print(f"[run] {runner.name} {runner.precision} {device_label} "
          f"batch={args.batch} warmup={args.warmup} iters={args.iters} "
          f"repeats={args.repeats}")

    rounds = []
    for r in range(args.repeats):
        t = time.perf_counter()
        res = run_once(runner, imgs, args.batch, args.warmup, args.iters,
                       not args.no_postprocess)
        s = summarize(res["inf"], args.batch)
        s["pre_mean_ms"] = statistics.mean(res["pre"]) / args.batch
        s["post_mean_ms"] = statistics.mean(res["post"]) / args.batch
        s["e2e_mean_ms"] = s["mean_ms"] + s["pre_mean_ms"] + s["post_mean_ms"]
        s["gpu"] = gpu_state()
        rounds.append(s)
        print(f"  รอบ {r+1}/{args.repeats}: p50 {s['p50_ms']:.3f}ms  "
              f"p99 {s['p99_ms']:.3f}ms  {s['fps']:.1f} FPS  "
              f"e2e {s['e2e_mean_ms']:.3f}ms  "
              f"({time.perf_counter()-t:.1f}s, GPU {s['gpu'].get('temp_c','?')}°C)")

    state_after = gpu_state()
    means = [r["mean_ms"] for r in rounds]
    p99s = [r["p99_ms"] for r in rounds]

    # เขียนแบบ append เพื่อให้สะสมผลจากหลาย config ไว้ไฟล์เดียวให้ make_report.py อ่าน
    # NOTE: รัน config เดิมซ้ำจะได้แถวซ้ำในตาราง ต้องลบไฟล์เองก่อนวัดรอบใหม่ทั้งชุด
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": runner.name,
        "precision": runner.precision,
        "device": device_label,
        "model": args.model,
        "batch": args.batch,
        "tag": args.tag,
        "config": {"warmup": args.warmup, "iters": args.iters,
                   "repeats": args.repeats, "num_images": len(imgs),
                   "postprocess_included": not args.no_postprocess},
        # p50/p99 ที่รายงานคือค่าเฉลี่ยของ p50/p99 รายรอบ ไม่ใช่ percentile ของ
        # ตัวอย่างทั้งหมดรวมกัน — ตั้งใจ เพราะแต่ละรอบมีสภาพ GPU ต่างกัน
        # การรวมตัวอย่างข้ามรอบจะกลบความต่างนั้น ส่วน *_std_across_repeats
        # คือตัวที่บอกว่าความต่างระหว่างรอบใหญ่แค่ไหน
        "latency_ms_per_image": {
            "mean": statistics.mean(means),
            "std_across_repeats": statistics.pstdev(means) if len(means) > 1 else 0.0,
            "p50": statistics.mean([r["p50_ms"] for r in rounds]),
            "p99": statistics.mean(p99s),
            "p99_std_across_repeats": statistics.pstdev(p99s) if len(p99s) > 1 else 0.0,
        },
        "fps": statistics.mean([r["fps"] for r in rounds]),
        # ค่าเดียวกับ fps เป๊ะ (latency ถูกหารด้วย batch แล้ว) เก็บสองชื่อไว้เพราะ
        # คนอ่านตารางมักหาคำว่า throughput ไม่ใช่ FPS
        "throughput_img_per_s": statistics.mean([r["fps"] for r in rounds]),
        "preprocess_ms": statistics.mean([r["pre_mean_ms"] for r in rounds]),
        "postprocess_ms": statistics.mean([r["post_mean_ms"] for r in rounds]),
        "end_to_end_ms": statistics.mean([r["e2e_mean_ms"] for r in rounds]),
        "model_size_mb": runner.model_size_mb(args.model),
        "peak_vram_mb": runner.peak_vram_mb(),
        "gpu_before": state_before,
        "gpu_after": state_after,
        "rounds": rounds,
    }
    if args.runtime == "onnx":
        record["providers"] = runner.providers

    with open(args.out, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    L = record["latency_ms_per_image"]
    print(f"\n{'-'*60}")
    print(f"  {runner.name} / {runner.precision} / {device_label} / batch {args.batch}")
    print(f"  inference-only : p50 {L['p50']:.3f} ms | p99 {L['p99']:.3f} ms")
    print(f"                   mean {L['mean']:.3f} ± {L['std_across_repeats']:.3f} ms")
    print(f"  end-to-end     : {record['end_to_end_ms']:.3f} ms "
          f"(pre {record['preprocess_ms']:.3f} / post {record['postprocess_ms']:.3f})")
    print(f"  throughput     : {record['fps']:.1f} img/s")
    print(f"  model size     : {record['model_size_mb']:.2f} MB")
    # 15°C เป็นเกณฑ์ที่ตั้งเอา ยังไม่ได้ยืนยันว่าตรงกับจุดที่การ์ดใบนี้เริ่ม throttle จริง
    # NOTES: GPU idle อยู่ที่ 37-40°C ยังไม่ได้วัดตอนโหลดเต็ม
    # TODO: เอา gpu_before/gpu_after จากการรันจริงมาปรับเกณฑ์นี้
    if state_before.get("temp_c") and state_after.get("temp_c"):
        d = state_after["temp_c"] - state_before["temp_c"]
        print(f"  GPU temp       : {state_before['temp_c']:.0f} -> "
              f"{state_after['temp_c']:.0f} °C ({d:+.0f})")
        if d > 15:
            print("  [warn] อุณหภูมิขึ้นเยอะ — อาจมี throttle ระหว่างวัด "
                  "ลองรันให้ร้อนคงที่ก่อน หรือ nvidia-smi -lgc")
    print(f"  -> ต่อท้ายลง {args.out}\n{'-'*60}")

    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()

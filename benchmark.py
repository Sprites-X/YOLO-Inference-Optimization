from __future__ import annotations

import argparse
import ctypes
import json
import os
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
# What nvidia-smi is asked for, in order. index has to come first so the rows can be
# told apart on a multi-GPU host; the three slowdown reasons are what turn "the GPU got
# hot" into "the GPU actually throttled".
_GPU_FIELDS = ["index", "temperature.gpu", "clocks.sm", "power.draw", "memory.used",
               "clocks_throttle_reasons.sw_thermal_slowdown",
               "clocks_throttle_reasons.hw_thermal_slowdown",
               "clocks_throttle_reasons.sw_power_cap"]


def _row_for_this_process(rows: dict) -> list | None:
    """The nvidia-smi row for the GPU this process will run on."""
    if not rows:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        # CUDA_VISIBLE_DEVICES remaps device ordinals, so this process's cuda:0 is
        # whatever it lists first. A UUID cannot be matched against the index column,
        # so fall through rather than report another card's temperature as this one's.
        first = visible.split(",")[0].strip()
        if first in rows:
            return rows[first]
    return rows[sorted(rows)[0]]


def gpu_state() -> dict:
    # Record temperature, clocks and whether the card throttled alongside every round,
    # so a round that came out slower than its neighbours can be pinned on throttling
    # rather than measurement noise. Without it there is nothing to go back and look at.
    #
    # Parsed line by line rather than by splitting the whole stdout on ",". With two
    # cards nvidia-smi returns two lines, and the old version handed float() '512\n38',
    # which the bare except swallowed — every telemetry field vanished from the record
    # and nothing said why.
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        rows = {}
        for line in out:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) == len(_GPU_FIELDS):
                rows[parts[0]] = parts
        parts = _row_for_this_process(rows)
        if parts is None:
            return {}

        # nvidia-smi spells these "Active" / "Not Active".
        reasons = [name.split(".")[-1] for name, val in
                   zip(_GPU_FIELDS[5:], parts[5:]) if val.lower() == "active"]
        return {"gpu_index": int(parts[0]),
                "temp_c": float(parts[1]), "sm_clock_mhz": float(parts[2]),
                "power_w": float(parts[3]), "mem_used_mb": float(parts[4]),
                "throttled": bool(reasons), "throttle_reasons": reasons}
    except Exception:
        return {}


def cpu_governor() -> str | None:
    """The scaling governor, or None if the kernel does not expose one.

    results.jsonl already carries gpu_before/gpu_after so a slow round can be blamed
    on throttling rather than noise. The CPU row had no equivalent, even though its
    validity rests entirely on this setting: under 'powersave' the CPU baseline reads
    slower than the hardware really is, which inflates every GPU speedup measured
    against it. It resets to the distro default on reboot, so knowing it was checked
    once is not enough — it has to be recorded per run.

    Reads every core rather than cpu0 alone: they are normally uniform, but a machine
    part-way through a governor change would otherwise be recorded as whatever cpu0
    happened to be.
    """
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    govs = set()
    for gp in paths:
        try:
            govs.add(gp.read_text().strip())
        except OSError:
            continue
    if not govs:
        return None
    return govs.pop() if len(govs) == 1 else "mixed:" + ",".join(sorted(govs))


# ONNX Runtime's CUDA provider links libcublasLt/libcublas/libcurand/libcufft/
# libcudart/libcudnn by soname, but pip puts them under site-packages/nvidia/*/lib,
# which the dynamic loader does not search. Loading the provider then fails with
# "libcublasLt.so.12: cannot open shared object file" and ORT quietly drops to CPU.
#
# It works anyway whenever torch was imported first, because torch dlopens the same
# files with RTLD_GLOBAL on its way in. That made the failure look like it depended on
# the script rather than the environment: verify_env.py checks torch before ORT and
# passed, check_parity.py builds the PyTorch reference first and passed, while
# `evaluate.py --runtime onnx --device cuda` — which never imports torch — could not
# get a CUDA session at all.
#
# Deliberately not LD_LIBRARY_PATH: pointing that at these directories shadows the
# TensorRT .so files that `import tensorrt` resolves from inside its own package, and
# trades a broken ONNX row for broken TensorRT rows.
#
# Ordered so each library is loaded before the ones that link against it.
_ORT_CUDA_DEPS = ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12",
                  "libcudnn.so.9", "libcurand.so.10", "libcufft.so.11")


def preload_ort_cuda_deps() -> list[str]:
    """dlopen ORT's CUDA dependencies from the nvidia pip packages. Returns what loaded."""
    try:
        import nvidia
    except ImportError:
        return []      # CUDA libs installed system-wide; the loader can find them itself

    root = Path(nvidia.__file__).parent
    loaded = []
    for soname in _ORT_CUDA_DEPS:
        cand = next(iter(sorted(root.glob(f"*/lib/{soname}"))), None)
        if cand is None:
            continue
        try:
            # RTLD_GLOBAL so the provider .so resolves against these once ORT dlopens
            # it; the default RTLD_LOCAL would keep them private to this module.
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            loaded.append(cand.name)
        except OSError:
            # Not fatal here: whatever is missing surfaces as ONNXRunner refusing to
            # hand back a CUDA session, which is the error worth reading.
            pass
    return loaded


# ==========================================================================
# Runners 
# ==========================================================================
# All three classes expose the same infer/sync/peak_vram_mb/model_size_mb, so
# run_once can time them with one code path. Runtime differences belong inside the
# class, never in the timing loop.
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
        # Use yolo.model directly rather than yolo.predict(), because predict wraps
        # ultralytics' own pre/postprocess and that makes it incomparable to ONNX/TRT.
        # All three runtimes have to go through the same common.preprocess/postprocess.
        self.model = yolo.model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Reset before measuring: YOLO() grabs scratch memory while loading, and
        # without this the peak reflects load time rather than inference.
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        # non_blocking=False on purpose — the H2D copy has to sit inside the timed
        # window, same as TRT and ORT pay for. Otherwise the PyTorch row gets an unfair
        # advantage by hiding the copy outside the clock.
        t = self.torch.from_numpy(batch).to(self.device, non_blocking=False)
        if self.half:
            t = t.half()
        with self.torch.inference_mode():
            out = self.model(t)
        # In eval mode the Detect head returns (y, x): y is the decoded output
        # (B, 84, 8400), x the raw per-scale feature maps that postprocess never uses.
        out = out[0] if isinstance(out, (list, tuple)) else out
        # .float() on the way out: leave it half and postprocess de-letterboxes in
        # float16, putting boxes off by ~1 px (see the NOTE in common.postprocess).
        return out.float().cpu().numpy()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def peak_vram_mb(self):
        # Counts only tensors torch's allocator reserved, excluding the CUDA context
        # (~300-600MB), so it is not directly comparable to the TensorRT figure, which
        # comes from nvidia-smi. record["vram_mb"] is the comparable one; see the note
        # on TensorRTRunner.peak_vram_mb.
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1024 / 1024
        return None

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024


# ONNX spells its tensor types like this; the table wants the short name.
_ORT_PRECISION = {"tensor(float)": "FP32", "tensor(float16)": "FP16",
                  "tensor(bfloat16)": "BF16", "tensor(double)": "FP64",
                  "tensor(int8)": "INT8", "tensor(uint8)": "INT8"}


class ONNXRunner:
    name = "ONNX Runtime"

    def __init__(self, model_path: str, device: str):
        if device == "cuda":
            preload_ort_cuda_deps()
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(model_path, opts, providers=providers)
        used = self.sess.get_providers()

        # This check is the main reason verify_env.py exists.
        # ORT does not treat falling back to CPU as an error — it just goes quiet and
        # runs 20-50x slower. Let it through and the "ONNX Runtime GPU" row is CPU
        # numbers wearing a GPU label, which is exactly the kind of mistake you cannot
        # spot from the table. It has to raise here.
        if device == "cuda" and "CUDAExecutionProvider" not in used:
            raise RuntimeError(
                f"asked for CUDA but ORT is using {used} — letting this through would "
                f"make this table row CPU numbers wearing a GPU label"
            )
        self.providers = used
        self.iname = self.sess.get_inputs()[0].name
        self.device = device
        self.precision = self._graph_precision()

    def _graph_precision(self) -> str:
        """Precision label taken from the graph, not assumed.

        It used to be the constant "FP32", which was true of every export this project
        makes and would have quietly mislabelled the first FP16 one — a wrong precision
        in the table is invisible, because the latency it sits next to looks plausible
        either way.

        Read off the output rather than the input: a converted model often keeps an FP32
        input for convenience, and the output dtype is both what the graph computed in
        and what common.postprocess has to de-letterbox.
        """
        out_type = self.sess.get_outputs()[0].type
        return _ORT_PRECISION.get(out_type, out_type)

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.iname: batch})[0]

    def sync(self):
        pass  # sess.run() already blocks; this exists so run_once can call sync() on any runtime

    def peak_vram_mb(self):
        return None

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024


class TensorRTRunner:
    name = "TensorRT"

    def __init__(self, engine_path: str, batch: int = 1, size: int = IMG_SIZE):
        import tensorrt as trt
        # cuda.cudart is deprecated; it moved to cuda.bindings.runtime. The fallback
        # is kept for machines still on an older cuda-python (this one is 12.9.7).
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
                f"could not load engine: {engine_path}\n"
                f"an engine is tied to the GPU and TensorRT version that built it — change either and you have to rebuild"
            )
        self.ctx = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        self._ck(err)

        # TRT 10 API: iterate num_io_tensors and address tensors by name, not the
        # TRT 8 bindings[] array. (verify_env.py has already checked for TRT >= 10.)
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
                        f"engine is static batch {shape[0]} but batch {batch} was asked for "
                        f"— rebuild the engine with --min/opt/max-batch"
                    )
                self.inputs.append(nm)
            else:
                self.outputs.append(nm)

        self.precision = self._detect_precision(engine_path)
        self.batch = batch
        self.host_in, self.host_out = {}, {}
        self.dev_mem, self.host_mem = [], []

        # Allocate once at init and reuse every iteration. Allocating and freeing each
        # round would fold cudaMalloc into the measured latency, which is not what a real
        # system does. get_tensor_shape has to come from ctx, not engine, because the
        # dynamic batch was only just set above.
        for nm in self.inputs + self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(nm))
            dtype = trt.nptype(self.engine.get_tensor_dtype(nm))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, ptr = cudart.cudaMalloc(nbytes)
            self._ck(err)
            self.ptrs[nm] = (ptr, nbytes, shape, dtype)
            self.dev_mem.append(ptr)
            self.ctx.set_tensor_address(nm, int(ptr))
            # Page-locked on both ends. cudaMemcpyAsync only becomes a real async DMA
            # out of pinned memory; from pageable memory the driver stages through its
            # own buffer, and the copy is neither async nor free. It also removes the
            # hazard the input side used to carry — a temporary array could in principle
            # be freed before its DMA finished.
            host = self._pinned(shape, dtype, nbytes)
            (self.host_out if nm in self.outputs else self.host_in)[nm] = host

    def _pinned(self, shape, dtype, nbytes) -> np.ndarray:
        """A numpy view over page-locked host memory, freed in close()."""
        ptr = self._ck(self.cudart.cudaHostAlloc(
            nbytes, self.cudart.cudaHostAllocDefault))
        self.host_mem.append(ptr)
        buf = (ctypes.c_byte * nbytes).from_address(int(ptr))
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def _ck(self, err):
        # cuda-python returns (error, value) tuples rather than a bare value, and what
        # comes back differs per function:
        #   cudaMalloc / cudaStreamCreate       -> (err, value)  caller unpacks first
        #   cudaMemcpyAsync / StreamSynchronize -> (err,)        whole tuple arrives here
        # Compare a tuple against cudaSuccess directly and it can never match, so it
        # raises on every call even when the call worked — the symptom is the
        # self-contradictory RuntimeError: CUDA error: (<cudaError_t.cudaSuccess: 0>,).
        # (build_engine._check() already unpacks correctly; same logic lifted here.)
        if isinstance(err, tuple):
            err, *rest = err
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA error: {err}")
            return rest[0] if len(rest) == 1 else rest
        if err != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")

    def _detect_precision(self, path: str) -> str:
        # There is no single engine field that reports precision, because TRT mixes
        # layer precisions inside one engine anyway (an INT8 engine always has FP16/FP32
        # layers in it). So read it off the filename build_engine.py chose
        # (_fp16.engine / _int8.engine). Check int8 before fp16, since an INT8 engine
        # has the FP16 flag set too.
        p = Path(path).name.lower()
        for tag, label in (("int8", "INT8"), ("fp16", "FP16"),
                           ("half", "FP16"), ("fp32", "FP32")):
            if tag in p:
                return label
        return "unknown"

    def infer(self, batch: np.ndarray) -> np.ndarray:
        # All three steps (H2D → execute → D2H) share one stream, so they order
        # themselves and need no sync in between. The caller must sync() before reading
        # the result (run_once does that at t3).
        cudart = self.cudart
        nm_in = self.inputs[0]
        ptr, nbytes, _, _ = self.ptrs[nm_in]

        # Staged through the pinned buffer allocated at init. copyto casts if the engine
        # takes a narrower input dtype, and the destination outlives the copy, so there
        # is no temporary that could be freed while its DMA is still in flight.
        arr = self.host_in[nm_in]
        np.copyto(arr, batch, casting="unsafe")
        self._ck(cudart.cudaMemcpyAsync(
            ptr, arr.ctypes.data, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))

        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")

        nm_out = self.outputs[0]
        optr, onbytes, _, _ = self.ptrs[nm_out]
        # Returns the same buffer every call rather than a fresh copy. That is
        # deliberate — it keeps malloc out of the measured time — but it means the
        # previous result is overwritten the moment infer runs again. Anyone keeping a
        # result around has to copy it (evaluate.py:69 does, with np.array).
        host = self.host_out[nm_out]
        self._ck(cudart.cudaMemcpyAsync(
            host.ctypes.data, optr, onbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        return host

    def sync(self):
        self._ck(self.cudart.cudaStreamSynchronize(self.stream))

    def peak_vram_mb(self):
        # nvidia-smi memory.used: the whole card, every process and the CUDA context
        # included. Not comparable to the PyTorch row, which counts allocator tensors
        # only, nor to ONNX, which reports nothing — that is why the number the table
        # shows is record["vram_mb"], the difference between a sample taken before the
        # runtime loaded and one taken after the run, measured the same way for all
        # three. This stays as the runtime's own view.
        return gpu_state().get("mem_used_mb")

    def model_size_mb(self, path):
        return Path(path).stat().st_size / 1024 / 1024

    def close(self):
        for p in self.dev_mem:
            self.cudart.cudaFree(p)
        self.dev_mem = []
        # The numpy views in host_in/host_out point into this memory, so drop them
        # first: touching one after cudaFreeHost is a use-after-free, not an exception.
        self.host_in, self.host_out = {}, {}
        for p in self.host_mem:
            self.cudart.cudaFreeHost(p)
        self.host_mem = []
        self.cudart.cudaStreamDestroy(self.stream)


# ==========================================================================
# harness
# ==========================================================================
def load_images(d: str, limit: int) -> list[np.ndarray]:
    files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)[:limit]
    if not files:
        raise FileNotFoundError(f"no images found in {d}")
    imgs = [cv2.imread(str(p)) for p in files]
    imgs = [i for i in imgs if i is not None]
    print(f"[data] loaded {len(imgs)} images into RAM (keeps disk I/O out of the timing)")
    return imgs


def pct(vals: list[float], q: float) -> float:
    # nearest-rank, not np.percentile — percentile interpolates and hands back a value
    # that never actually happened, which is wrong for p99 latency where the point is to
    # name one genuinely slow iteration, not the average of two.
    s = sorted(vals)
    k = min(int(round(q / 100 * (len(s) - 1))), len(s) - 1)
    return s[k]


def run_once(runner, imgs, batch, warmup, iters, do_post):
    """One measurement round. Returns a dict of per-stage latencies in milliseconds."""
    n = len(imgs)
    # Wrap around the image list so every iteration sees a different image. Firing the
    # same one repeatedly warms the cache unnaturally and flatters the numbers.
    batches = [imgs[i % n:i % n + batch] if i % n + batch <= n
               else (imgs[i % n:] + imgs[:batch - (n - i % n)])
               for i in range(0, n)]

    # Warm-up is not ritual. The first iterations include cuDNN/TRT kernel autotuning,
    # loading kernels onto the GPU, and clocks still climbing out of idle. Keep them and
    # the mean skews while p99 just becomes the warm-up time.
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

        runner.sync()                 # settle leftover GPU work so it does not leak into inf_ms
        t2 = time.perf_counter()
        raw = runner.infer(x)
        runner.sync()                 # infer is async; without this you only time the enqueue
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
    # Divide everything by batch so the batch 1 and batch 8 rows share a unit
    # (ms per image); otherwise they cannot be compared in the table.
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
    # 3 rounds is not about a more accurate mean, it is to expose the std between
    # rounds. If that std is larger than the difference you are about to claim, the
    # claim does not hold up yet.
    ap.add_argument("--repeats", type=int, default=3, help="rounds to run; reports mean±std across them")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--tag", default="", help="extra label stored with the result")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    imgs = load_images(args.images, args.limit)

    # Sampled before the runner exists, so the delta taken after the run is the VRAM
    # this runtime needed, CUDA context and all, measured identically for all three.
    # The per-runner peak_vram_mb() numbers cannot be compared to each other: PyTorch
    # counts allocator tensors only, TensorRT reads the whole card, ONNX reports
    # nothing at all.
    vram_before = gpu_state().get("mem_used_mb")

    if args.runtime == "pytorch":
        runner = PyTorchRunner(args.model, args.device, args.half)
        device_label = "GPU" if args.device == "cuda" else "CPU"
    elif args.runtime == "onnx":
        runner = ONNXRunner(args.model, args.device)
        device_label = "GPU" if args.device == "cuda" else "CPU"
        print(f"[onnx] providers actually in use: {runner.providers}")
    else:
        runner = TensorRTRunner(args.model, args.batch)
        device_label = "GPU"

    state_before = gpu_state()
    governor = cpu_governor()
    print(f"[gpu] before: {state_before}")
    # Only worth interrupting for on a CPU run: that is the row the governor distorts.
    if args.device == "cpu" and governor not in (None, "performance"):
        print(f"  [warn] CPU governor is '{governor}', not 'performance' — this baseline "
              f"will read slower than the hardware really is, and every GPU speedup "
              f"measured against it will look better than it is "
              f"(sudo cpupower frequency-set -g performance)")
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
        print(f"  round {r+1}/{args.repeats}: p50 {s['p50_ms']:.3f}ms  "
              f"p99 {s['p99_ms']:.3f}ms  {s['fps']:.1f} FPS  "
              f"e2e {s['e2e_mean_ms']:.3f}ms  "
              f"({time.perf_counter()-t:.1f}s, GPU {s['gpu'].get('temp_c','?')}°C)")

    state_after = gpu_state()
    vram_after = state_after.get("mem_used_mb")
    vram_delta = (round(vram_after - vram_before, 1)
                  if device_label == "GPU" and None not in (vram_before, vram_after)
                  else None)
    means = [r["mean_ms"] for r in rounds]
    p50s = [r["p50_ms"] for r in rounds]
    p99s = [r["p99_ms"] for r in rounds]

    # Append so results from several configs pile up in one file for make_report.py.
    # NOTE: re-running the same config gives you a duplicate row in the table — delete
    # the file yourself before starting a fresh full sweep.
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
        # The reported p50/p99 are the mean of each round's p50/p99, not a percentile
        # over all samples pooled together. That is deliberate: every round runs under
        # different GPU conditions, and pooling would hide exactly that. The
        # *_std_across_repeats fields are what tell you how big the round-to-round
        # spread was.
        "latency_ms_per_image": {
            "mean": statistics.mean(means),
            "std_across_repeats": statistics.pstdev(means) if len(means) > 1 else 0.0,
            "p50": statistics.mean(p50s),
            # Recorded because make_report draws the error bar on the p50 bar, and the
            # spread of the mean is a different statistic from the spread of p50.
            "p50_std_across_repeats": statistics.pstdev(p50s) if len(p50s) > 1 else 0.0,
            "p99": statistics.mean(p99s),
            "p99_std_across_repeats": statistics.pstdev(p99s) if len(p99s) > 1 else 0.0,
        },
        "fps": statistics.mean([r["fps"] for r in rounds]),
        # Identical to fps (latency is already per-image), kept under both names
        # because people reading the table look for "throughput" rather than FPS.
        "throughput_img_per_s": statistics.mean([r["fps"] for r in rounds]),
        "preprocess_ms": statistics.mean([r["pre_mean_ms"] for r in rounds]),
        "postprocess_ms": statistics.mean([r["post_mean_ms"] for r in rounds]),
        "end_to_end_ms": statistics.mean([r["e2e_mean_ms"] for r in rounds]),
        "model_size_mb": runner.model_size_mb(args.model),
        # The comparable one. None on CPU runs, where card usage says nothing about
        # what the run needed.
        "vram_mb": vram_delta,
        # The runtime's own view, kept because it answers a different question for
        # PyTorch (tensors the allocator reserved, no context).
        "peak_vram_mb": runner.peak_vram_mb(),
        "gpu_before": state_before,
        "gpu_after": state_after,
        "cpu_governor": governor,
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
    if state_before.get("temp_c") and state_after.get("temp_c"):
        d = state_after["temp_c"] - state_before["temp_c"]
        print(f"  GPU temp       : {state_before['temp_c']:.0f} -> "
              f"{state_after['temp_c']:.0f} °C ({d:+.0f})")
    if record["vram_mb"] is not None:
        print(f"  VRAM           : {record['vram_mb']:.0f} MB over baseline")

    # This used to warn on a temperature rise over 15°C, a number picked without ever
    # checking where this card throttles — it would have fired on a run that was fine
    # and stayed quiet on one that was not. nvidia-smi reports the slowdown flags
    # directly, so ask instead of inferring: gpu_state() collects them every round.
    hot = [(i, r["gpu"].get("throttle_reasons", []))
           for i, r in enumerate(rounds) if r["gpu"].get("throttled")]
    if hot:
        detail = "; ".join(f"round {i}: {', '.join(reasons)}" for i, reasons in hot)
        print(f"  [warn] the GPU throttled during this run ({detail}) — these numbers "
              f"are below what the card does at steady state. Let it settle, or lock "
              f"clocks with nvidia-smi -lgc")
    print(f"  -> appended to {args.out}\n{'-'*60}")

    if hasattr(runner, "close"):
        runner.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
# cuda.cudart is deprecated and moved to cuda.bindings.runtime. This machine runs
# cuda-python 12.9.7 and takes the new path; the fallback covers older installs
# elsewhere (benchmark.py does the same).
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

sys.path.insert(0, str(Path(__file__).parent))
from common import IMG_SIZE, preprocess  

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# yolov8's detect head is module 22, and TensorRT keeps the node names the ONNX carried,
# so the whole head is addressable by prefix. Counted on yolov8n.onnx: 73 of 299 layers
# match, spanning index 138 to 298, and they include all 19 convolutions in the head —
# the cv2.* box branch, the cv3.* class branch, and the DFL conv at 230.
HEAD_PREFIX = "/model.22/"


def _check(err):
    # cuda-python returns (error, value) tuples, unlike the C API which returns only
    # an error and hands the value back through a pointer — cudaMalloc gives
    # (err, ptr), cudaMemcpy gives (err,). So this has to accept both shapes and
    # return just the value.
    if isinstance(err, tuple):
        err, *rest = err
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")
        return rest[0] if len(rest) == 1 else rest
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def _calib_fingerprint(onnx_path, calib_dir, size, num, batch) -> str:
    # The cache holds raw dynamic ranges and nothing about what produced them, so we
    # attach our own fingerprint of the conditions. Change the model, image set, size
    # or count and the old ranges no longer apply.
    key = "|".join([str(Path(onnx_path).resolve()), str(Path(calib_dir).resolve()),
                    str(size), str(num), str(batch)])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _check_calib_cache(cache_path: str, fingerprint: str, describe: str) -> None:
    # Has to run before the calibrator is handed to TRT. It cannot live in a
    # calibrator callback, because TRT calls those from C++ and Python exceptions
    # never make it back out.
    if not os.path.exists(cache_path):
        return
    meta_path = cache_path + ".meta.json"
    if not os.path.exists(meta_path):
        # A cache from before the meta system existed: its origin cannot be checked,
        # only warned about.
        print(f"[calib] warning: {cache_path} has no .meta.json beside it, so its origin "
              f"cannot be verified — delete it and recalibrate if unsure")
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"cache {cache_path} was built under different conditions\n"
            f"  in the file : {meta.get('describe')}\n"
            f"  asked now   : {describe}\n"
            f"dynamic ranges do not carry across models, image sizes or calib sets "
            f"— delete this file, or pass a different --calib-cache name"
        )


# --------------------------------------------------------------------------
# IInt8EntropyCalibrator2 has been deprecated since TRT 10 (the modern route is
# explicit quantization with Q/DQ nodes in the ONNX), but it is still used here
# because it is post-training calibration that needs no changes to the ONNX — and
# it is one reason requirements pin tensorrt-cu12==10.16.1.11 (let pip wander to
# TRT 11 and this API may already be gone).
#
# Entropy2 rather than MinMax because Entropy2 is the recommended choice for CNNs;
# MinMax is more sensitive to outliers in the calibration images.
class ImageCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calib_dir: str, cache_path: str, num_images: int = 500,
                 batch_size: int = 8, size: int = IMG_SIZE, seed: int = 0,
                 fingerprint: str = "", describe: str = ""):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.size = size
        self.fingerprint = fingerprint
        self.describe = describe

        files = sorted(p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no images found in {calib_dir}")
        # Shuffle before slicing num_images: sorted() gives consecutive COCO ids,
        # which are not spread across image types. The seed is fixed so every build
        # uses the same set — otherwise measured INT8 mAP moves because the
        # calibration set changed, not because the code did.
        random.Random(seed).shuffle(files) 
        self.files = files[:num_images]
        self.index = 0

        # 4 = bytes per float32 value. The engine's input stays FP32 even in an INT8
        # build, because quantization happens inside the graph, not at the input.
        nbytes = batch_size * 3 * size * size * 4
        self.d_input = _check(cudart.cudaMalloc(nbytes))
        self.nbytes = nbytes
        print(f"[calib] using {len(self.files)} images, batch {batch_size} "
              f"-> {(len(self.files) + batch_size - 1) // batch_size} rounds")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # Returning None tells TRT we are out of batches; it is not an error.
        if self.index >= len(self.files):
            return None
        import cv2

        chunk = self.files[self.index:self.index + self.batch_size]
        arrs = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # Must be the same common.preprocess used at inference. Calibrate through
            # a different preprocess and the dynamic ranges will not match the data the
            # model actually sees.
            arrs.append(preprocess(img, self.size)[0])
        if not arrs:
            self.index += self.batch_size
            return self.get_batch(names)

        # Pad the final chunk up to a full batch by repeating the first image, since
        # TRT always reads the whole buffer.
        # NOTE: this over-counts that image's histogram by at most batch_size-1
        # (7 out of 500 at the defaults). Not measured against mAP; likely tiny.
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
        # The cache makes rebuilds much faster by skipping the forward pass over 500
        # images entirely.
        #
        # The default name used to be a generic "calibration.cache", so building a
        # second model in the same folder picked up the first model's dynamic ranges
        # and printed "skipping calibration" as though all were well — mAP broken, no
        # error. The default name now carries a fingerprint, and the paired
        # .meta.json is checked as a second layer for when --calib-cache is passed
        # explicitly.
        if not os.path.exists(self.cache_path):
            return None

        # Fingerprint mismatches are already caught by _check_calib_cache() during
        # build(). Do not move that check here: TRT calls this method from C++ and
        # swallows the Python exception, so raising would not stop the build — TRT
        # would decide there is no cache and recalibrate over the existing file
        # (confirmed by testing).
        print(f"[calib] found existing cache {self.cache_path} — skipping calibration")
        with open(self.cache_path, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        with open(self.cache_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint, "describe": self.describe},
                      f, ensure_ascii=False, indent=2)
        print(f"[calib] wrote cache -> {self.cache_path}")

    def free(self):
        if getattr(self, "d_input", None):
            cudart.cudaFree(self.d_input)
            self.d_input = None


# --------------------------------------------------------------------------
def build(args):
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH is deprecated in TRT 10 (networks are always explicit batch now).
    # Still passed so the intent reads clearly, and in case this runs on an older TRT.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            # Have to walk num_errors by hand: the parser keeps errors inside and
            # returns only False. Without printing them you are left with a bare
            # "parse failed" that cannot be debugged.
            for i in range(parser.num_errors):
                print(f"[onnx] {parser.get_error(i)}", file=sys.stderr)
            raise RuntimeError("failed to parse the ONNX")

    config = builder.create_builder_config()
    # workspace is the ceiling TRT gets for trying kernels during autotuning, not the
    # memory the engine uses at runtime. Set it too low and some fast kernels are
    # silently dropped from consideration. << 30 = GB -> bytes.
    #
    # MemoryPoolType, not MemoryPoolFlag — the latter does not exist in tensorrt
    # 10.16.1.11. (It was written wrong once, and build_engine.py threw AttributeError
    # from this line for every precision, before even reaching calibration.)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)

    # Off by default. At the default verbosity the engine keeps layer *names* only, and
    # the compiler backend has already rewritten those to things like __mye48100_myl0_0,
    # so an inspector dump says nothing about precision or tactic. DETAILED keeps the
    # rest, at the cost of a larger engine — a diagnosis switch, not a build setting.
    if args.detailed_layers:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        print("[build] DETAILED profiling verbosity — engine carries full layer info")

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
        engine_max_batch = args.max_batch
        print(f"[net] dynamic batch min={args.min_batch} opt={args.opt_batch} "
              f"max={args.max_batch}")
    else:
        engine_max_batch = int(inp.shape[0])
        if args.max_batch > 1:
            print("[net] ONNX is static batch — --max-batch has no effect "
                  "(re-export with dynamic=True to test batching)")

    # Resolved once here so the build and the engine name cannot disagree about what
    # was pinned. --fp16-head is the shorthand the results in the README were built with.
    pin_prefixes = tuple(args.fp16_prefix or ())
    if args.fp16_head:
        pin_prefixes = (HEAD_PREFIX,) + pin_prefixes
    if pin_prefixes and args.precision != "int8":
        raise SystemExit(
            f"--fp16-head/--fp16-prefix does nothing for --precision {args.precision}\n"
            f"they exist to take layers back out of INT8; an fp16 build already "
            f"has them in FP16, and an fp32 build has them in FP32"
        )

    calibrator = None
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[warn] platform has no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    elif args.precision == "int8":
        if not builder.platform_has_fast_int8:
            print("[warn] platform has no fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        # Always enable FP16 alongside INT8: the INT8 flag does not force every layer
        # to INT8. Where quantizing makes a layer slower or badly less accurate, TRT
        # picks another precision for it. Without FP16 the only fallback left is FP32,
        # which is slower than it needs to be.
        config.set_flag(trt.BuilderFlag.FP16)

        if not args.calib_dir:
            raise SystemExit(
                "INT8 requires --calib-dir\n"
                "without images TensorRT invents its own dynamic ranges and mAP breaks with no error"
            )
        # TRT calibrates at whatever batch the optimization profile allows, not at the
        # calibrator's get_batch_size(). If the calibrator hands over 8 images but the
        # profile maxes at 1, TRT reads only the first image of each chunk and silently
        # discards the rest (--calib-batch 8 with --max-batch 1 = calibrating on 63
        # images out of 500).
        #
        # Measured on TRT 10.16.1.11 with a tiny model: made slots 1-7 of every chunk
        # differ by 50x and watched whether the calibration cache changed.
        #   max_batch=1 calib_batch=8                -> cache unchanged (first image only)
        #   max_batch=8 calib_batch=8                -> cache changes (all read)
        #   max_batch=8 calib_batch=8 + calib prof 1 -> cache unchanged
        # That last line is why config.set_calibration_profile() is not the fix — it
        # does not help, and set wrong it makes things worse. Max batch is what governs.
        # Pinned to the engine's max batch rather than clamped in one direction, because
        # both directions are wrong:
        #   calib_batch > max — TRT reads only the first max images of each chunk and
        #     drops the rest, quietly calibrating on a fraction of --calib-num
        #     (--calib-batch 8 with --max-batch 1 = 63 images out of 500).
        #   calib_batch < max — TRT reads max images anyway, but ImageCalibrator sized
        #     its device buffer for calib_batch, so the read runs off the end of the
        #     allocation. Measured with --max-batch 8 --calib-batch 2: dies on the first
        #     chunk with CUDA 700 (illegal memory access), reported as a cascade of TRT
        #     errors about calibrator.cpp, deallocation and Myelin teardown that name
        #     neither batch size. Nothing in it points at --calib-batch.
        calib_batch = args.calib_batch
        if calib_batch != engine_max_batch:
            print(f"[calib] --calib-batch {calib_batch} -> {engine_max_batch} to match the "
                  f"engine's max batch (same {args.calib_num} images, different chunk size)")
            calib_batch = engine_max_batch

        fp = _calib_fingerprint(args.onnx, args.calib_dir, args.size,
                                args.calib_num, calib_batch)
        describe = (f"onnx={Path(args.onnx).name} calib_dir={args.calib_dir} "
                    f"size={args.size} num={args.calib_num} batch={calib_batch}")
        cache_path = args.calib_cache or (
            Path(args.onnx).with_suffix("").name + f"_calib_{fp}.cache")
        _check_calib_cache(cache_path, fp, describe)

        calibrator = ImageCalibrator(
            args.calib_dir, cache_path, args.calib_num,
            calib_batch, args.size, fingerprint=fp, describe=describe,
        )
        config.int8_calibrator = calibrator

        if pin_prefixes:
            _force_fp16_layers(network, config, pin_prefixes)

    # The name carries _fp16head as well as the precision, because an INT8 engine with
    # the head pinned is a different engine from one without and the two are meant to be
    # compared. Sharing a filename would have the second build overwrite the first.
    # benchmark.py still reads INT8 off this, since it checks for int8 before fp16.
    # _fp16head stays the name for the head shorthand, because run_all.sh and the README
    # refer to that file. Sweep builds get a name off their prefixes instead, so twenty
    # of them in a row do not all land on one filename.
    if args.fp16_head and not args.fp16_prefix:
        pin_tag = "_fp16head"
    elif pin_prefixes:
        pin_tag = "_fp16" + "-".join(
            pre.strip("/").replace("model.", "m").replace("/", "_") for pre in pin_prefixes)
    else:
        pin_tag = ""
    tag = f"_{args.precision}" + pin_tag
    out = args.engine or str(Path(args.onnx).with_suffix("").name + tag + ".engine")

    # The filename has to carry _fp16/_int8, because benchmark.py reads precision off
    # it (TensorRTRunner._detect_precision) — an engine does not report its own.
    print(f"[build] building {args.precision} — the first run takes several minutes, which is normal")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0

    if calibrator:
        calibrator.free()
    if serialized is None:
        raise RuntimeError("build failed")

    with open(out, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"[build] done in {dt:.1f}s -> {out} ({size_mb:.2f} MB)")
    print("[note] this engine is tied to this GPU and this TensorRT version — rebuild on another machine")


def _force_fp16_layers(network, config, prefixes=(HEAD_PREFIX,)) -> int:
    """Pin every layer whose name contains one of `prefixes` to FP16.

    Defaults to the detect head, which is what this was written for: the head regresses
    coordinates, which needs finer resolution than class scores do, so it is the first
    thing worth taking back out of INT8. Arbitrary prefixes are reachable from
    --fp16-prefix, which is how the per-block sensitivity sweep is run — pin one
    `/model.N/` block at a time and the mAP delta is that block's share of the INT8
    loss.

    Layers are selected by name, not by counting back from the end of the network. The
    previous version took the last n layers, and on this model that is the box *decode*
    arithmetic — Add/Sub/Div/Mul over values the convolutions already produced. At the
    documented n=10 it pinned six elementwise ops and reached no convolution at all;
    the last conv in the head sits at index 230 of 299, so counting back would have to
    reach n=69 before touching one. It could never have done what it was for.

    OBEY_PRECISION_CONSTRAINTS makes the build fail outright if TensorRT cannot honour
    what was asked, which beats it quietly choosing another precision while we believe
    the constraint took.

    Measured on yolov8n over the 500-image set, calibrated on 500 train2017 images:
    INT8 alone scores mAP50-95 0.2898 against the FP32 baseline's 0.4008, and pinning
    the head brings it to 0.3519 — a bit over half the loss recovered, for 51 layers out
    of 299 left in FP16.
    """
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Never retype a tensor the engine hands back. common.postprocess de-letterboxes in
    # whatever dtype reaches it, and in float16 that puts boxes about 1 px out on a
    # 1080p frame. On yolov8n this guard never fires — the head ends in a concatenation,
    # which the skip list below already excludes, and all three engines were checked to
    # emit FLOAT. It is here for the model or skip list that does not end that way.
    net_outputs = {network.get_output(i).name for i in range(network.num_outputs)}

    forced = kept_fp32_out = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not any(pre in layer.name for pre in prefixes):
            continue
        # Layers that work on shapes and indices rather than values: forcing those to
        # FP16 gains nothing and often fails the build under OBEY_PRECISION_CONSTRAINTS.
        if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT,
                          trt.LayerType.CONCATENATION, trt.LayerType.GATHER,
                          trt.LayerType.SLICE, trt.LayerType.SHUFFLE):
            continue
        layer.precision = trt.float16
        for j in range(layer.num_outputs):
            if layer.get_output(j).name in net_outputs:
                kept_fp32_out += 1
                continue
            layer.set_output_type(j, trt.float16)
        forced += 1

    msg = (f"[mixed] pinned {forced} {'/'.join(prefixes)} layers to FP16 "
           f"({network.num_layers} in the network)")
    # Only claimed when it actually happened. On yolov8n the head's last layer is a
    # concatenation, which is skipped anyway, so nothing needs holding back — but a
    # model whose head ends in a compute layer would, and the engine has to keep
    # returning FP32 either way.
    if kept_fp32_out:
        msg += f", holding {kept_fp32_out} network output(s) at FP32"
    print(msg)
    if forced == 0:
        raise SystemExit(
            f"matched no layers on {'/'.join(prefixes)}\n"
            f"TensorRT takes layer names from the ONNX, so a different model or a "
            f"re-export that renames modules needs a different prefix. A block that is "
            f"only concat/upsample also lands here — it has no layer worth pinning."
        )
    return forced


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
    # The calib set must not overlap the eval set, or INT8 mAP flatters itself by
    # calibrating on the very images it is scored against. They are already separate
    # COCO splits: calib from data/train_pool (train2017), eval from data/val500
    # (val2017).
    ap.add_argument("--calib-dir", default=None)
    # Defaults to None and gets a fingerprinted name in build(). The old generic name
    # ("calibration.cache") let different models share dynamic ranges with nothing to
    # flag it.
    ap.add_argument("--calib-cache", default=None)
    ap.add_argument("--calib-num", type=int, default=500)
    # Always clamped down to the engine's max batch in build(); reasoning is there.
    ap.add_argument("--calib-batch", type=int, default=8)
    # Was a layer count, which selected the wrong layers entirely — see _force_fp16_layers.
    ap.add_argument("--fp16-head", action="store_true",
                    help="pin the detect head to FP16 (INT8 only, when INT8 costs too much mAP)")
    # The sweep handle: --fp16-prefix /model.4/ leaves that block in FP16 and everything
    # else in INT8, so the mAP delta against a plain INT8 build is that block's share of
    # the loss. Repeatable, because the useful follow-up is pinning the worst few together.
    ap.add_argument("--fp16-prefix", action="append", default=None, metavar="PREFIX",
                    help="pin layers whose name contains PREFIX to FP16 (INT8 only, repeatable)")
    ap.add_argument("--detailed-layers", action="store_true",
                    help="keep full per-layer info in the engine for inspection (larger engine)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

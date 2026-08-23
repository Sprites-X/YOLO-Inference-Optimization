from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
# cuda.cudart ถูก deprecate ย้ายไป cuda.bindings.runtime — เครื่องนี้ cuda-python 12.9.7
# ใช้ตัวใหม่ได้ fallback ไว้เผื่อเครื่องอื่นที่ยังเป็นรุ่นเก่า (benchmark.py ทำเหมือนกัน)
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

sys.path.insert(0, str(Path(__file__).parent))
from common import IMG_SIZE, preprocess  

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _check(err):
    # cuda-python คืน (error, value) เป็น tuple ไม่เหมือน C API ที่คืน error อย่างเดียว
    # แล้วส่ง value ผ่าน pointer — cudaMalloc คืน (err, ptr), cudaMemcpy คืน (err,)
    # ฟังก์ชันนี้เลยต้องรับทั้งสองแบบแล้วคืนเฉพาะ value ออกไป
    if isinstance(err, tuple):
        err, *rest = err
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")
        return rest[0] if len(rest) == 1 else rest
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def _calib_fingerprint(onnx_path, calib_dir, size, num, batch) -> str:
    # cache เก็บแค่ dynamic range ดิบ ไม่มีข้อมูลว่าสร้างมาจากอะไร เลยต้องผูก
    # ลายนิ้วมือของเงื่อนไขไว้เอง — เปลี่ยนโมเดล/ชุดภาพ/ขนาด/จำนวน แล้ว range เดิมใช้ไม่ได้
    key = "|".join([str(Path(onnx_path).resolve()), str(Path(calib_dir).resolve()),
                    str(size), str(num), str(batch)])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _check_calib_cache(cache_path: str, fingerprint: str, describe: str) -> None:
    # ต้องเรียกก่อนส่ง calibrator ให้ TRT — ตรวจในคอลแบ็กของ calibrator ไม่ได้
    # เพราะ TRT เรียกจาก C++ แล้ว exception ของ Python ไม่ทะลุออกมา
    if not os.path.exists(cache_path):
        return
    meta_path = cache_path + ".meta.json"
    if not os.path.exists(meta_path):
        # cache ที่สร้างก่อนมีระบบ meta — ตรวจที่มาไม่ได้ ได้แค่เตือน
        print(f"[calib] เตือน: {cache_path} ไม่มี .meta.json คู่กัน ตรวจที่มาไม่ได้ "
              f"— ถ้าไม่แน่ใจให้ลบทิ้งแล้ว calibrate ใหม่")
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"cache {cache_path} สร้างจากเงื่อนไขคนละชุด\n"
            f"  ในไฟล์    : {meta.get('describe')}\n"
            f"  ที่ขอตอนนี้ : {describe}\n"
            f"dynamic range ใช้ข้ามโมเดล/ขนาดภาพ/ชุด calib ไม่ได้ "
            f"— ลบไฟล์นี้ทิ้ง หรือส่ง --calib-cache ชื่ออื่น"
        )


# --------------------------------------------------------------------------
# IInt8EntropyCalibrator2 ถูก deprecate ตั้งแต่ TRT 10 แล้ว (ทางใหม่คือ explicit
# quantization ด้วย Q/DQ node ใน ONNX) แต่ยังใช้ตัวนี้เพราะเป็น post-training
# calibration ที่ไม่ต้องแก้ ONNX — และเป็นเหตุผลหนึ่งที่ requirements pin
# tensorrt-cu12==10.16.1.11 ไว้ (NOTES ปัญหา 4: pip ไล่ไปเจอ TRT 11 ซึ่งอาจถอด API นี้แล้ว)
#
# เลือก Entropy2 ไม่ใช่ MinMax เพราะ Entropy2 เป็นค่าแนะนำสำหรับ CNN
# MinMax ไวต่อ outlier ในภาพ calibration มากกว่า
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
            raise FileNotFoundError(f"ไม่เจอรูปใน {calib_dir}")
        # shuffle ก่อนตัด num_images เพราะ sorted() จะได้ COCO id เรียงกัน ซึ่งไม่ได้
        # กระจายตามชนิดภาพ — seed คงที่เพื่อให้ทุก build ใช้ชุดเดิม ไม่งั้น INT8 mAP
        # ที่วัดได้จะขยับเพราะ calibration set เปลี่ยน ไม่ใช่เพราะโค้ดเปลี่ยน
        random.Random(seed).shuffle(files) 
        self.files = files[:num_images]
        self.index = 0

        # 4 = ขนาด float32 ต่อค่า (input ของ engine เป็น FP32 เสมอแม้จะ build INT8
        # เพราะ quantize เกิดข้างในกราฟ ไม่ใช่ที่ขา input)
        nbytes = batch_size * 3 * size * size * 4
        self.d_input = _check(cudart.cudaMalloc(nbytes))
        self.nbytes = nbytes
        print(f"[calib] ใช้ {len(self.files)} รูป, batch {batch_size} "
              f"-> {(len(self.files) + batch_size - 1) // batch_size} รอบ")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # คืน None = บอก TRT ว่าหมดแล้ว ไม่ใช่ error
        if self.index >= len(self.files):
            return None
        import cv2

        chunk = self.files[self.index:self.index + self.batch_size]
        arrs = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # ต้องใช้ common.preprocess ตัวเดียวกับตอน inference — ถ้า calibrate ด้วย
            # การ preprocess คนละแบบ dynamic range ที่ได้จะไม่ตรงกับข้อมูลจริงที่โมเดลเจอ
            arrs.append(preprocess(img, self.size)[0])
        if not arrs:
            self.index += self.batch_size
            return self.get_batch(names)

        # เติมก้อนสุดท้ายให้ครบ batch ด้วยภาพแรกซ้ำ เพราะ TRT อ่านบัฟเฟอร์เต็มขนาดเสมอ
        # NOTE: ทำให้ histogram ของภาพนั้นถูกนับเกินจริง มากสุด batch_size-1 ครั้ง
        # (7 จาก 500 ที่ค่า default) ยังไม่ได้วัดว่ากระทบ mAP แค่ไหน น่าจะน้อยมาก
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
        # cache ทำให้ build ซ้ำเร็วขึ้นมาก เพราะข้าม forward pass 500 ภาพไปเลย
        #
        # แต่เดิมชื่อ default เป็น "calibration.cache" กลางๆ ทำให้ build โมเดลที่สอง
        # ในโฟลเดอร์เดียวกันหยิบ dynamic range ของโมเดลแรกมาใช้ แล้วพิมพ์ว่า
        # "ข้าม calibration" เหมือนทุกอย่างปกติ — mAP พังโดยไม่มี error
        # ตอนนี้ชื่อ default ผูกกับลายนิ้วมือแล้ว และเทียบ .meta.json คู่กันซ้ำอีกชั้น
        # เผื่อกรณีที่ส่ง --calib-cache มาเอง
        if not os.path.exists(self.cache_path):
            return None

        # ความไม่ตรงของลายนิ้วมือถูกตรวจไปแล้วใน _check_calib_cache() ตอน build()
        # ห้ามย้ายมาตรวจตรงนี้: TRT เรียกเมธอดนี้จากฝั่ง C++ แล้วกลืน exception ของ
        # Python ทิ้ง — raise ที่นี่จะไม่หยุด build แต่จะกลายเป็นว่า TRT คิดว่าไม่มี
        # cache แล้ว calibrate ใหม่ทับไฟล์เดิม (ทดสอบแล้วเป็นแบบนั้นจริง)
        print(f"[calib] เจอ cache เดิม {self.cache_path} — ข้าม calibration")
        with open(self.cache_path, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        with open(self.cache_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint, "describe": self.describe},
                      f, ensure_ascii=False, indent=2)
        print(f"[calib] เขียน cache -> {self.cache_path}")

    def free(self):
        if getattr(self, "d_input", None):
            cudart.cudaFree(self.d_input)
            self.d_input = None


# --------------------------------------------------------------------------
def build(args):
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH ถูก deprecate ใน TRT 10 (network เป็น explicit batch เสมอแล้ว)
    # ยังส่งอยู่เพื่อให้อ่านออกว่าตั้งใจ และเผื่อรันบน TRT รุ่นเก่ากว่า
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            # ต้องวน num_errors เอง — parser เก็บ error ไว้ข้างในแล้วคืนแค่ False
            # ถ้าไม่พิมพ์ออกมาจะเหลือแค่ "parse ไม่ผ่าน" ที่ debug ต่อไม่ได้
            for i in range(parser.num_errors):
                print(f"[onnx] {parser.get_error(i)}", file=sys.stderr)
            raise RuntimeError("parse ONNX ไม่ผ่าน")

    config = builder.create_builder_config()
    # workspace คือเพดานที่ TRT ใช้ลอง kernel ตอน autotune ไม่ใช่หน่วยความจำที่
    # engine จะกินตอนรัน — ให้น้อยไปแปลว่า kernel เร็วบางตัวถูกตัดออกจากตัวเลือกเงียบๆ
    # << 30 = GB -> bytes
    #
    # MemoryPoolType ไม่ใช่ MemoryPoolFlag — ชื่อหลังไม่มีอยู่ใน tensorrt 10.16.1.11
    # (เคยเขียนผิดไว้ ทำให้ build_engine.py โยน AttributeError ตั้งแต่บรรทัดนี้
    #  ทุก precision ก่อนจะไปถึง calibration ด้วยซ้ำ)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)

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
        # เปิด FP16 คู่กับ INT8 ด้วยเสมอ: แฟล็ก INT8 ไม่ได้บังคับให้ทุก layer เป็น INT8
        # layer ที่ quantize แล้วช้าลงหรือแม่นยำตกหนัก TRT จะเลือก precision อื่นให้
        # ถ้าไม่เปิด FP16 ทางเลือกสำรองเหลือแค่ FP32 ซึ่งช้ากว่าที่ควร
        config.set_flag(trt.BuilderFlag.FP16)

        if not args.calib_dir:
            raise SystemExit(
                "INT8 ต้องมี --calib-dir\n"
                "ถ้าไม่ให้รูป TensorRT จะเดา dynamic range เอง แล้ว mAP จะพังโดยไม่มี error"
            )
        # TRT calibrate ที่ batch ซึ่ง optimization profile ยอมให้ ไม่ใช่ที่
        # get_batch_size() ของ calibrator — ถ้า calibrator ส่งมา 8 ภาพแต่ profile
        # max เป็น 1 มันจะอ่านแค่ภาพแรกของทุกก้อนแล้วทิ้งที่เหลือเงียบๆ
        # (--calib-batch 8 กับ --max-batch 1 = calibrate ด้วย 63 ภาพจาก 500)
        #
        # วัดจริงบน TRT 10.16.1.11 ด้วยโมเดลจิ๋ว: ทำให้ช่อง 1-7 ของทุกก้อนมีค่า
        # ต่างกัน 50 เท่าแล้วดูว่า calibration cache เปลี่ยนไหม
        #   max_batch=1 calib_batch=8              -> cache เหมือนเดิม (อ่านแค่ภาพแรก)
        #   max_batch=8 calib_batch=8              -> cache ต่าง (อ่านครบ)
        #   max_batch=8 calib_batch=8 + calib prof 1 -> cache เหมือนเดิม
        # ข้อสุดท้ายคือเหตุผลที่ไม่ใช้ config.set_calibration_profile() แก้ —
        # มันไม่ได้ช่วย และตั้งผิดยิ่งทำให้แย่ลง ตัวที่กำหนดจริงคือ max batch
        calib_batch = args.calib_batch
        if calib_batch > engine_max_batch:
            print(f"[calib] ลด --calib-batch {calib_batch} -> {engine_max_batch} "
                  f"ให้เท่า max batch ของ engine ไม่งั้น TRT จะอ่านแค่ภาพแรกของทุกก้อน "
                  f"(ใช้จริง {args.calib_num} ภาพเท่าเดิม แค่แบ่งเป็นก้อนเล็กลง)")
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

        if args.fp16_head:
            _force_fp16_output_layers(network, config, n=args.fp16_head)

    out = args.engine or str(
        Path(args.onnx).with_suffix("").name + f"_{args.precision}.engine"
    )

    # ชื่อไฟล์ต้องมี _fp16/_int8 ติดไป เพราะ benchmark.py อ่าน precision จากชื่อไฟล์
    # (TensorRTRunner._detect_precision) — engine ไม่ได้บอก precision ในตัวมันเอง
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
    # ใช้เมื่อ INT8 ทำ mAP ตกหนัก: หัว detect ทำ box regression ซึ่งไวต่อ quantization
    # มากกว่าชั้น conv ทั่วไป บังคับให้ layer ท้ายๆ อยู่ FP16 มักคืน mAP มาได้
    # โดยเสียความเร็วไม่มาก
    #
    # OBEY_PRECISION_CONSTRAINTS ทำให้ build "ล้มเหลว" ถ้าทำตามที่สั่งไม่ได้
    # ซึ่งดีกว่าปล่อยให้ TRT เงียบๆ เลือก precision อื่นแล้วเราเข้าใจผิดว่าบังคับสำเร็จ
    #
    # TODO: ยังไม่เคยรันจริง (ยังไม่ถึง Phase 3) สองอย่างที่ต้องเช็กตอนใช้ครั้งแรก
    #   1. set_output_type(float16) ที่ layer สุดท้ายอาจทำให้ engine คาย output เป็น
    #      FP16 → common.postprocess จะ de-letterbox ด้วย float16 แล้วกล่องคลาด ~1 px
    #      (ดู NOTE ท้าย common.postprocess) ถ้าเป็นแบบนั้นต้อง cast ก่อน postprocess
    #   2. n=10 เป็นค่าที่เดาเอา ยังไม่รู้ว่าหัว detect ของ yolov8n กินกี่ layer จริง
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    total = network.num_layers
    count = 0
    for i in range(max(0, total - n), total):
        layer = network.get_layer(i)
        # ข้าม layer ที่ทำงานกับ shape/index ไม่ใช่ค่าจริง — บังคับพวกนี้เป็น FP16
        # ไม่ได้ประโยชน์ และมักทำให้ build ล้มเพราะ OBEY_PRECISION_CONSTRAINTS
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
    # TODO: calib set ต้องไม่ทับกับ eval set ไม่งั้น INT8 mAP จะดูดีเกินจริง
    # เพราะ calibrate ด้วยภาพเดียวกับที่ใช้วัด — ยังไม่ได้แบ่ง ดู open items ใน NOTES.md
    # (แผนคือ commit eval_list.txt / calib_list.txt แล้วให้สคริปต์อ่านจากไฟล์)
    ap.add_argument("--calib-dir", default=None)
    # default เป็น None แล้วไปตั้งชื่อจากลายนิ้วมือใน build() — ชื่อกลางๆ แบบเดิม
    # ("calibration.cache") ทำให้โมเดลคนละตัวใช้ dynamic range ทับกันโดยไม่มีอะไรฟ้อง
    ap.add_argument("--calib-cache", default=None)
    ap.add_argument("--calib-num", type=int, default=500)
    # ถูก clamp ลงให้ไม่เกิน max batch ของ engine ใน build() เสมอ ดูเหตุผลตรงนั้น
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--fp16-head", type=int, default=0,
                    help="จำนวน layer ท้ายที่บังคับเป็น FP16 (ลองใช้เมื่อ INT8 mAP ตกมาก)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations
 
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
 
REPORT = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": {}, "versions": {}}
FAILURES: list[str] = []
WARNINGS: list[str] = []
 
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
 
 
def ok(name, detail=""):
    print(f"  {GREEN}PASS{RESET}  {name} {DIM}{detail}{RESET}")
    REPORT["checks"][name] = {"status": "pass", "detail": detail}
 
 
def fail(name, detail=""):
    print(f"  {RED}FAIL{RESET}  {name} {DIM}{detail}{RESET}")
    REPORT["checks"][name] = {"status": "fail", "detail": detail}
    FAILURES.append(name)
 
 
def warn(name, detail=""):
    print(f"  {YELLOW}WARN{RESET}  {name} {DIM}{detail}{RESET}")
    REPORT["checks"][name] = {"status": "warn", "detail": detail}
    WARNINGS.append(name)
 
 
def sh(cmd: str) -> str | None:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        return None
 
 
# --------------------------------------------------------------------------
def check_driver():
    print("\n[1] NVIDIA driver / GPU")
    out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,"
             "power.limit,clocks.max.sm --format=csv,noheader")
    if not out:
        fail("nvidia-smi", "เรียกไม่ได้ — ไม่มี driver หรือไม่มี GPU")
        return
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    REPORT["versions"]["gpu_name"] = parts[0]
    REPORT["versions"]["driver"] = parts[1]
    REPORT["versions"]["vram_total"] = parts[2]
    ok("nvidia-smi", " | ".join(parts))
 
    try:
        major = int(parts[1].split(".")[0])
        if major < 570:
            fail("driver >= 570", f"เจอ {parts[1]} — Blackwell ต้อง 570 ขึ้นไป")
        else:
            ok("driver >= 570", parts[1])
    except ValueError:
        warn("driver version", f"parse ไม่ได้: {parts[1]}")
 
 
def check_torch():
    print("\n[2] PyTorch + CUDA kernel")
    try:
        import torch
    except ImportError:
        fail("import torch", "ยังไม่ได้ติดตั้ง")
        return
 
    REPORT["versions"]["torch"] = torch.__version__
    REPORT["versions"]["torch_cuda"] = torch.version.cuda
    ok("torch", f"{torch.__version__} (built for CUDA {torch.version.cuda})")
 
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available()", "False")
        return
 
    cap = torch.cuda.get_device_capability()
    REPORT["versions"]["compute_capability"] = f"{cap[0]}.{cap[1]}"
    if cap == (12, 0):
        ok("compute capability", "(12, 0) = sm_120 Blackwell")
    else:
        warn("compute capability", f"{cap} — guideline นี้เขียนสำหรับ (12, 0)")

    try:
        a = torch.randn(512, 512, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        ok("รัน CUDA kernel จริง", f"matmul สำเร็จ (checksum {b:.1f})")
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        fail("รัน CUDA kernel จริง", msg)
        if "no kernel image" in msg:
            print(f"       {DIM}-> ลง PyTorch ผิด build: "
                  f"pip install torch --index-url https://download.pytorch.org/whl/cu128{RESET}")
        return

    try:
        h = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        (h @ h).sum().item()
        torch.cuda.synchronize()
        ok("FP16 matmul", "ใช้ได้")
    except RuntimeError as e:
        warn("FP16 matmul", str(e).split("\n")[0])
 
 
def check_onnxruntime():
    print("\n[3] ONNX Runtime (จุดที่ fallback เงียบบ่อยที่สุด)")
    try:
        import onnxruntime as ort
    except ImportError:
        fail("import onnxruntime", "ยังไม่ได้ติดตั้ง onnxruntime-gpu")
        return
 
    REPORT["versions"]["onnxruntime"] = ort.__version__
    avail = ort.get_available_providers()
    ok("onnxruntime", f"{ort.__version__} | providers: {avail}")
 
    if "CUDAExecutionProvider" not in avail:
        fail("CUDAExecutionProvider ใน build", "wheel นี้เป็น CPU-only — ต้องลง onnxruntime-gpu")
        return
 
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper
 
        node = helper.make_node("Add", ["x", "y"], ["z"])
        graph = helper.make_graph(
            [node], "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [64, 64]),
             helper.make_tensor_value_info("y", TensorProto.FLOAT, [64, 64])],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [64, 64])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
 
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        used = sess.get_providers()
        if "CUDAExecutionProvider" in used:
            x = np.ones((64, 64), np.float32)
            sess.run(None, {"x": x, "y": x})
            ok("ORT ใช้ CUDA จริง", str(used))
        else:
            fail("ORT ใช้ CUDA จริง",
                 f"ตกไป {used} — แถว 'ONNX Runtime GPU' ในตารางจะเป็นตัวเลข CPU")
    except Exception as e:
        fail("ORT smoke test", str(e).split("\n")[0])
 
 
def check_tensorrt():
    print("\n[4] TensorRT")
    try:
        import tensorrt as trt
    except ImportError:
        fail("import tensorrt", "ยังไม่ได้ติดตั้ง")
        return
 
    REPORT["versions"]["tensorrt"] = trt.__version__
    parts = [int(p) for p in trt.__version__.split(".")[:2]]
    if (parts[0], parts[1]) >= (10, 8):
        ok("tensorrt >= 10.8", trt.__version__)
    else:
        fail("tensorrt >= 10.8", f"เจอ {trt.__version__} — Blackwell ต้อง 10.8 ขึ้นไป")
 
    if parts[0] >= 10:
        ok("TensorRT 10 API", "ใช้ execute_async_v3 + set_tensor_address (ไม่ใช่ execute_v2)")
 
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        trt.Builder(logger)
        ok("สร้าง trt.Builder", "ได้")
    except Exception as e:
        fail("สร้าง trt.Builder", str(e).split("\n")[0])
 
    try:
        try:
            from cuda.bindings import runtime as cudart  
            api = "cuda.bindings.runtime"
        except ImportError:
            from cuda import cudart  
            api = "cuda.cudart (เก่า)"
        try:
            from importlib.metadata import version
            ver = version("cuda-python")
        except Exception:
            ver = "unknown"
        REPORT["versions"]["cuda_python"] = ver
        ok("cuda-python", f"{ver} via {api}")
    except ImportError:
        fail("cuda-python", "pip install cuda-python — สคริปต์ชุดนี้ใช้ cudart ไม่ใช่ pycuda")
 
    if sh("which trtexec"):
        ok("trtexec ใน PATH", sh("trtexec --version | head -1") or "")
    else:
        warn("trtexec ใน PATH",
             "หาไม่เจอ — ปกติอยู่ที่ /usr/src/tensorrt/bin หรือใน site-packages/tensorrt_libs")
 
 
def check_misc():
    print("\n[5] ไลบรารีอื่น + สภาพเครื่อง")
    for mod in ("ultralytics", "cv2", "numpy", "pycocotools"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None:                      
                from importlib.metadata import version as _v
                dist = {"cv2": "opencv-python"}.get(mod, mod)
                try:
                    v = _v(dist)
                except Exception:
                    v = "unknown"
            REPORT["versions"][mod] = v
            ok(mod, v)
        except ImportError:
            (warn if mod == "pycocotools" else fail)(mod, "ยังไม่ได้ติดตั้ง")
 
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    REPORT["versions"]["cpu_governor"] = gov
    if gov == "performance":
        ok("CPU governor", gov)
    elif gov:
        warn("CPU governor", f"'{gov}' — baseline CPU จะช้ากว่าจริง "
                             f"(sudo cpupower frequency-set -g performance)")
    else:
        warn("CPU governor", "อ่านไม่ได้")
 
    pstate = sh("nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader")
    if pstate:
        ok("GPU state ตอนนี้", pstate)
 
    REPORT["versions"]["os"] = platform.platform()
    REPORT["versions"]["python"] = sys.version.split()[0]
    REPORT["versions"]["cpu"] = (sh("lscpu | grep 'Model name' | cut -d: -f2") or "").strip()
    ok("system", f"{REPORT['versions']['cpu']} | py{REPORT['versions']['python']}")
 
 
def main():
    print(f"\n{'='*68}\n  Phase 0 — Environment Verification\n{'='*68}")
 
    if sys.prefix == sys.base_prefix:
        print(f"\n  {RED}ไม่ได้อยู่ใน venv{RESET} — รัน `source .venv/bin/activate` ก่อน")
        print(f"  {DIM}ตอนนี้ python คือ {sys.executable}{RESET}\n")
        return 1
    print(f"  {DIM}venv: {sys.prefix}{RESET}")
 
    pypath = os.environ.get("PYTHONPATH", "")
    if pypath:
        entries = [p for p in pypath.split(":") if p]
        warn("PYTHONPATH ว่าง", f"เจอ {len(entries)} รายการ: {entries[0]}"
                                f"{' ...' if len(entries) > 1 else ''}")
        print(f"       {DIM}-> รัน `env -u PYTHONPATH .venv/bin/pip freeze > "
              f"requirements.lock.txt` เพื่อไม่ให้ lock ปน{RESET}")
    else:
        ok("PYTHONPATH ว่าง", "ไม่มีอะไร shadow venv")
 
    check_driver()
    check_torch()
    check_onnxruntime()
    check_tensorrt()
    check_misc()
 
    with open("env_report.json", "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False)
 
    print(f"\n{'='*68}")
    if FAILURES:
        print(f"{RED}  {len(FAILURES)} FAIL{RESET} — อย่าเพิ่งไป Phase 1")
        for f_ in FAILURES:
            print(f"    - {f_}")
    else:
        print(f"{GREEN}  ผ่านทุกข้อ{RESET} — เริ่ม Phase 1 ได้")
    if WARNINGS:
        print(f"{YELLOW}  {len(WARNINGS)} WARN{RESET} — วัดได้ แต่ตัวเลขอาจแกว่ง")
    print(f"  บันทึกลง env_report.json (แนบใน README)\n{'='*68}\n")
    return 1 if FAILURES else 0
 
 
if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations
 
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
 
# Gate that runs before any measurement, writing env_report.json to go alongside the
# README. Benchmark numbers mean nothing without knowing what they were measured on.
#
# Every check here came from a problem hit during setup, not from checking things just
# in case. Where each one came from is noted at the check itself.
REPORT = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": {}, "versions": {}}
FAILURES: list[str] = []
WARNINGS: list[str] = []
 
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
 
 
# FAIL = do not measure, the results will be wrong.
# WARN = you can measure, but numbers may wobble or not compare across machines.
# Two levels rather than one, because if everything is a FAIL people start skipping
# all of it.
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
 
 
# Swallowing every exception is deliberate: a diagnostic script that dies when a tool
# is missing is useless. Every check has to end in a FAIL or WARN, never a traceback.
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
        fail("nvidia-smi", "could not run it — no driver, or no GPU")
        return
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    REPORT["versions"]["gpu_name"] = parts[0]
    REPORT["versions"]["driver"] = parts[1]
    REPORT["versions"]["vram_total"] = parts[2]
    ok("nvidia-smi", " | ".join(parts))
 
    try:
        # 570 is the first driver with Blackwell (sm_120) support; this machine has 595.84.
        major = int(parts[1].split(".")[0])
        if major < 570:
            fail("driver >= 570", f"found {parts[1]} — Blackwell needs 570 or newer")
        else:
            ok("driver >= 570", parts[1])
    except ValueError:
        warn("driver version", f"could not parse: {parts[1]}")
 
 
def check_torch():
    print("\n[2] PyTorch + CUDA kernel")
    try:
        import torch
    except ImportError:
        fail("import torch", "not installed")
        return
 
    REPORT["versions"]["torch"] = torch.__version__
    REPORT["versions"]["torch_cuda"] = torch.version.cuda
    ok("torch", f"{torch.__version__} (built for CUDA {torch.version.cuda})")
 
    # is_available() can be True while nothing actually works — it only checks that a
    # driver and CUDA runtime exist, not that the installed wheel ships sm_120 kernels.
    # Hence the real matmul below, watching for "no kernel image" as the tell.
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available()", "False")
        return
 
    cap = torch.cuda.get_device_capability()
    REPORT["versions"]["compute_capability"] = f"{cap[0]}.{cap[1]}"
    if cap == (12, 0):
        ok("compute capability", "(12, 0) = sm_120 Blackwell")
    else:
        warn("compute capability", f"{cap} — these scripts were checked against (12, 0)")

    try:
    # This is the check that actually catches "wrong torch build": a wheel built for
    # sm_90 imports fine and passes is_available(), then every kernel fails on call.
    # The error reads "no kernel image is available for execution", which does not tell
    # you to install cu128 — so the right command is printed underneath.
        a = torch.randn(512, 512, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        ok("real CUDA kernel", f"matmul succeeded (checksum {b:.1f})")
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        fail("real CUDA kernel", msg)
        if "no kernel image" in msg:
            print(f"       {DIM}-> wrong PyTorch build: "
                  f"pip install torch --index-url https://download.pytorch.org/whl/cu128{RESET}")
        return

    try:
        h = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        (h @ h).sum().item()
        torch.cuda.synchronize()
        ok("FP16 matmul", "works")
    except RuntimeError as e:
        warn("FP16 matmul", str(e).split("\n")[0])
 
 
# The newest torch ships with cu130 while the onnxruntime-gpu release is still CUDA
# 12.x, so the whole stack was held back to cu128. This check is what catches someone
# cloning the repo and installing a newer torch over the top — ORT would quietly fall
# back to CPU.
def check_onnxruntime():
    print("\n[3] ONNX Runtime (most common place for a silent fallback)")
    try:
        import onnxruntime as ort
    except ImportError:
        fail("import onnxruntime", "onnxruntime-gpu not installed")
        return
 
    REPORT["versions"]["onnxruntime"] = ort.__version__
    avail = ort.get_available_providers()
    ok("onnxruntime", f"{ort.__version__} | providers: {avail}")
 
    # get_available_providers() only says what this wheel was built with, not which
    # provider a real session would pick — hence the actual model test below.
    if "CUDAExecutionProvider" not in avail:
        fail("CUDAExecutionProvider in build", "this wheel is CPU-only — install onnxruntime-gpu")
        return
 
    try:
        import numpy as np
        from onnx import TensorProto, helper
 
        node = helper.make_node("Add", ["x", "y"], ["z"])
        graph = helper.make_graph(
            [node], "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [64, 64]),
             helper.make_tensor_value_info("y", TensorProto.FLOAT, [64, 64])],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [64, 64])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        # Pin ir_version to 9: an onnx package newer than ORT can write an ir_version
        # the installed ORT does not support, and the smoke test would then fail for
        # the wrong reason.
        model.ir_version = 9
 
        # Ask for CUDAExecutionProvider alone, with no CPU fallback listed. Include CPU
        # and ORT quietly falls back to it, and this check passes when it should fail.
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        used = sess.get_providers()
        if "CUDAExecutionProvider" in used:
            x = np.ones((64, 64), np.float32)
            sess.run(None, {"x": x, "y": x})
            ok("ORT really on CUDA", str(used))
        else:
            fail("ORT really on CUDA",
                 f"fell back to {used} — the 'ONNX Runtime GPU' row would be CPU numbers")
    except Exception as e:
        fail("ORT smoke test", str(e).split("\n")[0])
 
 
def check_tensorrt():
    print("\n[4] TensorRT")
    try:
        import tensorrt as trt
    except ImportError:
        fail("import tensorrt", "not installed")
        return
 
    REPORT["versions"]["tensorrt"] = trt.__version__
    # 10.8 is the first TRT with Blackwell kernels. Below that a build still succeeds
    # but falls back to PTX JIT, which is slower and can give different results.
    # (tensorrt_libs/ does contain libnvinfer_builder_resource_sm120.so.10.16.1, so the
    #  builder has real sm_120 kernels and needs no JIT.)
    parts = [int(p) for p in trt.__version__.split(".")[:2]]
    if (parts[0], parts[1]) >= (10, 8):
        ok("tensorrt >= 10.8", trt.__version__)
    else:
        fail("tensorrt >= 10.8", f"found {trt.__version__} — Blackwell needs 10.8 or newer")
 
    if parts[0] >= 10:
        ok("TensorRT 10 API", "uses execute_async_v3 + set_tensor_address (not execute_v2)")
 
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        trt.Builder(logger)
        ok("create trt.Builder", "ok")
    except Exception as e:
        fail("create trt.Builder", str(e).split("\n")[0])
 
    try:
        try:
            # cuda.cudart is deprecated and moved to cuda.bindings.runtime.
            # build_engine.py and benchmark.py carry the same fallback.
            from cuda.bindings import runtime as cudart  
            api = "cuda.bindings.runtime"
        except ImportError:
            from cuda import cudart  
            api = "cuda.cudart (legacy)"
        try:
            from importlib.metadata import version
            ver = version("cuda-python")
        except Exception:
            ver = "unknown"
        # TODO: the version is only recorded, never checked against the < 13 that
        # requirements pins (torch cu128 pins cuda-bindings<13; left alone, pip installs
        # 13.x and they clash). This should be a FAIL, since it is the same condition
        # requirements.txt already enforces.
        REPORT["versions"]["cuda_python"] = ver
        ok("cuda-python", f"{ver} via {api}")
    except ImportError:
        fail("cuda-python", "pip install cuda-python — these scripts use cudart, not pycuda")
 
    # WARN rather than FAIL because trtexec does not ship with the pip wheel, only in
    # the GA tarball. (The GitHub source repo has no bin/ folder, so downloading the
    # wrong file leaves you without trtexec.) It is only used as a cross-check;
    # measurement works fine without it.
    #
    # Verified: trtexec reports v101601, matching the pip tensorrt-cu12 exactly
    # (10.16.1.11). That matters because an engine is tied to its TensorRT version — if
    # the numbers differ, trtexec cannot load engines built by these scripts.
    #
    # The GA tarball lives outside the repo and is not permanently on PATH. If you add
    # it, add only $TRT_ROOT/bin. Do not touch LD_LIBRARY_PATH: it would shadow the .so
    # files pip's tensorrt import relies on, and that currently works.
    #
    # --help rather than --version, because trtexec has no --version flag.
    if sh("which trtexec"):
        # Cut everything from " #" onward: trtexec echoes its own full path on the
        # first line, which is tied to the home directory of whoever ran it and would
        # ride along into the committed env_report.json. The version is all we need.
        ok("trtexec on PATH",
           sh("trtexec --help 2>&1 | head -1 | sed 's/ #.*//'") or "")
    else:
        warn("trtexec on PATH",
             "not found — usually in /usr/src/tensorrt/bin or site-packages/tensorrt_libs")
 
 
def check_misc():
    print("\n[5] Other libraries + machine state")
    for mod in ("ultralytics", "cv2", "numpy", "pycocotools"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            # Some cv2 builds have no __version__, and the dist name does not match
            # the module name, so ask importlib.metadata by package name instead.
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
            # pycocotools is only a WARN because it is used solely by evaluate.py;
            # benchmark.py, which is the main job, runs without it.
            (warn if mod == "pycocotools" else fail)(mod, "not installed")
 
    # A powersave governor makes the CPU baseline row slower than the hardware really
    # is, which flatters the GPU speedup. The setting only lasts until reboot, so it
    # has to be redone each time.
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    REPORT["versions"]["cpu_governor"] = gov
    if gov == "performance":
        ok("CPU governor", gov)
    elif gov:
        warn("CPU governor", f"'{gov}' — CPU baseline will read slower than reality "
                             f"(sudo cpupower frequency-set -g performance)")
    else:
        warn("CPU governor", "could not read it")
 
    pstate = sh("nvidia-smi --query-gpu=pstate,clocks.sm --format=csv,noheader")
    if pstate:
        ok("GPU state now", pstate)
 
    REPORT["versions"]["os"] = platform.platform()
    REPORT["versions"]["python"] = sys.version.split()[0]
    REPORT["versions"]["cpu"] = (sh("lscpu | grep 'Model name' | cut -d: -f2") or "").strip()
    ok("system", f"{REPORT['versions']['cpu']} | py{REPORT['versions']['python']}")
 
 
def main():
    print(f"\n{'='*68}\n  Environment Verification\n{'='*68}")
 
    # Two venv mistakes already made: installing into ~/.local without noticing, and
    # running the verify script from a terminal that was never activated. Versions then
    # appear to go backwards (ultralytics 8.4.126 -> 8.4.104, cv2 5.0.0 -> 4.13.0,
    # numpy 2.2.6 -> 1.26.4) with nothing saying anything is wrong. Hence a guard in
    # code rather than relying on memory — and it has to come before every other check.
    if sys.prefix == sys.base_prefix:
        print(f"\n  {RED}not inside the venv{RESET} — run `source .venv/bin/activate` first")
        print(f"  {DIM}python is currently {sys.executable}{RESET}\n")
        return 1
    print(f"  {DIM}venv: {sys.prefix}{RESET}")
 
    # ROS 2's setup.bash sets PYTHONPATH, which a venv does not strip, so pip freeze
    # pulls roughly 150 ROS packages into the lock file. Being "in a venv" does not
    # mean actually isolated.
    #
    # TODO: the check is named "PYTHONPATH clean" in both the passing and failing case,
    # so it prints "WARN PYTHONPATH clean — found 5 entries", which contradicts itself.
    # The names should differ.
    pypath = os.environ.get("PYTHONPATH", "")
    if pypath:
        entries = [p for p in pypath.split(":") if p]
        warn("PYTHONPATH clean", f"found {len(entries)} entries: {entries[0]}"
                                f"{' ...' if len(entries) > 1 else ''}")
        print(f"       {DIM}-> run `env -u PYTHONPATH .venv/bin/pip freeze > "
              f"requirements.lock.txt` to keep the lock file clean{RESET}")
    else:
        ok("PYTHONPATH clean", "nothing shadowing the venv")
 
    check_driver()
    check_torch()
    check_onnxruntime()
    check_tensorrt()
    check_misc()
 
    # TODO: nothing here catches nvidia-* wheels mixing cu12 and cu13 in one venv.
    # Right now that is only caught indirectly, through the ORT smoke test. To add it:
    # walk importlib.metadata for packages starting with nvidia- and check their cu
    # suffixes all agree.
    with open("env_report.json", "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False)
 
    print(f"\n{'='*68}")
    if FAILURES:
        print(f"{RED}  {len(FAILURES)} FAIL{RESET} — fix these before measuring anything")
        for f_ in FAILURES:
            print(f"    - {f_}")
    else:
        print(f"{GREEN}  all checks passed{RESET} — ready to measure")
    if WARNINGS:
        print(f"{YELLOW}  {len(WARNINGS)} WARN{RESET} — measurable, but numbers may wobble")
    print(f"  written to env_report.json (attach it in the README)\n{'='*68}\n")
    return 1 if FAILURES else 0
 
 
if __name__ == "__main__":
    sys.exit(main())

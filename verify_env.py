from __future__ import annotations
 
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
 
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
def check_nvidia_wheel_majors():
    """Every cu-suffixed nvidia wheel in this venv has to agree on the CUDA major.

    Two majors side by side means two wheels shipping the same sonames, and whichever
    the loader reaches first wins. Until this existed the condition was only caught
    indirectly, by the ONNX Runtime smoke test failing once the wrong one won — which
    points the blame at ONNX Runtime rather than at the install.

    Counts only what this interpreter can import. Packages in the user site
    (~/.local/lib) show up in `pip list` even from inside a venv, but are not on
    sys.path when include-system-site-packages is false, so they cannot affect a run
    and must not fail this check. On this machine that is the whole cu13 set.
    """
    import re
    from importlib.metadata import distributions

    by_major = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if not name.startswith(("nvidia-", "tensorrt")):
            continue
        # tensorrt_cu12 uses an underscore where nvidia-cublas-cu12 uses a dash.
        m = re.search(r"[-_]cu(\d+)", name)
        if m:
            by_major.setdefault(m.group(1), set()).add(name)

    if not by_major:
        warn("nvidia wheel CUDA majors", "no cu-suffixed nvidia wheels found to check")
        return
    if len(by_major) > 1:
        detail = "; ".join(
            f"cu{k}: {', '.join(sorted(v)[:3])}{' ...' if len(v) > 3 else ''}"
            for k, v in sorted(by_major.items()))
        fail("nvidia wheel CUDA majors",
             f"two CUDA majors importable in one venv — {detail}")
        return
    major, pkgs = next(iter(by_major.items()))
    REPORT["versions"]["nvidia_wheel_cuda_major"] = f"cu{major}"
    ok("nvidia wheel CUDA majors", f"all {len(pkgs)} importable wheels are cu{major}")


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
 
    # The floor depends on the card, so ask the card rather than assuming this one.
    # 570 is the first driver with Blackwell (sm_120) support and is what this machine
    # needs; demanding it everywhere would fail a perfectly good Ampere box on 535.
    # 525 is the CUDA 12.x minimum, which is what the cu128 wheels need below Blackwell.
    cap = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    cap_major = 0
    try:
        cap_major = int(float((cap or "0").splitlines()[0]))
    except (ValueError, IndexError):
        pass
    floor = 570 if cap_major >= 12 else 525
    why = "Blackwell (sm_120)" if cap_major >= 12 else f"CUDA 12.x on sm_{cap_major}x"
    try:
        major = int(parts[1].split(".")[0])
        if major < floor:
            fail(f"driver >= {floor}", f"found {parts[1]} — {why} needs {floor} or newer")
        else:
            ok(f"driver >= {floor}", f"{parts[1]} (floor set by {why})")
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
 
    # Run the session test in a subprocess that never imports torch.
    #
    # It used to run here, and passed — but only because check_torch() runs first and
    # torch dlopens ORT's CUDA dependencies (libcublasLt and friends, which pip leaves
    # in site-packages/nvidia/*/lib where the loader does not look) with RTLD_GLOBAL on
    # the way in. So this reported "ORT really on CUDA" while
    # `evaluate.py --runtime onnx --device cuda` could not get a CUDA session at all.
    # A gate that only passes because of what another check imported first is not
    # testing the machine.
    #
    # benchmark.preload_ort_cuda_deps() is what the ONNX runner itself calls, so this
    # checks the real path: no torch, ORT's libraries loaded the way production loads
    # them.
    try:
        r = subprocess.run([sys.executable, "-c", SUBPROCESS_ORT_PROBE], capture_output=True,
                           text=True, timeout=180,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    except Exception as e:
        fail("ORT smoke test", str(e).split("\n")[0])
        return
 
    if line.startswith("CUDA-OK"):
        ok("ORT really on CUDA", line[len("CUDA-OK "):] + " (no torch in that process)")
    elif line.startswith("CUDA-NO"):
        fail("ORT really on CUDA",
             f"fell back to {line[len('CUDA-NO '):]} — the 'ONNX Runtime GPU' row would be CPU numbers")
    else:
        fail("ORT smoke test", (r.stderr.strip().splitlines() or ["no output"])[-1][:200])
 
 
# Kept as source text rather than a function because it has to run in a process where
# torch was never imported; anything importing this module would drag torch in through
# the runners.
SUBPROCESS_ORT_PROBE = r"""
import sys

import numpy as np
from onnx import TensorProto, helper

from benchmark import preload_ort_cuda_deps

preload_ort_cuda_deps()
import onnxruntime as ort

# The whole point of this process: prove ORT finds CUDA on its own, without torch
# having dlopened its dependencies first.
assert "torch" not in sys.modules, "torch got imported — this probe would prove nothing"

node = helper.make_node("Add", ["x", "y"], ["z"])
graph = helper.make_graph(
    [node], "t",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [64, 64]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [64, 64])],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [64, 64])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
# Pin ir_version to 9: an onnx package newer than ORT can write an ir_version the
# installed ORT does not support, and the smoke test would then fail for the wrong
# reason.
model.ir_version = 9

# Ask for CUDAExecutionProvider alone, with no CPU fallback listed. Include CPU and ORT
# quietly falls back to it, and this check passes when it should fail.
sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
used = sess.get_providers()
if "CUDAExecutionProvider" in used:
    x = np.ones((64, 64), np.float32)
    sess.run(None, {"x": x, "y": x})
    print("CUDA-OK", used)
else:
    print("CUDA-NO", used)
"""
 
 
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
        REPORT["versions"]["cuda_python"] = ver
        # requirements.txt pins cuda-python<13, because torch cu128 pins
        # cuda-bindings<13 and pip left to itself installs 13.x, which clashes. Checked
        # here as well: an environment drifts after install, and a pin in a file nobody
        # re-runs is not a guarantee.
        major = ver.split(".")[0]
        if major.isdigit() and int(major) >= 13:
            fail("cuda-python < 13",
                 f"{ver} clashes with torch cu128 — pip install 'cuda-python<13'")
        else:
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
 
    # Activating the venv is not the same as the venv's `yolo` winning. Activation puts
    # .venv/bin on PATH, but does not guarantee it comes first: here ~/.local/bin sits
    # ahead of it, so bare `yolo` was ultralytics 8.4.104 while every import in this venv
    # gets 8.4.126. It cost a sweep — the export step ran from the wrong install, and on
    # a machine where that install merely works rather than crashes, the ONNX would have
    # been exported by one version while everything measuring it used another, silently.
    # run_all.sh now calls the entrypoint through python and does not depend on PATH;
    # this is a WARN for the README's step-by-step commands, which do run `yolo`.
    # Same shape as the yolo check below, and the one that costs more when it is wrong.
    # `pip` on PATH is not necessarily this interpreter's pip: with ~/.local/bin ahead of
    # .venv/bin, bare `pip` installs into ~/.local, which a venv does not read
    # (ENABLE_USER_SITE is false inside one). Everything appears to install and then
    # every import here fails, which reads as a broken install rather than a misdirected
    # one. `python -m pip` cannot miss, and is what the README tells you to use.
    venv_root = Path(sys.prefix).resolve()
    pip_cli = shutil.which("pip")
    if pip_cli and not Path(pip_cli).resolve().is_relative_to(venv_root):
        warn("pip on PATH", f"{pip_cli} is outside this venv — use `python -m pip` so "
                            f"installs land here and not in ~/.local")
    elif pip_cli:
        ok("pip on PATH", pip_cli)

    cli = shutil.which("yolo")
    if cli is None:
        warn("yolo CLI", "not on PATH — run_all.sh does not need it, the README's "
                         "individual export commands do")
    elif not Path(cli).resolve().is_relative_to(venv_root):
        # tail -1 because a mismatched install often prints an import traceback first.
        cli_ver = sh(f"'{cli}' version 2>&1 | tail -1") or "unreadable"
        warn("yolo CLI", f"{cli} is outside this venv (says {cli_ver!r}, venv has "
                         f"{REPORT['versions'].get('ultralytics')!r}) — running "
                         f"`yolo export` by hand would use that one")
    else:
        ok("yolo CLI", cli)

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
    # Named for the thing being checked, not for the outcome. It used to be called
    # "PYTHONPATH clean" in both branches and printed "WARN PYTHONPATH clean — found 5
    # entries", contradicting itself. One neutral name also keeps the key stable in
    # env_report.json, so two reports can be diffed.
    pypath = os.environ.get("PYTHONPATH", "")
    if pypath:
        entries = [p for p in pypath.split(":") if p]
        warn("PYTHONPATH", f"{len(entries)} entries shadowing the venv: {entries[0]}"
                           f"{' ...' if len(entries) > 1 else ''}")
        print(f"       {DIM}-> run `env -u PYTHONPATH .venv/bin/pip freeze > "
              f"requirements.lock.txt` to keep the lock file clean{RESET}")
    else:
        ok("PYTHONPATH", "clean — nothing shadowing the venv")
 
    check_driver()
    check_torch()
    check_onnxruntime()
    check_tensorrt()
    check_misc()
 
    check_nvidia_wheel_majors()

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

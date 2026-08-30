#!/usr/bin/env bash
# One-command setup from a fresh clone: venv, dependencies, datasets, then the gate.
# Safe to re-run — every step checks whether it is already done and skips if so.
#
# Deliberately not `set -u`: the venv's activate script reads $PYTHONPATH unguarded and
# dies under it. `set -e` still stops at the first real failure.
set -eo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
step () { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
have () { [ -e "$1" ]; }

step "Python"
$PY -c 'import sys; assert sys.version_info[:2] >= (3, 10), sys.version' \
    || { echo "need Python 3.10+; set PYTHON=/path/to/python3.10 and re-run"; exit 1; }
echo "  $($PY --version) at $(command -v $PY)"

step "Virtual environment (.venv)"
if have .venv/bin/activate; then
    echo "  already exists — leaving it alone"
else
    $PY -m venv .venv && echo "  created"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# python -m pip, never bare pip: with ~/.local/bin ahead of .venv/bin on PATH, bare pip
# installs into ~/.local, which a venv does not read. The install looks fine and then
# every import fails. `python -m pip` targets the interpreter that is running.
PIP="python -m pip"
$PIP install -q --upgrade pip >/dev/null

step "Dependencies"
if python -c 'import torch, tensorrt, ultralytics, pycocotools' 2>/dev/null; then
    echo "  already satisfied — skipping (delete .venv to force a reinstall)"
else
    # torch/torchvision are pinned to +cu128 builds that exist only on the PyTorch
    # index; plain PyPI returns 404 for them, so they cannot come from the lock alone.
    echo "  torch + torchvision (CUDA 12.8 build, ~2.5 GB)"
    $PIP install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    echo "  tensorrt"
    $PIP install --extra-index-url https://pypi.nvidia.com \
        tensorrt-cu12==10.16.1.11 tensorrt-cu12-libs==10.16.1.11 \
        tensorrt-cu12-bindings==10.16.1.11
    echo "  the rest, pinned"
    $PIP install -r requirements.lock.txt
fi

step "Measurement set — 500 images from COCO val2017"
if have data/val500 && [ "$(ls data/val500 | wc -l)" -ge 500 ]; then
    echo "  data/val500 already populated — skipping"
else
    if ! have data/val2017; then
        echo "  downloading val2017.zip (~780 MB)"
        curl -# -O http://images.cocodataset.org/zips/val2017.zip
        unzip -q val2017.zip -d data/ && rm -f val2017.zip
    fi
    if ! have annotations/instances_val2017.json; then
        echo "  downloading annotations (~250 MB)"
        curl -# -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -qj annotations_trainval2017.zip annotations/instances_val2017.json \
            -d annotations/ && rm -f annotations_trainval2017.zip
    fi
    python prepare_data.py --src data/val2017
fi

step "INT8 calibration pool — 2000 images from COCO train2017"
if have data/train_pool && [ "$(ls data/train_pool | wc -l)" -ge 2000 ]; then
    echo "  data/train_pool already populated — skipping"
else
    echo "  fetching ~315 MB, one file at a time"
    python fetch_train_pool.py
fi

step "Environment check"
python verify_env.py || {
    echo
    echo "verify_env reported failures. Fix those before measuring — the numbers from a"
    echo "half-working environment look plausible and are wrong."
    exit 1
}

cat <<'DONE'

==> Ready.

    source .venv/bin/activate
    ./run_all.sh --fresh

--fresh clears the measurements committed in this repo first, so the report is
yours and not a mix of both. Roughly 20 minutes.
DONE

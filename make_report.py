from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(p: str) -> list[dict]:
    # Return [] rather than blowing up when the file is missing — accuracy.jsonl often
    # does not exist yet right after a benchmark run, and you should still get the
    # timing table with the mAP column left blank.
    path = Path(p)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_of(r: dict) -> tuple:
    # Join key between results.jsonl and accuracy.jsonl. Batch is left out on purpose:
    # mAP does not depend on batch size, so the b1 and b8 rows can share one accuracy
    # value. (Must be spelled exactly as benchmark.py and evaluate.py write it — see
    # evaluate.py:record.)
    return (r.get("runtime"), r.get("precision"), r.get("device"))


def label_of(r: dict) -> str:
    b = r.get("batch", 1)
    suffix = f" b{b}" if b and b > 1 else ""
    return f"{r['runtime']}\n{r['precision']} {r['device']}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--accuracy", default="accuracy.jsonl")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    results = load_jsonl(args.results)
    accs = load_jsonl(args.accuracy)
    if not results:
        raise SystemExit(f"no data in {args.results} — run benchmark.py first")

    # NOTE: both files are append-only. Re-running the same config gives a duplicate
    # row in the table, and acc_map silently keeps the last one (the dict comprehension
    # overwrites). Delete the files before starting a fresh sweep.
    acc_map = {key_of(a): a for a in accs}
    outdir = Path(args.outdir)

    # ---------------- table ----------------
    hdr = ("| Runtime | Precision | Device | Batch | p50 (ms) | p99 (ms) | "
           "mean ± std (ms) | FPS | E2E (ms) | mAP50-95 | Size (MB) | VRAM (MB) |")
    sep = "|" + "---|" * 12
    lines = [hdr, sep]

    # precision == "FP32" is load-bearing, not decoration. The heading below states
    # "vs PyTorch GPU FP32", so the baseline has to actually be that row. Without the
    # check, running --half first puts an FP16 row earlier in the jsonl and every
    # speedup silently gets measured against FP16 under a label claiming FP32.
    baseline_fps = None
    for r in results:
        if (r["runtime"] == "PyTorch" and r["device"] == "GPU"
                and r["precision"] == "FP32" and r.get("batch", 1) == 1):
            baseline_fps = r["fps"]
            break

    # Slowest to fastest, so reading the table top to bottom tells the optimisation
    # story (PyTorch -> ONNX -> TRT FP16 -> TRT INT8) instead of making you hunt.
    for r in sorted(results, key=lambda x: -x["latency_ms_per_image"]["p50"]):
        L = r["latency_ms_per_image"]
        a = acc_map.get(key_of(r))
        vram = r.get("peak_vram_mb")
        cells = [
            r["runtime"],
            r["precision"],
            r["device"],
            str(r.get("batch", 1)),
            f"{L['p50']:.2f}",
            f"{L['p99']:.2f}",
            f"{L['mean']:.2f} ± {L['std_across_repeats']:.2f}",
            f"{r['fps']:.1f}",
            f"{r['end_to_end_ms']:.2f}",
            # "—" does not mean mAP is zero. It means evaluate.py has not been run for
            # this config, or key_of() does not match between the two files.
            f"{a['mAP50_95']:.4f}" if a else "—",
            f"{r['model_size_mb']:.1f}",
            # NOTE: this column is not comparable across rows — PyTorch reports
            # tensors only, TensorRT reports the whole card via nvidia-smi, and ONNX
            # reports nothing. Full reasoning in
            # benchmark.py TensorRTRunner.peak_vram_mb.
            f"{vram:.0f}" if vram else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)

    if baseline_fps:
        table += "\n\n**Speedup vs PyTorch GPU FP32 (batch 1):**\n\n"
        table += "| Config | Speedup |\n|---|---|\n"
        for r in sorted(results, key=lambda x: -x["fps"]):
            table += (f"| {r['runtime']} {r['precision']} {r['device']} "
                      f"b{r.get('batch',1)} | {r['fps']/baseline_fps:.2f}x |\n")

    (outdir / "report_table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\n-> {outdir/'report_table.md'}")

    # ---------------- figures ----------------
    # Imported here because matplotlib is optional. The table is the real output; the
    # figures are a bonus and should not take the whole script down when missing.
    try:
        import matplotlib
        # Must come before importing pyplot: benchmark machines usually have no
        # display, and without Agg pyplot hunts for a GUI backend and dies on import.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] no matplotlib — skipping figures (pip install matplotlib)")
        return

    rs = sorted(results, key=lambda x: x["latency_ms_per_image"]["p50"])
    labels = [label_of(r) for r in rs]
    p50 = [r["latency_ms_per_image"]["p50"] for r in rs]
    p99 = [r["latency_ms_per_image"]["p99"] for r in rs]
    err = [r["latency_ms_per_image"]["std_across_repeats"] for r in rs]

    x = np.arange(len(rs))
    w = 0.38
    # Always plot p50 next to p99, never the mean alone — the gap between the two bars
    # is tail latency, which matters more in deployment than the average does.
    fig, ax = plt.subplots(figsize=(max(8, len(rs) * 1.5), 5))
    # TODO: err is the std of the mean across rounds, but it is drawn as the error bar
    # on the p50 bar, which is a different statistic. It should be the std of p50
    # across rounds (not recorded in the jsonl yet — benchmark.py currently stores only
    # p99_std_across_repeats).
    ax.bar(x - w / 2, p50, w, yerr=err, capsize=3, label="p50", color="#4C78A8")
    ax.bar(x + w / 2, p99, w, label="p99", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency per image (ms) — lower is better")
    ax.set_title("Inference latency: p50 vs p99")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(p50):
        ax.text(i - w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig_latency.png", dpi=150)
    print(f"-> {outdir/'fig_latency.png'}")

    # This figure is the whole project's conclusion: how much mAP each speedup costs.
    # A single number in the table cannot say whether the trade was worth it — you have
    # to see both axes at once.
    pts = [(r, acc_map[key_of(r)]) for r in results if key_of(r) in acc_map]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for r, a in pts:
            ax.scatter(r["fps"], a["mAP50_95"], s=110, zorder=3)
            b = r.get("batch", 1)
            tag = f"{r['runtime']} {r['precision']}\n{r['device']}" + (f" b{b}" if b > 1 else "")
            ax.annotate(tag, (r["fps"], a["mAP50_95"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.set_xlabel("Throughput (FPS) — higher is better")
        ax.set_ylabel("mAP50-95 — higher is better")
        ax.set_title("Accuracy vs Speed trade-off")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "fig_tradeoff.png", dpi=150)
        print(f"-> {outdir/'fig_tradeoff.png'}")
    else:
        print("[warn] no accuracy data matched up yet — run evaluate.py for every config")


if __name__ == "__main__":
    main()

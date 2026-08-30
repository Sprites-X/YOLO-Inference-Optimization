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


def row_key(r: dict) -> tuple:
    """Identity of one measured row, for collapsing re-runs of the same config.

    Wider than key_of(), which is only the accuracy join. Batch and tag are what put
    two rows on separate lines of the table: the batch sweep runs a dynamic-shape
    engine that shares runtime/precision/device with the static one while being a
    different engine with different numbers.
    """
    return (r.get("runtime"), r.get("precision"), r.get("device"),
            r.get("batch", 1), r.get("tag", ""))


def newest_per_config(results: list[dict]) -> tuple[list[dict], list[tuple]]:
    """One row per config, the last one measured. Returns (kept, collapsed).

    results.jsonl is append-only, so a sweep run without run_all.sh --fresh appends a
    second set of rows rather than replacing the first, and the table then prints every
    config twice — once with each run's numbers, nothing saying which is which. That
    happened, and the doubled table is not obviously wrong at a glance, which is the
    problem with it.

    Ordered by timestamp rather than file position, so rows concatenated out of order
    still resolve to the genuinely newest. Rows written before benchmark.py recorded a
    timestamp sort as empty and fall back to file order, which is the best available
    answer for them.
    """
    best: dict[tuple, tuple] = {}
    counts: dict[tuple, int] = {}
    for i, r in enumerate(results):
        k = row_key(r)
        counts[k] = counts.get(k, 0) + 1
        rank = (r.get("timestamp") or "", i)
        if k not in best or rank > best[k][0]:
            best[k] = (rank, i, r)
    kept = [r for _, _, r in sorted(best.values(), key=lambda v: v[1])]
    collapsed = [(k, n) for k, n in counts.items() if n > 1]
    return kept, collapsed


def label_of(r: dict) -> str:
    b = r.get("batch", 1)
    suffix = f" b{b}" if b and b > 1 else ""
    tag = f"\n{r['tag']}" if r.get("tag") else ""
    return f"{r['runtime']}\n{r['precision']} {r['device']}{suffix}{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--accuracy", default="accuracy.jsonl")
    ap.add_argument("--outdir", default=".")
    # accuracy.jsonl also carries reference measurements taken on other image sets —
    # the full-val2017 PyTorch row is one. Those share a key with the val500 rows, so
    # they have to be filtered out rather than left to the join.
    ap.add_argument("--eval-images", type=int, default=500,
                    help="only join accuracy rows measured on this many images")
    args = ap.parse_args()

    results = load_jsonl(args.results)
    accs = load_jsonl(args.accuracy)
    if not results:
        raise SystemExit(f"no data in {args.results} — run benchmark.py first")

    # Both files are append-only, so a sweep run without run_all.sh --fresh leaves two
    # measurements of every config in here. Collapse to the newest of each rather than
    # printing both: two rows for one config is not a comparison, it is the same table
    # twice with no way to tell which run a row came from.
    results, collapsed = newest_per_config(results)
    if collapsed:
        print(f"[dedupe] {sum(n for _, n in collapsed)} rows for "
              f"{len(collapsed)} configs — keeping the newest measurement of each:")
        for (runtime, precision, device, batch, tag), n in sorted(collapsed):
            label = f"{runtime} {precision} {device} b{batch}" + (f" [{tag}]" if tag else "")
            print(f"           {label}: {n} rows")
        print("           (run_all.sh --fresh clears the old ones instead)")

    # The mAP column used to read 0.3651 for PyTorch FP32 GPU — the full-val2017 number,
    # joined onto a latency row timed on val500, where that same config scores 0.4008.
    # Nothing in the table said the two came from different image sets. Rows measured on
    # anything other than --eval-images are dropped before the join for that reason.
    graded = [a for a in accs if a.get("num_images") == args.eval_images]
    acc_map = {}
    for a in graded:
        prev = acc_map.get(key_of(a))
        # accuracy.jsonl carries no timestamp, so file order is the only ordering there
        # is and the later row wins. Re-measuring the same model is the ordinary reason
        # for a duplicate and needs no comment; two *different* models under one key is
        # not — the mAP column would then report the wrong model, so that one is said
        # out loud.
        if prev and prev.get("model") != a.get("model"):
            print(f"[warn] two accuracy rows for {key_of(a)} from different models: "
                  f"{prev['model']} {prev['mAP50_95']:.4f} then {a['model']} "
                  f"{a['mAP50_95']:.4f} — using the later one")
        acc_map[key_of(a)] = a

    # Created rather than assumed: --outdir is how a run writes its table somewhere
    # other than the repo, and the failure without this is a FileNotFoundError from
    # pathlib after the whole table has already been built and printed.
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        # vram_mb is the comparable number: card usage measured before the runtime
        # loaded and again after the run, the same way for every row. peak_vram_mb is
        # each runtime's own measure and means something different in every row, so it
        # is only a fallback for rows recorded before vram_mb existed.
        vram = r.get("vram_mb") or r.get("peak_vram_mb")
        cells = [
            # The tag distinguishes rows the table would otherwise print identically:
            # the batch sweep runs a dynamic-shape engine at batch 1 as its own
            # reference point, which collides with the static engine's batch 1 row on
            # every visible column while being a different engine with different
            # numbers. benchmark.py --tag is what sets it.
            r["runtime"] + (f" ({r['tag']})" if r.get("tag") else ""),
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
            # Comparable for rows carrying vram_mb. An older row falls back to
            # peak_vram_mb, which is not comparable to anything — PyTorch counted
            # allocator tensors, TensorRT read the whole card, ONNX reported nothing.
            f"{vram:.0f}" if vram else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)

    if baseline_fps:
        table += "\n\n**Speedup vs PyTorch GPU FP32 (batch 1):**\n\n"
        table += "| Config | Speedup |\n|---|---|\n"
        for r in sorted(results, key=lambda x: -x["fps"]):
            tag = f" {r['tag']}" if r.get("tag") else ""
            table += (f"| {r['runtime']}{tag} {r['precision']} {r['device']} "
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
    # The bar sits on p50, so the spread has to be the spread of p50 across rounds.
    # It used to be std_across_repeats, which is the spread of the *mean* — a different
    # statistic drawn on the p50 bar. Rows written before benchmark.py recorded
    # p50_std_across_repeats get no bar, which is better than a misleading one.
    err = [r["latency_ms_per_image"].get("p50_std_across_repeats", 0.0) for r in rs]

    x = np.arange(len(rs))
    w = 0.38
    # Always plot p50 next to p99, never the mean alone — the gap between the two bars
    # is tail latency, which matters more in deployment than the average does.
    fig, ax = plt.subplots(figsize=(max(8, len(rs) * 1.5), 5))
    ax.bar(x - w / 2, p50, w, yerr=err, capsize=3, label="p50", color="#4C78A8")
    ax.bar(x + w / 2, p99, w, label="p99", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency per image (ms) — lower is better, log scale")
    ax.set_title("Inference latency: p50 vs p99")
    # Log scale because the CPU rows are ~15x the GPU rows. On a linear axis the five
    # GPU bars collapse into one indistinguishable clump at the bottom — 1.4 ms and
    # 2.8 ms are a 2x difference and the whole point of the chart, and they looked
    # identical. Log keeps every row on one axis and readable.
    ax.set_yscale("log")
    # A log axis defaults to labelling only the decades, which here is the single tick
    # "10^1" — useless for reading a value off. Label the actual range instead.
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50) if t <= max(p99) * 1.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    # Both series get a number, since on a log axis the bar heights no longer let you
    # read the ratio off the picture.
    for i, (a, b) in enumerate(zip(p50, p99)):
        ax.text(i - w / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig_latency.png", dpi=150)
    print(f"-> {outdir/'fig_latency.png'}")

    # This figure is the whole project's conclusion: how much mAP each speedup costs.
    # A single number in the table cannot say whether the trade was worth it — you have
    # to see both axes at once.
    pts = [(r, acc_map[key_of(r)]) for r in results if key_of(r) in acc_map]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        # Points near the top of the range had their labels drawn over the title, and
        # two GPU rows at similar accuracy overprinted each other. Labelling below the
        # marker for the highest-scoring rows separates both cases.
        # The near-lossless rows all sit within a hair of the same mAP, so their labels
        # would print over each other however they are anchored. They get dropped below
        # the marker (clear of the title) and stepped down in turn, left to right.
        top = max(a["mAP50_95"] for _, a in pts)
        high_x = sorted(r["fps"] for r, a in pts if a["mAP50_95"] > top - 0.005)
        for r, a in pts:
            ax.scatter(r["fps"], a["mAP50_95"], s=110, zorder=3)
            b = r.get("batch", 1)
            tag = f"{r['runtime']} {r['precision']}\n{r['device']}" + (f" b{b}" if b > 1 else "")
            if a["mAP50_95"] > top - 0.005:
                # Three levels, not two: the batch sweep puts four near-lossless points
                # into the top-right corner and a two-level stagger still overlapped.
                dy = -24 - 20 * (high_x.index(r["fps"]) % 3)
                ax.annotate(tag, (r["fps"], a["mAP50_95"]), textcoords="offset points",
                            xytext=(0, dy), fontsize=8, ha="center")
            else:
                ax.annotate(tag, (r["fps"], a["mAP50_95"]), textcoords="offset points",
                            xytext=(0, 10), fontsize=8, ha="center")
        ax.margins(x=0.16, y=0.16)
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

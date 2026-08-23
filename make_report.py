from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(p: str) -> list[dict]:
    # คืน [] แทนที่จะพังถ้าไฟล์ยังไม่มี — accuracy.jsonl มักยังไม่เกิดตอนรัน
    # benchmark เสร็จใหม่ๆ ควรได้ตารางเวลาออกมาก่อนโดยที่คอลัมน์ mAP ว่างไว้
    path = Path(p)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_of(r: dict) -> tuple:
    # join key ระหว่าง results.jsonl กับ accuracy.jsonl — ไม่รวม batch โดยตั้งใจ
    # เพราะ mAP ไม่ขึ้นกับ batch size แถว b1 กับ b8 จึงใช้ค่า accuracy ตัวเดียวกันได้
    # (ต้องสะกดตรงกับที่ benchmark.py และ evaluate.py เขียนลงไฟล์ ดู evaluate.py:record)
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
        raise SystemExit(f"ไม่มีข้อมูลใน {args.results} — รัน benchmark.py ก่อน")

    # NOTE: ทั้งสองไฟล์เป็น append-only รัน config เดิมซ้ำจะได้แถวซ้ำในตาราง
    # และ acc_map จะเก็บอันหลังสุดเงียบๆ (dict comprehension ทับของเดิม)
    # ถ้าจะวัดใหม่ทั้งชุดให้ลบไฟล์ทิ้งก่อน
    acc_map = {key_of(a): a for a in accs}
    outdir = Path(args.outdir)

    # ---------------- ตาราง ----------------
    hdr = ("| Runtime | Precision | Device | Batch | p50 (ms) | p99 (ms) | "
           "mean ± std (ms) | FPS | E2E (ms) | mAP50-95 | Size (MB) | VRAM (MB) |")
    sep = "|" + "---|" * 12
    lines = [hdr, sep]

    # TODO: เลือกแถวแรกที่เป็น PyTorch/GPU/batch 1 โดยไม่ได้เช็ก precision
    # แต่หัวตารางข้างล่างเขียนตายตัวว่า "เทียบ PyTorch GPU FP32"
    # ถ้ารัน --half ก่อนแล้วบรรทัดนั้นมาก่อนใน jsonl ตัวเลข speedup ทุกแถว
    # จะเทียบกับ FP16 ใต้ป้ายที่บอกว่า FP32 — ต้องเพิ่มเงื่อนไข precision == "FP32"
    baseline_fps = None
    for r in results:
        if r["runtime"] == "PyTorch" and r["device"] == "GPU" and r.get("batch", 1) == 1:
            baseline_fps = r["fps"]
            break

    # เรียงช้าไปเร็ว เพื่อให้อ่านตารางจากบนลงล่างแล้วเห็นเรื่องราวของการ optimize
    # (PyTorch -> ONNX -> TRT FP16 -> TRT INT8) แทนที่จะต้องไล่หาเอง
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
            # "—" ไม่ได้แปลว่า mAP เป็นศูนย์ แปลว่ายังไม่ได้รัน evaluate.py สำหรับ
            # config นี้ หรือ key_of() ไม่ตรงกันระหว่างสองไฟล์
            f"{a['mAP50_95']:.4f}" if a else "—",
            f"{r['model_size_mb']:.1f}",
            # NOTE: คอลัมน์นี้เทียบข้ามแถวไม่ได้ — PyTorch รายงานเฉพาะ tensor,
            # TensorRT รายงานทั้งการ์ดจาก nvidia-smi, ONNX ไม่รายงานเลย
            # ดูเหตุผลเต็มที่ benchmark.py TensorRTRunner.peak_vram_mb
            # ต้องเขียนกำกับใน Limitations ไม่งั้นคนอ่านจะเทียบตัวเลขกันตรงๆ
            f"{vram:.0f}" if vram else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)

    if baseline_fps:
        table += "\n\n**Speedup เทียบ PyTorch GPU FP32 (batch 1):**\n\n"
        table += "| Config | Speedup |\n|---|---|\n"
        for r in sorted(results, key=lambda x: -x["fps"]):
            table += (f"| {r['runtime']} {r['precision']} {r['device']} "
                      f"b{r.get('batch',1)} | {r['fps']/baseline_fps:.2f}x |\n")

    (outdir / "report_table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\n-> {outdir/'report_table.md'}")

    # ---------------- กราฟ ----------------
    # import ในนี้เพราะ matplotlib เป็น optional — ตารางคือผลลัพธ์หลัก
    # กราฟเป็นของแถม ไม่ควรทำให้ทั้งสคริปต์พังถ้าไม่มี
    try:
        import matplotlib
        # ต้องเรียกก่อน import pyplot — เครื่องที่วัด benchmark มักไม่มี display
        # ถ้าไม่ตั้ง Agg pyplot จะพยายามหา GUI backend แล้วพังตอน import
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] ไม่มี matplotlib — ข้ามการวาดกราฟ (pip install matplotlib)")
        return

    rs = sorted(results, key=lambda x: x["latency_ms_per_image"]["p50"])
    labels = [label_of(r) for r in rs]
    p50 = [r["latency_ms_per_image"]["p50"] for r in rs]
    p99 = [r["latency_ms_per_image"]["p99"] for r in rs]
    err = [r["latency_ms_per_image"]["std_across_repeats"] for r in rs]

    x = np.arange(len(rs))
    w = 0.38
    # วาด p50 คู่ p99 เสมอ ไม่ใช่ mean อย่างเดียว — ระยะห่างระหว่างสองแท่งคือ
    # tail latency ซึ่งเป็นตัวที่สำคัญจริงตอน deploy มากกว่าค่าเฉลี่ย
    fig, ax = plt.subplots(figsize=(max(8, len(rs) * 1.5), 5))
    # TODO: err คือ std ของ mean ระหว่างรอบ แต่เอามาวางเป็น error bar ของแท่ง p50
    # ซึ่งเป็นคนละสถิติกัน ควรใช้ std ของ p50 ระหว่างรอบ (ยังไม่ได้เก็บลง jsonl —
    # ตอนนี้ benchmark.py เก็บแค่ p99_std_across_repeats)
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

    # กราฟนี้คือข้อสรุปของทั้งโปรเจกต์: เร็วขึ้นแลกกับ mAP ที่หายไปเท่าไร
    # ตัวเลขเดี่ยวๆ ในตารางตอบไม่ได้ว่าคุ้มไหม ต้องเห็นทั้งสองแกนพร้อมกัน
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
        print("[warn] ยังไม่มีข้อมูล accuracy ที่จับคู่ได้ — รัน evaluate.py ให้ครบทุก config")


if __name__ == "__main__":
    main()

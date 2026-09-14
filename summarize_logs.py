#!/usr/bin/env python3
"""Collect the per-class results from a directory of attack logs into summary.csv/.txt."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MARKER = "===== FINAL PER-CLASS SUMMARY ====="

RE_CLASS = re.compile(r"^Class (\d+):\s*(.+)$")
RE_FLIP = re.compile(r"Flip=\(layer (\d+), head (\d+), dim (\d+)\)")
RE_KV = re.compile(r"(\w+)=([-\d.]+)")
RE_VICTIM = re.compile(r"Victim model:\s*(\S+)")
RE_DATASET = re.compile(r"Dataset:\s*(\w+)(?:\s*\|\s*Calibration dataset:\s*(\w+))?")
RE_THREAT = re.compile(r"Threat model:\s*(\w+)")
RE_BASELINE = re.compile(r"\[Baseline\].*KV Accuracy:\s*([\d.]+)")

FIELDS = [
    "log", "victim", "dataset", "calib_dataset", "threat_model",
    "class", "baseline_acc", "acc_under_attack", "best_asr", "avg_asr",
    "layer", "head", "dim", "calib_asr",
]


def parse_log(path: Path) -> tuple[list[dict], str | None]:
    text = path.read_text(errors="ignore")
    idx = text.rfind(MARKER)
    if idx == -1:
        return [], None

    block = text[idx:].split("===================================")[0]

    def first(pattern, group=1, default=""):
        m = pattern.search(text)
        return m.group(group) if m and m.group(group) else default

    victim = Path(first(RE_VICTIM)).name
    m = RE_DATASET.search(text)
    dataset = m.group(1) if m else ""
    calib_dataset = m.group(2) if m and m.group(2) else ""
    baseline = first(RE_BASELINE)

    rows = []
    for line in block.splitlines():
        cm = RE_CLASS.match(line.strip())
        if not cm:
            continue
        rest = cm.group(2)
        kv = dict(RE_KV.findall(rest.replace("Flip=(", "").replace(")", "")))
        fm = RE_FLIP.search(rest)
        rows.append({
            "log": path.name,
            "victim": victim,
            "dataset": dataset,
            "calib_dataset": calib_dataset,
            "threat_model": first(RE_THREAT),
            "class": cm.group(1),
            "baseline_acc": baseline,
            "acc_under_attack": kv.get("Accuracy", ""),
            "best_asr": kv.get("BestEvalASR", kv.get("ASR", "")),
            "avg_asr": kv.get("AvgEvalASR", ""),
            "layer": fm.group(1) if fm else "",
            "head": fm.group(2) if fm else "",
            "dim": fm.group(3) if fm else "",
            "calib_asr": kv.get("Calib_ASR", ""),
        })
    return rows, block.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, required=True, help="Directory of *.log files.")
    args = parser.parse_args()

    if not args.log_dir.exists():
        raise SystemExit(f"Log directory not found: {args.log_dir}")

    rows, blocks, skipped = [], [], []
    for log_path in sorted(args.log_dir.glob("*.log")):
        parsed, block = parse_log(log_path)
        if not parsed:
            skipped.append(log_path.name)
            continue
        rows.extend(parsed)
        blocks.append(f"===== {log_path.name} =====\n{block}\n")

    for name in skipped:
        print(f"[warn] no summary block in {name} (crashed or still running?)")

    if not rows:
        raise SystemExit("No completed attack logs found.")

    csv_path = args.log_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = args.log_dir / "summary.txt"
    txt_path.write_text("\n".join(blocks))

    header = f"{'victim':<38} {'cls':>3} {'base':>6} {'atk_acc':>8} {'ASR':>6}  flip"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        flip = f"L{r['layer']}/H{r['head']}/D{r['dim']}" if r["layer"] else "-"
        print(f"{r['victim']:<38} {r['class']:>3} {r['baseline_acc']:>6} "
              f"{r['acc_under_attack']:>8} {r['best_asr']:>6}  {flip}")

    print(f"\nWrote {csv_path} ({len(rows)} rows) and {txt_path}")


if __name__ == "__main__":
    main()

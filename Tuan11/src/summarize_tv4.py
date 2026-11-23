"""Summarize TV4 run histories and mean ± sample std across the final seeds.

Author: Lê Thị Thanh Tuyền – 3123411332 | TV4: results analysis | 24/11/2026

Example from Tuan11/: python src/summarize_tv4.py --run_dirs results/runs/E4-03_best_seed42_s42 results/runs/E4-03_best_seed43_s43 results/runs/E4-03_best_seed44_s44
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_best(run_dir: Path):
    with (run_dir / "history.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty history.csv: {run_dir}")
    best = min(rows, key=lambda row: float(row["val_loss"]))
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return {
        "run": run_dir.name,
        "exp_id": cfg.get("exp_id", ""),
        "seed": cfg.get("seed", ""),
        "best_epoch": int(float(best["epoch"])),
        "train_acc": float(best.get("train_acc", "nan")),
        "val_loss": float(best["val_loss"]),
        "val_acc": float(best["val_acc"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/summary_TV4_seeds.csv"))
    args = parser.parse_args()
    rows = [read_best(path) for path in args.run_dirs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        if len(rows) >= 2:
            writer.writerow({"run": "MEAN ± SAMPLE STD", "val_acc": f"{statistics.mean(r['val_acc'] for r in rows):.4f} ± {statistics.stdev(r['val_acc'] for r in rows):.4f}", "val_loss": f"{statistics.mean(r['val_loss'] for r in rows):.4f} ± {statistics.stdev(r['val_loss'] for r in rows):.4f}"})
    print(f"Saved seed summary: {args.out.resolve()}")


if __name__ == "__main__":
    main()

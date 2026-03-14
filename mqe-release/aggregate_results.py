"""
aggregate_results.py
Reads the eval CSV logs from all seeds and reports the mean ± stderr success rate,
matching how Table 4 of the MQE paper is computed.

Usage:
    python aggregate_results.py --exp_dir exp/OGBench/visual_cube_triple_play_reproduce
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        default="exp/OGBench/visual_cube_triple_play_reproduce",
        help="Directory containing per-seed subdirectories with eval.csv files.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step to read results from. Defaults to the last logged step.",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.exp_dir, "**", "eval.csv"), recursive=True))
    if not csv_paths:
        print(f"No eval.csv files found under {args.exp_dir}")
        return

    overall_scores = []
    task_scores = {}  # task_name -> list of success rates across seeds

    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        if args.step is not None:
            row = df[df["step"] == args.step]
        else:
            row = df.iloc[[-1]]  # last row

        if row.empty:
            print(f"  WARNING: step {args.step} not found in {path}")
            continue

        overall_col = [c for c in df.columns if "overall_success" in c]
        if overall_col:
            val = float(row[overall_col[0]].values[0]) * 100.0
            overall_scores.append(val)
            print(f"  {path}: overall_success = {val:.1f}%")

        task_cols = [c for c in df.columns if c.startswith("evaluation/") and c.endswith("_success")]
        for col in task_cols:
            task = col.replace("evaluation/", "").replace("_success", "")
            val = float(row[col].values[0]) * 100.0
            task_scores.setdefault(task, []).append(val)

    print("\n=== Per-task results (mean ± stderr across seeds) ===")
    per_task_means = []
    for task, vals in sorted(task_scores.items()):
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        per_task_means.append(m)
        print(f"  {task}: {m:.1f} ± {se:.1f}%  (n={len(vals)})")

    print("\n=== Aggregate (mean of per-task means) ===")
    if overall_scores:
        m = np.mean(overall_scores)
        se = np.std(overall_scores, ddof=1) / np.sqrt(len(overall_scores)) if len(overall_scores) > 1 else 0.0
        print(f"  Overall success: {m:.1f} ± {se:.1f}%  (n={len(overall_scores)} seeds)")
    elif per_task_means:
        m = np.mean(per_task_means)
        se = np.std(per_task_means, ddof=1) / np.sqrt(len(per_task_means)) if len(per_task_means) > 1 else 0.0
        print(f"  Mean across tasks: {m:.1f} ± {se:.1f}%")

    print("\n  Paper target (Table 4): 19.8 ± 0.9%")


if __name__ == "__main__":
    main()

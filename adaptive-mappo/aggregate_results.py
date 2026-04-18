#!/usr/bin/env python3
"""Aggregate BenchMARL results across seeds and print a summary table."""

import json
import os
from collections import defaultdict
from pathlib import Path
import statistics

def extract_final_return(seed_data: dict) -> float:
    """Walk the seed dict to find the final evaluation return."""
    # BenchMARL stores eval returns under various keys — try common ones
    for top_key in seed_data:
        val = seed_data[top_key]
        if isinstance(val, dict):
            for metric_key in val:
                metric_val = val[metric_key]
                if isinstance(metric_val, list) and len(metric_val) > 0:
                    return metric_val[-1]  # final value
                if isinstance(metric_val, (int, float)):
                    return metric_val
        if isinstance(val, list) and len(val) > 0:
            return val[-1]
        if isinstance(val, (int, float)):
            return val
    return None


def get_all_metrics(seed_data: dict, prefix="") -> dict:
    """Recursively collect all scalar/list metrics."""
    results = {}
    for k, v in seed_data.items():
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, list) and len(v) > 0 and isinstance(v[-1], (int, float)):
            results[full_key] = v[-1]
        elif isinstance(v, (int, float)):
            results[full_key] = v
        elif isinstance(v, dict):
            results.update(get_all_metrics(v, full_key))
    return results


def main():
    base_dir = Path(__file__).parent
    json_files = list(base_dir.glob("*/*.json"))

    if not json_files:
        print("No JSON result files found.")
        return

    # Collect: {(env, algo, metric): [values across seeds]}
    data = defaultdict(list)
    found_metrics = set()

    for jf in json_files:
        try:
            with open(jf) as f:
                d = json.load(f)
        except Exception:
            continue

        # Structure: d['vmas'][env][algo][seed_X][...]
        for framework, envs in d.items():
            for env, algos in envs.items():
                for algo, seeds in algos.items():
                    for seed_key, seed_data in seeds.items():
                        metrics = get_all_metrics(seed_data)
                        found_metrics.update(metrics.keys())
                        for metric, value in metrics.items():
                            data[(env, algo, metric)].append(value)

    if not data:
        print("No data extracted. Printing raw keys from first file:")
        with open(json_files[0]) as f:
            d = json.load(f)
        print(json.dumps(d, indent=2)[:2000])
        return

    # Pick the most informative metric (prefer eval/return related keys)
    priority_keywords = ["eval", "return", "reward", "episode"]
    all_metrics = sorted(found_metrics)
    chosen_metrics = [m for m in all_metrics if any(k in m.lower() for k in priority_keywords)]
    if not chosen_metrics:
        chosen_metrics = all_metrics[:3]

    print(f"\nAvailable metrics: {all_metrics}\n")

    for metric in chosen_metrics:
        print(f"\n{'='*60}")
        print(f"Metric: {metric}")
        print(f"{'='*60}")
        print(f"{'Env':<15} {'Algo':<20} {'Seeds':>6} {'Mean':>10} {'Std':>10}")
        print("-" * 60)

        rows = {}
        for (env, algo, m), values in data.items():
            if m != metric:
                continue
            rows[(env, algo)] = values

        for (env, algo), values in sorted(rows.items()):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{env:<15} {algo:<20} {len(values):>6} {mean:>10.4f} {std:>10.4f}")


if __name__ == "__main__":
    main()

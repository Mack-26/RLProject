#!/usr/bin/env python3
"""Aggregate BenchMARL results — prints final-step return per algo/env."""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def get_final_return(seed_data: dict) -> float:
    """Extract the return from the last evaluation step."""
    # Keys are like "step_1", "step_2", ..., "step_50"
    step_keys = sorted(
        (k for k in seed_data if k.startswith("step_")),
        key=lambda k: int(k.split("_")[1]),
    )
    if not step_keys:
        return None
    last = seed_data[step_keys[-1]]
    # Try common return keys
    for key in ("agents_return", "return", "reward", "episode_reward"):
        if key in last:
            v = last[key]
            return v[-1] if isinstance(v, list) else v
    # Fallback: first numeric value
    for v in last.values():
        if isinstance(v, (int, float)):
            return v
    return None


def main():
    base_dir = Path(__file__).parent
    json_files = sorted(base_dir.glob("*/*.json"))

    # {(env, algo, seed_key): (mtime, final_return)} — keep latest run per seed
    seen = {}

    for jf in sorted(json_files, key=lambda f: f.stat().st_mtime):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        mtime = jf.stat().st_mtime
        for framework, envs in d.items():
            for env, algos in envs.items():
                for algo, seeds in algos.items():
                    for seed_key, seed_data in seeds.items():
                        val = get_final_return(seed_data)
                        if val is not None:
                            seen[(env, algo, seed_key)] = val

    results = defaultdict(list)
    for (env, algo, seed_key), val in seen.items():
        results[(env, algo)].append(val)

    if not results:
        print("No results found.")
        return

    print(f"\n{'Env':<15} {'Algorithm':<20} {'Seeds':>6} {'Mean Return':>12} {'Std':>10}")
    print("=" * 65)
    for (env, algo), values in sorted(results.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{env:<15} {algo:<20} {len(values):>6} {mean:>12.4f} {std:>10.4f}")


if __name__ == "__main__":
    main()

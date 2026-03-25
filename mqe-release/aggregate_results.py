"""
aggregate_results.py
Reads eval CSV logs and reports mean ± stderr success rate per (agent, env),
matching the format of Table 4 in the MQE paper.

Usage:
    python aggregate_results.py                        # all agents/envs
    python aggregate_results.py --agents mqe           # MQE only
    python aggregate_results.py --agents mqe,hiql --envs visual-cube-triple-play-v0

Results are picked as: for each seed prefix (sd000, sd001, ...), use the run
with the highest last logged step (handles multiple partial/resumed runs).
"""

import argparse
import glob
import os

import numpy as np

PAPER_TARGETS = {
    ('mqe',  'visual-cube-triple-play-v0'): 19.8,
    ('hiql', 'visual-cube-triple-play-v0'): 21.0,
    ('mqe',  'visual-scene-play-v0'):       38.1,
    ('hiql', 'visual-scene-play-v0'):       49.9,
}

ENV_SHORTS = {
    'visual-cube-triple-play-v0': 'visual_cube_triple',
    'visual-scene-play-v0':       'visual_scene_play',
}

DEFAULT_AGENTS = ['mqe', 'hiql']
DEFAULT_ENVS   = ['visual-cube-triple-play-v0', 'visual-scene-play-v0']


def read_csv(path):
    """Return list of dicts from a CSV file."""
    rows = []
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        return rows
    headers = lines[0].split(',')
    for line in lines[1:]:
        vals = line.split(',')
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return rows


def best_run_for_seed(seed_dirs):
    """Given a list of run directories for one seed, return (total_steps, overall_success)
    treating the directories as a sequential chain sorted by timestamp.
    Resumed runs reset their step counter to 1, so we sum steps across all runs
    and report the final run's last eval as the result."""
    def dir_timestamp(d):
        # dir format: sd000_s_JOBID.0.YYYYMMDD_HHMMSS
        parts = os.path.basename(d).split('.')
        return parts[-1] if len(parts) >= 3 else os.path.basename(d)

    total_steps = 0
    final_success = None
    for d in sorted(seed_dirs, key=dir_timestamp):
        csv = os.path.join(d, 'eval.csv')
        if not os.path.exists(csv):
            continue
        rows = read_csv(csv)
        if not rows:
            continue
        last = rows[-1]
        try:
            step = int(last.get('step', -1))
            success = float(last.get('evaluation/overall_success', -1))
        except ValueError:
            continue
        total_steps += step
        final_success = success * 100.0
    return total_steps, final_success


def gather_results(exp_base, agent, env):
    """Return list of (seed, last_step, overall_success_pct) for all seeds found."""
    env_short = ENV_SHORTS.get(env, env.replace('-', '_').replace('_v0', ''))
    run_group = f'{agent}_{env_short}'
    group_dir = os.path.join(exp_base, run_group)

    if not os.path.isdir(group_dir):
        return []

    # Group directories by seed prefix (sd000, sd001, ...)
    all_dirs = sorted(glob.glob(os.path.join(group_dir, 'sd*')))
    seed_map = {}
    for d in all_dirs:
        name = os.path.basename(d)
        prefix = name[:5]  # e.g. "sd000"
        seed_map.setdefault(prefix, []).append(d)

    results = []
    for prefix, dirs in sorted(seed_map.items()):
        seed = int(prefix[2:])  # "sd000" -> 0
        step, success = best_run_for_seed(dirs)
        if success is not None:
            results.append((seed, step, success))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='impls/exp/OGBench',
                        help='Base experiment directory.')
    parser.add_argument('--agents', default=','.join(DEFAULT_AGENTS),
                        help='Comma-separated list of agents.')
    parser.add_argument('--envs', default=','.join(DEFAULT_ENVS),
                        help='Comma-separated list of environments.')
    args = parser.parse_args()

    agents = args.agents.split(',')
    envs   = args.envs.split(',')

    col_w = 22
    header = f"{'':30s}" + ''.join(f'{a:^{col_w}s}' for a in agents)
    print(header)
    print('-' * (30 + col_w * len(agents)))

    for env in envs:
        row = f'{env:30s}'
        for agent in agents:
            results = gather_results(args.exp_dir, agent, env)
            if not results:
                row += f"{'no data':^{col_w}s}"
                continue
            successes = [r[2] for r in results]
            m  = np.mean(successes)
            se = np.std(successes, ddof=1) / np.sqrt(len(successes)) if len(successes) > 1 else 0.0
            cell = f'{m:.1f}±{se:.1f} n={len(successes)}'
            row += f'{cell:^{col_w}s}'
        print(row)

    print()
    print('Per-run details:')
    for agent in agents:
        for env in envs:
            results = gather_results(args.exp_dir, agent, env)
            if not results:
                continue
            target = PAPER_TARGETS.get((agent, env), '?')
            print(f'\n  {agent} / {env}  (paper target: {target}%)')
            for seed, step, success in results:
                status = 'COMPLETE' if step >= 500000 else f'partial ({step // 1000}k steps)'
                print(f'    seed {seed}: {success:.1f}%  [{status}]')


if __name__ == '__main__':
    main()

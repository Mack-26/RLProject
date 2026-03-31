"""
plot_results.py — Bar chart of final performance for MQE, HIQL, TMD on OGBench visual envs.

Reads local eval CSVs, correctly stitches resumed runs, reports mean ± stderr at final step.

Usage:
    python plot_results.py                    # reads from impls/exp/OGBench
    python plot_results.py --exp_dir PATH
    python plot_results.py --out results.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
AGENTS = ['mqe', 'hiql', 'tmd']
ENVS   = ['visual-cube-triple-play-v0', 'visual-scene-play-v0']

ENV_SHORTS = {
    'visual-cube-triple-play-v0': 'visual_cube_triple',
    'visual-scene-play-v0':       'visual_scene_play',
}
ENV_DISPLAY = {
    'visual-cube-triple-play-v0': 'Visual Cube Triple Play',
    'visual-scene-play-v0':       'Visual Scene Play',
}
AGENT_DISPLAY = {'mqe': 'MQE', 'hiql': 'HIQL', 'tmd': 'TMD'}
COLORS        = {'mqe': '#2196F3', 'hiql': '#4CAF50', 'tmd': '#FF9800'}

# MQE cube-triple was run before unified naming
SPECIAL_RUN_GROUPS = {
    ('mqe', 'visual-cube-triple-play-v0'): 'visual_cube_triple_play_reproduce',
}

# Paper-reported targets (Table 4). TMD not evaluated in paper on visual envs.
PAPER_TARGETS = {
    ('mqe',  'visual-cube-triple-play-v0'): 19.8,
    ('hiql', 'visual-cube-triple-play-v0'): 21.0,
    ('mqe',  'visual-scene-play-v0'):       38.1,
    ('hiql', 'visual-scene-play-v0'):       49.9,
}


# ── CSV helpers ───────────────────────────────────────────────────────────────
def _dir_ts(d):
    parts = os.path.basename(d).split('.')
    return parts[-1] if len(parts) >= 3 else os.path.basename(d)


def _read_csv(path):
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split(',')
    rows = []
    for line in lines[1:]:
        vals = line.split(',')
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return rows


def _final_success(seed_dirs):
    """Return final overall success % for a seed, stitching resumed runs."""
    step_offset = 0
    final = None
    for d in sorted(seed_dirs, key=_dir_ts):
        csv = os.path.join(d, 'eval.csv')
        if not os.path.exists(csv):
            continue
        rows = _read_csv(csv)
        last_valid = 0
        for row in rows:
            try:
                step    = int(row['step'])
                success = float(row['evaluation/overall_success'])
            except (KeyError, ValueError):
                continue
            if step <= 1:
                continue
            final = success * 100.0
            last_valid = step
        step_offset += last_valid
    return final


def gather_finals(exp_base, agent, env):
    """Return list of final success % values, one per seed."""
    env_short = ENV_SHORTS.get(env, env.replace('-', '_').replace('_v0', ''))
    run_group = SPECIAL_RUN_GROUPS.get((agent, env), f'{agent}_{env_short}')
    group_dir = os.path.join(exp_base, run_group)
    if not os.path.isdir(group_dir):
        return []

    seed_map = {}
    for d in sorted(glob.glob(os.path.join(group_dir, 'sd*'))):
        prefix = os.path.basename(d)[:5]
        seed_map.setdefault(prefix, []).append(d)

    results = []
    for prefix, dirs in sorted(seed_map.items()):
        val = _final_success(dirs)
        if val is not None:
            results.append(val)
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(exp_base='impls/exp/OGBench', out='results_plot.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bar_width = 0.22
    x = np.arange(len(AGENTS))

    for ax, env in zip(axes, ENVS):
        for i, agent in enumerate(AGENTS):
            vals = gather_finals(exp_base, agent, env)
            if not vals:
                continue
            mean = np.mean(vals)
            se   = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            color = COLORS[agent]

            bar = ax.bar(i, mean, bar_width * 2.5,
                         color=color, alpha=0.85, zorder=3,
                         label=f'{AGENT_DISPLAY[agent]} ({mean:.1f}±{se:.1f}%, n={len(vals)})')
            ax.errorbar(i, mean, yerr=se,
                        fmt='none', color='black', capsize=5, linewidth=1.5, zorder=4)

        # Paper target lines
        for agent in ['mqe', 'hiql']:
            tgt = PAPER_TARGETS.get((agent, env))
            if tgt:
                ax.axhline(tgt, color=COLORS[agent], linestyle='--',
                           linewidth=1.5, alpha=0.7,
                           label=f'{AGENT_DISPLAY[agent]} paper ({tgt}%)')

        ax.set_title(ENV_DISPLAY[env], fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels([AGENT_DISPLAY[a] for a in AGENTS], fontsize=11)
        ax.set_ylabel('Overall Success Rate (%)', fontsize=10)
        ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        'Goal-Conditioned RL on OGBench Visual Environments\n'
        'Final performance at 500k steps (mean ± std error across seeds)\n'
        'Dashed lines: paper-reported targets (Table 4)',
        fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='impls/exp/OGBench')
    parser.add_argument('--out',     default='results_plot.png')
    args = parser.parse_args()
    plot(args.exp_dir, args.out)

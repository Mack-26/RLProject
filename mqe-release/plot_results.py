"""
plot_results.py — Learning curves for MQE, HIQL, TMD on OGBench visual environments.

Reads local eval CSVs, correctly stitches resumed runs, plots mean ± stderr across seeds.

Usage:
    python plot_results.py                    # reads from impls/exp/OGBench
    python plot_results.py --exp_dir PATH
    python plot_results.py --out results.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
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

# MQE cube-triple was run before unified naming — lives in a different dir
SPECIAL_RUN_GROUPS = {
    ('mqe', 'visual-cube-triple-play-v0'): 'visual_cube_triple_play_reproduce',
}

PAPER_TARGETS = {
    ('mqe',  'visual-cube-triple-play-v0'): 19.8,
    ('hiql', 'visual-cube-triple-play-v0'): 21.0,
    ('mqe',  'visual-scene-play-v0'):       38.1,
    ('hiql', 'visual-scene-play-v0'):       49.9,
}

TARGET_STEPS = np.array([100_000, 200_000, 300_000, 400_000, 500_000])


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def _seed_curve(seed_dirs):
    """Return list of (absolute_step, success_pct) stitching sequential resumed runs."""
    step_offset = 0
    points = []
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
            points.append((step_offset + step, success * 100.0))
            last_valid = step
        step_offset += last_valid
    return points


def _interp(points, steps):
    """Linearly interpolate a curve to given step values; NaN outside range."""
    if not points:
        return [np.nan] * len(steps)
    xs, ys = zip(*sorted(points))
    xs, ys = np.array(xs, float), np.array(ys, float)
    out = []
    for s in steps:
        if s < xs[0] or s > xs[-1]:
            out.append(np.nan)
        else:
            out.append(float(np.interp(s, xs, ys)))
    return out


def gather(exp_base, agent, env):
    """Return {seed: [(step, pct), ...]} for one (agent, env) combo."""
    env_short  = ENV_SHORTS.get(env, env.replace('-', '_').replace('_v0', ''))
    run_group  = SPECIAL_RUN_GROUPS.get((agent, env), f'{agent}_{env_short}')
    group_dir  = os.path.join(exp_base, run_group)
    if not os.path.isdir(group_dir):
        return {}

    seed_map = {}
    for d in sorted(glob.glob(os.path.join(group_dir, 'sd*'))):
        prefix = os.path.basename(d)[:5]
        seed_map.setdefault(prefix, []).append(d)

    curves = {}
    for prefix, dirs in sorted(seed_map.items()):
        pts = _seed_curve(dirs)
        if pts:
            curves[int(prefix[2:])] = pts
    return curves


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(exp_base='impls/exp/OGBench', out='results_plot.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    steps_k = TARGET_STEPS / 1_000

    for ax, env in zip(axes, ENVS):
        for agent in AGENTS:
            curves = gather(exp_base, agent, env)
            if not curves:
                continue

            matrix = np.array([_interp(pts, TARGET_STEPS)
                                for pts in curves.values()])   # (n_seeds, n_steps)
            mean = np.nanmean(matrix, axis=0)
            se   = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(
                       np.sum(~np.isnan(matrix), axis=0).clip(1))
            valid = ~np.isnan(mean)
            color = COLORS[agent]

            ax.plot(steps_k[valid], mean[valid],
                    color=color, linewidth=2, marker='o', markersize=4,
                    label=f'{AGENT_DISPLAY[agent]} (n={len(curves)})')
            ax.fill_between(steps_k[valid],
                            mean[valid] - se[valid],
                            mean[valid] + se[valid],
                            color=color, alpha=0.18)

        # Dashed paper-target lines
        for agent in ['mqe', 'hiql']:
            tgt = PAPER_TARGETS.get((agent, env))
            if tgt:
                ax.axhline(tgt, color=COLORS[agent], linestyle='--',
                           linewidth=1.2, alpha=0.55,
                           label=f'{AGENT_DISPLAY[agent]} paper target')

        ax.set_title(ENV_DISPLAY[env], fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Steps (k)', fontsize=10)
        ax.set_ylabel('Overall Success (%)', fontsize=10)
        ax.set_xlim(50, 520)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        'Goal-Conditioned RL on OGBench Visual Environments\n'
        'Mean ± Std Error across seeds',
        fontsize=13, fontweight='bold', y=1.01)
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

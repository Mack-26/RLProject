#!/usr/bin/env python3
"""Training entry point for AdaptiveMappo / ScheduledMappo experiments.

Usage examples:
  # Baseline MAPPO
  python train.py --algo mappo --seed 0

  # Adaptive alpha with all features
  python train.py --algo adaptive --seed 0

  # Adaptive alpha ablation: only global+local+disagree
  python train.py --algo adaptive --seed 0 --features global,local,disagree

  # Scheduled alpha, linear MAPPO→IPPO over training
  python train.py --algo scheduled --seed 0 --alpha_schedule linear --alpha_start 1.0 --alpha_end 0.0

  # Scheduled constant IPPO baseline
  python train.py --algo scheduled --seed 0 --alpha_schedule constant --alpha_start 0.0

  # Different environment
  python train.py --algo adaptive --env wheel --seed 1
"""

import argparse
import sys
from pathlib import Path

# Make algorithms/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from benchmarl.algorithms import MappoConfig, IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from algorithms.adaptive_mappo import AdaptiveMappoConfig
from algorithms.scheduled_mappo import ScheduledMappoConfig

ALGO_CONFIG_CLS = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "adaptive": AdaptiveMappoConfig,
    "scheduled": ScheduledMappoConfig,
}

ALGO_YAML = {
    "adaptive": "conf/algorithm/adaptive_mappo.yaml",
    "scheduled": "conf/algorithm/scheduled_mappo.yaml",
    # mappo/ippo use BenchMARL's built-in defaults (pass None)
    "mappo": None,
    "ippo": None,
}

VMAS_TASKS = {
    "navigation": VmasTask.NAVIGATION,
    "wheel": VmasTask.WHEEL,
    "balance": VmasTask.BALANCE,
    "transport": VmasTask.TRANSPORT,
    "give_way": VmasTask.GIVE_WAY,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGO_CONFIG_CLS), default="adaptive")
    p.add_argument("--env", choices=list(VMAS_TASKS), default="navigation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_n_frames", type=int, default=None, help="Override max training frames")
    # Adaptive-specific
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature subset for AdaptiveAlphaNetwork "
                        "(e.g. 'global,local,disagree')")
    # Scheduled-specific
    p.add_argument("--alpha_schedule", choices=["linear", "cosine", "constant"], default=None)
    p.add_argument("--alpha_start", type=float, default=None)
    p.add_argument("--alpha_end", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    conf_dir = Path(__file__).parent / "conf"

    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml(
        str(conf_dir / "experiment" / "base_experiment.yaml")
    )
    if args.max_n_frames is not None:
        experiment_config.max_n_frames = args.max_n_frames

    # Algorithm config
    yaml_path = ALGO_YAML[args.algo]
    algo_config = ALGO_CONFIG_CLS[args.algo].get_from_yaml(
        str(conf_dir / "algorithm" / Path(yaml_path).name) if yaml_path else None
    )

    if args.algo == "adaptive" and args.features is not None:
        algo_config.features = args.features.split(",")

    if args.algo == "scheduled":
        if args.alpha_schedule is not None:
            algo_config.alpha_schedule = args.alpha_schedule
        if args.alpha_start is not None:
            algo_config.alpha_start = args.alpha_start
        if args.alpha_end is not None:
            algo_config.alpha_end = args.alpha_end

    # Task and model
    task = VMAS_TASKS[args.env].get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()

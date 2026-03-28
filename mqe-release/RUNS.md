# Experiment Run Log
ECE 567 — Reinforcement Learning Project
Paper: *Multistep Quasimetric Learning for Scalable Goal-Conditioned RL* (ICLR 2026)
Cluster: Great Lakes (UMich), Account: `ece567w26_class`, Partition: `gpu`, QOS: `class`
GPU: V100, Max wall time: 7h30m per job

---

## Target Results (Table 4, paper)
| Environment               | MQE         | HIQL        |
|---------------------------|-------------|-------------|
| visual-cube-triple-play   | 19.8 ± 0.9% | 21.0 ± 0.2% |
| visual-scene-play         | 38.1 ± 3.2% | 49.9 ± 0.6% |

---

## MQE — visual-cube-triple-play-v0
*Already completed before unified script. Run via `run_mqe_visual_cube_triple_play.sh`.*

| Seed | Job ID   | Date       | Duration | Last Step | Overall | Task1 (single PnP) | Task2 (triple PnP) | Task3 (PnP stack) | Task4 (cycle) | Task5 (stack) | Status   |
|------|----------|------------|----------|-----------|---------|--------------------|--------------------|--------------------|---------------|---------------|----------|
| 0    | 45276609 | 2026-03-16 | ~8h      | 400,000   | 11.2%   | 56%                | 0%                 | 0%                 | 0%            | 0%            | Partial  |
| 1    | 45276610 | 2026-03-16 | ~8h      | 400,000   | 15.2%   | 76%                | 0%                 | 0%                 | 0%            | 0%            | Partial  |
| 2    | 45323108 | 2026-03-16 | ~8h      | 500,000   | 10.4%   | 52%                | 0%                 | 0%                 | 0%            | 0%            | Complete |
| 3    | 45323105 | 2026-03-16 | ~8h      | 500,000   | 17.2%   | 86%                | 0%                 | 0%                 | 0%            | 0%            | Complete |

**Mean (all 4 seeds): 13.5 ± 1.6%** | Paper target: 19.8 ± 0.9%
*Note: Seeds 0 and 1 hit the 8h wall time at 400k/500k steps. Need re-run for full 500k.*

Failed/crashed runs (not counted):
- 45274827, 45342976 — crashed at step 1 (env/setup issues during debugging)
- 45272xxx, 45273xxx, 45274xxx, 45276408, 45342773, 45342893, 45385224 — no eval.csv (crashed before first eval)

---

## MQE — visual-scene-play-v0
*Run via `run_experiment.sh` (unified script). Stopped at seed 0 — result well below paper target.*

| Seed | Job ID | Date | Duration | Last Step | Overall | Task1 | Task2 | Task3 | Task4 | Task5 | Status |
|------|--------|------|----------|-----------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 45772681 | 2026-03-21 | ~7.5h | 500,000 | 17.2% | 84% | 2% | 0% | 0% | 0% | Complete |

**Mean (seed 0 only): 17.2%** | Paper target: 38.1 ± 3.2%
*Note: Stopped at 1 seed — result (17.2%) is well below paper target. MQE may require more tuning for scene env.*

---

## HIQL — visual-cube-triple-play-v0
*Run via `run_experiment.sh` (unified script). Seed 0 timed out at 300k, resumed via job 45839179.*

| Seed | Job ID | Date | Duration | Last Step | Overall | Task1 | Task2 | Task3 | Task4 | Task5 | Status |
|------|--------|------|----------|-----------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 45772682 + 45839179 | 2026-03-22 | ~15h total | 500,000 | 20.8% | 100% | 0% | 4% | 0% | 0% | Complete |
| 1    | 46119349 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |
| 2    | 46119350 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |
| 3    | 46119351 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |

**Mean (seed 0 only): 20.8%** | Paper target: 21.0 ± 0.2%

---

## HIQL — visual-scene-play-v0
*Run via `run_experiment.sh` (unified script). Seed 0 timed out at 300k, resumed via job 45839193.*

| Seed | Job ID | Date | Duration | Last Step | Overall | Task1 | Task2 | Task3 | Task4 | Task5 | Status |
|------|--------|------|----------|-----------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 45772683 + 45839193 | 2026-03-22 | ~15h total | 500,000 | 42.4% | 66% | 38% | 64% | 36% | 8% | Complete |
| 1    | 46119353 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |
| 2    | 46119354 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |
| 3    | 46119365 | 2026-03-25 | —        | —         | —       | —     | —     | —     | —     | —     | Queued |

**Mean (seed 0 only): 42.4%** | Paper target: 49.9 ± 0.6%

---

## TMD — visual-cube-triple-play-v0
*Run via `run_experiment.sh`. Seeds split across 3 accounts for parallel execution.*

| Seed | Job ID | Account | Date | Duration | Last Step | Overall | Task1 | Task2 | Task3 | Task4 | Task5 | Status |
|------|--------|---------|------|----------|-----------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 46119366 | ece567w26_class    | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |
| 1    | 46281518 | eecs504s001w26_class | 2026-03-28 | — | — | — | — | — | — | — | — | Running |
| 2    | 46281521 | eecs504s001w26_class | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |
| 3    | 46281526 | eecs545w26_class   | 2026-03-28 | — | — | — | — | — | — | — | — | Running |

**Mean (all 4 seeds): — ± —%** | Paper target: N/A (not in Table 4)

---

## TMD — visual-scene-play-v0
*Run via `run_experiment.sh`. Seeds split across 3 accounts for parallel execution.*

| Seed | Job ID | Account | Date | Duration | Last Step | Overall | Task1 | Task2 | Task3 | Task4 | Task5 | Status |
|------|--------|---------|------|----------|-----------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 46119370 | ece567w26_class    | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |
| 1    | 46281522 | eecs504s001w26_class | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |
| 2    | 46281527 | eecs545w26_class   | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |
| 3    | 46281528 | eecs545w26_class   | 2026-03-28 | — | — | — | — | — | — | — | — | Queued |

**Mean (all 4 seeds): — ± —%** | Paper target: N/A (not in Table 4)

---

## Setup & Debugging History

| Date       | Issue | Cause | Fix |
|------------|-------|-------|-----|
| 2026-03-15 | `No module named 'jax'` | `conda activate` fails in SLURM non-interactive shell | Use `PYTHON=/home/aromanan/.conda/envs/mqe/bin/python` |
| 2026-03-15 | `cuSPARSE not found` / CPU fallback | `module load cuda/12.1.1` overrides pip-installed CUDA | Remove cuda module load |
| 2026-03-15 | `CUDNN_STATUS_INTERNAL_ERROR` on V100 | cuDNN 9.x has V100 (sm_70) regression | Use JAX 0.4.28+cuda12.cudnn89 |
| 2026-03-15 | Segfault | flax 0.10.7 binary-incompatible with JAX 0.4.28 | Downgrade flax to 0.8.4 |
| 2026-03-15 | OOM killed | 32GB RAM insufficient for 4.8GB pixel dataset | Increase `--mem` to 64G |
| 2026-03-16 | `$(dirname $0)` resolves to SLURM spool dir | SLURM copies script to spool before running | Hardcode `REPO_DIR=$HOME/RLProject/mqe-release` |
| 2026-03-21 | `AssocMaxWallDurationPerJobLimit` | `--time=12:00:00` exceeds class account 8h limit | Reduce to `07:30:00`, add `--qos=class` |

---

## Working SLURM Config (as of 2026-03-16)
```bash
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --qos=class
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:30:00

module load python3.10-anaconda/2023.03
# DO NOT load cuda/12.1.1 — conflicts with JAX pip-installed CUDA
PYTHON=/home/aromanan/.conda/envs/mqe/bin/python
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export MUJOCO_GL=egl
```

## JAX/CUDA Package Versions (V100-compatible)
```
jaxlib==0.4.28+cuda12.cudnn89
jax==0.4.28
flax==0.8.4
optax==0.2.2
distrax==0.1.5
```

# ECE 567 RL Project — MQE Reproduction

Reproducing Table 4 of **"Multistep Quasimetric Learning for Scalable Goal-Conditioned RL"** (ICLR 2026)
Target: `visual-cube-triple-play-v0` on OGBench → **19.8 ± 0.9%** success rate (4 seeds)

---

## GitHub

```
https://github.com/Mack-26/RLProject.git
```

---

## Great Lakes Setup

**Login:**
```bash
ssh aromanan@greatlakes.arc-ts.umich.edu
```

**Repo location:**
```
~/RLProject/mqe-release/
```

**Conda environment:** `mqe` (Python 3.10)
```bash
conda activate mqe
```

**Dataset location:**
```
~/ogbench_data/visual-cube-triple-play-v0.npz       (~4.8 GB, train)
~/ogbench_data/visual-cube-triple-play-v0-val.npz   (val)
```

**Logs output:**
```
~/RLProject/mqe-release/logs/mqe_vct_<jobid>_<seed>.out
~/RLProject/mqe-release/logs/mqe_vct_<jobid>_<seed>.err
```

---

## Slurm Account

| Account | GPU limit | Max walltime | Notes |
|---------|-----------|-------------|-------|
| `ece567w26_class` | 1 GPU at a time | 8:00:00 | Use this for class project |
| `engin1` | 2 GPUs | default | Backup |

---

## Running Jobs

**First time only — download dataset:**
```bash
conda activate mqe
python ~/RLProject/mqe-release/download_dataset.py --dataset_dir ~/ogbench_data
```

**Submit all 4 seeds:**
```bash
cd ~/RLProject/mqe-release
mkdir -p logs
sbatch run_mqe_visual_cube_triple_play.sh
```

Seeds run sequentially due to 1 GPU limit on class account. Total runtime ~30h.

**Submit a single seed (for testing):**
```bash
sbatch --export=SEED=0 --array=0 run_mqe_visual_cube_triple_play.sh
```

---

## Monitoring Jobs

```bash
squeue -u aromanan          # running/pending jobs
sacct -u aromanan           # completed job history
scancel <jobid>             # cancel a job
```

---

## Getting Results

After all 4 seeds finish:
```bash
conda activate mqe
python ~/RLProject/mqe-release/aggregate_results.py
```

Prints mean ± stderr across seeds. Target: **19.8 ± 0.9%**

---

## Key Files

| File | Purpose |
|------|---------|
| `mqe-release/impls/agents/mqe.py` | MQE agent (bug fix: `use_latent=False` in `get_config()`) |
| `mqe-release/impls/main.py` | Training entry point |
| `mqe-release/run_mqe_visual_cube_triple_play.sh` | SLURM job script |
| `mqe-release/setup_greatlakes.sh` | One-time env setup (already done) |
| `mqe-release/download_dataset.py` | Dataset download (already done) |
| `mqe-release/aggregate_results.py` | Parse eval CSVs → mean ± stderr |

---

## Key Hyperparameters (visual-cube-triple-play)

| Param | Value |
|-------|-------|
| encoder | impala_small |
| latent_dim | 512 |
| batch_size | 256 |
| discount (γ) | 0.995 |
| lambda (λ) | 0.95 |
| next_state_sample (p) | 0.1 |
| alpha (BC coeff) | 3.0 |
| p_aug | 0.5 |
| train_steps | 500,000 |
| eval_episodes | 50 |

---

## Updating Code

**Mac → push:**
```bash
cd "/Users/mananarora/Desktop/Manan/ECE 567 - Reinforcement Learning/RLProject"
git add .
git commit -m "your message"
git push
```

**Great Lakes → pull:**
```bash
cd ~/RLProject && git pull
```

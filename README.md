# Sequential Aircraft Maintenance Scheduling with PCL

Reinforcement learning approach to aircraft maintenance scheduling using Path Consistency Learning (PCL) with a Transformer architecture.

## Problem

Schedule A-checks and C-checks for a fleet of 45 aircraft over a 6-year horizon to minimize wasted flight hours while respecting:
- Maintenance interval limits (FH and calendar days)
- Hangar capacity constraints
- Peak season blackout periods

## Files

### Core Environment
- `sequential_env.py` - Environment with step-by-step scheduling decisions
- `data_loader.py` - Load aircraft data from Excel or create synthetic environment

### Baselines
- `paper_dp.py` - Greedy (earliest-deadline-first) and Forward DP baselines

### PCL Implementations
- `pcl.py` - Original PCL implementation
- `pcl_improved.py` - Enhanced PCL with n-step returns, replay buffer, reward shaping
- `pcl_transformer.py` - **Transformer-based PCL** (recommended for GPU)

### Imitation Learning
- `bc_transformer.py` - Behavioral cloning to learn from greedy expert
- `imitation_learning.py` - BC and DAgger implementations

### Training Scripts
- `train.py` - Original training script
- `train_improved.py` - Training with improved PCL

## Quick Start

### 1. Setup Environment

```bash
# On Mila cluster
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For GPU (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Baselines

```bash
# Compare greedy vs random
python paper_dp.py
```

Expected output:
- Greedy: ~612K wasted FH
- Random: ~780K wasted FH

### 3. Train Transformer PCL

```bash
# On GPU node
python pcl_transformer.py \
    --iterations 500 \
    --eval-every 50 \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --lr 1e-4 \
    --entropy 0.2
```

### 4. Behavioral Cloning + Fine-tuning

```bash
# First learn from greedy (fast)
python bc_transformer.py --bc-epochs 100 --n-expert-traj 20

# Then fine-tune with PCL (slower)
python bc_transformer.py --bc-epochs 50 --n-expert-traj 20 --fine-tune --fine-tune-iters 200
```

## SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=aircraft-pcl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long

module load python/3.10
source venv/bin/activate

python pcl_transformer.py \
    --iterations 1000 \
    --eval-every 100 \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --lr 1e-4
```

## Key Challenges

1. **Long horizon**: 2190 days × ~2 decisions/day ≈ 4000+ steps
2. **Sparse rewards**: Only negative reward when scheduling (wasted FH)
3. **Near-optimal baseline**: Greedy (EDF) is classically optimal for this problem class

## Current Results

| Method | Wasted FH | Notes |
|--------|-----------|-------|
| Greedy | 612K | Earliest-deadline-first |
| Random | 782K | Uniform random valid action |
| PCL (MLP) | ~1.1M | Struggles with long horizon |
| Transformer PCL | TBD | Training on cluster |
| BC → PCL | TBD | Most promising approach |

## Architecture

The Transformer architecture processes the fleet state as:
1. Per-aircraft features (15 dims): remaining days, utilization, eligibility, urgency
2. Self-attention over aircraft: learns to compare and prioritize
3. Cross-attention with global context: time, capacity, fleet statistics
4. Policy heads: C-check scores, A-check scores, wait logit
5. Value head: pooled aircraft embeddings → single value

## References

- PCL: [Nachum et al. 2017](https://arxiv.org/abs/1702.08892)
- Problem formulation: "Sep 25 2017 (Case Study 2018-2021).xlsx"


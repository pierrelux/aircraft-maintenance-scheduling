# Session Notes: PCL for Aircraft Maintenance Scheduling

## Project Goal
Train a reinforcement learning agent using Path Consistency Learning (PCL) to schedule aircraft maintenance (A-checks and C-checks) that **beats or matches the greedy baseline**.

## Current Status: Ready for Cluster Training

### What We Tried (Locally)

1. **Original PCL (MLP)** - `pcl.py`, `train.py`
   - Result: **Failed** - Returns ~932K vs greedy's 612K
   - Problem: Couldn't learn anything useful over 4000+ step horizon

2. **Improved PCL with fixes** - `pcl_improved.py`
   - Added: Enhanced state encoding with "remaining days until deadline"
   - Added: N-step returns (n=20) for better credit assignment
   - Added: Replay buffer for batch updates
   - Added: Temperature annealing (15→1)
   - Added: Reward shaping (potential-based)
   - Result: **Still failed** - Returns got WORSE with shaping enabled (-1M → -2.5M)
   - Without shaping: Returns stable at ~-1.1M but no improvement

3. **Transformer PCL** - `pcl_transformer.py`
   - Architecture: Self-attention over aircraft fleet + cross-attention with global state
   - Hypothesis: Attention can handle long-range dependencies better
   - Result: **Too slow locally** (~100s/iteration vs 20s for MLP)
   - Needs GPU training on cluster

4. **Behavioral Cloning** - `bc_transformer.py`
   - Strategy: Learn from greedy expert, then fine-tune with PCL
   - Result: **Too slow locally** - Training started but didn't complete
   - Most promising approach for cluster

### Key Findings

| Method | Wasted FH | Notes |
|--------|-----------|-------|
| **Greedy (baseline)** | 612K | Earliest-deadline-first, near-optimal |
| Random | 782K | 22% worse than greedy |
| Forward DP | 2.8M | Surprisingly worse than greedy |
| PCL (any variant) | ~1.1M+ | Could not improve beyond random |

### Why PCL Struggles

1. **Horizon too long**: 4000+ steps makes credit assignment nearly impossible
2. **Greedy is near-optimal**: Earliest-deadline-first is classically optimal for single-machine deadline scheduling
3. **Sparse scheduling decisions**: Most actions are "wait", few actual scheduling choices
4. **Reward structure**: Only negative rewards (wasted FH) when scheduling

### Recommended Next Steps (on Cluster)

1. **First: Run Behavioral Cloning**
   ```bash
   sbatch run_cluster.sh bc
   ```
   This trains a transformer to imitate greedy. Should achieve ~612K (matching greedy).

2. **Then: Fine-tune with PCL**
   ```bash
   sbatch run_cluster.sh bc-finetune
   ```
   Starting from a good initialization may help discover improvements over greedy.

3. **Alternative: Pure Transformer PCL (exploratory)**
   ```bash
   sbatch run_cluster.sh transformer
   ```
   With GPU acceleration, training is feasible but may still struggle with the fundamental problem structure.

### Files Summary

```
Core Environment:
├── sequential_env.py      # Main environment (step-by-step decisions)
├── data_loader.py         # Load Excel data / synthetic env

Baselines:
├── paper_dp.py            # Greedy + Forward DP

PCL Implementations:
├── pcl.py                 # Original (broken)
├── pcl_improved.py        # Enhanced but still doesn't work
├── pcl_transformer.py     # Transformer architecture (needs GPU)

Imitation Learning:
├── bc_transformer.py      # BC + optional PCL fine-tune (RECOMMENDED)
├── imitation_learning.py  # BC/DAgger with MLP

Training Scripts:
├── train.py               # Original training
├── train_improved.py      # Improved training
├── run_cluster.sh         # SLURM job script

Data:
├── Sep 25 2017 (Case Study 2018-2021).xlsx  # Aircraft fleet data
```

### Quick Verification Commands

```bash
# Test environment loads correctly
python -c "from data_loader import load_env; e = load_env('Sep 25 2017 (Case Study 2018-2021).xlsx'); print(f'{e.n_aircraft} aircraft, {e.horizon} days')"

# Run greedy baseline
python paper_dp.py

# Expected output: Greedy cost ~612K
```

### Key Hyperparameters (for Transformer)

```python
d_model = 256        # Embedding dimension
n_heads = 8          # Attention heads
n_layers = 4         # Transformer layers
lr = 1e-4            # Learning rate
entropy_coeff = 0.2  # Exploration bonus
temperature = 10→1   # Annealing schedule
n_steps = 10         # N-step returns
batch_size = 16      # Replay buffer batch
```

### Open Questions

1. Can BC + PCL fine-tuning beat greedy by learning context-dependent thresholds?
2. Would a different RL algorithm (PPO, SAC) work better?
3. Is there value in model-based planning using the known environment dynamics?
4. Could curriculum learning with shorter horizons help?

---
*Last updated: Nov 26, 2025*


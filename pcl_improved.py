"""
Improved Path Consistency Learning (PCL) for Aircraft Maintenance Scheduling.

Key improvements over base PCL:
1. Enhanced state encoding with "remaining days until deadline" 
2. Reward shaping with potential-based function
3. Multi-step (n-step) consistency for better credit assignment
4. Curriculum learning with increasing horizon
5. Batch updates from trajectory segments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
from tqdm import tqdm

from sequential_env import (
    SequentialSchedulingEnv,
    SchedulingState,
    AircraftStatus,
    MaintenanceConfig,
)


# =============================================================================
# Enhanced State Encoding
# =============================================================================

def compute_remaining_days(aircraft: AircraftStatus, check_type: str, config: MaintenanceConfig) -> float:
    """Compute remaining days until check is due (what greedy uses!)."""
    if check_type == 'A':
        remaining_dy = config.a_interval_dy - aircraft.dy_a
        remaining_fh = (config.a_interval_fh - aircraft.fh_a) / max(aircraft.daily_fh, 0.1)
        remaining_fc = (config.a_interval_fc - aircraft.fc_a) / max(aircraft.daily_fc, 0.1)
    else:  # 'C'
        remaining_dy = config.c_interval_dy - aircraft.dy_c
        remaining_fh = (config.c_interval_fh - aircraft.fh_c) / max(aircraft.daily_fh, 0.1)
        remaining_fc = (config.c_interval_fc - aircraft.fc_c) / max(aircraft.daily_fc, 0.1)
    return min(remaining_dy, remaining_fh, remaining_fc)


def compute_utilization(aircraft: AircraftStatus, check_type: str, config: MaintenanceConfig) -> float:
    """Compute utilization ratio [0, 1+] where 1 = at interval limit."""
    if check_type == 'A':
        dy_ratio = aircraft.dy_a / config.a_interval_dy if config.a_interval_dy > 0 else 0
        fh_ratio = aircraft.fh_a / config.a_interval_fh if config.a_interval_fh > 0 else 0
        fc_ratio = aircraft.fc_a / config.a_interval_fc if config.a_interval_fc > 0 else 0
    else:  # 'C'
        dy_ratio = aircraft.dy_c / config.c_interval_dy if config.c_interval_dy > 0 else 0
        fh_ratio = aircraft.fh_c / config.c_interval_fh if config.c_interval_fh > 0 else 0
        fc_ratio = aircraft.fc_c / config.c_interval_fc if config.c_interval_fc > 0 else 0
    return max(dy_ratio, fh_ratio, fc_ratio)


def encode_state_enhanced(
    state: SchedulingState, 
    n_aircraft: int, 
    horizon: int,
    config: MaintenanceConfig
) -> np.ndarray:
    """
    Enhanced state encoding with critical features for scheduling.
    
    KEY ADDITIONS vs original:
    - Remaining days until A-check due (normalized)
    - Remaining days until C-check due (normalized)
    - Utilization ratios (how much of interval used)
    - Fleet statistics (min/mean/max remaining days)
    - Days remaining in horizon
    - Priority ranks (what greedy uses!)
    """
    # Pre-compute remaining days and priority ranks
    a_remaining_list = []
    c_remaining_list = []
    for i in range(n_aircraft):
        aircraft = state.fleet[i]
        if not aircraft.in_maintenance:
            a_remaining_list.append((i, compute_remaining_days(aircraft, 'A', config)))
            c_remaining_list.append((i, compute_remaining_days(aircraft, 'C', config)))
    
    # Compute priority ranks (lower remaining = higher priority = lower rank number)
    a_sorted = sorted(a_remaining_list, key=lambda x: x[1])
    c_sorted = sorted(c_remaining_list, key=lambda x: x[1])
    
    a_priority_rank = {aid: rank for rank, (aid, _) in enumerate(a_sorted)}
    c_priority_rank = {aid: rank for rank, (aid, _) in enumerate(c_sorted)}
    
    features = []
    
    # Time features
    features.append(state.current_day / horizon)  # Progress through horizon
    features.append((horizon - state.current_day) / horizon)  # Remaining fraction
    
    # Per-aircraft features (15 features each)
    for i in range(n_aircraft):
        aircraft = state.fleet[i]
        
        # Basic counters (normalized)
        features.extend([
            aircraft.dy_a / 120,      # Normalized A-check days
            aircraft.fh_a / 750,      # Normalized A-check FH
            aircraft.dy_c / 730,      # Normalized C-check days
            aircraft.fh_c / 7500,     # Normalized C-check FH
        ])
        
        # CRITICAL: Remaining days until check is due (what greedy uses!)
        remaining_a = compute_remaining_days(aircraft, 'A', config)
        remaining_c = compute_remaining_days(aircraft, 'C', config)
        features.extend([
            remaining_a / 120,   # Normalized remaining days for A
            remaining_c / 730,   # Normalized remaining days for C
        ])
        
        # Utilization (how close to needing check)
        util_a = compute_utilization(aircraft, 'A', config)
        util_c = compute_utilization(aircraft, 'C', config)
        features.extend([
            util_a,  # 0-1+ where 1 = at limit
            util_c,
        ])
        
        # Binary: needs check soon (within eligibility threshold)
        features.extend([
            float(remaining_a <= config.a_eligibility_threshold),  # A-check eligible
            float(remaining_c <= config.c_eligibility_threshold),  # C-check eligible
        ])
        
        # Binary: in maintenance
        features.append(float(aircraft.in_maintenance))
        
        # Urgency signal (negative remaining = overdue!)
        features.extend([
            min(1.0, max(-1.0, -remaining_a / 30)),  # Urgency for A (-1 to 1)
            min(1.0, max(-1.0, -remaining_c / 90)),  # Urgency for C
        ])
        
        # Priority rank (normalized 0-1, lower = more urgent)
        a_rank = a_priority_rank.get(i, n_aircraft) / n_aircraft
        c_rank = c_priority_rank.get(i, n_aircraft) / n_aircraft
        features.extend([a_rank, c_rank])
    
    # Scheduled today: binary indicator per aircraft
    scheduled = [0] * n_aircraft
    if state.current_day in state.schedule:
        for check_type, aircraft_id in state.schedule[state.current_day]:
            scheduled[aircraft_id] = 1
    features.extend(scheduled)
    
    # Capacity remaining today (normalized)
    features.append(state.c_checks_today / max(3, state.c_checks_today + 1))
    features.append(state.a_checks_today / max(3, state.a_checks_today + 1))
    
    # Fleet statistics (for global context)
    if a_remaining_list:
        all_a_rem = [r for _, r in a_remaining_list]
        features.extend([
            min(all_a_rem) / 120,  # Most urgent A-check
            np.mean(all_a_rem) / 120,
        ])
    else:
        features.extend([1.0, 1.0])
    
    if c_remaining_list:
        all_c_rem = [r for _, r in c_remaining_list]
        features.extend([
            min(all_c_rem) / 730,  # Most urgent C-check
            np.mean(all_c_rem) / 730,
        ])
    else:
        features.extend([1.0, 1.0])
    
    return np.array(features, dtype=np.float32)


def get_enhanced_state_dim(n_aircraft: int) -> int:
    """Compute state dimension for enhanced encoding."""
    # Time: 2, Per-aircraft: 15, Scheduled: n_aircraft, Capacity: 2, Fleet stats: 4
    return 2 + 15 * n_aircraft + n_aircraft + 2 + 4


# =============================================================================
# Potential-Based Reward Shaping
# =============================================================================

def compute_potential(
    state: SchedulingState,
    config: MaintenanceConfig,
    gamma: float = 0.99
) -> float:
    """
    Potential function for reward shaping.
    
    Key insight: We want to maximize utilization before scheduling.
    Potential should be HIGH when aircraft are close to optimal scheduling time.
    
    Φ(s) = -sum of expected wasted FH if scheduled now
    
    This creates shaping reward: γΦ(s') - Φ(s)
    - If s' has higher utilization, shaping reward is positive
    - Incentivizes waiting until closer to deadline
    """
    total_potential = 0.0
    
    for aircraft in state.fleet.values():
        if aircraft.in_maintenance:
            continue
        
        # A-check potential: how much FH would be wasted if scheduled now
        a_remaining = config.a_interval_fh - aircraft.fh_a
        a_potential = -max(0, a_remaining)  # Negative because waste is bad
        
        # C-check potential: same logic
        c_remaining = config.c_interval_fh - aircraft.fh_c  
        c_potential = -max(0, c_remaining)
        
        # Weight C-checks more (they waste more FH)
        total_potential += a_potential + 10 * c_potential
    
    # Normalize by fleet size
    return total_potential / len(state.fleet)


def shaped_reward(
    state: SchedulingState,
    action: int,
    next_state: SchedulingState,
    base_reward: float,
    config: MaintenanceConfig,
    gamma: float = 0.99,
    shaping_coeff: float = 0.01
) -> float:
    """
    Add potential-based shaping to reward.
    
    r_shaped = r + γΦ(s') - Φ(s)
    
    This preserves optimal policy but provides denser learning signal.
    """
    phi_s = compute_potential(state, config, gamma)
    phi_s_next = compute_potential(next_state, config, gamma)
    
    shaping = gamma * phi_s_next - phi_s
    
    return base_reward + shaping_coeff * shaping


# =============================================================================
# Improved PCL Agent
# =============================================================================

class ImprovedPCLAgent(nn.Module):
    """
    PCL agent with:
    - Deeper network with residual connections
    - Separate value and policy encoders
    - Layer normalization for stability
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 512,
        temperature: float = 10.0,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = temperature
        
        # Shared encoder with residual connections
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for _ in range(num_layers)
        ])
        
        # Value head (predicts V(s))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy head (predicts logits)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Input projection
        h = self.input_norm(F.relu(self.input_proj(state_tensor)))
        
        # Residual encoder layers
        for layer in self.encoder_layers:
            h = h + layer(h)  # Residual connection
        
        value = self.value_head(h).squeeze(-1)
        logits = self.policy_head(h)
        
        if squeeze_output:
            value = value.squeeze(0)
            logits = logits.squeeze(0)
        
        return value, logits
    
    def sample_action(
        self,
        state_tensor: torch.Tensor,
        valid_mask: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        with torch.no_grad():
            value, logits = self.forward(state_tensor)
            logits = logits.clone()
            logits[~torch.from_numpy(valid_mask)] = -1e9
            probs = F.softmax(logits / self.tau, dim=-1)  # Temperature in sampling!
            
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            
            log_prob = torch.log(probs[action] + 1e-10).item()
        
        return action, log_prob


# =============================================================================
# Multi-Step PCL Loss
# =============================================================================

def nstep_pcl_loss(
    agent: ImprovedPCLAgent,
    trajectory: List[Dict],
    n_steps: int = 10,
    gamma: float = 0.99,
    entropy_coeff: float = 0.1,  # Increased from 0.01
) -> Tuple[torch.Tensor, Dict]:
    """
    N-step PCL consistency loss.
    
    Instead of: V(s) + τ·log π(a|s) = r + V(s')
    We use:     V(s_t) + τ·Σ_k log π(a_{t+k}|s_{t+k}) = Σ_k γ^k r_{t+k} + γ^n V(s_{t+n})
    
    This helps with credit assignment over long horizons.
    """
    consistency_losses = []
    entropy_losses = []
    T = len(trajectory)
    
    for t in range(T):
        s = trajectory[t]['state']
        valid_mask = torch.from_numpy(trajectory[t]['valid_mask'])
        
        # Compute V(s_t) and log π(a_t|s_t)
        V_s, logits_s = agent(s)
        logits_s_masked = logits_s.clone()
        logits_s_masked[~valid_mask] = -1e9
        log_probs = F.log_softmax(logits_s_masked, dim=-1)
        
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        entropy_losses.append(entropy)
        
        # Compute n-step return
        n = min(n_steps, T - t)
        cumulative_log_prob = 0.0
        cumulative_reward = 0.0
        
        for k in range(n):
            a_k = trajectory[t + k]['action']
            r_k = trajectory[t + k]['reward']
            valid_k = torch.from_numpy(trajectory[t + k]['valid_mask'])
            
            # Get log prob for action at step t+k
            s_k = trajectory[t + k]['state']
            _, logits_k = agent(s_k)
            logits_k_masked = logits_k.clone()
            logits_k_masked[~valid_k] = -1e9
            log_probs_k = F.log_softmax(logits_k_masked, dim=-1)
            
            cumulative_log_prob = cumulative_log_prob + log_probs_k[a_k]
            cumulative_reward = cumulative_reward + (gamma ** k) * r_k
        
        # Bootstrap value at t+n
        if t + n < T and not trajectory[t + n - 1]['done']:
            s_n = trajectory[t + n]['state']
            with torch.no_grad():
                V_s_n, _ = agent(s_n)
            target = cumulative_reward + (gamma ** n) * V_s_n.detach()
        else:
            target = cumulative_reward
        
        # PCL consistency: V(s) + τ·Σ log π = Σ γ^k r + γ^n V(s')
        predicted = V_s + agent.tau * cumulative_log_prob
        # Handle both tensor and float targets
        if isinstance(target, torch.Tensor):
            consistency_loss = (predicted - target.detach()) ** 2
        else:
            consistency_loss = (predicted - target) ** 2
        
        consistency_losses.append(consistency_loss)
    
    consistency_loss_mean = torch.stack(consistency_losses).mean()
    entropy_loss_mean = torch.stack(entropy_losses).mean()
    
    # Normalize consistency loss to be on similar scale to entropy
    consistency_loss_normalized = consistency_loss_mean / (consistency_loss_mean.detach() + 1e-8)
    
    total_loss = consistency_loss_normalized - entropy_coeff * entropy_loss_mean
    
    return total_loss, {
        'consistency_loss': consistency_loss_mean.item(),
        'entropy': entropy_loss_mean.item(),
        'total_loss': total_loss.item(),
    }


# =============================================================================
# Trajectory Sampling with Reward Shaping
# =============================================================================

def sample_trajectory_enhanced(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    use_shaping: bool = True,
    shaping_coeff: float = 0.01,
    deterministic: bool = False,
    max_steps: int = None,
    verbose: bool = False
) -> Tuple[List[Dict], float]:
    """Sample trajectory with enhanced state encoding and reward shaping."""
    state = env.reset()
    trajectory = []
    total_return = 0.0
    done = False
    step = 0
    max_steps = max_steps or (env.horizon * 3)
    
    state_dim = get_enhanced_state_dim(env.n_aircraft)
    
    pbar = None
    if verbose:
        pbar = tqdm(total=env.horizon, desc="  Sampling", 
                    bar_format='{desc}: day {n}/{total} |{bar}| {percentage:3.0f}%')
    
    last_day = 0
    while not done and step < max_steps:
        state_tensor = torch.from_numpy(
            encode_state_enhanced(state, env.n_aircraft, env.horizon, config)
        ).float()
        
        valid_mask = env.get_valid_actions(state)
        action, log_prob = agent.sample_action(state_tensor, valid_mask, deterministic)
        next_state, base_reward, done, info = env.step(action)
        
        # Apply reward shaping
        if use_shaping:
            reward = shaped_reward(
                state, action, next_state, base_reward, 
                config, shaping_coeff=shaping_coeff
            )
        else:
            reward = base_reward
        
        next_state_tensor = torch.from_numpy(
            encode_state_enhanced(next_state, env.n_aircraft, env.horizon, config)
        ).float()
        
        trajectory.append({
            'state': state_tensor,
            'action': action,
            'reward': reward,
            'base_reward': base_reward,
            'next_state': next_state_tensor,
            'done': done,
            'log_prob': log_prob,
            'valid_mask': valid_mask
        })
        
        total_return += reward
        
        if pbar and next_state.current_day > last_day:
            pbar.update(next_state.current_day - last_day)
            last_day = next_state.current_day
        
        state = next_state
        step += 1
    
    if pbar:
        pbar.close()
    
    return trajectory, total_return


# =============================================================================
# Training with Curriculum
# =============================================================================

# =============================================================================
# Replay Buffer (Issue #5: Sample Efficiency)
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for PCL.
    
    Stores trajectory segments and samples mini-batches for training.
    This provides:
    - Decorrelated updates (samples from different trajectories)
    - Better sample efficiency (reuse past experience)
    - More stable training
    """
    
    def __init__(self, capacity: int = 100000, segment_length: int = 64):
        self.capacity = capacity
        self.segment_length = segment_length
        self.buffer: List[List[Dict]] = []
        self.position = 0
        self.total_transitions = 0
    
    def add_trajectory(self, trajectory: List[Dict]):
        """Add a trajectory to the buffer, splitting into segments."""
        T = len(trajectory)
        
        # Split trajectory into segments
        for start in range(0, T - self.segment_length + 1, self.segment_length // 2):
            segment = trajectory[start:start + self.segment_length]
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(segment)
            else:
                self.buffer[self.position] = segment
            
            self.position = (self.position + 1) % self.capacity
            self.total_transitions += len(segment)
    
    def sample(self, batch_size: int) -> List[List[Dict]]:
        """Sample batch_size segments from buffer."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, min_segments: int = 10) -> bool:
        """Check if buffer has enough data for training."""
        return len(self.buffer) >= min_segments


def train_pcl_improved(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    num_iterations: int = 500,
    lr: float = 1e-4,
    n_steps: int = 20,
    entropy_coeff: float = 0.1,
    eval_every: int = 50,
    n_eval_episodes: int = 3,
    temperature_schedule: str = 'annealing',
    use_shaping: bool = True,
    shaping_coeff: float = 0.01,
    save_path: str = None,
    verbose: bool = True,
    # Replay buffer parameters (Issue #5)
    buffer_capacity: int = 50000,
    segment_length: int = 64,
    batch_size: int = 32,
    gradient_steps_per_iter: int = 16,
    warmup_trajectories: int = 3,
):
    """
    Train improved PCL agent with replay buffer.
    
    Addresses all 5 critical issues:
    1. N-step consistency loss for credit assignment
    2. Reward shaping for wait action
    3. Higher entropy + loss normalization
    4. Enhanced state encoding with remaining days
    5. Replay buffer for sample-efficient batch updates
    """
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations, eta_min=1e-6)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, segment_length=segment_length)
    
    best_eval_return = -float('inf')
    training_history = defaultdict(list)
    
    initial_tau = agent.tau
    final_tau = 1.0  # Anneal to lower temperature (more deterministic)
    
    print("="*80)
    print("IMPROVED PCL TRAINING WITH REPLAY BUFFER")
    print("="*80)
    print(f"Environment: {env.n_aircraft} aircraft, {env.horizon} days ({env.horizon/365:.1f} years)")
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    print(f"Temperature: {initial_tau:.2f} → {final_tau:.2f}")
    print(f"N-step: {n_steps}, LR: {lr}, Entropy: {entropy_coeff}")
    print(f"Reward shaping: {use_shaping} (coeff={shaping_coeff})")
    print(f"Replay buffer: capacity={buffer_capacity}, segment_len={segment_length}")
    print(f"Batch: {batch_size} segments × {gradient_steps_per_iter} updates/iter")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Warmup: collect initial trajectories
    print(f"Warming up replay buffer with {warmup_trajectories} trajectories...")
    for i in range(warmup_trajectories):
        traj, ret = sample_trajectory_enhanced(
            agent, env, config,
            use_shaping=use_shaping,
            shaping_coeff=shaping_coeff,
            verbose=False
        )
        replay_buffer.add_trajectory(traj)
        print(f"  Warmup {i+1}/{warmup_trajectories}: {len(traj)} steps, return: {ret:.1f}")
    print(f"  Buffer size: {len(replay_buffer)} segments\n")
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Temperature annealing
        if temperature_schedule == 'annealing':
            progress = iteration / num_iterations
            agent.tau = initial_tau * (1 - progress) + final_tau * progress
        
        print(f"\n--- Iteration {iteration+1}/{num_iterations} (τ={agent.tau:.2f}) ---")
        
        # Sample new trajectory and add to buffer
        print("  Sampling trajectory...")
        traj_start = time.time()
        trajectory, total_return = sample_trajectory_enhanced(
            agent, env, config, 
            use_shaping=use_shaping,
            shaping_coeff=shaping_coeff,
            verbose=verbose
        )
        traj_time = time.time() - traj_start
        replay_buffer.add_trajectory(trajectory)
        print(f"  → {len(trajectory)} steps, return: {total_return:.1f} ({traj_time:.1f}s)")
        print(f"  → Buffer: {len(replay_buffer)} segments ({replay_buffer.total_transitions} total transitions)")
        
        # Multiple gradient updates from replay buffer
        print(f"  Training on {gradient_steps_per_iter} batches of {batch_size} segments...")
        batch_losses = []
        batch_entropies = []
        
        for _ in range(gradient_steps_per_iter):
            # Sample batch from replay buffer
            segments = replay_buffer.sample(batch_size)
            
            # Accumulate gradients over batch
            optimizer.zero_grad()
            total_loss = 0.0
            total_entropy = 0.0
            
            for segment in segments:
                loss, metrics = nstep_pcl_loss(
                    agent, segment, 
                    n_steps=n_steps,
                    entropy_coeff=entropy_coeff
                )
                # Scale loss for gradient accumulation
                (loss / len(segments)).backward()
                total_loss += metrics['total_loss']
                total_entropy += metrics['entropy']
            
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            batch_losses.append(total_loss / len(segments))
            batch_entropies.append(total_entropy / len(segments))
        
        scheduler.step()
        
        avg_loss = np.mean(batch_losses)
        avg_entropy = np.mean(batch_entropies)
        print(f"  → Avg loss: {avg_loss:.4f}, Avg entropy: {avg_entropy:.3f}")
        
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        remaining = (num_iterations - iteration - 1) * iter_time
        
        training_history['return'].append(total_return)
        training_history['loss'].append(avg_loss)
        training_history['entropy'].append(avg_entropy)
        training_history['temperature'].append(agent.tau)
        training_history['buffer_size'].append(len(replay_buffer))
        
        print(f"  → Iter time: {iter_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
        
        # Evaluation
        if (iteration + 1) % eval_every == 0:
            print(f"\n  Evaluating ({n_eval_episodes} episodes)...")
            eval_returns = []
            eval_base_returns = []
            
            for _ in range(n_eval_episodes):
                traj, ret = sample_trajectory_enhanced(
                    agent, env, config,
                    use_shaping=False,  # Evaluate on true reward
                    deterministic=False,
                    verbose=False
                )
                eval_returns.append(ret)
                base_ret = sum(t['base_reward'] for t in traj)
                eval_base_returns.append(base_ret)
            
            mean_return = np.mean(eval_returns)
            mean_base = np.mean(eval_base_returns)
            print(f"  → Eval return: {mean_return:.1f} ± {np.std(eval_returns):.1f}")
            print(f"  → Eval base return: {mean_base:.1f}")
            
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                if save_path:
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': agent.state_dict(),
                        'best_eval_return': best_eval_return,
                        'temperature': agent.tau,
                    }, save_path)
                    print(f"  → Saved best model!")
            
            # Checkpoint
            if save_path:
                checkpoint_path = save_path.replace('.pt', f'_iter{iteration+1}.pt')
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.state_dict(),
                    'eval_return': mean_return,
                    'temperature': agent.tau,
                }, checkpoint_path)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best eval return: {best_eval_return:.1f}")
    print(f"Final buffer: {len(replay_buffer)} segments")
    print(f"{'='*80}")
    
    return training_history


# =============================================================================
# Evaluation against Baselines
# =============================================================================

def evaluate_against_baselines(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    n_episodes: int = 5
):
    """Compare improved PCL with baselines."""
    from paper_dp import greedy_schedule
    
    print("\n" + "="*80)
    print("EVALUATION: IMPROVED PCL vs BASELINES")
    print("="*80)
    
    # 1. Greedy baseline
    print("\n1. Paper's Greedy (earliest-deadline-first):")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=False)
    c_greedy = sum(1 for day, checks in greedy_sched.items() for t, _ in checks if t == 'C')
    a_greedy = sum(1 for day, checks in greedy_sched.items() for t, _ in checks if t == 'A')
    print(f"   Cost: {greedy_cost:.0f} | C: {c_greedy} | A: {a_greedy}")
    
    # 2. PCL stochastic
    print(f"\n2. Improved PCL (stochastic, {n_episodes} episodes):")
    pcl_costs = []
    for _ in range(n_episodes):
        traj, _ = sample_trajectory_enhanced(
            agent, env, config,
            use_shaping=False,
            deterministic=False,
            verbose=False
        )
        # Compute wasted FH cost
        cost = 0.0
        for t in traj:
            action = t['action']
            if action < env.n_aircraft:
                # This is approximate - we'd need to track state
                pass
        # Use base return as proxy (it's negative wasted FH)
        cost = -sum(t['base_reward'] for t in traj)
        pcl_costs.append(cost)
    
    pcl_mean = np.mean(pcl_costs)
    pcl_std = np.std(pcl_costs)
    print(f"   Cost: {pcl_mean:.0f} ± {pcl_std:.0f}")
    
    # 3. PCL greedy
    print("\n3. Improved PCL (deterministic):")
    traj, _ = sample_trajectory_enhanced(
        agent, env, config,
        use_shaping=False,
        deterministic=True,
        verbose=False
    )
    pcl_det_cost = -sum(t['base_reward'] for t in traj)
    print(f"   Cost: {pcl_det_cost:.0f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'Wasted FH':>12} {'vs Greedy':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12}")
    print(f"{'Paper Greedy':<30} {greedy_cost:>12.0f} {'(baseline)':>12}")
    
    imp = (greedy_cost - pcl_det_cost) / greedy_cost * 100
    print(f"{'Improved PCL (deterministic)':<30} {pcl_det_cost:>12.0f} {imp:>+11.1f}%")
    
    imp_stoch = (greedy_cost - pcl_mean) / greedy_cost * 100
    print(f"{'Improved PCL (stochastic)':<30} {pcl_mean:>12.0f} {imp_stoch:>+11.1f}%")
    
    print("="*80)


# =============================================================================
# Main Training Script
# =============================================================================

if __name__ == "__main__":
    import argparse
    from data_loader import load_env, create_synthetic_env
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=15.0)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--entropy', type=float, default=0.1)
    parser.add_argument('--shaping-coeff', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=50)
    # Replay buffer parameters
    parser.add_argument('--buffer-capacity', type=int, default=50000)
    parser.add_argument('--segment-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gradient-steps', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load environment
    try:
        print("\nLoading data from Excel...")
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except Exception as e:
        print(f"Could not load Excel: {e}")
        print("\nUsing synthetic environment...")
        env = create_synthetic_env(n_aircraft=45, horizon_days=1461)
    
    config = env.config
    state_dim = get_enhanced_state_dim(env.n_aircraft)
    
    print(f"\nEnvironment ready:")
    print(f"  - Aircraft: {env.n_aircraft}")
    print(f"  - Horizon: {env.horizon} days")
    print(f"  - Enhanced state dim: {state_dim}")
    print(f"  - Action space: {env.n_actions}")
    
    # Create improved agent
    agent = ImprovedPCLAgent(
        state_dim=state_dim,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        num_layers=3
    )
    
    print(f"\nAgent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    
    # Train with replay buffer
    history = train_pcl_improved(
        agent=agent,
        env=env,
        config=config,
        num_iterations=args.iterations,
        lr=args.lr,
        n_steps=args.n_steps,
        entropy_coeff=args.entropy,
        eval_every=args.eval_every,
        use_shaping=True,
        shaping_coeff=args.shaping_coeff,
        save_path='best_pcl_improved.pt',
        verbose=True,
        # Replay buffer settings
        buffer_capacity=args.buffer_capacity,
        segment_length=args.segment_length,
        batch_size=args.batch_size,
        gradient_steps_per_iter=args.gradient_steps,
        warmup_trajectories=args.warmup,
    )
    
    # Load best and evaluate
    checkpoint = torch.load('best_pcl_improved.pt', weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    evaluate_against_baselines(agent, env, config, n_episodes=5)


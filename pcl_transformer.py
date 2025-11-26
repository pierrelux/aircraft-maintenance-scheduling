"""
Transformer-based PCL for Aircraft Maintenance Scheduling.

Key insight: Use self-attention over aircraft to learn which ones need scheduling.
The transformer can directly compare aircraft states and prioritize based on urgency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple
import time
from collections import defaultdict
from tqdm import tqdm

from sequential_env import (
    SequentialSchedulingEnv,
    SchedulingState,
    MaintenanceConfig,
)
from pcl_improved import (
    compute_remaining_days,
    compute_utilization,
    ReplayBuffer,
)


# =============================================================================
# Per-Aircraft Feature Encoding
# =============================================================================

def encode_aircraft_features(
    aircraft,
    config: MaintenanceConfig,
    current_day: int,
    horizon: int,
    scheduled_today: bool,
) -> np.ndarray:
    """
    Encode features for a single aircraft (15 features).
    """
    remaining_a = compute_remaining_days(aircraft, 'A', config)
    remaining_c = compute_remaining_days(aircraft, 'C', config)
    util_a = compute_utilization(aircraft, 'A', config)
    util_c = compute_utilization(aircraft, 'C', config)
    
    features = [
        # Basic counters (normalized)
        aircraft.dy_a / 120,
        aircraft.fh_a / 750,
        aircraft.dy_c / 730,
        aircraft.fh_c / 7500,
        # Remaining days (critical feature!)
        remaining_a / 120,
        remaining_c / 730,
        # Utilization
        util_a,
        util_c,
        # Eligibility
        float(remaining_a <= config.a_eligibility_threshold),
        float(remaining_c <= config.c_eligibility_threshold),
        # In maintenance
        float(aircraft.in_maintenance),
        # Urgency signals
        min(1.0, max(-1.0, -remaining_a / 30)),
        min(1.0, max(-1.0, -remaining_c / 90)),
        # Scheduled today
        float(scheduled_today),
        # Aircraft daily rate (normalized)
        aircraft.daily_fh / 12.0,
    ]
    return np.array(features, dtype=np.float32)


def encode_global_features(
    state: SchedulingState,
    n_aircraft: int,
    horizon: int,
    config: MaintenanceConfig,
) -> np.ndarray:
    """
    Encode global state features (8 features).
    """
    # Time
    progress = state.current_day / horizon
    remaining_frac = (horizon - state.current_day) / horizon
    
    # Fleet statistics
    a_remaining = []
    c_remaining = []
    for ac in state.fleet.values():
        if not ac.in_maintenance:
            a_remaining.append(compute_remaining_days(ac, 'A', config))
            c_remaining.append(compute_remaining_days(ac, 'C', config))
    
    if a_remaining:
        min_a = min(a_remaining) / 120
        mean_a = np.mean(a_remaining) / 120
    else:
        min_a = mean_a = 1.0
    
    if c_remaining:
        min_c = min(c_remaining) / 730
        mean_c = np.mean(c_remaining) / 730
    else:
        min_c = mean_c = 1.0
    
    features = [
        progress,
        remaining_frac,
        state.c_checks_today / 3,
        state.a_checks_today / 3,
        min_a,
        mean_a,
        min_c,
        mean_c,
    ]
    return np.array(features, dtype=np.float32)


def encode_state_for_transformer(
    state: SchedulingState,
    n_aircraft: int,
    horizon: int,
    config: MaintenanceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode state for transformer: (aircraft_features, global_features).
    
    Returns:
        aircraft_features: (n_aircraft, 15)
        global_features: (8,)
    """
    scheduled_today = set()
    if state.current_day in state.schedule:
        for _, aid in state.schedule[state.current_day]:
            scheduled_today.add(aid)
    
    aircraft_features = []
    for i in range(n_aircraft):
        ac = state.fleet[i]
        features = encode_aircraft_features(
            ac, config, state.current_day, horizon,
            i in scheduled_today
        )
        aircraft_features.append(features)
    
    aircraft_features = np.stack(aircraft_features, axis=0)  # (n_aircraft, 15)
    global_features = encode_global_features(state, n_aircraft, horizon, config)
    
    return aircraft_features, global_features


# =============================================================================
# Transformer Architecture
# =============================================================================

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for aircraft IDs."""
    
    def __init__(self, d_model: int, max_aircraft: int = 100):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_aircraft, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_aircraft, d_model)"""
        batch_size, n_aircraft, _ = x.shape
        positions = torch.arange(n_aircraft, device=x.device)
        pos_emb = self.pos_embedding(positions)  # (n_aircraft, d_model)
        return x + pos_emb.unsqueeze(0)


class TransformerPCLAgent(nn.Module):
    """
    Transformer-based PCL agent for aircraft scheduling.
    
    Architecture:
    1. Per-aircraft features → aircraft embeddings
    2. Self-attention over aircraft (learns to compare/prioritize)
    3. Global features → context embedding
    4. Cross-attention: aircraft attend to global context
    5. Output heads: value and policy
    """
    
    def __init__(
        self,
        aircraft_feature_dim: int = 15,
        global_feature_dim: int = 8,
        n_aircraft: int = 45,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 10.0,
    ):
        super().__init__()
        
        self.n_aircraft = n_aircraft
        self.d_model = d_model
        self.tau = temperature
        
        # Aircraft feature projection
        self.aircraft_proj = nn.Sequential(
            nn.Linear(aircraft_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Positional encoding for aircraft IDs
        self.pos_encoding = PositionalEncoding(d_model, n_aircraft)
        
        # Transformer encoder (self-attention over aircraft)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Cross-attention: aircraft attend to global context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Value head (single value for state)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Policy heads
        # C-check logits: per-aircraft score
        self.c_check_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # A-check logits: per-aircraft score  
        self.a_check_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Wait action logit
        self.wait_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        aircraft_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            aircraft_features: (batch, n_aircraft, 15)
            global_features: (batch, 8)
        
        Returns:
            value: (batch,)
            logits: (batch, 2*n_aircraft + 1)
        """
        batch_size = aircraft_features.shape[0]
        
        # Project aircraft features
        ac_emb = self.aircraft_proj(aircraft_features)  # (batch, n_aircraft, d_model)
        ac_emb = self.pos_encoding(ac_emb)
        
        # Project global features
        global_emb = self.global_proj(global_features)  # (batch, d_model)
        global_emb = global_emb.unsqueeze(1)  # (batch, 1, d_model)
        
        # Self-attention over aircraft
        ac_emb = self.transformer(ac_emb)  # (batch, n_aircraft, d_model)
        
        # Cross-attention with global context
        ac_emb, _ = self.cross_attn(ac_emb, global_emb, global_emb)
        
        # Value: pool aircraft embeddings
        pooled = ac_emb.mean(dim=1)  # (batch, d_model)
        value = self.value_head(pooled).squeeze(-1)  # (batch,)
        
        # Policy logits
        c_logits = self.c_check_head(ac_emb).squeeze(-1)  # (batch, n_aircraft)
        a_logits = self.a_check_head(ac_emb).squeeze(-1)  # (batch, n_aircraft)
        wait_logit = self.wait_head(pooled)  # (batch, 1)
        
        # Combine: [C-check for each, A-check for each, wait]
        logits = torch.cat([c_logits, a_logits, wait_logit], dim=-1)  # (batch, 2N+1)
        
        return value, logits
    
    def sample_action(
        self,
        aircraft_features: torch.Tensor,
        global_features: torch.Tensor,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Sample action with masking."""
        with torch.no_grad():
            # Add batch dimension if needed
            if aircraft_features.dim() == 2:
                aircraft_features = aircraft_features.unsqueeze(0)
                global_features = global_features.unsqueeze(0)
            
            value, logits = self.forward(aircraft_features, global_features)
            logits = logits.squeeze(0)  # Remove batch dim
            
            # Mask invalid actions
            logits = logits.clone()
            logits[~torch.from_numpy(valid_mask)] = -1e9
            
            probs = F.softmax(logits / self.tau, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            
            log_prob = torch.log(probs[action] + 1e-10).item()
        
        return action, log_prob


# =============================================================================
# Training
# =============================================================================

def sample_trajectory_transformer(
    agent: TransformerPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    deterministic: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict], float]:
    """Sample trajectory with transformer agent."""
    state = env.reset()
    trajectory = []
    total_return = 0.0
    done = False
    step = 0
    
    pbar = None
    if verbose:
        pbar = tqdm(total=env.horizon, desc="  Sampling", 
                    bar_format='{desc}: day {n}/{total} |{bar}| {percentage:3.0f}%')
    
    last_day = 0
    while not done and step < env.horizon * 3:
        ac_features, global_features = encode_state_for_transformer(
            state, env.n_aircraft, env.horizon, config
        )
        ac_tensor = torch.from_numpy(ac_features).float()
        global_tensor = torch.from_numpy(global_features).float()
        
        valid_mask = env.get_valid_actions(state)
        action, log_prob = agent.sample_action(
            ac_tensor, global_tensor, valid_mask, deterministic
        )
        
        next_state, reward, done, info = env.step(action)
        
        next_ac_features, next_global_features = encode_state_for_transformer(
            next_state, env.n_aircraft, env.horizon, config
        )
        
        trajectory.append({
            'ac_features': ac_tensor,
            'global_features': global_tensor,
            'action': action,
            'reward': reward,
            'next_ac_features': torch.from_numpy(next_ac_features).float(),
            'next_global_features': torch.from_numpy(next_global_features).float(),
            'done': done,
            'log_prob': log_prob,
            'valid_mask': valid_mask,
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


def pcl_loss_transformer(
    agent: TransformerPCLAgent,
    segment: List[Dict],
    n_steps: int = 10,
    gamma: float = 0.99,
    entropy_coeff: float = 0.1,
) -> Tuple[torch.Tensor, Dict]:
    """PCL loss for transformer agent."""
    T = len(segment)
    consistency_losses = []
    entropy_losses = []
    
    for t in range(T):
        ac_feat = segment[t]['ac_features'].unsqueeze(0)
        global_feat = segment[t]['global_features'].unsqueeze(0)
        valid_mask = torch.from_numpy(segment[t]['valid_mask'])
        
        V_s, logits_s = agent(ac_feat, global_feat)
        V_s = V_s.squeeze(0)
        logits_s = logits_s.squeeze(0)
        
        logits_s_masked = logits_s.clone()
        logits_s_masked[~valid_mask] = -1e9
        log_probs = F.log_softmax(logits_s_masked, dim=-1)
        
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        entropy_losses.append(entropy)
        
        # N-step return
        n = min(n_steps, T - t)
        cumulative_log_prob = 0.0
        cumulative_reward = 0.0
        
        for k in range(n):
            a_k = segment[t + k]['action']
            r_k = segment[t + k]['reward']
            valid_k = torch.from_numpy(segment[t + k]['valid_mask'])
            
            ac_k = segment[t + k]['ac_features'].unsqueeze(0)
            global_k = segment[t + k]['global_features'].unsqueeze(0)
            _, logits_k = agent(ac_k, global_k)
            logits_k = logits_k.squeeze(0)
            logits_k_masked = logits_k.clone()
            logits_k_masked[~valid_k] = -1e9
            log_probs_k = F.log_softmax(logits_k_masked, dim=-1)
            
            cumulative_log_prob = cumulative_log_prob + log_probs_k[a_k]
            cumulative_reward = cumulative_reward + (gamma ** k) * r_k
        
        # Bootstrap
        if t + n < T and not segment[t + n - 1]['done']:
            ac_n = segment[t + n]['ac_features'].unsqueeze(0)
            global_n = segment[t + n]['global_features'].unsqueeze(0)
            with torch.no_grad():
                V_s_n, _ = agent(ac_n, global_n)
                V_s_n = V_s_n.squeeze(0)
            target = cumulative_reward + (gamma ** n) * V_s_n.detach()
        else:
            target = cumulative_reward
        
        predicted = V_s + agent.tau * cumulative_log_prob
        if isinstance(target, torch.Tensor):
            consistency_loss = (predicted - target.detach()) ** 2
        else:
            consistency_loss = (predicted - target) ** 2
        
        consistency_losses.append(consistency_loss)
    
    consistency_loss_mean = torch.stack(consistency_losses).mean()
    entropy_loss_mean = torch.stack(entropy_losses).mean()
    
    # Normalize
    consistency_normalized = consistency_loss_mean / (consistency_loss_mean.detach() + 1e-8)
    total_loss = consistency_normalized - entropy_coeff * entropy_loss_mean
    
    return total_loss, {
        'consistency_loss': consistency_loss_mean.item(),
        'entropy': entropy_loss_mean.item(),
        'total_loss': total_loss.item(),
    }


class TransformerReplayBuffer:
    """Replay buffer for transformer agent."""
    
    def __init__(self, capacity: int = 50000, segment_length: int = 32):
        self.capacity = capacity
        self.segment_length = segment_length
        self.buffer: List[List[Dict]] = []
        self.position = 0
    
    def add_trajectory(self, trajectory: List[Dict]):
        T = len(trajectory)
        for start in range(0, T - self.segment_length + 1, self.segment_length // 2):
            segment = trajectory[start:start + self.segment_length]
            if len(self.buffer) < self.capacity:
                self.buffer.append(segment)
            else:
                self.buffer[self.position] = segment
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[List[Dict]]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def train_transformer_pcl(
    agent: TransformerPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    num_iterations: int = 300,
    lr: float = 1e-4,
    n_steps: int = 10,
    entropy_coeff: float = 0.2,
    eval_every: int = 50,
    batch_size: int = 8,
    gradient_steps: int = 8,
    segment_length: int = 32,
    warmup: int = 3,
    save_path: str = None,
    verbose: bool = True,
):
    """Train transformer PCL agent."""
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations, eta_min=1e-6)
    
    replay_buffer = TransformerReplayBuffer(capacity=30000, segment_length=segment_length)
    
    best_eval = -float('inf')
    history = defaultdict(list)
    
    initial_tau = agent.tau
    final_tau = 1.0
    
    print("="*80)
    print("TRANSFORMER PCL TRAINING")
    print("="*80)
    print(f"Environment: {env.n_aircraft} aircraft, {env.horizon} days")
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    print(f"Temperature: {initial_tau:.1f} → {final_tau:.1f}")
    print(f"N-step: {n_steps}, LR: {lr}, Entropy: {entropy_coeff}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Warmup
    print(f"Warming up with {warmup} trajectories...")
    for i in range(warmup):
        traj, ret = sample_trajectory_transformer(agent, env, config, verbose=False)
        replay_buffer.add_trajectory(traj)
        print(f"  Warmup {i+1}/{warmup}: {len(traj)} steps, return: {ret:.1f}")
    print(f"  Buffer: {len(replay_buffer)} segments\n")
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Temperature annealing
        progress = iteration / num_iterations
        agent.tau = initial_tau * (1 - progress) + final_tau * progress
        
        print(f"--- Iteration {iteration+1}/{num_iterations} (τ={agent.tau:.2f}) ---")
        
        # Sample trajectory
        traj, total_return = sample_trajectory_transformer(
            agent, env, config, verbose=verbose
        )
        replay_buffer.add_trajectory(traj)
        print(f"  → {len(traj)} steps, return: {total_return:.1f}")
        print(f"  → Buffer: {len(replay_buffer)} segments")
        
        # Train
        batch_losses = []
        batch_entropies = []
        
        for _ in range(gradient_steps):
            segments = replay_buffer.sample(batch_size)
            
            optimizer.zero_grad()
            total_loss = 0.0
            total_entropy = 0.0
            
            for segment in segments:
                loss, metrics = pcl_loss_transformer(
                    agent, segment, n_steps, entropy_coeff=entropy_coeff
                )
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
        
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        eta = (num_iterations - iteration - 1) * iter_time
        
        print(f"  → Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.3f}")
        print(f"  → Iter: {iter_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        
        history['return'].append(total_return)
        history['loss'].append(avg_loss)
        history['entropy'].append(avg_entropy)
        
        # Evaluation
        if (iteration + 1) % eval_every == 0:
            print(f"\n  Evaluating...")
            eval_returns = []
            for _ in range(3):
                traj, ret = sample_trajectory_transformer(
                    agent, env, config, deterministic=False, verbose=False
                )
                eval_returns.append(ret)
            
            mean_ret = np.mean(eval_returns)
            print(f"  → Eval return: {mean_ret:.1f} ± {np.std(eval_returns):.1f}")
            
            if mean_ret > best_eval:
                best_eval = mean_ret
                if save_path:
                    torch.save({
                        'model_state_dict': agent.state_dict(),
                        'iteration': iteration,
                        'best_return': best_eval,
                    }, save_path)
                    print(f"  → Saved best model!")
    
    print(f"\n{'='*80}")
    print(f"Training complete! Time: {(time.time()-start_time)/60:.1f}min")
    print(f"Best eval return: {best_eval:.1f}")
    print(f"{'='*80}")
    
    return history


def evaluate_transformer(
    agent: TransformerPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
):
    """Evaluate transformer agent vs greedy."""
    from paper_dp import greedy_schedule
    
    print("\n" + "="*80)
    print("EVALUATION: TRANSFORMER vs GREEDY")
    print("="*80)
    
    # Greedy
    print("\n1. Greedy baseline:")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=False)
    print(f"   Cost: {greedy_cost:.0f}")
    
    # Transformer
    print("\n2. Transformer agent:")
    costs = []
    for _ in range(5):
        state = env.reset()
        done = False
        total_cost = 0.0
        
        while not done:
            ac_feat, global_feat = encode_state_for_transformer(
                state, env.n_aircraft, env.horizon, config
            )
            valid_mask = env.get_valid_actions(state)
            action, _ = agent.sample_action(
                torch.from_numpy(ac_feat).float(),
                torch.from_numpy(global_feat).float(),
                valid_mask,
                deterministic=True
            )
            
            if action < env.n_aircraft:
                wasted = config.c_interval_fh - state.fleet[action].fh_c
                total_cost += max(0, wasted)
            elif action < 2 * env.n_aircraft:
                aid = action - env.n_aircraft
                wasted = config.a_interval_fh - state.fleet[aid].fh_a
                total_cost += max(0, wasted)
            
            state, _, done, _ = env.step(action)
        
        costs.append(total_cost)
    
    mean_cost = np.mean(costs)
    print(f"   Cost: {mean_cost:.0f} ± {np.std(costs):.0f}")
    
    improvement = (greedy_cost - mean_cost) / greedy_cost * 100
    print(f"\nImprovement vs greedy: {improvement:+.1f}%")
    print("="*80)


if __name__ == "__main__":
    import argparse
    from data_loader import load_env, create_synthetic_env
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--n-steps', type=int, default=10)
    parser.add_argument('--entropy', type=float, default=0.2)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gradient-steps', type=int, default=8)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load environment
    try:
        print("Loading data from Excel...")
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except Exception as e:
        print(f"Could not load Excel: {e}")
        print("Using synthetic environment...")
        env = create_synthetic_env(n_aircraft=45, horizon_days=1461)
    
    config = env.config
    
    print(f"\nEnvironment: {env.n_aircraft} aircraft, {env.horizon} days")
    
    # Create transformer agent
    agent = TransformerPCLAgent(
        aircraft_feature_dim=15,
        global_feature_dim=8,
        n_aircraft=env.n_aircraft,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        temperature=args.temperature,
    )
    
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    
    # Train
    history = train_transformer_pcl(
        agent=agent,
        env=env,
        config=config,
        num_iterations=args.iterations,
        lr=args.lr,
        n_steps=args.n_steps,
        entropy_coeff=args.entropy,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        save_path='best_transformer_pcl.pt',
        verbose=True,
    )
    
    # Load best and evaluate
    checkpoint = torch.load('best_transformer_pcl.pt', weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    evaluate_transformer(agent, env, config)


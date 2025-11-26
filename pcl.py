"""
Path Consistency Learning (PCL) for Aircraft Maintenance Scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import time
from collections import defaultdict
from tqdm import tqdm

from sequential_env import (
    SequentialSchedulingEnv,
    SchedulingState,
    encode_state,
)


# =============================================================================
# PCL Agent
# =============================================================================

class PCLAgent(nn.Module):
    """
    PCL agent for sequential schedule construction.
    
    Learns:
    - V(s): soft value function
    - π(a|s): policy distribution
    
    Trained with path consistency:
        V(s) + τ·log π(a|s) ≈ r(s,a) + V(s')
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        temperature: float = 10.0
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = temperature
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
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
        
        h = self.encoder(state_tensor)
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
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            
            log_prob = torch.log(probs[action] + 1e-10).item()
        
        return action, log_prob


# =============================================================================
# Trajectory Sampling
# =============================================================================

def sample_trajectory(
    agent: PCLAgent,
    env: SequentialSchedulingEnv,
    deterministic: bool = False,
    verbose: bool = False
) -> Tuple[List[Dict], float]:
    """Sample one complete trajectory."""
    state = env.reset()
    trajectory = []
    total_return = 0.0
    done = False
    step = 0
    
    # Progress bar for long trajectories
    pbar = None
    if verbose:
        pbar = tqdm(total=env.horizon, desc="  Sampling trajectory", 
                    bar_format='{desc}: day {n}/{total} |{bar}| {percentage:3.0f}%')
    
    last_day = 0
    while not done and step < 100000:
        state_tensor = torch.from_numpy(
            encode_state(state, env.n_aircraft, env.horizon)
        ).float()
        
        valid_mask = env.get_valid_actions(state)
        action, log_prob = agent.sample_action(state_tensor, valid_mask, deterministic)
        next_state, reward, done, info = env.step(action)
        
        next_state_tensor = torch.from_numpy(
            encode_state(next_state, env.n_aircraft, env.horizon)
        ).float()
        
        trajectory.append({
            'state': state_tensor,
            'action': action,
            'reward': reward,
            'next_state': next_state_tensor,
            'done': done,
            'log_prob': log_prob,
            'valid_mask': valid_mask
        })
        
        total_return += reward
        
        # Update progress bar when day changes
        if pbar and next_state.current_day > last_day:
            pbar.update(next_state.current_day - last_day)
            last_day = next_state.current_day
        
        state = next_state
        step += 1
    
    if pbar:
        pbar.close()
    
    return trajectory, total_return


# =============================================================================
# PCL Loss
# =============================================================================

def pcl_loss(
    agent: PCLAgent,
    trajectory: List[Dict],
    entropy_coeff: float = 0.01,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """Path Consistency Learning loss."""
    consistency_losses = []
    entropy_losses = []
    
    iterator = trajectory
    if verbose:
        iterator = tqdm(trajectory, desc="  Computing loss", leave=False)
    
    for transition in iterator:
        s = transition['state']
        a = transition['action']
        r = transition['reward']
        s_next = transition['next_state']
        done = transition['done']
        valid_mask = torch.from_numpy(transition['valid_mask'])
        
        V_s, logits_s = agent(s)
        logits_s_masked = logits_s.clone()
        logits_s_masked[~valid_mask] = -1e9
        log_probs = F.log_softmax(logits_s_masked, dim=-1)
        log_pi_a = log_probs[a]
        
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        
        if not done:
            with torch.no_grad():
                V_s_next, _ = agent(s_next)
        else:
            V_s_next = torch.tensor(0.0)
        
        target = r + V_s_next
        predicted = V_s + agent.tau * log_pi_a
        consistency_loss = (predicted - target) ** 2
        
        consistency_losses.append(consistency_loss)
        entropy_losses.append(entropy)
    
    consistency_loss_mean = torch.stack(consistency_losses).mean()
    entropy_loss_mean = torch.stack(entropy_losses).mean()
    total_loss = consistency_loss_mean - entropy_coeff * entropy_loss_mean
    
    return total_loss, {
        'consistency_loss': consistency_loss_mean.item(),
        'entropy': entropy_loss_mean.item(),
        'total_loss': total_loss.item()
    }


# =============================================================================
# Training
# =============================================================================

def train_pcl(
    agent: PCLAgent,
    env: SequentialSchedulingEnv,
    num_iterations: int = 100,
    lr: float = 3e-4,
    entropy_coeff: float = 0.01,
    eval_every: int = 50,
    n_eval_episodes: int = 5,
    temperature_schedule: str = 'constant',
    save_path: str = None,
    verbose: bool = True
):
    """Train PCL agent."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    best_eval_return = -float('inf')
    training_history = defaultdict(list)
    
    print("="*80)
    print("TRAINING PCL AGENT")
    print("="*80)
    print(f"Environment: {env.n_aircraft} aircraft, {env.horizon} days ({env.horizon/365:.1f} years)")
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    print(f"Temperature: {agent.tau:.2f}, LR: {lr}, Entropy: {entropy_coeff}")
    print(f"Iterations: {num_iterations}, Eval every: {eval_every}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Temperature schedule
        if temperature_schedule == 'linear':
            agent.tau = 20.0 + (1.0 - 20.0) * (iteration / num_iterations)
        elif temperature_schedule == 'exponential':
            agent.tau = max(1.0, 20.0 * (0.9999 ** iteration))
        
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # Sample trajectory with progress
        print("  Sampling trajectory...")
        traj_start = time.time()
        trajectory, total_return = sample_trajectory(agent, env, verbose=verbose)
        traj_time = time.time() - traj_start
        print(f"  → {len(trajectory)} steps, return: {total_return:.1f} ({traj_time:.1f}s)")
        
        # Compute loss
        print("  Computing loss...")
        loss_start = time.time()
        loss, metrics = pcl_loss(agent, trajectory, entropy_coeff, verbose=verbose)
        loss_time = time.time() - loss_start
        print(f"  → Loss: {metrics['total_loss']:.3f} ({loss_time:.1f}s)")
        
        # Optimize
        print("  Optimizing...")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        remaining = (num_iterations - iteration - 1) * iter_time
        
        training_history['return'].append(total_return)
        training_history['loss'].append(metrics['total_loss'])
        training_history['trajectory_length'].append(len(trajectory))
        
        print(f"  → Done! Iter time: {iter_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
        
        # Evaluation
        if (iteration + 1) % eval_every == 0:
            print(f"\n  Evaluating ({n_eval_episodes} episodes)...")
            eval_metrics = evaluate_agent(agent, env, n_eval_episodes, verbose=False)
            print(f"  → Eval return: {eval_metrics['mean_return']:.1f} ± {eval_metrics['std_return']:.1f}")
            
            if eval_metrics['mean_return'] > best_eval_return:
                best_eval_return = eval_metrics['mean_return']
                if save_path:
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': agent.state_dict(),
                        'best_eval_return': best_eval_return,
                    }, save_path)
                    print(f"  → Saved best model!")
            
            # Always save a checkpoint at each eval (for recovery)
            if save_path:
                checkpoint_path = save_path.replace('.pt', f'_iter{iteration+1}.pt')
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.state_dict(),
                    'eval_return': eval_metrics['mean_return'],
                }, checkpoint_path)
                print(f"  → Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")
    return training_history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_agent(
    agent: PCLAgent,
    env: SequentialSchedulingEnv,
    n_episodes: int = 5,
    verbose: bool = True
) -> Dict:
    """Evaluate agent."""
    returns = []
    lengths = []
    
    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating")
    
    for _ in iterator:
        trajectory, total_return = sample_trajectory(agent, env, deterministic=False, verbose=False)
        returns.append(total_return)
        lengths.append(len(trajectory))
    
    metrics = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_length': np.mean(lengths),
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("EVALUATION")
        print(f"{'='*80}")
        print(f"Episodes: {n_episodes}")
        print(f"Return:   {metrics['mean_return']:8.1f} ± {metrics['std_return']:6.1f}")
        print(f"          [{metrics['min_return']:8.1f}, {metrics['max_return']:8.1f}]")
        print(f"{'='*80}\n")
    
    return metrics


def compare_with_baseline(
    agent: PCLAgent,
    env: SequentialSchedulingEnv,
    n_episodes: int = 5
):
    """Compare with baseline policies."""
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*80}\n")
    
    # PCL stochastic
    print("Evaluating PCL (stochastic)...")
    pcl_returns = []
    for _ in tqdm(range(n_episodes), desc="PCL stochastic"):
        _, ret = sample_trajectory(agent, env, False, verbose=False)
        pcl_returns.append(ret)
    print(f"  → {np.mean(pcl_returns):.1f} ± {np.std(pcl_returns):.1f}")
    
    # PCL greedy
    print("Evaluating PCL (greedy)...")
    greedy_returns = []
    for _ in tqdm(range(n_episodes), desc="PCL greedy"):
        _, ret = sample_trajectory(agent, env, True, verbose=False)
        greedy_returns.append(ret)
    print(f"  → {np.mean(greedy_returns):.1f} ± {np.std(greedy_returns):.1f}")
    
    # Random
    print("Evaluating Random...")
    random_returns = []
    for _ in tqdm(range(n_episodes), desc="Random"):
        state = env.reset()
        total_return = 0.0
        done = False
        while not done:
            valid_mask = env.get_valid_actions(state)
            action = np.random.choice(np.where(valid_mask)[0])
            state, reward, done, _ = env.step(action)
            total_return += reward
        random_returns.append(total_return)
    print(f"  → {np.mean(random_returns):.1f} ± {np.std(random_returns):.1f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Mean Return':>15}")
    print(f"{'-'*25} {'-'*15}")
    print(f"{'PCL (stochastic)':<25} {np.mean(pcl_returns):>15.1f}")
    print(f"{'PCL (greedy)':<25} {np.mean(greedy_returns):>15.1f}")
    print(f"{'Random':<25} {np.mean(random_returns):>15.1f}")
    print(f"{'='*80}\n")

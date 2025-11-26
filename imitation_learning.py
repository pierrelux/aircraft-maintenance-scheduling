"""
Imitation Learning from Greedy Baseline.

Key insight: The greedy baseline is already near-optimal for this problem.
Instead of learning from scratch with RL, we:
1. Generate expert trajectories using greedy policy
2. Train policy via supervised learning (behavioral cloning)
3. Optionally fine-tune with PCL/RL

This approach is much more sample-efficient and provides a strong starting point.
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
    MaintenanceConfig,
)
from pcl_improved import (
    ImprovedPCLAgent,
    encode_state_enhanced,
    get_enhanced_state_dim,
    compute_remaining_days,
    sample_trajectory_enhanced,
    nstep_pcl_loss,
)
from paper_dp import compute_remaining_days as dp_remaining_days


# =============================================================================
# Expert Policy (Greedy)
# =============================================================================

def greedy_action(
    state: SchedulingState,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    r_a_threshold: float = 30.0,
    r_c_threshold: float = 90.0,
) -> int:
    """
    Compute greedy action following paper's earliest-deadline-first rule.
    
    Priority:
    1. C-check for aircraft with earliest deadline (if within threshold)
    2. A-check for aircraft with earliest deadline (if within threshold)
    3. Advance to next day
    """
    valid_mask = env.get_valid_actions(state)
    
    # Check if any C-check is valid and urgent
    c_candidates = []
    for i in range(env.n_aircraft):
        if valid_mask[i]:  # C-check for aircraft i is valid
            aircraft = state.fleet[i]
            remaining = compute_remaining_days(aircraft, 'C', config)
            if remaining <= r_c_threshold:
                c_candidates.append((i, remaining))
    
    if c_candidates:
        # Schedule the most urgent C-check
        c_candidates.sort(key=lambda x: x[1])
        return c_candidates[0][0]
    
    # Check if any A-check is valid and urgent
    a_candidates = []
    for i in range(env.n_aircraft):
        action_idx = env.n_aircraft + i
        if valid_mask[action_idx]:  # A-check for aircraft i is valid
            aircraft = state.fleet[i]
            remaining = compute_remaining_days(aircraft, 'A', config)
            if remaining <= r_a_threshold:
                a_candidates.append((action_idx, remaining))
    
    if a_candidates:
        # Schedule the most urgent A-check
        a_candidates.sort(key=lambda x: x[1])
        return a_candidates[0][0]
    
    # Default: advance to next day
    return 2 * env.n_aircraft


def collect_expert_trajectories(
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    n_trajectories: int = 10,
    verbose: bool = True
) -> List[Dict]:
    """Collect trajectories using greedy expert policy."""
    all_data = []
    
    iterator = range(n_trajectories)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting expert data")
    
    for _ in iterator:
        state = env.reset()
        done = False
        trajectory_data = []
        
        while not done:
            state_tensor = encode_state_enhanced(
                state, env.n_aircraft, env.horizon, config
            )
            valid_mask = env.get_valid_actions(state)
            
            # Get expert action
            expert_action = greedy_action(state, env, config)
            
            trajectory_data.append({
                'state': torch.from_numpy(state_tensor).float(),
                'action': expert_action,
                'valid_mask': valid_mask,
            })
            
            state, _, done, _ = env.step(expert_action)
        
        all_data.extend(trajectory_data)
    
    if verbose:
        print(f"Collected {len(all_data)} state-action pairs from {n_trajectories} trajectories")
    
    return all_data


# =============================================================================
# Behavioral Cloning
# =============================================================================

def behavioral_cloning(
    agent: ImprovedPCLAgent,
    expert_data: List[Dict],
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """
    Train policy to imitate expert actions via supervised learning.
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    n_samples = len(expert_data)
    indices = np.arange(n_samples)
    
    history = defaultdict(list)
    
    if verbose:
        print("\n" + "="*80)
        print("BEHAVIORAL CLONING")
        print("="*80)
        print(f"Expert samples: {n_samples}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
        print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            # Prepare batch
            states = torch.stack([expert_data[i]['state'] for i in batch_indices])
            actions = torch.tensor([expert_data[i]['action'] for i in batch_indices])
            valid_masks = torch.tensor(
                np.array([expert_data[i]['valid_mask'] for i in batch_indices])
            )
            
            # Forward pass
            _, logits = agent(states)
            
            # Mask invalid actions
            logits = logits.clone()
            logits[~valid_masks] = -1e9
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, actions)
            
            # Accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == actions).float().mean().item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")
    
    if verbose:
        print(f"\nBehavioral cloning complete!")
        print(f"Final accuracy: {history['accuracy'][-1]:.3f}")
    
    return history


# =============================================================================
# Fine-tuning with PCL
# =============================================================================

def finetune_with_pcl(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    num_iterations: int = 100,
    lr: float = 1e-5,  # Lower LR for fine-tuning
    n_steps: int = 20,
    entropy_coeff: float = 0.05,  # Lower entropy to stay close to expert
    save_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Fine-tune behavioral cloned policy with PCL.
    
    Key: Start from a good policy (from BC) and improve with RL.
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    history = defaultdict(list)
    best_return = -float('inf')
    
    if verbose:
        print("\n" + "="*80)
        print("FINE-TUNING WITH PCL")
        print("="*80)
        print(f"Iterations: {num_iterations}, LR: {lr}")
        print("="*80 + "\n")
    
    for iteration in range(num_iterations):
        # Sample trajectory
        trajectory, total_return = sample_trajectory_enhanced(
            agent, env, config,
            use_shaping=True,
            deterministic=False,
            verbose=False
        )
        
        # Compute n-step PCL loss
        loss, metrics = nstep_pcl_loss(
            agent, trajectory,
            n_steps=n_steps,
            entropy_coeff=entropy_coeff
        )
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  # Tighter clipping
        optimizer.step()
        
        history['return'].append(total_return)
        history['loss'].append(metrics['total_loss'])
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1:3d}: Return={total_return:.1f}, Loss={metrics['total_loss']:.4f}")
        
        if total_return > best_return:
            best_return = total_return
            if save_path:
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.state_dict(),
                    'best_return': best_return,
                }, save_path)
    
    if verbose:
        print(f"\nFine-tuning complete! Best return: {best_return:.1f}")
    
    return history


# =============================================================================
# DAgger (Dataset Aggregation)
# =============================================================================

def dagger_training(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    num_rounds: int = 10,
    trajectories_per_round: int = 5,
    bc_epochs_per_round: int = 20,
    beta_schedule: str = 'linear',  # How to mix expert and agent
    verbose: bool = True
) -> Dict:
    """
    DAgger: Iteratively collect data with agent, label with expert, retrain.
    
    This helps the agent learn to recover from its own distribution.
    """
    all_data = []
    history = defaultdict(list)
    
    if verbose:
        print("\n" + "="*80)
        print("DAgger TRAINING")
        print("="*80)
        print(f"Rounds: {num_rounds}, Trajectories per round: {trajectories_per_round}")
        print("="*80 + "\n")
    
    for round_idx in range(num_rounds):
        # Beta: probability of using expert action (decreases over time)
        if beta_schedule == 'linear':
            beta = 1.0 - round_idx / num_rounds
        else:
            beta = 0.5 ** round_idx
        
        if verbose:
            print(f"\n--- Round {round_idx+1}/{num_rounds} (β={beta:.2f}) ---")
        
        # Collect trajectories with mixed policy
        round_data = []
        for _ in range(trajectories_per_round):
            state = env.reset()
            done = False
            
            while not done:
                state_tensor = encode_state_enhanced(
                    state, env.n_aircraft, env.horizon, config
                )
                valid_mask = env.get_valid_actions(state)
                
                # Get expert label
                expert_action = greedy_action(state, env, config)
                
                # Choose action (agent or expert based on beta)
                if np.random.random() < beta:
                    action = expert_action
                else:
                    action, _ = agent.sample_action(
                        torch.from_numpy(state_tensor).float(),
                        valid_mask,
                        deterministic=False
                    )
                
                # Store with EXPERT label (key insight of DAgger)
                round_data.append({
                    'state': torch.from_numpy(state_tensor).float(),
                    'action': expert_action,  # Always label with expert
                    'valid_mask': valid_mask,
                })
                
                state, _, done, _ = env.step(action)
        
        all_data.extend(round_data)
        
        if verbose:
            print(f"  Collected {len(round_data)} samples, total: {len(all_data)}")
        
        # Retrain with all collected data
        bc_history = behavioral_cloning(
            agent, all_data,
            num_epochs=bc_epochs_per_round,
            batch_size=256,
            lr=1e-3 * (0.9 ** round_idx),  # Decay LR
            verbose=False
        )
        
        history['accuracy'].append(bc_history['accuracy'][-1])
        history['loss'].append(bc_history['loss'][-1])
        
        if verbose:
            print(f"  BC: Loss={bc_history['loss'][-1]:.4f}, Acc={bc_history['accuracy'][-1]:.3f}")
    
    return history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_imitation_agent(
    agent: ImprovedPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    n_episodes: int = 5
):
    """Evaluate imitation learning agent against baselines."""
    from paper_dp import greedy_schedule
    
    print("\n" + "="*80)
    print("EVALUATION: IMITATION LEARNING AGENT")
    print("="*80)
    
    # 1. Greedy baseline
    print("\n1. Paper's Greedy (expert):")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=False)
    print(f"   Cost: {greedy_cost:.0f}")
    
    # 2. Imitation agent (deterministic)
    print("\n2. Imitation Agent (deterministic):")
    costs = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_cost = 0.0
        
        while not done:
            state_tensor = torch.from_numpy(
                encode_state_enhanced(state, env.n_aircraft, env.horizon, config)
            ).float()
            valid_mask = env.get_valid_actions(state)
            action, _ = agent.sample_action(state_tensor, valid_mask, deterministic=True)
            
            # Track cost
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
    std_cost = np.std(costs)
    print(f"   Cost: {mean_cost:.0f} ± {std_cost:.0f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    improvement = (greedy_cost - mean_cost) / greedy_cost * 100
    print(f"Greedy (expert):     {greedy_cost:>10.0f}")
    print(f"Imitation Agent:     {mean_cost:>10.0f} ({improvement:+.1f}%)")
    print("="*80)
    
    return {'greedy': greedy_cost, 'imitation': mean_cost}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    from data_loader import load_env, create_synthetic_env
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['bc', 'dagger', 'bc+pcl'], default='bc')
    parser.add_argument('--n-expert-traj', type=int, default=20)
    parser.add_argument('--bc-epochs', type=int, default=100)
    parser.add_argument('--dagger-rounds', type=int, default=10)
    parser.add_argument('--finetune-iterations', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=512)
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
    state_dim = get_enhanced_state_dim(env.n_aircraft)
    
    print(f"\nEnvironment: {env.n_aircraft} aircraft, {env.horizon} days")
    print(f"State dim: {state_dim}, Actions: {env.n_actions}")
    
    # Create agent
    agent = ImprovedPCLAgent(
        state_dim=state_dim,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        temperature=10.0,
        num_layers=3
    )
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    
    if args.method == 'bc':
        # Pure behavioral cloning
        print("\n*** Behavioral Cloning ***")
        expert_data = collect_expert_trajectories(env, config, args.n_expert_traj)
        bc_history = behavioral_cloning(agent, expert_data, args.bc_epochs)
        
    elif args.method == 'dagger':
        # DAgger
        print("\n*** DAgger Training ***")
        dagger_history = dagger_training(
            agent, env, config,
            num_rounds=args.dagger_rounds,
            trajectories_per_round=5
        )
        
    elif args.method == 'bc+pcl':
        # BC then fine-tune with PCL
        print("\n*** BC + PCL Fine-tuning ***")
        expert_data = collect_expert_trajectories(env, config, args.n_expert_traj)
        bc_history = behavioral_cloning(agent, expert_data, args.bc_epochs)
        finetune_history = finetune_with_pcl(
            agent, env, config,
            num_iterations=args.finetune_iterations,
            save_path='best_imitation_pcl.pt'
        )
    
    # Save final model
    torch.save({
        'model_state_dict': agent.state_dict(),
        'method': args.method,
    }, f'imitation_{args.method}.pt')
    
    # Evaluate
    evaluate_imitation_agent(agent, env, config, n_episodes=5)


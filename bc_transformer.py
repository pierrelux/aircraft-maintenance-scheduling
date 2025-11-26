"""
Behavioral Cloning with Transformer for Aircraft Scheduling.

Strategy:
1. Collect expert trajectories from greedy baseline
2. Train transformer to predict greedy actions (BC)  
3. Optionally fine-tune with PCL to potentially beat greedy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
from collections import defaultdict

from sequential_env import SequentialSchedulingEnv, SchedulingState, MaintenanceConfig
from pcl_transformer import (
    TransformerPCLAgent,
    encode_state_for_transformer,
    encode_aircraft_features,
    encode_global_features,
)
from pcl_improved import compute_remaining_days, compute_utilization


def greedy_action(
    state: SchedulingState,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    r_a_threshold: float = 30.0,
    r_c_threshold: float = 90.0,
) -> int:
    """Get greedy action for current state."""
    # Check valid actions
    valid_mask = env.get_valid_actions(state)
    
    # C-check priorities (higher priority = more urgent)
    c_priorities = []
    for aid in range(env.n_aircraft):
        if not valid_mask[aid]:
            c_priorities.append((-float('inf'), aid))
            continue
        ac = state.fleet[aid]
        remaining = compute_remaining_days(ac, 'C', config)
        if remaining <= r_c_threshold:
            c_priorities.append((remaining, aid))
        else:
            c_priorities.append((-float('inf'), aid))
    
    c_priorities.sort(key=lambda x: x[0])
    
    # Schedule C-check if any urgent
    for remaining, aid in c_priorities:
        if remaining > -float('inf') and valid_mask[aid]:
            return aid  # C-check action
    
    # A-check priorities
    a_priorities = []
    for aid in range(env.n_aircraft):
        action = env.n_aircraft + aid
        if not valid_mask[action]:
            a_priorities.append((-float('inf'), aid))
            continue
        ac = state.fleet[aid]
        remaining = compute_remaining_days(ac, 'A', config)
        if remaining <= r_a_threshold:
            a_priorities.append((remaining, aid))
        else:
            a_priorities.append((-float('inf'), aid))
    
    a_priorities.sort(key=lambda x: x[0])
    
    # Schedule A-check if any urgent
    for remaining, aid in a_priorities:
        if remaining > -float('inf') and valid_mask[env.n_aircraft + aid]:
            return env.n_aircraft + aid  # A-check action
    
    # Otherwise, advance to next day
    return 2 * env.n_aircraft


def collect_bc_data(
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    n_trajectories: int = 20,
) -> List[Dict]:
    """Collect (state, action) pairs from greedy expert."""
    print(f"\nCollecting {n_trajectories} expert trajectories...")
    
    all_data = []
    
    for traj_idx in tqdm(range(n_trajectories), desc="  Collecting"):
        state = env.reset()
        done = False
        
        while not done:
            ac_features, global_features = encode_state_for_transformer(
                state, env.n_aircraft, env.horizon, config
            )
            valid_mask = env.get_valid_actions(state)
            expert_action = greedy_action(state, env, config)
            
            all_data.append({
                'ac_features': torch.from_numpy(ac_features).float(),
                'global_features': torch.from_numpy(global_features).float(),
                'action': expert_action,
                'valid_mask': valid_mask.copy(),
            })
            
            state, _, done, _ = env.step(expert_action)
    
    print(f"  Collected {len(all_data)} state-action pairs")
    return all_data


def train_bc(
    agent: TransformerPCLAgent,
    bc_data: List[Dict],
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> Dict:
    """Train transformer with behavioral cloning."""
    print(f"\n{'='*60}")
    print("BEHAVIORAL CLONING TRAINING")
    print(f"{'='*60}")
    print(f"Data: {len(bc_data)} examples")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    n_batches = len(bc_data) // batch_size
    history = defaultdict(list)
    
    for epoch in range(epochs):
        np.random.shuffle(bc_data)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_samples = 0
        
        for batch_idx in range(n_batches):
            batch = bc_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # Stack batch
            ac_feat = torch.stack([b['ac_features'] for b in batch])
            global_feat = torch.stack([b['global_features'] for b in batch])
            actions = torch.tensor([b['action'] for b in batch], dtype=torch.long)
            valid_masks = np.stack([b['valid_mask'] for b in batch])
            
            # Forward
            _, logits = agent(ac_feat, global_feat)
            
            # Mask invalid actions
            mask_tensor = torch.from_numpy(~valid_masks).bool()
            logits = logits.masked_fill(mask_tensor, -1e9)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, actions)
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == actions).float().mean().item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
            epoch_acc += acc * len(batch)
            n_samples += len(batch)
        
        scheduler.step()
        
        epoch_loss /= n_samples
        epoch_acc /= n_samples
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.1%}")
    
    print(f"\nBC training complete. Final accuracy: {epoch_acc:.1%}")
    return history


def evaluate_bc_agent(
    agent: TransformerPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    n_episodes: int = 5,
) -> Tuple[float, float]:
    """Evaluate BC-trained agent."""
    costs = []
    
    for _ in range(n_episodes):
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
            
            # Track wasted FH
            if action < env.n_aircraft:
                wasted = config.c_interval_fh - state.fleet[action].fh_c
                total_cost += max(0, wasted)
            elif action < 2 * env.n_aircraft:
                aid = action - env.n_aircraft
                wasted = config.a_interval_fh - state.fleet[aid].fh_a
                total_cost += max(0, wasted)
            
            state, _, done, _ = env.step(action)
        
        costs.append(total_cost)
    
    return np.mean(costs), np.std(costs)


def fine_tune_with_pcl(
    agent: TransformerPCLAgent,
    env: SequentialSchedulingEnv,
    config: MaintenanceConfig,
    num_iterations: int = 100,
    lr: float = 1e-5,
    n_steps: int = 10,
    entropy_coeff: float = 0.05,
    eval_every: int = 25,
):
    """Fine-tune BC-pretrained agent with PCL."""
    from pcl_transformer import (
        TransformerReplayBuffer,
        sample_trajectory_transformer,
        pcl_loss_transformer,
    )
    
    print(f"\n{'='*60}")
    print("PCL FINE-TUNING")
    print(f"{'='*60}")
    
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-5)
    replay_buffer = TransformerReplayBuffer(capacity=20000, segment_length=32)
    
    initial_tau = 5.0
    final_tau = 1.0
    agent.tau = initial_tau
    
    # Warmup
    print("Warming up...")
    for i in range(2):
        traj, ret = sample_trajectory_transformer(agent, env, config, verbose=False)
        replay_buffer.add_trajectory(traj)
        print(f"  Warmup {i+1}: {len(traj)} steps, return: {ret:.1f}")
    
    for iteration in range(num_iterations):
        # Temperature annealing
        progress = iteration / num_iterations
        agent.tau = initial_tau * (1 - progress) + final_tau * progress
        
        # Sample
        traj, total_return = sample_trajectory_transformer(
            agent, env, config, verbose=False
        )
        replay_buffer.add_trajectory(traj)
        
        # Train
        segments = replay_buffer.sample(8)
        optimizer.zero_grad()
        
        total_loss = 0.0
        for segment in segments:
            loss, _ = pcl_loss_transformer(agent, segment, n_steps, entropy_coeff=entropy_coeff)
            (loss / len(segments)).backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration+1:3d}/{num_iterations}: return={total_return:.1f}, τ={agent.tau:.2f}")
        
        # Evaluate
        if (iteration + 1) % eval_every == 0:
            mean_cost, std_cost = evaluate_bc_agent(agent, env, config, n_episodes=3)
            print(f"  → Eval: {mean_cost:.0f} ± {std_cost:.0f}")


def main():
    import argparse
    from data_loader import load_env, create_synthetic_env
    from paper_dp import greedy_schedule
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc-epochs', type=int, default=50)
    parser.add_argument('--bc-lr', type=float, default=1e-3)
    parser.add_argument('--n-expert-traj', type=int, default=20)
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--fine-tune-iters', type=int, default=100)
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
    
    # Create agent
    agent = TransformerPCLAgent(
        aircraft_feature_dim=15,
        global_feature_dim=8,
        n_aircraft=env.n_aircraft,
        d_model=128,
        n_heads=4,
        n_layers=3,
        temperature=1.0,  # Low temp for deterministic BC
    )
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} parameters")
    
    # Greedy baseline
    print("\n" + "="*60)
    print("GREEDY BASELINE")
    print("="*60)
    _, greedy_cost = greedy_schedule(env, verbose=False)
    print(f"Greedy cost: {greedy_cost:.0f}")
    
    # Collect BC data
    bc_data = collect_bc_data(env, config, n_trajectories=args.n_expert_traj)
    
    # Train BC
    train_bc(agent, bc_data, epochs=args.bc_epochs, lr=args.bc_lr)
    
    # Evaluate BC
    print("\n" + "="*60)
    print("BC AGENT EVALUATION")
    print("="*60)
    bc_cost, bc_std = evaluate_bc_agent(agent, env, config, n_episodes=5)
    print(f"BC agent cost: {bc_cost:.0f} ± {bc_std:.0f}")
    print(f"Greedy cost:   {greedy_cost:.0f}")
    
    match_pct = 100 * (1 - abs(bc_cost - greedy_cost) / greedy_cost)
    print(f"Match with greedy: {match_pct:.1f}%")
    
    # Fine-tune with PCL
    if args.fine_tune:
        fine_tune_with_pcl(
            agent, env, config,
            num_iterations=args.fine_tune_iters,
        )
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION (after PCL fine-tuning)")
        print("="*60)
        final_cost, final_std = evaluate_bc_agent(agent, env, config, n_episodes=5)
        print(f"Final cost: {final_cost:.0f} ± {final_std:.0f}")
        print(f"Greedy:     {greedy_cost:.0f}")
        
        improvement = (greedy_cost - final_cost) / greedy_cost * 100
        print(f"Improvement over greedy: {improvement:+.1f}%")
    
    # Save model
    torch.save({
        'model_state_dict': agent.state_dict(),
        'bc_cost': bc_cost,
        'greedy_cost': greedy_cost,
    }, 'bc_transformer.pt')
    print("\nModel saved to bc_transformer.pt")


if __name__ == "__main__":
    main()


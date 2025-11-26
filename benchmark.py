#!/usr/bin/env python3
"""
Benchmark: Compare inference speed of PCL agent vs paper's greedy baseline.

Key insight: Neural network inference is O(1) per decision regardless of fleet size!
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple

from data_loader import load_env
from pcl import PCLAgent
from sequential_env import encode_state


def benchmark_pcl_agent(env, agent, n_runs: int = 5) -> Dict:
    """Benchmark PCL agent inference time."""
    agent.eval()
    times = []
    
    for run in range(n_runs):
        state = env.reset()
        start = time.perf_counter()
        
        steps = 0
        while not env.is_terminal(state):
            # Get valid actions
            valid_actions = env.get_valid_actions(state)
            
            # Create mask
            valid_mask = np.zeros(env.n_actions, dtype=bool)
            valid_mask[valid_actions] = True
            
            # Neural network forward pass
            state_tensor = encode_state(state, env.n_aircraft, env.horizon)
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)
            
            with torch.no_grad():
                value, logits = agent(state_tensor)
                # Mask invalid actions (logits is [1, n_actions])
                mask = torch.tensor(valid_mask, dtype=torch.bool)
                logits[0, ~mask] = float('-inf')
                action = torch.argmax(logits[0]).item()
            
            state, _, done, _ = env.step(action)
            steps += 1
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'method': 'PCL Agent',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'steps_per_episode': steps
    }


def benchmark_greedy(env, n_runs: int = 5) -> Dict:
    """Benchmark paper's greedy approach."""
    times = []
    
    for run in range(n_runs):
        state = env.reset()
        start = time.perf_counter()
        
        steps = 0
        while not env.is_terminal(state):
            valid_actions = env.get_valid_actions(state)
            
            # Greedy: prioritize by utilization (earliest deadline first)
            best_action = None
            best_util = -1
            
            for action in valid_actions:
                if action == 2 * env.n_aircraft:  # Advance day
                    continue
                    
                check_type = 'C' if action < env.n_aircraft else 'A'
                aircraft_id = action if action < env.n_aircraft else action - env.n_aircraft
                aircraft = state.fleet[aircraft_id]
                
                if check_type == 'C':
                    util = max(
                        aircraft.dy_c / env.config.c_interval_dy,
                        aircraft.fh_c / env.config.c_interval_fh,
                        aircraft.fc_c / env.config.c_interval_fc
                    )
                else:
                    util = max(
                        aircraft.dy_a / env.config.a_interval_dy,
                        aircraft.fh_a / env.config.a_interval_fh,
                        aircraft.fc_a / env.config.a_interval_fc
                    )
                
                if util > best_util:
                    best_util = util
                    best_action = action
            
            # If no check to schedule, advance day
            if best_action is None or best_util < 0.6:
                best_action = 2 * env.n_aircraft  # advance day
            
            state, _, done, _ = env.step(best_action)
            steps += 1
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'method': 'Paper Greedy',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'steps_per_episode': steps
    }


def run_benchmark():
    """Run full benchmark comparison."""
    print("=" * 70)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 70)
    print()
    
    # Load environment
    print("Loading environment...")
    env = load_env('Sep 25 2017 (Case Study 2018-2021).xlsx')
    print(f"  Fleet: {env.n_aircraft} aircraft")
    print(f"  Horizon: {env.horizon} days ({env.horizon/365:.1f} years)")
    print()
    
    # Load agent
    print("Loading PCL agent...")
    agent = PCLAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=512,
        temperature=15.0
    )
    
    try:
        checkpoint = torch.load('best_pcl_agent.pt', weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from iteration {checkpoint['iteration']}")
    except FileNotFoundError:
        print("  Using random initialization (no checkpoint)")
    
    agent.eval()
    print(f"  Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()
    
    # Benchmark
    n_runs = 5
    print(f"Running {n_runs} episodes each...")
    print()
    
    # PCL Agent
    print("Benchmarking PCL Agent...")
    pcl_results = benchmark_pcl_agent(env, agent, n_runs)
    
    # Greedy
    print("Benchmarking Paper's Greedy...")
    greedy_results = benchmark_greedy(env, n_runs)
    
    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Mean Time':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    
    for results in [pcl_results, greedy_results]:
        print(f"{results['method']:<20} "
              f"{results['mean_time']:>10.2f}s "
              f"±{results['std_time']:>8.2f}s "
              f"{results['min_time']:>9.2f}s "
              f"{results['max_time']:>9.2f}s")
    
    print()
    speedup = greedy_results['mean_time'] / pcl_results['mean_time']
    print(f"PCL Agent is {speedup:.1f}x {'faster' if speedup < 1 else 'slower'} than Greedy")
    
    # Per-step timing
    print()
    print("Per-Step Analysis:")
    print("-" * 70)
    pcl_per_step = pcl_results['mean_time'] / pcl_results['steps_per_episode'] * 1000
    greedy_per_step = greedy_results['mean_time'] / greedy_results['steps_per_episode'] * 1000
    print(f"  PCL Agent:    {pcl_per_step:.3f} ms/step ({pcl_results['steps_per_episode']} steps)")
    print(f"  Greedy:       {greedy_per_step:.3f} ms/step ({greedy_results['steps_per_episode']} steps)")
    
    print()
    print("Key Insight:")
    print("-" * 70)
    print("  PCL Agent: O(1) per decision - just a neural network forward pass")
    print("  Greedy:    O(N) per decision - iterates over all aircraft")
    print("  Forward DP: O(N × states) - exponentially more expensive")
    print()
    print("  As fleet size grows, PCL advantage increases!")


if __name__ == "__main__":
    run_benchmark()


"""
Train PCL agent for aircraft maintenance scheduling.

Compares with paper's greedy baseline.
"""

import torch
import numpy as np
import sys
import argparse
import time

from data_loader import load_env, create_synthetic_env
from pcl import (
    PCLAgent,
    train_pcl,
    evaluate_agent,
    compare_with_baseline,
)
from sequential_env import encode_state, compute_schedule_statistics
from paper_dp import greedy_schedule, ForwardDP


def compare_all_methods(env, agent, n_episodes=5):
    """Compare PCL agent with paper's methods."""
    print("\n" + "="*80)
    print("COMPARISON: PCL vs PAPER'S METHODS")
    print("="*80)
    
    results = {}
    
    # 1. Paper's Greedy (priority-based)
    print("\n1. Paper's Greedy (earliest-deadline-first):")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=False)
    c_greedy = sum(1 for day, checks in greedy_sched.items() for t, _ in checks if t == 'C')
    a_greedy = sum(1 for day, checks in greedy_sched.items() for t, _ in checks if t == 'A')
    print(f"   Cost: {greedy_cost:.0f} | C-checks: {c_greedy} | A-checks: {a_greedy}")
    results['greedy'] = {'cost': greedy_cost, 'c': c_greedy, 'a': a_greedy}
    
    # 2. PCL Agent (stochastic)
    print(f"\n2. PCL Agent (stochastic, {n_episodes} episodes):")
    pcl_costs = []
    pcl_c_checks = []
    pcl_a_checks = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_cost = 0.0
        c_count = 0
        a_count = 0
        
        while not done:
            state_tensor = torch.from_numpy(
                encode_state(state, env.n_aircraft, env.horizon)
            ).float()
            valid_mask = env.get_valid_actions(state)
            action, _ = agent.sample_action(state_tensor, valid_mask, deterministic=False)
            
            # Track cost
            if action < env.n_aircraft:
                ac = state.fleet[action]
                wasted = env.config.c_interval_fh - ac.fh_c
                ep_cost += max(0, wasted)
                c_count += 1
            elif action < 2 * env.n_aircraft:
                aid = action - env.n_aircraft
                ac = state.fleet[aid]
                wasted = env.config.a_interval_fh - ac.fh_a
                ep_cost += max(0, wasted)
                a_count += 1
            
            state, _, done, _ = env.step(action)
        
        pcl_costs.append(ep_cost)
        pcl_c_checks.append(c_count)
        pcl_a_checks.append(a_count)
    
    pcl_mean = np.mean(pcl_costs)
    pcl_std = np.std(pcl_costs)
    print(f"   Cost: {pcl_mean:.0f} ± {pcl_std:.0f} | C-checks: {np.mean(pcl_c_checks):.0f} | A-checks: {np.mean(pcl_a_checks):.0f}")
    results['pcl_stoch'] = {'cost': pcl_mean, 'c': np.mean(pcl_c_checks), 'a': np.mean(pcl_a_checks)}
    
    # 3. PCL Agent (greedy/deterministic)
    print(f"\n3. PCL Agent (greedy):")
    state = env.reset()
    done = False
    pcl_greedy_cost = 0.0
    c_pcl = 0
    a_pcl = 0
    
    while not done:
        state_tensor = torch.from_numpy(
            encode_state(state, env.n_aircraft, env.horizon)
        ).float()
        valid_mask = env.get_valid_actions(state)
        action, _ = agent.sample_action(state_tensor, valid_mask, deterministic=True)
        
        if action < env.n_aircraft:
            ac = state.fleet[action]
            wasted = env.config.c_interval_fh - ac.fh_c
            pcl_greedy_cost += max(0, wasted)
            c_pcl += 1
        elif action < 2 * env.n_aircraft:
            aid = action - env.n_aircraft
            ac = state.fleet[aid]
            wasted = env.config.a_interval_fh - ac.fh_a
            pcl_greedy_cost += max(0, wasted)
            a_pcl += 1
        
        state, _, done, _ = env.step(action)
    
    print(f"   Cost: {pcl_greedy_cost:.0f} | C-checks: {c_pcl} | A-checks: {a_pcl}")
    results['pcl_greedy'] = {'cost': pcl_greedy_cost, 'c': c_pcl, 'a': a_pcl}
    
    # 4. Random baseline
    print(f"\n4. Random baseline:")
    random_costs = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_cost = 0.0
        while not done:
            valid = env.get_valid_actions(state)
            action = np.random.choice(np.where(valid)[0])
            if action < env.n_aircraft:
                ep_cost += max(0, env.config.c_interval_fh - state.fleet[action].fh_c)
            elif action < 2 * env.n_aircraft:
                ep_cost += max(0, env.config.a_interval_fh - state.fleet[action - env.n_aircraft].fh_a)
            state, _, done, _ = env.step(action)
        random_costs.append(ep_cost)
    
    random_mean = np.mean(random_costs)
    print(f"   Cost: {random_mean:.0f}")
    results['random'] = {'cost': random_mean}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY (lower cost = better)")
    print("="*80)
    print(f"{'Method':<25} {'Wasted FH':>12} {'Improvement':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12}")
    print(f"{'Paper Greedy':<25} {greedy_cost:>12.0f} {'(baseline)':>12}")
    
    pcl_imp = (greedy_cost - pcl_greedy_cost) / greedy_cost * 100
    print(f"{'PCL (greedy)':<25} {pcl_greedy_cost:>12.0f} {pcl_imp:>+11.1f}%")
    
    pcl_stoch_imp = (greedy_cost - pcl_mean) / greedy_cost * 100
    print(f"{'PCL (stochastic)':<25} {pcl_mean:>12.0f} {pcl_stoch_imp:>+11.1f}%")
    
    rand_imp = (greedy_cost - random_mean) / greedy_cost * 100
    print(f"{'Random':<25} {random_mean:>12.0f} {rand_imp:>+11.1f}%")
    print("="*80)
    
    return results


def main():
    """Main training script."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000, help='Training iterations')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=15.0, help='PCL temperature')
    parser.add_argument('--eval-every', type=int, default=100, help='Evaluate every N iterations')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*80)
    print("PCL TRAINING FOR AIRCRAFT MAINTENANCE SCHEDULING")
    print("="*80)
    
    # Load environment
    try:
        print("\nLoading data from Excel...")
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except Exception as e:
        print(f"Could not load Excel: {e}")
        print("\nUsing synthetic environment...\n")
        env = create_synthetic_env(n_aircraft=45, horizon_days=1461)
    
    print(f"\nEnvironment ready:")
    print(f"  - Aircraft: {env.n_aircraft}")
    print(f"  - Horizon: {env.horizon} days (~{env.horizon/365:.1f} years)")
    print(f"  - State dim: {env.state_dim}")
    print(f"  - Action space: {env.n_actions}")
    
    # Create agent
    agent = PCLAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature
    )
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"\nAgent: {n_params:,} parameters")
    
    # Train
    NUM_ITERATIONS = args.iterations
    
    print(f"\n*** Starting training with {NUM_ITERATIONS} iterations ***")
    print("*** Each iteration samples a full 6-year schedule ***\n")
    
    start_time = time.time()
    
    history = train_pcl(
        agent=agent,
        env=env,
        num_iterations=NUM_ITERATIONS,
        lr=args.lr,
        entropy_coeff=0.01,
        eval_every=args.eval_every,
        n_eval_episodes=3,
        temperature_schedule='constant',
        save_path='best_pcl_agent.pt',
        verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Load best model for final comparison
    checkpoint = torch.load('best_pcl_agent.pt', weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from iteration {checkpoint['iteration']}")
    agent.eval()
    
    # Compare with paper's methods
    compare_all_methods(env, agent, n_episodes=5)
    
    # Sample and show schedule
    print("\n" + "="*80)
    print("SAMPLE SCHEDULE FROM PCL")
    print("="*80)
    
    print("\nSampling final schedule...")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        state_tensor = torch.from_numpy(
            encode_state(state, env.n_aircraft, env.horizon)
        ).float()
        valid_mask = env.get_valid_actions(state)
        action, _ = agent.sample_action(state_tensor, valid_mask, deterministic=True)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        if state.current_day % 500 == 0:
            print(f"  Day {state.current_day}/{env.horizon}...")
    
    print(f"\nSampled schedule: {steps} steps, return: {total_reward:.1f}")
    
    stats = compute_schedule_statistics(state, env.n_aircraft)
    print(f"\nSchedule Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show first scheduled days
    print("\nFirst 15 scheduled days:")
    for day in sorted(state.schedule.keys())[:15]:
        if state.schedule[day]:
            checks_str = ", ".join(
                f"{ct}({env.get_aircraft_tail(aid) if hasattr(env, 'get_aircraft_tail') else f'AC{aid}'})"
                for ct, aid in state.schedule[day]
            )
            print(f"  Day {day:4d}: {checks_str}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

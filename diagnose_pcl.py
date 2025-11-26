"""
Diagnostic tools to understand why PCL isn't beating greedy.

This script helps visualize:
1. What actions the PCL agent takes vs greedy
2. Value function over a trajectory
3. Policy entropy over time
4. Action distribution analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from data_loader import load_env, create_synthetic_env
from sequential_env import encode_state, compute_schedule_statistics
from pcl import PCLAgent
from pcl_improved import (
    ImprovedPCLAgent,
    encode_state_enhanced,
    get_enhanced_state_dim,
    compute_remaining_days,
    compute_utilization,
)
from paper_dp import greedy_schedule


def load_pcl_agent(model_path: str, env, enhanced: bool = False):
    """Load trained PCL agent."""
    if enhanced:
        state_dim = get_enhanced_state_dim(env.n_aircraft)
        agent = ImprovedPCLAgent(
            state_dim=state_dim,
            n_actions=env.n_actions,
            hidden_dim=512,
            temperature=15.0,
            num_layers=3
        )
    else:
        agent = PCLAgent(
            state_dim=env.state_dim,
            n_actions=env.n_actions,
            hidden_dim=512,
            temperature=15.0
        )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    return agent


def diagnose_trajectory(
    agent,
    env,
    config,
    enhanced: bool = False,
    max_steps: int = 5000
):
    """
    Run a trajectory and collect diagnostic info.
    """
    state = env.reset()
    done = False
    step = 0
    
    diagnostics = {
        'days': [],
        'actions': [],
        'action_types': [],  # 'C', 'A', 'wait'
        'values': [],
        'log_probs': [],
        'entropies': [],
        'c_remaining': [],  # Min remaining days for C-check
        'a_remaining': [],  # Min remaining days for A-check
        'greedy_actions': [],
        'matches_greedy': [],
    }
    
    while not done and step < max_steps:
        if enhanced:
            state_tensor = torch.from_numpy(
                encode_state_enhanced(state, env.n_aircraft, env.horizon, config)
            ).float()
        else:
            state_tensor = torch.from_numpy(
                encode_state(state, env.n_aircraft, env.horizon)
            ).float()
        
        valid_mask = env.get_valid_actions(state)
        
        with torch.no_grad():
            value, logits = agent(state_tensor)
            logits_masked = logits.clone()
            logits_masked[~torch.from_numpy(valid_mask)] = -1e9
            probs = F.softmax(logits_masked, dim=-1)
            log_probs = F.log_softmax(logits_masked, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            
            action = torch.argmax(probs).item()
            log_prob = log_probs[action].item()
        
        # Determine action type
        if action < env.n_aircraft:
            action_type = 'C'
        elif action < 2 * env.n_aircraft:
            action_type = 'A'
        else:
            action_type = 'wait'
        
        # Greedy action
        greedy_action = get_greedy_action(state, env, config)
        
        # Compute remaining days
        c_remaining = []
        a_remaining = []
        for ac in state.fleet.values():
            if not ac.in_maintenance:
                c_remaining.append(compute_remaining_days(ac, 'C', config))
                a_remaining.append(compute_remaining_days(ac, 'A', config))
        
        diagnostics['days'].append(state.current_day)
        diagnostics['actions'].append(action)
        diagnostics['action_types'].append(action_type)
        diagnostics['values'].append(value.item())
        diagnostics['log_probs'].append(log_prob)
        diagnostics['entropies'].append(entropy)
        diagnostics['c_remaining'].append(min(c_remaining) if c_remaining else 0)
        diagnostics['a_remaining'].append(min(a_remaining) if a_remaining else 0)
        diagnostics['greedy_actions'].append(greedy_action)
        diagnostics['matches_greedy'].append(action == greedy_action)
        
        state, _, done, _ = env.step(action)
        step += 1
        
        if step % 500 == 0:
            print(f"  Step {step}, Day {state.current_day}")
    
    return diagnostics


def get_greedy_action(state, env, config, r_a_threshold=30.0, r_c_threshold=90.0):
    """Get greedy action for comparison."""
    valid_mask = env.get_valid_actions(state)
    
    # C-checks first
    c_candidates = []
    for i in range(env.n_aircraft):
        if valid_mask[i]:
            remaining = compute_remaining_days(state.fleet[i], 'C', config)
            if remaining <= r_c_threshold:
                c_candidates.append((i, remaining))
    
    if c_candidates:
        c_candidates.sort(key=lambda x: x[1])
        return c_candidates[0][0]
    
    # A-checks
    a_candidates = []
    for i in range(env.n_aircraft):
        if valid_mask[env.n_aircraft + i]:
            remaining = compute_remaining_days(state.fleet[i], 'A', config)
            if remaining <= r_a_threshold:
                a_candidates.append((env.n_aircraft + i, remaining))
    
    if a_candidates:
        a_candidates.sort(key=lambda x: x[1])
        return a_candidates[0][0]
    
    return 2 * env.n_aircraft  # wait


def plot_diagnostics(diagnostics, save_path='pcl_diagnostics.png'):
    """Plot diagnostic visualizations."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Value function over trajectory
    ax = axes[0, 0]
    ax.plot(diagnostics['days'], diagnostics['values'], alpha=0.7)
    ax.set_xlabel('Day')
    ax.set_ylabel('V(s)')
    ax.set_title('Value Function Over Trajectory')
    ax.grid(True, alpha=0.3)
    
    # 2. Entropy over trajectory
    ax = axes[0, 1]
    ax.plot(diagnostics['days'], diagnostics['entropies'], alpha=0.7, color='orange')
    ax.set_xlabel('Day')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy Over Trajectory')
    ax.grid(True, alpha=0.3)
    
    # 3. Action distribution
    ax = axes[1, 0]
    action_counts = defaultdict(int)
    for at in diagnostics['action_types']:
        action_counts[at] += 1
    labels = list(action_counts.keys())
    values = [action_counts[l] for l in labels]
    colors = {'C': 'red', 'A': 'blue', 'wait': 'green'}
    ax.bar(labels, values, color=[colors[l] for l in labels])
    ax.set_xlabel('Action Type')
    ax.set_ylabel('Count')
    ax.set_title('Action Distribution')
    
    # 4. Agreement with greedy
    ax = axes[1, 1]
    window = 100
    matches = diagnostics['matches_greedy']
    rolling_acc = []
    for i in range(0, len(matches) - window, window // 2):
        rolling_acc.append(np.mean(matches[i:i+window]))
    ax.plot(range(len(rolling_acc)), rolling_acc, alpha=0.7, color='purple')
    ax.axhline(y=np.mean(matches), color='red', linestyle='--', label=f'Overall: {np.mean(matches):.2%}')
    ax.set_xlabel('Window')
    ax.set_ylabel('Agreement with Greedy')
    ax.set_title('PCL vs Greedy Agreement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Remaining days when scheduling
    ax = axes[2, 0]
    c_actions = [(d, r) for d, at, r in zip(diagnostics['days'], diagnostics['action_types'], diagnostics['c_remaining']) if at == 'C']
    a_actions = [(d, r) for d, at, r in zip(diagnostics['days'], diagnostics['action_types'], diagnostics['a_remaining']) if at == 'A']
    
    if c_actions:
        c_days, c_rem = zip(*c_actions)
        ax.scatter(c_days, c_rem, alpha=0.5, label='C-check', c='red', s=10)
    if a_actions:
        a_days, a_rem = zip(*a_actions)
        ax.scatter(a_days, a_rem, alpha=0.5, label='A-check', c='blue', s=10)
    
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='C threshold')
    ax.axhline(y=30, color='blue', linestyle='--', alpha=0.5, label='A threshold')
    ax.set_xlabel('Day')
    ax.set_ylabel('Min Remaining Days')
    ax.set_title('Remaining Days When Scheduling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Text summary
    ax = axes[2, 1]
    ax.axis('off')
    
    summary = f"""
    DIAGNOSTIC SUMMARY
    ==================
    
    Total Steps: {len(diagnostics['actions'])}
    Total Days: {max(diagnostics['days'])}
    
    Action Distribution:
      - C-checks: {action_counts['C']} ({action_counts['C']/len(diagnostics['actions'])*100:.1f}%)
      - A-checks: {action_counts['A']} ({action_counts['A']/len(diagnostics['actions'])*100:.1f}%)
      - Wait: {action_counts['wait']} ({action_counts['wait']/len(diagnostics['actions'])*100:.1f}%)
    
    Greedy Agreement: {np.mean(diagnostics['matches_greedy'])*100:.1f}%
    
    Value Function:
      - Mean: {np.mean(diagnostics['values']):.1f}
      - Std: {np.std(diagnostics['values']):.1f}
      - Min: {np.min(diagnostics['values']):.1f}
      - Max: {np.max(diagnostics['values']):.1f}
    
    Entropy:
      - Mean: {np.mean(diagnostics['entropies']):.3f}
      - Std: {np.std(diagnostics['entropies']):.3f}
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved diagnostics to {save_path}")
    plt.close()


def analyze_scheduling_timing(diagnostics, config):
    """Analyze when PCL schedules checks vs optimal timing."""
    print("\n" + "="*80)
    print("SCHEDULING TIMING ANALYSIS")
    print("="*80)
    
    # When does PCL schedule C-checks?
    c_timings = []
    for at, c_rem in zip(diagnostics['action_types'], diagnostics['c_remaining']):
        if at == 'C':
            c_timings.append(c_rem)
    
    if c_timings:
        print(f"\nC-check scheduling (remaining days):")
        print(f"  Mean: {np.mean(c_timings):.1f} days before deadline")
        print(f"  Std:  {np.std(c_timings):.1f}")
        print(f"  Min:  {np.min(c_timings):.1f}")
        print(f"  Max:  {np.max(c_timings):.1f}")
        print(f"\n  Optimal: Schedule as close to 0 as possible without grounding")
        
        # What fraction scheduled early vs late?
        very_early = sum(1 for t in c_timings if t > 60) / len(c_timings)
        early = sum(1 for t in c_timings if 30 < t <= 60) / len(c_timings)
        optimal = sum(1 for t in c_timings if 10 < t <= 30) / len(c_timings)
        late = sum(1 for t in c_timings if t <= 10) / len(c_timings)
        
        print(f"\n  Very early (>60 days): {very_early:.1%}")
        print(f"  Early (30-60 days):    {early:.1%}")
        print(f"  Optimal (10-30 days):  {optimal:.1%}")
        print(f"  Late (<10 days):       {late:.1%}")
    
    # A-check analysis
    a_timings = []
    for at, a_rem in zip(diagnostics['action_types'], diagnostics['a_remaining']):
        if at == 'A':
            a_timings.append(a_rem)
    
    if a_timings:
        print(f"\nA-check scheduling (remaining days):")
        print(f"  Mean: {np.mean(a_timings):.1f} days before deadline")
        print(f"  Std:  {np.std(a_timings):.1f}")


def compare_pcl_vs_greedy_detailed(agent, env, config, enhanced=False):
    """Detailed comparison of PCL vs Greedy decisions."""
    print("\n" + "="*80)
    print("PCL vs GREEDY: DETAILED COMPARISON")
    print("="*80)
    
    state = env.reset()
    done = False
    step = 0
    
    disagreements = []
    pcl_schedule = defaultdict(list)
    greedy_schedule_local = defaultdict(list)
    
    while not done and step < 10000:
        if enhanced:
            state_tensor = torch.from_numpy(
                encode_state_enhanced(state, env.n_aircraft, env.horizon, config)
            ).float()
        else:
            state_tensor = torch.from_numpy(
                encode_state(state, env.n_aircraft, env.horizon)
            ).float()
        
        valid_mask = env.get_valid_actions(state)
        
        # PCL action
        pcl_action, _ = agent.sample_action(state_tensor, valid_mask, deterministic=True)
        
        # Greedy action
        greedy_action = get_greedy_action(state, env, config)
        
        if pcl_action != greedy_action and step < 100:  # Log first 100 disagreements
            # Decode actions
            def decode_action(a):
                if a < env.n_aircraft:
                    return f"C-check(aircraft {a})"
                elif a < 2 * env.n_aircraft:
                    return f"A-check(aircraft {a - env.n_aircraft})"
                else:
                    return "wait"
            
            disagreements.append({
                'step': step,
                'day': state.current_day,
                'pcl': decode_action(pcl_action),
                'greedy': decode_action(greedy_action),
            })
        
        # Track schedules
        if pcl_action < env.n_aircraft:
            pcl_schedule[state.current_day].append(('C', pcl_action))
        elif pcl_action < 2 * env.n_aircraft:
            pcl_schedule[state.current_day].append(('A', pcl_action - env.n_aircraft))
        
        if greedy_action < env.n_aircraft:
            greedy_schedule_local[state.current_day].append(('C', greedy_action))
        elif greedy_action < 2 * env.n_aircraft:
            greedy_schedule_local[state.current_day].append(('A', greedy_action - env.n_aircraft))
        
        state, _, done, _ = env.step(pcl_action)
        step += 1
    
    # Print disagreements
    print("\nFirst 20 disagreements:")
    print(f"{'Step':>6} {'Day':>6} {'PCL':>25} {'Greedy':>25}")
    print("-" * 70)
    for d in disagreements[:20]:
        print(f"{d['step']:>6} {d['day']:>6} {d['pcl']:>25} {d['greedy']:>25}")
    
    # Summary
    total_pcl_c = sum(len([c for c in checks if c[0] == 'C']) for checks in pcl_schedule.values())
    total_pcl_a = sum(len([c for c in checks if c[0] == 'A']) for checks in pcl_schedule.values())
    
    total_greedy_c = sum(len([c for c in checks if c[0] == 'C']) for checks in greedy_schedule_local.values())
    total_greedy_a = sum(len([c for c in checks if c[0] == 'A']) for checks in greedy_schedule_local.values())
    
    print(f"\nCheck counts comparison:")
    print(f"  PCL:    C={total_pcl_c}, A={total_pcl_a}")
    print(f"  Greedy: C={total_greedy_c}, A={total_greedy_a}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_pcl_agent.pt')
    parser.add_argument('--enhanced', action='store_true')
    parser.add_argument('--output', type=str, default='pcl_diagnostics.png')
    args = parser.parse_args()
    
    # Load environment
    try:
        print("Loading environment...")
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except:
        print("Using synthetic environment...")
        env = create_synthetic_env(n_aircraft=45, horizon_days=1461)
    
    config = env.config
    
    # Load agent
    print(f"Loading agent from {args.model}...")
    agent = load_pcl_agent(args.model, env, enhanced=args.enhanced)
    
    # Run diagnostics
    print("\nRunning diagnostic trajectory...")
    diagnostics = diagnose_trajectory(agent, env, config, enhanced=args.enhanced)
    
    # Plot
    plot_diagnostics(diagnostics, args.output)
    
    # Analysis
    analyze_scheduling_timing(diagnostics, config)
    compare_pcl_vs_greedy_detailed(agent, env, config, enhanced=args.enhanced)
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()


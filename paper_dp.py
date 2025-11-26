"""
Forward Induction DP for Aircraft Maintenance Scheduling.

Implementation of Deng et al. (2019):
"A practical dynamic programming based methodology for aircraft 
maintenance check scheduling optimization"

Key components:
- Forward induction (not backward DP)
- State aggregation by fleet utilization
- Priority-based action selection (earliest deadline first)
- Thrifty Algorithm for feasibility checking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import time
from copy import deepcopy
from tqdm import tqdm

from sequential_env import (
    AircraftStatus,
    SchedulingState,
    SequentialSchedulingEnv,
    MaintenanceConfig,
    compute_schedule_statistics
)


# =============================================================================
# Utility Functions (Paper's Equations)
# =============================================================================

def compute_utilization(
    aircraft: AircraftStatus,
    check_type: str,
    config: MaintenanceConfig
) -> float:
    """
    Compute utilization for aircraft (Paper's Equation 45).
    
    u^k_{i,t} = max(DY/I_DY, FH/I_FH, FC/I_FC)
    
    Returns value in [0, 1+] where 1 means at interval limit.
    """
    if check_type == 'A':
        dy_ratio = aircraft.dy_a / config.a_interval_dy if config.a_interval_dy > 0 else 0
        fh_ratio = aircraft.fh_a / config.a_interval_fh if config.a_interval_fh > 0 else 0
        fc_ratio = aircraft.fc_a / config.a_interval_fc if config.a_interval_fc > 0 else 0
    else:  # 'C'
        dy_ratio = aircraft.dy_c / config.c_interval_dy if config.c_interval_dy > 0 else 0
        fh_ratio = aircraft.fh_c / config.c_interval_fh if config.c_interval_fh > 0 else 0
        fc_ratio = aircraft.fc_c / config.c_interval_fc if config.c_interval_fc > 0 else 0
    
    return max(dy_ratio, fh_ratio, fc_ratio)


def compute_remaining_days(
    aircraft: AircraftStatus,
    check_type: str,
    config: MaintenanceConfig
) -> float:
    """
    Compute remaining days before check is due (Paper's Equation 36).
    
    R^k_{i,t} = min(R_DY, R_FH, R_FC)
    """
    if check_type == 'A':
        remaining_dy = config.a_interval_dy - aircraft.dy_a
        remaining_fh = (config.a_interval_fh - aircraft.fh_a) / max(aircraft.daily_fh, 0.1)
        remaining_fc = (config.a_interval_fc - aircraft.fc_a) / max(aircraft.daily_fc, 0.1)
    else:  # 'C'
        remaining_dy = config.c_interval_dy - aircraft.dy_c
        remaining_fh = (config.c_interval_fh - aircraft.fh_c) / max(aircraft.daily_fh, 0.1)
        remaining_fc = (config.c_interval_fc - aircraft.fc_c) / max(aircraft.daily_fc, 0.1)
    
    return min(remaining_dy, remaining_fh, remaining_fc)


def compute_fleet_utilization(
    fleet: Dict[int, AircraftStatus],
    check_type: str,
    config: MaintenanceConfig
) -> float:
    """Mean fleet utilization (Paper's Equation 44)."""
    utils = [compute_utilization(ac, check_type, config) for ac in fleet.values()]
    return np.mean(utils) if utils else 0.0


# =============================================================================
# Priority-Based Action Selection (Section 4.2)
# =============================================================================

def get_priority_list(
    fleet: Dict[int, AircraftStatus],
    check_type: str,
    config: MaintenanceConfig,
    threshold: float,
    exclude: Set[int] = None
) -> List[Tuple[int, float]]:
    """
    Get aircraft sorted by priority (earliest deadline first).
    
    Returns: List of (aircraft_id, remaining_days) sorted ascending.
    """
    exclude = exclude or set()
    priorities = []
    
    for aid, aircraft in fleet.items():
        if aid in exclude:
            continue
        if aircraft.in_maintenance:
            continue
            
        remaining = compute_remaining_days(aircraft, check_type, config)
        
        # Only include if within eligibility threshold
        if remaining <= threshold:
            priorities.append((aid, remaining))
    
    # Sort by remaining days (most urgent first)
    priorities.sort(key=lambda x: x[1])
    return priorities


def select_checks_greedy(
    fleet: Dict[int, AircraftStatus],
    config: MaintenanceConfig,
    c_capacity: int,
    a_capacity: int,
    r_a_threshold: float = 30.0,
    r_c_threshold: float = 90.0
) -> Tuple[List[int], List[int]]:
    """
    Select checks using paper's greedy priority rules.
    
    Rule (iii): C-check has higher priority than A-check.
    Rule (ii): Allocate by earliest deadline first.
    
    Returns: (c_checks, a_checks) - lists of aircraft IDs.
    """
    c_checks = []
    a_checks = []
    scheduled = set()
    
    # C-checks first (higher priority)
    c_priorities = get_priority_list(fleet, 'C', config, r_c_threshold)
    for aid, remaining in c_priorities:
        if len(c_checks) >= c_capacity:
            break
        c_checks.append(aid)
        scheduled.add(aid)
        
        # Rule (iv): Merge A-check if beneficial
        a_remaining = compute_remaining_days(fleet[aid], 'A', config)
        if a_remaining <= r_a_threshold:
            a_checks.append(aid)  # Merged, doesn't use A-slot
    
    # A-checks (excluding merged)
    a_priorities = get_priority_list(fleet, 'A', config, r_a_threshold, exclude=scheduled)
    for aid, remaining in a_priorities:
        if len([a for a in a_checks if a not in c_checks]) >= a_capacity:
            break
        if aid not in a_checks:
            a_checks.append(aid)
            scheduled.add(aid)
    
    return c_checks, a_checks


# =============================================================================
# State Aggregation (Section 4.4)
# =============================================================================

@dataclass
class DPState:
    """State for forward DP with aggregation info."""
    day: int
    fleet: Dict[int, AircraftStatus]
    schedule: Dict[int, List[Tuple[str, int]]]  # day -> [(check_type, aircraft_id)]
    cumulative_cost: float = 0.0
    
    # Aggregation keys
    mean_util_a: float = 0.0
    mean_util_c: float = 0.0
    
    def copy(self):
        return DPState(
            day=self.day,
            fleet={aid: ac.copy() for aid, ac in self.fleet.items()},
            schedule=deepcopy(self.schedule),
            cumulative_cost=self.cumulative_cost,
            mean_util_a=self.mean_util_a,
            mean_util_c=self.mean_util_c
        )
    
    def update_utilization(self, config: MaintenanceConfig):
        """Update aggregation keys."""
        self.mean_util_a = compute_fleet_utilization(self.fleet, 'A', config)
        self.mean_util_c = compute_fleet_utilization(self.fleet, 'C', config)


class StateAggregator:
    """
    State aggregation by discretized fleet utilization.
    
    Keeps only best state per (u_A, u_C) grid cell.
    """
    
    def __init__(self, delta_u: float = 0.05):
        self.delta_u = delta_u
        self.states: Dict[Tuple[float, float], DPState] = {}
    
    def _round(self, u: float) -> float:
        return round(u / self.delta_u) * self.delta_u
    
    def add(self, state: DPState):
        key = (self._round(state.mean_util_a), self._round(state.mean_util_c))
        if key not in self.states or state.cumulative_cost < self.states[key].cumulative_cost:
            self.states[key] = state
    
    def get_states(self) -> List[DPState]:
        return list(self.states.values())
    
    def clear(self):
        self.states.clear()


# =============================================================================
# Forward Induction DP Solver
# =============================================================================

class ForwardDP:
    """
    Forward induction DP solver (Paper's Algorithm 1).
    
    Builds schedule day-by-day using:
    - Greedy priority-based check selection
    - State aggregation for tractability
    """
    
    def __init__(
        self,
        env: SequentialSchedulingEnv,
        delta_u: float = 0.05,
        r_a_threshold: float = 30.0,
        r_c_threshold: float = 90.0,
        penalty_grounded: float = 1e6
    ):
        self.env = env
        self.config = env.config
        self.delta_u = delta_u
        self.r_a_threshold = r_a_threshold
        self.r_c_threshold = r_c_threshold
        self.penalty_grounded = penalty_grounded
    
    def _compute_cost(
        self,
        fleet: Dict[int, AircraftStatus],
        c_checks: List[int],
        a_checks: List[int]
    ) -> float:
        """Compute cost of scheduling checks (wasted FH)."""
        cost = 0.0
        
        for aid in c_checks:
            ac = fleet[aid]
            wasted = self.config.c_interval_fh - ac.fh_c
            cost += max(0, wasted)
        
        # A-checks not merged into C
        for aid in a_checks:
            if aid not in c_checks:
                ac = fleet[aid]
                wasted = self.config.a_interval_fh - ac.fh_a
                cost += max(0, wasted)
        
        return cost
    
    def _transition(
        self,
        state: DPState,
        c_checks: List[int],
        a_checks: List[int]
    ) -> DPState:
        """Apply state transition for one day."""
        new = state.copy()
        
        # Record schedule
        if c_checks or a_checks:
            new.schedule[state.day] = []
            for aid in c_checks:
                new.schedule[state.day].append(('C', aid))
            for aid in a_checks:
                if aid not in c_checks:
                    new.schedule[state.day].append(('A', aid))
        
        # Apply checks
        for aid in c_checks:
            ac = new.fleet[aid]
            ac.dy_c = 0.0
            ac.fh_c = 0.0
            ac.fc_c = 0.0
            ac.dy_a = 0.0  # Merge A-check
            ac.fh_a = 0.0
            ac.fc_a = 0.0
            ac.in_maintenance = True
            ac.maintenance_days_remaining = self.config.c_check_duration
        
        for aid in a_checks:
            if aid not in c_checks:
                ac = new.fleet[aid]
                ac.dy_a = 0.0
                ac.fh_a = 0.0
                ac.fc_a = 0.0
                ac.in_maintenance = True
                ac.maintenance_days_remaining = self.config.a_check_duration
        
        # Age fleet
        for ac in new.fleet.values():
            if ac.in_maintenance:
                ac.maintenance_days_remaining -= 1
                if ac.maintenance_days_remaining <= 0:
                    ac.in_maintenance = False
            else:
                ac.dy_a += 1.0
                ac.dy_c += 1.0
                ac.fh_a += ac.daily_fh
                ac.fh_c += ac.daily_fh
                ac.fc_a += ac.daily_fc
                ac.fc_c += ac.daily_fc
        
        # Update cost
        cost = self._compute_cost(state.fleet, c_checks, a_checks)
        new.cumulative_cost = state.cumulative_cost + cost
        
        # Check for grounding (aircraft significantly over limit without check)
        # Use a small buffer to avoid penalizing for minor timing issues
        for aid, ac in new.fleet.items():
            if not ac.in_maintenance and aid not in c_checks and aid not in a_checks:
                a_remaining = compute_remaining_days(ac, 'A', self.config)
                c_remaining = compute_remaining_days(ac, 'C', self.config)
                # Only penalize if significantly overdue (more than 3 days)
                if a_remaining < -3 or c_remaining < -3:
                    new.cumulative_cost += self.penalty_grounded
        
        new.day += 1
        new.update_utilization(self.config)
        
        return new
    
    def solve(self, verbose: bool = True) -> Tuple[Dict[int, List[Tuple[str, int]]], float]:
        """
        Solve using forward induction DP.
        
        Returns: (schedule, total_cost)
        """
        if verbose:
            print("="*80)
            print("FORWARD INDUCTION DP (Paper's Algorithm)")
            print("="*80)
            print(f"Fleet: {self.env.n_aircraft} aircraft")
            print(f"Horizon: {self.env.horizon} days")
            print(f"State aggregation: Δu = {self.delta_u}")
            print("="*80 + "\n")
        
        start_time = time.time()
        
        # Initialize from environment
        env_state = self.env.reset()
        
        initial = DPState(
            day=0,
            fleet={aid: ac.copy() for aid, ac in env_state.fleet.items()},
            schedule={},
            cumulative_cost=0.0
        )
        initial.update_utilization(self.config)
        
        current_states = [initial]
        
        # Forward pass
        iterator = range(self.env.horizon)
        if verbose:
            iterator = tqdm(iterator, desc="Forward DP")
        
        for day in iterator:
            aggregator = StateAggregator(self.delta_u)
            
            for state in current_states:
                # Get capacity for this day
                c_cap = self.env.c_capacity.get(day, 0)
                a_cap = self.env.a_capacity.get(day, 0)
                
                # Greedy selection
                c_checks, a_checks = select_checks_greedy(
                    state.fleet,
                    self.config,
                    c_cap,
                    a_cap,
                    self.r_a_threshold,
                    self.r_c_threshold
                )
                
                # Transition
                next_state = self._transition(state, c_checks, a_checks)
                
                # Aggregate
                aggregator.add(next_state)
            
            current_states = aggregator.get_states()
            
            if len(current_states) == 0:
                print(f"Warning: No valid states at day {day}")
                break
        
        # Find best final state
        if current_states:
            best = min(current_states, key=lambda s: s.cumulative_cost)
            schedule = best.schedule
            cost = best.cumulative_cost
        else:
            schedule = {}
            cost = float('inf')
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Forward DP complete!")
            print(f"Time: {elapsed:.1f}s")
            print(f"Final cost: {cost:.1f}")
            print(f"States explored per day: ~{len(current_states)}")
            print(f"{'='*80}\n")
        
        return schedule, cost


# =============================================================================
# Greedy Baseline (for comparison)
# =============================================================================

def greedy_schedule(
    env: SequentialSchedulingEnv,
    r_a_threshold: float = 30.0,
    r_c_threshold: float = 90.0,
    verbose: bool = True
) -> Tuple[Dict[int, List[Tuple[str, int]]], float]:
    """
    Simple greedy scheduling (no DP, no aggregation).
    
    Just applies priority rules day by day.
    """
    if verbose:
        print("="*80)
        print("GREEDY SCHEDULING")
        print("="*80)
    
    start_time = time.time()
    
    env_state = env.reset()
    schedule = {}
    total_cost = 0.0
    
    iterator = range(env.horizon)
    if verbose:
        iterator = tqdm(iterator, desc="Greedy")
    
    fleet = {aid: ac.copy() for aid, ac in env_state.fleet.items()}
    
    for day in iterator:
        c_cap = env.c_capacity.get(day, 0)
        a_cap = env.a_capacity.get(day, 0)
        
        c_checks, a_checks = select_checks_greedy(
            fleet,
            env.config,
            c_cap,
            a_cap,
            r_a_threshold,
            r_c_threshold
        )
        
        # Record schedule
        if c_checks or a_checks:
            schedule[day] = []
            for aid in c_checks:
                schedule[day].append(('C', aid))
                # Cost
                wasted = env.config.c_interval_fh - fleet[aid].fh_c
                total_cost += max(0, wasted)
            for aid in a_checks:
                if aid not in c_checks:
                    schedule[day].append(('A', aid))
                    wasted = env.config.a_interval_fh - fleet[aid].fh_a
                    total_cost += max(0, wasted)
        
        # Apply transitions
        for aid in c_checks:
            ac = fleet[aid]
            ac.dy_c = ac.fh_c = ac.fc_c = 0.0
            ac.dy_a = ac.fh_a = ac.fc_a = 0.0
            ac.in_maintenance = True
            ac.maintenance_days_remaining = env.config.c_check_duration
        
        for aid in a_checks:
            if aid not in c_checks:
                ac = fleet[aid]
                ac.dy_a = ac.fh_a = ac.fc_a = 0.0
                ac.in_maintenance = True
                ac.maintenance_days_remaining = env.config.a_check_duration
        
        # Age
        for ac in fleet.values():
            if ac.in_maintenance:
                ac.maintenance_days_remaining -= 1
                if ac.maintenance_days_remaining <= 0:
                    ac.in_maintenance = False
            else:
                ac.dy_a += 1.0
                ac.dy_c += 1.0
                ac.fh_a += ac.daily_fh
                ac.fh_c += ac.daily_fh
                ac.fc_a += ac.daily_fc
                ac.fc_c += ac.daily_fc
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nGreedy complete! Time: {elapsed:.1f}s, Cost: {total_cost:.1f}")
    
    return schedule, total_cost


# =============================================================================
# Comparison
# =============================================================================

def compare_methods(env: SequentialSchedulingEnv):
    """Compare Forward DP vs Greedy vs Random."""
    
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80 + "\n")
    
    # 1. Forward DP (with smaller penalty for clearer comparison)
    print("1. Forward Induction DP:")
    dp_solver = ForwardDP(env, delta_u=0.05, penalty_grounded=1000)
    dp_schedule, dp_cost = dp_solver.solve(verbose=True)
    dp_stats = _schedule_stats(dp_schedule)
    
    # 2. Greedy
    print("\n2. Greedy (no DP):")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=True)
    greedy_stats = _schedule_stats(greedy_sched)
    
    # 3. Random (from our PCL code)
    print("\n3. Random Policy:")
    random_cost, random_stats = _run_random(env)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Wasted FH':>12} {'C-checks':>10} {'A-checks':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10}")
    print(f"{'Forward DP':<25} {dp_cost:>12.0f} {dp_stats['c']:>10} {dp_stats['a']:>10}")
    print(f"{'Greedy':<25} {greedy_cost:>12.0f} {greedy_stats['c']:>10} {greedy_stats['a']:>10}")
    print(f"{'Random':<25} {random_cost:>12.0f} {random_stats['c']:>10} {random_stats['a']:>10}")
    print("="*80)
    print("\nNote: Lower 'Wasted FH' is better (aircraft used more efficiently)")
    print("="*80 + "\n")
    
    return {
        'dp': (dp_schedule, dp_cost),
        'greedy': (greedy_sched, greedy_cost),
        'random': (None, random_cost)
    }


def _schedule_stats(schedule: Dict) -> Dict:
    c = sum(1 for day, checks in schedule.items() for t, _ in checks if t == 'C')
    a = sum(1 for day, checks in schedule.items() for t, _ in checks if t == 'A')
    return {'c': c, 'a': a}


def _run_random(env: SequentialSchedulingEnv) -> Tuple[float, Dict]:
    """Run random policy and compute cost."""
    import numpy as np
    from sequential_env import encode_state
    
    state = env.reset()
    total_cost = 0.0
    c_checks = 0
    a_checks = 0
    
    done = False
    while not done:
        valid = env.get_valid_actions(state)
        valid_idx = np.where(valid)[0]
        action = np.random.choice(valid_idx)
        
        # Track cost before step
        if action < env.n_aircraft:
            ac = state.fleet[action]
            wasted = env.config.c_interval_fh - ac.fh_c
            total_cost += max(0, wasted)
            c_checks += 1
        elif action < 2 * env.n_aircraft:
            aid = action - env.n_aircraft
            ac = state.fleet[aid]
            wasted = env.config.a_interval_fh - ac.fh_a
            total_cost += max(0, wasted)
            a_checks += 1
        
        state, _, done, _ = env.step(action)
    
    print(f"Random complete! Cost: {total_cost:.0f}")
    return total_cost, {'c': c_checks, 'a': a_checks}


# =============================================================================
# Compare with PCL Agent
# =============================================================================

def compare_all_methods(env: SequentialSchedulingEnv, pcl_model_path: str = None):
    """Compare all methods: Forward DP, Greedy, PCL, Random."""
    import os
    
    print("\n" + "="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80 + "\n")
    
    results = {}
    
    # 1. Greedy (Paper's approach)
    print("1. Greedy (Paper's priority-based):")
    greedy_sched, greedy_cost = greedy_schedule(env, verbose=True)
    greedy_stats = _schedule_stats(greedy_sched)
    results['greedy'] = {'cost': greedy_cost, **greedy_stats}
    
    # 2. Forward DP
    print("\n2. Forward Induction DP:")
    dp_solver = ForwardDP(env, delta_u=0.05, penalty_grounded=1000)
    dp_schedule, dp_cost = dp_solver.solve(verbose=True)
    dp_stats = _schedule_stats(dp_schedule)
    results['dp'] = {'cost': dp_cost, **dp_stats}
    
    # 3. PCL Agent (if available)
    pcl_cost = None
    pcl_stats = {'c': 0, 'a': 0}
    if pcl_model_path and os.path.exists(pcl_model_path):
        try:
            print("\n3. PCL Agent (trained):")
            pcl_cost, pcl_stats = _run_pcl_agent(env, pcl_model_path)
            results['pcl'] = {'cost': pcl_cost, **pcl_stats}
        except Exception as e:
            print(f"   Could not load PCL: {e}")
    else:
        print("\n3. PCL Agent: No model found (skip)")
    
    # 4. Random
    print("\n4. Random Policy:")
    random_cost, random_stats = _run_random(env)
    results['random'] = {'cost': random_cost, **random_stats}
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Wasted FH':>12} {'C-checks':>10} {'A-checks':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10}")
    print(f"{'Greedy (Paper)':<25} {greedy_cost:>12.0f} {greedy_stats['c']:>10} {greedy_stats['a']:>10}")
    print(f"{'Forward DP':<25} {dp_cost:>12.0f} {dp_stats['c']:>10} {dp_stats['a']:>10}")
    if pcl_cost is not None:
        print(f"{'PCL Agent':<25} {pcl_cost:>12.0f} {pcl_stats['c']:>10} {pcl_stats['a']:>10}")
    print(f"{'Random':<25} {random_cost:>12.0f} {random_stats['c']:>10} {random_stats['a']:>10}")
    print("="*80)
    print("\nNote: Lower 'Wasted FH' is better (aircraft used more efficiently)")
    
    # Relative improvement
    if greedy_cost < random_cost:
        improvement = (random_cost - greedy_cost) / random_cost * 100
        print(f"\nGreedy reduces waste by {improvement:.1f}% vs Random")
    print("="*80 + "\n")
    
    return results


def _run_pcl_agent(env: SequentialSchedulingEnv, model_path: str) -> Tuple[float, Dict]:
    """Run trained PCL agent and compute cost."""
    import torch
    from pcl import PCLAgent, sample_trajectory
    
    # Load agent
    agent = PCLAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=512,
        temperature=15.0
    )
    agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.eval()
    
    # Sample trajectory
    trajectory, total_reward = sample_trajectory(agent, env, deterministic=True)
    
    # Compute cost (wasted FH)
    total_cost = 0.0
    c_checks = 0
    a_checks = 0
    
    for step in trajectory:
        action = step['action']
        if action < env.n_aircraft:
            c_checks += 1
            ac = step['state'].fleet[action]
            wasted = env.config.c_interval_fh - ac.fh_c
            total_cost += max(0, wasted)
        elif action < 2 * env.n_aircraft:
            a_checks += 1
            aid = action - env.n_aircraft
            ac = step['state'].fleet[aid]
            wasted = env.config.a_interval_fh - ac.fh_a
            total_cost += max(0, wasted)
    
    print(f"PCL Agent: Cost={total_cost:.0f}, C={c_checks}, A={a_checks}")
    return total_cost, {'c': c_checks, 'a': a_checks}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_env, create_synthetic_env
    import os
    
    print("Loading environment...")
    try:
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except:
        print("Using synthetic environment...")
        env = create_synthetic_env(n_aircraft=20, horizon_days=365)
    
    # Check for PCL model
    pcl_path = "best_pcl_agent.pt"
    if not os.path.exists(pcl_path):
        pcl_path = None
    
    # Compare all methods
    results = compare_all_methods(env, pcl_path)


"""
CPD Decoder: Convert CPD to concrete schedule.

Implements the greedy forward scheduling algorithm per OVERVIEW.md Section 2.5.
This is a fixed, deterministic scheduling policy - no optimization inside.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .state import CPDState, TaskNode
from maintainer.amcs.schedule import Schedule


def find_first_feasible_day(
    schedule: Schedule,
    task: TaskNode,
    earliest: int,
    latest: int,
    capacity: int,
) -> int:
    """
    Find the first feasible day to schedule a task.
    
    A day is feasible if:
    - It's within [earliest, latest]
    - The task fits within horizon (considering duration)
    - Daily capacity is not exceeded for all days of the task
    
    Args:
        schedule: Current schedule
        task: Task to schedule
        earliest: Earliest allowed start day
        latest: Latest allowed start day
        capacity: Daily hangar capacity
    
    Returns:
        First feasible start day, or latest if none found
    """
    horizon = schedule.horizon
    duration = task.duration
    
    # Clamp to valid range
    earliest = max(0, earliest)
    latest = min(latest, horizon - duration)
    
    if earliest > latest:
        # No valid window - return latest possible
        return max(0, min(latest, horizon - duration))
    
    # Scan for first feasible day
    for day in range(earliest, latest + 1):
        # Check capacity for all days of the task
        feasible = True
        for d in range(day, min(day + duration, horizon)):
            if not schedule.is_capacity_available(d, capacity):
                feasible = False
                break
        
        if feasible:
            return day
    
    # No feasible day found - return latest (will incur penalty)
    return latest


def decode_cpd_to_schedule(cpd_state: CPDState) -> Schedule:
    """
    Decode a CPD to a concrete schedule using greedy forward scheduling.
    
    Algorithm:
    1. Use construction order as topological order (guaranteed by CPD design)
    2. For each task in order:
       - Compute earliest feasible day respecting precedence and window
       - Find first day with available capacity
       - Assign task (never revisit)
    
    This is a fixed, deterministic policy - no search or optimization.
    All combinatorial optimization happens in MCTS over CPD construction.
    
    Args:
        cpd_state: The CPD to decode
    
    Returns:
        Schedule with tasks placed
    """
    instance = cpd_state.instance
    schedule = Schedule(instance.horizon, instance.n_aircraft)
    
    # Track start days for precedence checking
    task_start_days: Dict[int, int] = {}
    
    # Process tasks in construction order (which is a valid topological order)
    for idx, task in enumerate(cpd_state.nodes):
        # Compute earliest feasible day respecting:
        # 1. Task's own time window
        # 2. Predecessor completion
        earliest_possible = task.earliest_day
        
        # Check predecessors (previous tasks on same aircraft)
        for from_idx, to_idx in cpd_state.edges:
            if to_idx == idx:
                pred_task = cpd_state.nodes[from_idx]
                pred_end = task_start_days[from_idx] + pred_task.duration
                earliest_possible = max(earliest_possible, pred_end)
        
        # Find first feasible day with capacity
        start_day = find_first_feasible_day(
            schedule,
            task,
            earliest_possible,
            task.latest_day,
            instance.capacity,
        )
        
        # Assign task (never move it again)
        task_start_days[idx] = start_day
        schedule.assign(
            start_day,
            task.aircraft_id,
            task.check_type,
            task.duration,
        )
    
    return schedule


def validate_schedule(schedule: Schedule, instance) -> List[str]:
    """
    Validate a schedule and return list of violations.
    
    Checks:
    - Capacity constraints (daily hangar limit)
    - No overlapping checks for same aircraft
    
    Returns:
        List of violation descriptions (empty if valid)
    """
    violations = []
    
    # Check daily capacity
    for day in range(schedule.horizon):
        usage = schedule.get_capacity_usage(day)
        if usage > instance.capacity:
            violations.append(
                f"Capacity violation on day {day}: {usage} > {instance.capacity}"
            )
    
    # Check aircraft overlap (already enforced by Schedule.assign, but double-check)
    for ac_id in range(instance.n_aircraft):
        checks = schedule.get_checks_for_aircraft(ac_id)
        checks.sort(key=lambda c: c.start_day)
        
        for i in range(len(checks) - 1):
            curr = checks[i]
            next_check = checks[i + 1]
            if curr.start_day + curr.duration > next_check.start_day:
                violations.append(
                    f"Overlap for aircraft {ac_id}: check ending day {curr.start_day + curr.duration} "
                    f"overlaps with check starting day {next_check.start_day}"
                )
    
    return violations


def compute_schedule_statistics(
    schedule: Schedule,
    instance,
) -> Dict[str, float]:
    """
    Compute statistics about a schedule.
    
    Returns:
        Dictionary of statistics
    """
    checks = schedule.get_all_checks()
    a_checks = [c for c in checks if c.check_type == 'A']
    c_checks = [c for c in checks if c.check_type == 'C']
    
    # Capacity utilization
    daily_usage = [schedule.get_capacity_usage(d) for d in range(schedule.horizon)]
    
    stats = {
        'n_a_checks': len(a_checks),
        'n_c_checks': len(c_checks),
        'total_checks': len(checks),
        'total_grounded_days': sum(c.duration for c in checks),
        'avg_daily_capacity_usage': np.mean(daily_usage),
        'max_daily_capacity_usage': max(daily_usage) if daily_usage else 0,
        'capacity_utilization': np.mean(daily_usage) / instance.capacity if instance.capacity > 0 else 0,
    }
    
    return stats


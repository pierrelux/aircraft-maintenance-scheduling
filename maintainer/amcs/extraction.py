"""
CPD Extraction from Deng schedules.

Converts a schedule (from Deng solver) into a CPD and training trajectories.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .instance import AMCSInstance
from .schedule import Schedule, ScheduledCheck
from maintainer.core.state import CPDState, TaskNode, Action, ActionType, apply_action


@dataclass
class TrajectoryStep:
    """A single step in a training trajectory."""
    state: CPDState
    action: Action
    reward: float  # Final reward (same for all steps in trajectory)


Trajectory = List[TrajectoryStep]


def extract_tasks_from_schedule(
    schedule: Schedule,
    instance: AMCSInstance,
) -> List[Dict]:
    """
    Extract tasks from a schedule as a list of task dictionaries.
    
    Returns tasks sorted by scheduled day (tie-break by aircraft ID).
    This defines the canonical build order.
    
    Returns:
        List of dicts with keys: aircraft_id, check_type, start_day, duration
    """
    tasks = []
    
    for check in schedule.get_all_checks():
        tasks.append({
            'aircraft_id': check.aircraft_id,
            'check_type': check.check_type,
            'start_day': check.start_day,
            'duration': check.duration,
        })
    
    # Sort by start_day, then by aircraft_id (canonical order)
    tasks.sort(key=lambda t: (t['start_day'], t['aircraft_id']))
    
    return tasks


def extract_cpd_from_schedule(
    instance: AMCSInstance,
    schedule: Schedule,
) -> CPDState:
    """
    Convert a schedule to a CPD by replaying the canonical build order.
    
    Algorithm:
    1. Extract all tasks from schedule
    2. Sort by scheduled day (tie-break by aircraft ID) -> canonical order
    3. Build CPD by adding tasks in this order
    
    Args:
        instance: Problem instance
        schedule: Schedule to convert
    
    Returns:
        CPDState representing the schedule's task structure
    """
    # Extract and sort tasks
    tasks = extract_tasks_from_schedule(schedule, instance)
    
    # Build CPD by adding tasks in canonical order
    cpd_state = CPDState.empty(instance)
    
    for task in tasks:
        action = Action.add_task(task['aircraft_id'], task['check_type'])
        cpd_state = apply_action(cpd_state, action)
    
    # Mark as complete (we've added all tasks)
    cpd_state = apply_action(cpd_state, Action.stop())
    
    return cpd_state


def cpd_to_trajectory(
    cpd: CPDState,
    final_reward: float,
) -> Trajectory:
    """
    Convert a final CPD to a training trajectory.
    
    Replays the CPD construction sequence, recording (state, action, reward)
    at each step. The reward is the same for all steps (final schedule cost).
    
    Args:
        cpd: The final CPD (should be terminal)
        final_reward: The reward (typically -cost) for this CPD's schedule
    
    Returns:
        List of TrajectoryStep objects
    """
    trajectory = []
    
    # Start with empty CPD
    state = CPDState.empty(cpd.instance)
    
    # Replay construction by adding each task
    for task in cpd.nodes:
        action = Action.add_task(task.aircraft_id, task.check_type)
        
        # Record step (before applying action)
        trajectory.append(TrajectoryStep(
            state=state.copy(),
            action=action,
            reward=final_reward,
        ))
        
        # Apply action
        state = apply_action(state, action)
    
    # Record final Stop action
    stop_action = Action.stop()
    trajectory.append(TrajectoryStep(
        state=state.copy(),
        action=stop_action,
        reward=final_reward,
    ))
    
    return trajectory


def schedule_to_trajectory(
    instance: AMCSInstance,
    schedule: Schedule,
    cost: float,
) -> Trajectory:
    """
    Convert a schedule directly to a training trajectory.
    
    Combines extract_cpd_from_schedule and cpd_to_trajectory.
    
    Args:
        instance: Problem instance
        schedule: Schedule to convert
        cost: Cost of the schedule (will be negated for reward)
    
    Returns:
        Training trajectory
    """
    cpd = extract_cpd_from_schedule(instance, schedule)
    reward = -cost  # Lower cost = higher reward
    return cpd_to_trajectory(cpd, reward)


def validate_cpd_schedule_roundtrip(
    instance: AMCSInstance,
    schedule: Schedule,
) -> Tuple[bool, List[str]]:
    """
    Validate that CPD extraction and trajectory replay produce consistent results.
    
    This is a diagnostic to check:
    1. Extract CPD from schedule
    2. Replay trajectory from CPD
    3. Verify we get back to the same CPD structure
    
    Returns:
        (is_valid, list of issues)
    """
    from maintainer.core.decoder import decode_cpd_to_schedule
    
    issues = []
    
    # Extract CPD
    cpd = extract_cpd_from_schedule(instance, schedule)
    
    # Create trajectory and replay
    trajectory = cpd_to_trajectory(cpd, 0.0)
    
    # Replay trajectory
    replayed_state = CPDState.empty(instance)
    for step in trajectory:
        replayed_state = apply_action(replayed_state, step.action)
    
    # Check node count matches
    if len(replayed_state.nodes) != len(cpd.nodes):
        issues.append(
            f"Node count mismatch: replayed {len(replayed_state.nodes)} vs original {len(cpd.nodes)}"
        )
    
    # Check nodes match (in order)
    for i, (rep_node, orig_node) in enumerate(zip(replayed_state.nodes, cpd.nodes)):
        if rep_node.aircraft_id != orig_node.aircraft_id:
            issues.append(f"Node {i} aircraft mismatch")
        if rep_node.check_type != orig_node.check_type:
            issues.append(f"Node {i} check_type mismatch")
        if rep_node.sequence_index != orig_node.sequence_index:
            issues.append(f"Node {i} sequence_index mismatch")
    
    # Check edge count matches
    if len(replayed_state.edges) != len(cpd.edges):
        issues.append(
            f"Edge count mismatch: replayed {len(replayed_state.edges)} vs original {len(cpd.edges)}"
        )
    
    return len(issues) == 0, issues


def compute_trajectory_statistics(trajectory: Trajectory) -> Dict:
    """
    Compute statistics about a trajectory.
    
    Returns:
        Dictionary of statistics
    """
    n_steps = len(trajectory)
    n_add_task = sum(1 for s in trajectory if s.action.type == ActionType.ADD_TASK)
    n_a_checks = sum(1 for s in trajectory 
                     if s.action.type == ActionType.ADD_TASK and s.action.check_type == 'A')
    n_c_checks = sum(1 for s in trajectory 
                     if s.action.type == ActionType.ADD_TASK and s.action.check_type == 'C')
    
    # Aircraft distribution
    aircraft_counts = {}
    for step in trajectory:
        if step.action.type == ActionType.ADD_TASK:
            ac_id = step.action.aircraft_id
            aircraft_counts[ac_id] = aircraft_counts.get(ac_id, 0) + 1
    
    return {
        'n_steps': n_steps,
        'n_add_task': n_add_task,
        'n_a_checks': n_a_checks,
        'n_c_checks': n_c_checks,
        'aircraft_task_counts': aircraft_counts,
        'final_reward': trajectory[-1].reward if trajectory else 0.0,
    }


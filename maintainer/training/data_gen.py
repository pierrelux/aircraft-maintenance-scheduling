"""
Training data generation for Maintainer.

Generates behavior cloning data from the Deng solver.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import random
from tqdm import tqdm

from maintainer.amcs.instance import AMCSInstance
from maintainer.amcs.schedule import Schedule
from maintainer.amcs.bridge import solve_with_deng, evaluate_schedule, DENG_AVAILABLE
from maintainer.amcs.extraction import (
    Trajectory,
    TrajectoryStep,
    extract_cpd_from_schedule,
    cpd_to_trajectory,
    schedule_to_trajectory,
)
from maintainer.core.state import CPDState, Action, apply_action
from maintainer.core.decoder import decode_cpd_to_schedule


@dataclass
class DataGenConfig:
    """Configuration for data generation."""
    n_instances: int = 100
    n_aircraft_range: Tuple[int, int] = (5, 10)
    horizon_range: Tuple[int, int] = (180, 365)
    capacity_range: Tuple[int, int] = (1, 3)
    seed: Optional[int] = None


def generate_synthetic_schedule(instance: AMCSInstance) -> Schedule:
    """
    Generate a synthetic schedule using a simple greedy heuristic.
    
    This is used when Deng solver is not available.
    """
    # Build a simple CPD: add tasks in order of urgency
    state = CPDState.empty(instance)
    
    # Collect all needed tasks
    tasks_to_add = []
    for ac_id in range(instance.n_aircraft):
        ac = instance.get_aircraft(ac_id)
        
        # Add A-checks
        for k in range(ac.max_A_checks(instance.horizon)):
            earliest = ac.A_earliest_day(k)
            if earliest < instance.horizon:
                tasks_to_add.append({
                    'aircraft_id': ac_id,
                    'check_type': 'A',
                    'earliest': earliest,
                })
        
        # Add C-checks
        for k in range(ac.max_C_checks(instance.horizon)):
            earliest = ac.C_earliest_day(k)
            if earliest < instance.horizon:
                tasks_to_add.append({
                    'aircraft_id': ac_id,
                    'check_type': 'C',
                    'earliest': earliest,
                })
    
    # Sort by earliest day (simple priority rule)
    tasks_to_add.sort(key=lambda t: (t['earliest'], t['aircraft_id']))
    
    # Add tasks to CPD
    for task in tasks_to_add:
        action = Action.add_task(task['aircraft_id'], task['check_type'])
        try:
            state = apply_action(state, action)
        except ValueError:
            # Skip if action is illegal
            continue
    
    # Finalize
    state = apply_action(state, Action.stop())
    
    # Decode to schedule
    return decode_cpd_to_schedule(state)


def generate_training_data(
    config: DataGenConfig,
    use_deng: bool = True,
    verbose: bool = True,
) -> List[Tuple[AMCSInstance, Trajectory]]:
    """
    Generate training data from the Deng solver.
    
    Args:
        config: Data generation configuration
        use_deng: Whether to use Deng solver (requires real data)
        verbose: Show progress bar
    
    Returns:
        List of (instance, trajectory) pairs
    """
    if config.seed is not None:
        random.seed(config.seed)
    
    data = []
    
    iterator = range(config.n_instances)
    if verbose:
        iterator = tqdm(iterator, desc="Generating training data")
    
    for _ in iterator:
        # Generate random instance
        n_aircraft = random.randint(*config.n_aircraft_range)
        horizon = random.randint(*config.horizon_range)
        capacity = random.randint(*config.capacity_range)
        
        instance = AMCSInstance.generate_random(
            n_aircraft=n_aircraft,
            horizon=horizon,
            capacity=capacity,
        )
        
        # Solve with Deng or synthetic
        if use_deng and DENG_AVAILABLE:
            try:
                schedule = solve_with_deng(instance)
            except (ValueError, NotImplementedError):
                # Fall back to synthetic if Deng can't run on this instance
                schedule = generate_synthetic_schedule(instance)
        else:
            schedule = generate_synthetic_schedule(instance)
        
        # Evaluate cost
        cost = evaluate_schedule(instance, schedule)
        
        # Convert to trajectory
        trajectory = schedule_to_trajectory(instance, schedule, cost)
        
        data.append((instance, trajectory))
    
    return data


def generate_synthetic_data(
    n_instances: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Tuple[AMCSInstance, Trajectory]]:
    """
    Generate synthetic training data (no Deng solver required).
    
    Uses simple greedy heuristics as the expert.
    Good for testing the pipeline before integrating real data.
    """
    config = DataGenConfig(
        n_instances=n_instances,
        seed=seed,
    )
    return generate_training_data(config, use_deng=False, verbose=verbose)


def compute_dataset_statistics(
    data: List[Tuple[AMCSInstance, Trajectory]],
) -> dict:
    """
    Compute statistics about a training dataset.
    """
    n_instances = len(data)
    
    total_steps = sum(len(traj) for _, traj in data)
    total_a_actions = sum(
        sum(1 for s in traj if s.action.type.value == "add_task" and s.action.check_type == 'A')
        for _, traj in data
    )
    total_c_actions = sum(
        sum(1 for s in traj if s.action.type.value == "add_task" and s.action.check_type == 'C')
        for _, traj in data
    )
    total_stop_actions = sum(
        sum(1 for s in traj if s.action.type.value == "stop")
        for _, traj in data
    )
    
    rewards = [traj[-1].reward for _, traj in data if traj]
    
    return {
        'n_instances': n_instances,
        'total_steps': total_steps,
        'avg_steps_per_instance': total_steps / n_instances if n_instances > 0 else 0,
        'total_a_actions': total_a_actions,
        'total_c_actions': total_c_actions,
        'total_stop_actions': total_stop_actions,
        'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
        'min_reward': min(rewards) if rewards else 0,
        'max_reward': max(rewards) if rewards else 0,
    }


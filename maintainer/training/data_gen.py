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


def generate_training_data_from_excel(
    excel_path: str,
    n_instances: int = 100,
    horizon_days: int = 365 * 4,  # 4 years as in original dataset
    max_aircraft: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Tuple[AMCSInstance, Trajectory]]:
    """
    Generate training data from the real Deng Excel dataset.
    
    Args:
        excel_path: Path to the Deng Excel file
        n_instances: Number of training instances to generate (via data augmentation)
        horizon_days: Planning horizon in days
        max_aircraft: Limit number of aircraft (for testing)
        seed: Random seed
        verbose: Show progress bar
    
    Returns:
        List of (instance, trajectory) pairs
    """
    if not DENG_AVAILABLE:
        raise RuntimeError(
            "Deng solver not available. Cannot generate training data from real Excel file. "
            "Ensure aircraft_scheduler module is properly installed."
        )
    
    if seed is not None:
        random.seed(seed)
    
    # Load the real instance from Excel
    from maintainer.amcs.bridge import load_from_deng_data
    base_instance = load_from_deng_data(excel_path, max_aircraft=max_aircraft)
    
    if verbose:
        print(f"Loaded instance with {base_instance.n_aircraft} aircraft, horizon={base_instance.horizon} days")
    
    data = []
    
    iterator = range(n_instances)
    if verbose:
        iterator = tqdm(iterator, desc="Generating training data from Deng")
    
    for i in iterator:
        # For data augmentation, we can vary the horizon or use different subsets
        # For now, use the same instance with different horizons
        if i == 0:
            instance = base_instance
        else:
            # Vary horizon slightly for augmentation
            varied_horizon = max(180, base_instance.horizon + random.randint(-90, 90))
            instance = AMCSInstance(
                n_aircraft=base_instance.n_aircraft,
                horizon=varied_horizon,
                capacity=base_instance.capacity,
                aircraft=base_instance.aircraft,
                _source_data=base_instance._source_data,
            )
        
        # Solve with Deng - NO silent fallback!
        try:
            schedule = solve_with_deng(instance)
        except Exception as e:
            raise RuntimeError(
                f"Deng solver failed on instance {i}: {e}. "
                "Not falling back to synthetic data."
            )
        
        # Evaluate cost
        cost = evaluate_schedule(instance, schedule)
        
        # Convert to trajectory
        trajectory = schedule_to_trajectory(instance, schedule, cost)
        
        data.append((instance, trajectory))
    
    return data


def generate_training_data(
    config: DataGenConfig,
    use_deng: bool = True,
    verbose: bool = True,
) -> List[Tuple[AMCSInstance, Trajectory]]:
    """
    Generate training data from random synthetic instances.
    
    NOTE: This uses synthetic instances. For real data, use generate_training_data_from_excel().
    
    Args:
        config: Data generation configuration
        use_deng: Whether to use Deng solver (requires real data with _source_data)
        verbose: Show progress bar
    
    Returns:
        List of (instance, trajectory) pairs
    """
    if use_deng:
        raise ValueError(
            "Cannot use Deng solver with synthetic instances. "
            "Either use generate_training_data_from_excel() with real Excel data, "
            "or set use_deng=False to use synthetic heuristic."
        )
    
    if config.seed is not None:
        random.seed(config.seed)
    
    data = []
    
    iterator = range(config.n_instances)
    if verbose:
        iterator = tqdm(iterator, desc="Generating synthetic training data")
    
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
        
        # Use synthetic heuristic
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


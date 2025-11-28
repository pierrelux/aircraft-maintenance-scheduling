"""
Bridge module connecting simplified Maintainer interfaces to the existing Deng solver.

This handles conversion between:
- AMCSInstance <-> MaintenanceData/FleetState
- Schedule <-> ScheduleResult
"""

from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to import aircraft_scheduler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .instance import AMCSInstance, AircraftConfig
from .schedule import Schedule

# Try to import the existing scheduler
try:
    from aircraft_scheduler import (
        load_maintenance_data,
        MaintenanceData,
        GreedyScheduler,
        DPScheduler,
        ScheduleResult,
    )
    from aircraft_scheduler.core import (
        FleetState,
        create_initial_fleet_state,
    )
    DENG_AVAILABLE = True
except ImportError:
    DENG_AVAILABLE = False


def _convert_instance_to_maintenance_data(instance: AMCSInstance) -> "MaintenanceData":
    """
    Convert simplified AMCSInstance to full MaintenanceData.
    
    For now, this creates synthetic data matching the simplified instance.
    In production, you'd load real data and just use the instance for indexing.
    """
    if not DENG_AVAILABLE:
        raise RuntimeError("Deng solver not available. Install aircraft_scheduler module.")
    
    # If the instance was created from real data, return the source
    if instance._source_data is not None:
        return instance._source_data
    
    # Otherwise, we need to create synthetic MaintenanceData
    # This is a simplified stub - full implementation would create proper data structures
    raise NotImplementedError(
        "Creating MaintenanceData from synthetic AMCSInstance not yet implemented. "
        "Use load_from_deng_data() to create instances from real Excel data."
    )


def _convert_schedule_result_to_schedule(
    result: "ScheduleResult",
    instance: AMCSInstance,
) -> Schedule:
    """Convert ScheduleResult from Deng solver to simplified Schedule."""
    schedule = Schedule(instance.horizon, instance.n_aircraft)
    
    # Build a mapping from tail to aircraft_id
    # For now, assume sequential ordering
    if instance._source_data is not None:
        tail_to_id = {
            tail: i for i, tail in enumerate(instance._source_data.aircraft_tails)
        }
    else:
        tail_to_id = {}
    
    for check in result.checks:
        # Map tail to aircraft_id
        if check.tail in tail_to_id:
            aircraft_id = tail_to_id[check.tail]
        else:
            # Try to parse tail as integer
            try:
                aircraft_id = int(check.tail)
            except ValueError:
                continue
        
        if aircraft_id >= instance.n_aircraft:
            continue
        
        # Map check type
        if check.check_type in ('A',):
            check_type = 'A'
            duration = instance.aircraft[aircraft_id].A_duration
        elif check.check_type in ('C', 'D'):
            check_type = 'C'
            duration = check.duration if check.duration > 0 else instance.aircraft[aircraft_id].C_duration
        else:
            continue
        
        # Convert date to day index
        if instance._source_data is not None:
            start_day = (check.start_date - instance._source_data.config.begin_day).days
        else:
            start_day = 0  # Would need proper date handling
        
        if 0 <= start_day < instance.horizon:
            schedule.assign(start_day, aircraft_id, check_type, duration)
    
    return schedule


def load_from_deng_data(excel_path: str, max_aircraft: Optional[int] = None) -> AMCSInstance:
    """
    Load an AMCSInstance from the Deng Excel data format.
    
    Args:
        excel_path: Path to the Excel file
        max_aircraft: If set, limit to first N aircraft (for testing)
    
    Returns:
        AMCSInstance with _source_data pointing to full MaintenanceData
    """
    if not DENG_AVAILABLE:
        raise RuntimeError("Deng solver not available.")
    
    data = load_maintenance_data(excel_path)
    
    aircraft_tails = data.aircraft_tails
    if max_aircraft is not None:
        aircraft_tails = aircraft_tails[:max_aircraft]
    
    n_aircraft = len(aircraft_tails)
    horizon = data.config.total_years * 365
    capacity = data.config.max_c_check
    
    aircraft_configs = []
    for i, tail in enumerate(aircraft_tails):
        c_init = data.c_initial[tail]
        a_init = data.a_initial[tail]
        
        aircraft_configs.append(AircraftConfig(
            id=i,
            A_interval=a_init.a_ci_dy,
            A_duration=1,  # A-checks are typically overnight
            A_tolerance=a_init.a_tol_dy,
            C_interval=c_init.c_ci_dy,
            C_duration=data.config.c_e_t,  # Default C-check elapsed time
            C_tolerance=c_init.c_tol_dy,
            initial_days_since_A=a_init.dy_a,
            initial_days_since_C=c_init.dy_c,
        ))
    
    return AMCSInstance(
        n_aircraft=n_aircraft,
        horizon=horizon,
        capacity=capacity,
        aircraft=aircraft_configs,
        _source_data=data,
    )


def solve_with_deng(
    instance: AMCSInstance,
    algorithm: str = "dp",
    verbose: bool = False,
) -> Schedule:
    """
    Solve an AMCS instance using the Deng solver.
    
    Args:
        instance: The problem instance
        algorithm: 'greedy' or 'dp'
        verbose: Print progress
    
    Returns:
        Schedule from Deng solver
    """
    if not DENG_AVAILABLE:
        raise RuntimeError("Deng solver not available.")
    
    # Need source data to run Deng
    if instance._source_data is None:
        raise ValueError(
            "Cannot run Deng solver on synthetic instance. "
            "Use load_from_deng_data() to create instance from real data."
        )
    
    data = instance._source_data
    
    if algorithm == "greedy":
        scheduler = GreedyScheduler(data)
    else:
        scheduler = DPScheduler(data)
    
    result = scheduler.schedule(instance.horizon, verbose=verbose)
    
    return _convert_schedule_result_to_schedule(result, instance)


def evaluate_schedule(instance: AMCSInstance, schedule: Schedule) -> float:
    """
    Evaluate a schedule's cost.
    
    For now, uses a simplified cost function:
    - Grounded days (weighted by check type)
    - Capacity violations
    - Window violations (tasks outside their feasible window)
    
    Returns:
        Total cost (lower is better)
    """
    cost = 0.0
    
    # Cost 1: Grounded days
    # A-checks cost less than C-checks (typically shorter, less impact)
    a_grounded = 0
    c_grounded = 0
    
    for check in schedule.get_all_checks():
        if check.check_type == 'A':
            a_grounded += check.duration
        else:
            c_grounded += check.duration
    
    cost += a_grounded * 1.0  # Weight for A-check grounded days
    cost += c_grounded * 2.0  # Weight for C-check grounded days
    
    # Cost 2: Capacity violations
    for day in range(schedule.horizon):
        usage = schedule.get_capacity_usage(day)
        if usage > instance.capacity:
            cost += (usage - instance.capacity) * 1000  # Heavy penalty
    
    # Cost 3: Window violations
    # Check if tasks are scheduled within their feasible windows
    for aircraft_id in range(instance.n_aircraft):
        ac = instance.get_aircraft(aircraft_id)
        checks = schedule.get_checks_for_aircraft(aircraft_id)
        
        a_checks = [c for c in checks if c.check_type == 'A']
        c_checks = [c for c in checks if c.check_type == 'C']
        
        # Sort by start day
        a_checks.sort(key=lambda c: c.start_day)
        c_checks.sort(key=lambda c: c.start_day)
        
        # Check A-check windows
        for i, check in enumerate(a_checks):
            earliest = ac.A_earliest_day(i)
            latest = ac.A_latest_day(i, instance.horizon)
            
            if check.start_day < earliest:
                cost += (earliest - check.start_day) * 10  # Too early
            if check.start_day > latest:
                cost += (check.start_day - latest) * 100  # Too late (more severe)
        
        # Check C-check windows
        for i, check in enumerate(c_checks):
            earliest = ac.C_earliest_day(i)
            latest = ac.C_latest_day(i, instance.horizon)
            
            if check.start_day < earliest:
                cost += (earliest - check.start_day) * 10
            if check.start_day > latest:
                cost += (check.start_day - latest) * 100
    
    return cost


def evaluate_decoder_loss(instance: AMCSInstance) -> Tuple[float, float]:
    """
    Diagnostic: Compare Deng schedule cost with greedy-decoded CPD cost.
    
    This isolates how much cost difference comes from the greedy decoder
    vs. the CPD construction itself.
    
    Returns:
        (deng_cost, greedy_decoded_cost)
    """
    # This requires the extraction module (Phase 3)
    # For now, just return the Deng schedule evaluated twice
    deng_schedule = solve_with_deng(instance)
    deng_cost = evaluate_schedule(instance, deng_schedule)
    
    # TODO: Once extraction is implemented:
    # cpd = extract_cpd_from_schedule(instance, deng_schedule)
    # greedy_schedule = decode_cpd_to_schedule(cpd)
    # greedy_cost = evaluate_schedule(instance, greedy_schedule)
    
    return deng_cost, deng_cost  # Placeholder until extraction is ready


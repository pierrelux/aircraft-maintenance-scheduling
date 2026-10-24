"""
Aircraft Maintenance Check Scheduling Optimization Package.

Based on: Deng, Q., Santos, B.F., Curran, R. (2020). "A practical dynamic 
programming based methodology for aircraft maintenance check scheduling 
optimization." European Journal of Operational Research, 281(2), 256-273.
"""

from .data_loader import (
    MaintenanceData,
    load_maintenance_data,
    AircraftDInitial,
    AircraftCInitial,
    AircraftAInitial,
    FlightRates,
    CCheckInfo,
    Config,
    DateRange,
    PeakSeason,
    ReservedMaintenance,
    AdditionalSlots,
)

from .core import (
    AircraftState,
    FleetState,
    create_initial_fleet_state,
    CheckResult,
    CCheckResult,
    age_aircraft,
    advance_fleet_one_day,
    schedule_a_check,
    schedule_c_check,
    perform_a_check,
    perform_c_check,
    get_daily_fh,
    get_daily_fc,
    get_c_check_elapsed_time,
    get_next_c_check_code,
    ConstraintViolation,
    can_schedule_c_check,
    can_schedule_a_check,
    get_available_c_slots,
    get_available_a_slots,
    get_feasible_c_check_aircraft,
    get_feasible_a_check_aircraft,
    get_mandatory_c_check_aircraft,
    get_mandatory_a_check_aircraft,
    is_c_check_not_allowed,
    is_a_check_not_allowed,
    is_c_check_peak_season,
    is_public_holiday,
    validate_fleet_state,
    CheckCost,
    FleetCost,
    calculate_a_check_cost,
    calculate_c_check_cost,
    calculate_urgency_score,
    calculate_overdue_penalty,
    calculate_tolerance_penalty,
    calculate_ground_waiting_penalty,
    calculate_scheduling_benefit,
    calculate_fleet_cost,
    estimate_future_checks,
)

from .scheduler import (
    Scheduler,
    ScheduledCheck,
    ScheduleResult,
    Table3KPIs,
    FleetAction,
    PAPER_ACTIONS,
)

from .visualization import (
    plot_gantt_chart,
    plot_check_distribution,
    plot_capacity_utilization,
    plot_wasted_interval_analysis,
    plot_fleet_timeline,
    generate_report,
    generate_table3_report,
    plot_fh_distribution,
    plot_tolerance_analysis,
)

from .adp import (
    LookaheadADPScheduler,
    thrifty_rollout,
    remaining_utilization_days,
)

__version__ = "2.0.0"
__all__ = [
    # Data
    "MaintenanceData", "load_maintenance_data", "Config",
    # State
    "AircraftState", "FleetState", "create_initial_fleet_state",
    # Scheduler
    "Scheduler",
    "ScheduledCheck", "ScheduleResult", "Table3KPIs", "FleetAction", "PAPER_ACTIONS",
    # Visualization
    "plot_gantt_chart", "plot_check_distribution", "generate_report",
]

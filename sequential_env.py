"""
Sequential Scheduling Environment for Aircraft Maintenance.

Key idea: Build schedule incrementally, action by action.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from copy import deepcopy


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AircraftStatus:
    """
    Status of a single aircraft.
    
    Includes tolerance tracking (Paper's Eq. 18-20) and check labels.
    """
    
    # Identification
    aircraft_id: int
    
    # A-check counters (days/hours/cycles since last A-check)
    dy_a: float = 0.0
    fh_a: float = 0.0
    fc_a: float = 0.0
    
    # C-check counters
    dy_c: float = 0.0
    fh_c: float = 0.0
    fc_c: float = 0.0
    
    # Daily utilization rates (aircraft-specific)
    daily_fh: float = 10.5  # Flight hours per day
    daily_fc: float = 4.7   # Flight cycles per day
    
    # Maintenance status
    in_maintenance: bool = False
    maintenance_days_remaining: int = 0
    current_check_type: Optional[str] = None  # 'A' or 'C'
    
    # Tolerance tracking (Paper's Eq. 18-20)
    # When check exceeds interval, tolerance is consumed from NEXT interval
    tolerance_used_dy_a: float = 0.0  # Days tolerance used in last A-check
    tolerance_used_fh_a: float = 0.0  # FH tolerance used
    tolerance_used_fc_a: float = 0.0  # FC tolerance used
    tolerance_used_dy_c: float = 0.0  # Days tolerance used in last C-check
    tolerance_used_fh_c: float = 0.0  # FH tolerance used
    tolerance_used_fc_c: float = 0.0  # FC tolerance used
    
    # Check labels (Paper's Eq. 21): A1-A4 and C1-C12 cycles
    check_label_a: int = 1  # Current A-check label (1-4)
    check_label_c: int = 1  # Current C-check label (1-12)
    
    # Grounding status (Paper's Eq. 15)
    is_grounded: bool = False
    grounded_days: int = 0  # Days waiting for slot
    
    def copy(self):
        """Deep copy of aircraft status."""
        return AircraftStatus(
            aircraft_id=self.aircraft_id,
            dy_a=self.dy_a,
            fh_a=self.fh_a,
            fc_a=self.fc_a,
            dy_c=self.dy_c,
            fh_c=self.fh_c,
            fc_c=self.fc_c,
            daily_fh=self.daily_fh,
            daily_fc=self.daily_fc,
            in_maintenance=self.in_maintenance,
            maintenance_days_remaining=self.maintenance_days_remaining,
            current_check_type=self.current_check_type,
            tolerance_used_dy_a=self.tolerance_used_dy_a,
            tolerance_used_fh_a=self.tolerance_used_fh_a,
            tolerance_used_fc_a=self.tolerance_used_fc_a,
            tolerance_used_dy_c=self.tolerance_used_dy_c,
            tolerance_used_fh_c=self.tolerance_used_fh_c,
            tolerance_used_fc_c=self.tolerance_used_fc_c,
            check_label_a=self.check_label_a,
            check_label_c=self.check_label_c,
            is_grounded=self.is_grounded,
            grounded_days=self.grounded_days
        )
    
    def get_effective_interval_a(self, config) -> Tuple[float, float, float]:
        """Get effective A-check interval after tolerance deduction."""
        return (
            config.a_interval_dy - self.tolerance_used_dy_a,
            config.a_interval_fh - self.tolerance_used_fh_a,
            config.a_interval_fc - self.tolerance_used_fc_a
        )
    
    def get_effective_interval_c(self, config) -> Tuple[float, float, float]:
        """Get effective C-check interval after tolerance deduction."""
        return (
            config.c_interval_dy - self.tolerance_used_dy_c,
            config.c_interval_fh - self.tolerance_used_fh_c,
            config.c_interval_fc - self.tolerance_used_fc_c
        )


@dataclass
class SchedulingState:
    """
    State in sequential construction.
    
    Represents a partial schedule being built incrementally.
    """
    
    # Time cursor
    current_day: int
    
    # Fleet status: aircraft_id -> AircraftStatus
    fleet: Dict[int, AircraftStatus]
    
    # Schedule built so far: day -> list of (check_type, aircraft_id)
    schedule: Dict[int, List[Tuple[str, int]]]
    
    # Counters for today's scheduling
    c_checks_today: int = 0
    a_checks_today: int = 0
    
    def copy(self):
        """Deep copy of state."""
        return SchedulingState(
            current_day=self.current_day,
            fleet={aid: aircraft.copy() for aid, aircraft in self.fleet.items()},
            schedule=deepcopy(self.schedule),
            c_checks_today=self.c_checks_today,
            a_checks_today=self.a_checks_today
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MaintenanceConfig:
    """Configuration for maintenance requirements."""
    
    # A-check intervals (max allowed before check is due)
    a_interval_dy: float = 120    # Days
    a_interval_fh: float = 750    # Flight hours
    a_interval_fc: float = 750    # Flight cycles
    
    # C-check intervals
    c_interval_dy: float = 730    # ~2 years
    c_interval_fh: float = 7500
    c_interval_fc: float = 5000
    
    # Minimum time between checks (can't do back-to-back)
    a_min_interval: float = 30    # Days
    c_min_interval: float = 90    # Days
    
    # Maintenance durations
    a_check_duration: int = 1     # Days
    c_check_duration: int = 20    # Days
    
    # Eligibility thresholds (schedule when within this many days of due)
    a_eligibility_threshold: float = 30   # Can schedule A-check when <30 days remaining
    c_eligibility_threshold: float = 90   # Can schedule C-check when <90 days remaining
    
    # Maximum tolerance allowed (Paper's constraints)
    max_tolerance_dy_a: float = 10   # Max days tolerance for A-check
    max_tolerance_fh_a: float = 75   # Max FH tolerance (10% of interval)
    max_tolerance_fc_a: float = 75   # Max FC tolerance
    max_tolerance_dy_c: float = 30   # Max days tolerance for C-check
    max_tolerance_fh_c: float = 750  # Max FH tolerance
    max_tolerance_fc_c: float = 500  # Max FC tolerance
    
    # Penalties (Paper's cost function)
    # Reduced penalties to focus on wasted FH optimization
    penalty_tolerance: float = 10.0    # Cost per unit of tolerance used
    penalty_grounded: float = 100.0    # Cost per day grounded (was 10000)


# =============================================================================
# Sequential Scheduling Environment
# =============================================================================

class SequentialSchedulingEnv:
    """
    Sequential construction environment for aircraft maintenance.
    
    State: (current_day, fleet_status, partial_schedule)
    Action: Add check OR advance to next day
    """
    
    def __init__(
        self,
        n_aircraft: int = 10,
        horizon: int = 365,
        config: Optional[MaintenanceConfig] = None,
        c_capacity: Optional[Dict[int, int]] = None,
        a_capacity: Optional[Dict[int, int]] = None,
        seed: int = 42
    ):
        """
        Initialize environment.
        
        Args:
            n_aircraft: Number of aircraft in fleet
            horizon: Planning horizon (days)
            config: Maintenance configuration
            c_capacity: C-check capacity per day (default: 1 per day)
            a_capacity: A-check capacity per day (default: 1 per day)
            seed: Random seed
        """
        self.n_aircraft = n_aircraft
        self.horizon = horizon
        self.config = config or MaintenanceConfig()
        
        # Set capacities
        if c_capacity is None:
            # Default: 1 C-check slot per weekday, 0 on weekends
            self.c_capacity = {}
            for day in range(horizon):
                day_of_week = day % 7
                self.c_capacity[day] = 1 if day_of_week < 5 else 0
        else:
            self.c_capacity = c_capacity
        
        if a_capacity is None:
            # Default: 1 A-check slot per weekday, 0 on weekends
            self.a_capacity = {}
            for day in range(horizon):
                day_of_week = day % 7
                self.a_capacity[day] = 1 if day_of_week < 5 else 0
        else:
            self.a_capacity = a_capacity
        
        # Action space
        # Actions: [C-check for aircraft 0, ..., C-check for aircraft N-1,
        #           A-check for aircraft 0, ..., A-check for aircraft N-1,
        #           advance to next day]
        self.n_actions = 2 * n_aircraft + 1
        
        # State dimension (for neural network input)
        # Time (1) + Fleet status (12 per aircraft) + Scheduled today (N) + Capacity (2)
        # Per aircraft: 6 counters + in_maint + 3 tolerance + check_labels + grounded = 12
        self.state_dim = 1 + 12 * n_aircraft + n_aircraft + 2
        
        self.rng = np.random.RandomState(seed)
        
        # Current state
        self.state = None
    
    def reset(self) -> SchedulingState:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state with empty schedule
        """
        # Initialize fleet
        fleet = {}
        for i in range(self.n_aircraft):
            # Randomize initial status (aircraft are at different stages)
            aircraft = AircraftStatus(
                aircraft_id=i,
                dy_a=self.rng.uniform(0, self.config.a_interval_dy * 0.5),
                fh_a=self.rng.uniform(0, self.config.a_interval_fh * 0.5),
                fc_a=self.rng.uniform(0, self.config.a_interval_fc * 0.5),
                dy_c=self.rng.uniform(0, self.config.c_interval_dy * 0.3),
                fh_c=self.rng.uniform(0, self.config.c_interval_fh * 0.3),
                fc_c=self.rng.uniform(0, self.config.c_interval_fc * 0.3),
                daily_fh=self.rng.uniform(9.0, 12.0),  # Aircraft-specific utilization
                daily_fc=self.rng.uniform(4.0, 5.5),
            )
            fleet[i] = aircraft
        
        # Initial state: day 0, empty schedule
        self.state = SchedulingState(
            current_day=0,
            fleet=fleet,
            schedule={},
            c_checks_today=0,
            a_checks_today=0
        )
        
        return self.state
    
    def step(self, action: int) -> Tuple[SchedulingState, float, bool, dict]:
        """
        Take action in environment.
        
        Args:
            action: Action index
                [0, N-1]: Schedule C-check for aircraft
                [N, 2N-1]: Schedule A-check for aircraft
                2N: Advance to next day
        
        Returns:
            next_state: Next state
            reward: Immediate reward
            done: Whether episode is complete
            info: Additional information
        """
        assert self.state is not None, "Must call reset() first"
        
        # Decode action
        if action < self.n_aircraft:
            # C-check for aircraft `action`
            check_type = 'C'
            aircraft_id = action
        elif action < 2 * self.n_aircraft:
            # A-check for aircraft `action - n_aircraft`
            check_type = 'A'
            aircraft_id = action - self.n_aircraft
        else:
            # Advance to next day
            check_type = None
            aircraft_id = None
        
        # Compute reward
        reward = self._compute_reward(self.state, check_type, aircraft_id)
        
        # Apply transition
        next_state = self._transition(self.state, check_type, aircraft_id)
        
        # Check if done
        done = self.is_terminal(next_state)
        
        # Update current state
        self.state = next_state
        
        info = {
            'current_day': next_state.current_day,
            'check_type': check_type,
            'aircraft_id': aircraft_id
        }
        
        return next_state, reward, done, info
    
    def _transition(
        self,
        state: SchedulingState,
        check_type: Optional[str],
        aircraft_id: Optional[int]
    ) -> SchedulingState:
        """
        State transition function with tolerance tracking (Paper's Eq. 18-20).
        
        Sequential construction: build schedule incrementally!
        """
        next_state = state.copy()
        
        if check_type is None:
            # Action: Advance to next day
            
            # Apply today's scheduled checks to fleet
            if state.current_day in state.schedule:
                for scheduled_check_type, scheduled_aircraft_id in state.schedule[state.current_day]:
                    aircraft = next_state.fleet[scheduled_aircraft_id]
                    
                    if scheduled_check_type == 'C':
                        # Compute tolerance used (Paper's Eq. 18-20)
                        eff_dy, eff_fh, eff_fc = aircraft.get_effective_interval_c(self.config)
                        tol_dy = max(0, aircraft.dy_c - eff_dy)
                        tol_fh = max(0, aircraft.fh_c - eff_fh)
                        tol_fc = max(0, aircraft.fc_c - eff_fc)
                        
                        # Store tolerance for next interval
                        aircraft.tolerance_used_dy_c = min(tol_dy, self.config.max_tolerance_dy_c)
                        aircraft.tolerance_used_fh_c = min(tol_fh, self.config.max_tolerance_fh_c)
                        aircraft.tolerance_used_fc_c = min(tol_fc, self.config.max_tolerance_fc_c)
                        
                        # Reset C-check counters
                        aircraft.dy_c = 0.0
                        aircraft.fh_c = 0.0
                        aircraft.fc_c = 0.0
                        
                        # Also reset A-check (A-check merged into C-check)
                        aircraft.tolerance_used_dy_a = 0.0
                        aircraft.tolerance_used_fh_a = 0.0
                        aircraft.tolerance_used_fc_a = 0.0
                        aircraft.dy_a = 0.0
                        aircraft.fh_a = 0.0
                        aircraft.fc_a = 0.0
                        
                        # Update check label (C1->C2->...->C12->C1)
                        aircraft.check_label_c = (aircraft.check_label_c % 12) + 1
                        aircraft.check_label_a = 1  # Reset A-label on C-check
                        
                        # Put in maintenance
                        aircraft.in_maintenance = True
                        aircraft.is_grounded = False
                        aircraft.grounded_days = 0
                        aircraft.maintenance_days_remaining = self.config.c_check_duration
                        aircraft.current_check_type = 'C'
                    
                    elif scheduled_check_type == 'A':
                        # Compute tolerance used
                        eff_dy, eff_fh, eff_fc = aircraft.get_effective_interval_a(self.config)
                        tol_dy = max(0, aircraft.dy_a - eff_dy)
                        tol_fh = max(0, aircraft.fh_a - eff_fh)
                        tol_fc = max(0, aircraft.fc_a - eff_fc)
                        
                        # Store tolerance for next interval
                        aircraft.tolerance_used_dy_a = min(tol_dy, self.config.max_tolerance_dy_a)
                        aircraft.tolerance_used_fh_a = min(tol_fh, self.config.max_tolerance_fh_a)
                        aircraft.tolerance_used_fc_a = min(tol_fc, self.config.max_tolerance_fc_a)
                        
                        # Reset A-check counters
                        aircraft.dy_a = 0.0
                        aircraft.fh_a = 0.0
                        aircraft.fc_a = 0.0
                        
                        # Update check label (A1->A2->A3->A4->A1)
                        aircraft.check_label_a = (aircraft.check_label_a % 4) + 1
                        
                        # Put in maintenance
                        aircraft.in_maintenance = True
                        aircraft.is_grounded = False
                        aircraft.grounded_days = 0
                        aircraft.maintenance_days_remaining = self.config.a_check_duration
                        aircraft.current_check_type = 'A'
            
            # Check for grounding before aging (aircraft that exceeded limits)
            scheduled_today = set()
            if state.current_day in state.schedule:
                scheduled_today = {aid for _, aid in state.schedule[state.current_day]}
            
            for aircraft in next_state.fleet.values():
                if aircraft.in_maintenance or aircraft.aircraft_id in scheduled_today:
                    continue
                
                # Check if exceeds effective interval (grounded)
                eff_dy_a, eff_fh_a, eff_fc_a = aircraft.get_effective_interval_a(self.config)
                eff_dy_c, eff_fh_c, eff_fc_c = aircraft.get_effective_interval_c(self.config)
                
                a_overdue = (aircraft.dy_a > eff_dy_a + self.config.max_tolerance_dy_a or
                            aircraft.fh_a > eff_fh_a + self.config.max_tolerance_fh_a or
                            aircraft.fc_a > eff_fc_a + self.config.max_tolerance_fc_a)
                c_overdue = (aircraft.dy_c > eff_dy_c + self.config.max_tolerance_dy_c or
                            aircraft.fh_c > eff_fh_c + self.config.max_tolerance_fh_c or
                            aircraft.fc_c > eff_fc_c + self.config.max_tolerance_fc_c)
                
                if a_overdue or c_overdue:
                    aircraft.is_grounded = True
                    aircraft.grounded_days += 1
            
            # Age all aircraft by one day
            for aircraft in next_state.fleet.values():
                if aircraft.in_maintenance:
                    # In maintenance: count down
                    aircraft.maintenance_days_remaining -= 1
                    if aircraft.maintenance_days_remaining <= 0:
                        aircraft.in_maintenance = False
                        aircraft.current_check_type = None
                elif not aircraft.is_grounded:
                    # Operating: accumulate usage (only if not grounded)
                    aircraft.dy_a += 1.0
                    aircraft.dy_c += 1.0
                    aircraft.fh_a += aircraft.daily_fh
                    aircraft.fh_c += aircraft.daily_fh
                    aircraft.fc_a += aircraft.daily_fc
                    aircraft.fc_c += aircraft.daily_fc
            
            # Move to next day
            next_state.current_day += 1
            next_state.c_checks_today = 0
            next_state.a_checks_today = 0
        
        else:
            # Action: Add check to today's schedule
            
            # Add to schedule
            if state.current_day not in next_state.schedule:
                next_state.schedule[state.current_day] = []
            next_state.schedule[state.current_day].append((check_type, aircraft_id))
            
            # Update counters
            if check_type == 'C':
                next_state.c_checks_today += 1
            else:  # 'A'
                next_state.a_checks_today += 1
        
        return next_state
    
    def _compute_reward(
        self,
        state: SchedulingState,
        check_type: Optional[str],
        aircraft_id: Optional[int]
    ) -> float:
        """
        Compute immediate reward with tolerance and grounding penalties.
        
        Paper's cost function (Eq. 27-28):
        - Wasted interval (FH not used before check)
        - Tolerance penalty (using buffer reduces next interval)
        - Grounding penalty (aircraft waiting for slot)
        """
        reward = 0.0
        
        if check_type is None:
            # Penalty for grounded aircraft
            for aircraft in state.fleet.values():
                if aircraft.is_grounded:
                    reward -= self.config.penalty_grounded
            return reward
        
        aircraft = state.fleet[aircraft_id]
        
        if check_type == 'C':
            # Get effective interval (after tolerance deduction)
            eff_dy, eff_fh, eff_fc = aircraft.get_effective_interval_c(self.config)
            
            # Wasted FH for C-check (paper's objective function)
            wasted_fh = eff_fh - aircraft.fh_c
            reward -= max(0, wasted_fh)  # Direct FH penalty
            
            # Tolerance penalty (if using tolerance this check)
            tol_dy = max(0, aircraft.dy_c - eff_dy)
            tol_fh = max(0, aircraft.fh_c - eff_fh)
            tol_fc = max(0, aircraft.fc_c - eff_fc)
            if tol_dy > 0 or tol_fh > 0 or tol_fc > 0:
                reward -= self.config.penalty_tolerance * max(tol_dy, tol_fh / 10, tol_fc / 10)
        
        else:  # 'A'
            # Get effective interval
            eff_dy, eff_fh, eff_fc = aircraft.get_effective_interval_a(self.config)
            
            # Wasted FH for A-check (paper's objective function)
            wasted_fh = eff_fh - aircraft.fh_a
            reward -= max(0, wasted_fh)  # Direct FH penalty
            
            # Tolerance penalty
            tol_dy = max(0, aircraft.dy_a - eff_dy)
            tol_fh = max(0, aircraft.fh_a - eff_fh)
            tol_fc = max(0, aircraft.fc_a - eff_fc)
            if tol_dy > 0 or tol_fh > 0 or tol_fc > 0:
                reward -= self.config.penalty_tolerance * max(tol_dy, tol_fh / 10, tol_fc / 10)
        
        return reward
    
    def get_valid_actions(self, state: SchedulingState) -> np.ndarray:
        """
        Get mask of valid actions.
        
        Returns:
            Boolean array of shape (n_actions,)
        """
        valid = np.zeros(self.n_actions, dtype=bool)
        
        # Check if we can add C-checks
        if state.c_checks_today < self.c_capacity.get(state.current_day, 0):
            for i in range(self.n_aircraft):
                if self._can_schedule_c_check(state, i):
                    valid[i] = True
        
        # Check if we can add A-checks
        if state.a_checks_today < self.a_capacity.get(state.current_day, 0):
            for i in range(self.n_aircraft):
                if self._can_schedule_a_check(state, i):
                    valid[self.n_aircraft + i] = True
        
        # Can always advance to next day (unless at terminal)
        if state.current_day < self.horizon:
            valid[2 * self.n_aircraft] = True
        
        return valid
    
    def _can_schedule_c_check(self, state: SchedulingState, aircraft_id: int) -> bool:
        """Check if aircraft can be scheduled for C-check."""
        aircraft = state.fleet[aircraft_id]
        
        # Can't schedule if already in maintenance
        if aircraft.in_maintenance:
            return False
        
        # Can't schedule if already scheduled today
        if state.current_day in state.schedule:
            for check_type, aid in state.schedule[state.current_day]:
                if aid == aircraft_id:
                    return False
        
        # Must meet minimum interval since last C-check
        if aircraft.dy_c < self.config.c_min_interval:
            return False
        
        # Must be within eligibility threshold (close enough to due)
        remaining_dy = self.config.c_interval_dy - aircraft.dy_c
        remaining_fh = (self.config.c_interval_fh - aircraft.fh_c) / aircraft.daily_fh
        remaining_fc = (self.config.c_interval_fc - aircraft.fc_c) / aircraft.daily_fc
        remaining = min(remaining_dy, remaining_fh, remaining_fc)
        
        if remaining > self.config.c_eligibility_threshold:
            return False
        
        return True
    
    def _can_schedule_a_check(self, state: SchedulingState, aircraft_id: int) -> bool:
        """Check if aircraft can be scheduled for A-check."""
        aircraft = state.fleet[aircraft_id]
        
        # Can't schedule if already in maintenance
        if aircraft.in_maintenance:
            return False
        
        # Can't schedule if already scheduled today
        if state.current_day in state.schedule:
            for check_type, aid in state.schedule[state.current_day]:
                if aid == aircraft_id:
                    return False
        
        # Must meet minimum interval since last A-check
        if aircraft.dy_a < self.config.a_min_interval:
            return False
        
        # Must be within eligibility threshold
        remaining_dy = self.config.a_interval_dy - aircraft.dy_a
        remaining_fh = (self.config.a_interval_fh - aircraft.fh_a) / aircraft.daily_fh
        remaining_fc = (self.config.a_interval_fc - aircraft.fc_a) / aircraft.daily_fc
        remaining = min(remaining_dy, remaining_fh, remaining_fc)
        
        if remaining > self.config.a_eligibility_threshold:
            return False
        
        return True
    
    def is_terminal(self, state: SchedulingState) -> bool:
        """Check if state is terminal."""
        return state.current_day >= self.horizon
    
    def compute_total_wasted(self, state: SchedulingState) -> float:
        """
        Compute total wasted hours for complete schedule.
        
        This is for evaluation only (not used during training).
        """
        total_wasted = 0.0
        
        for day, checks in state.schedule.items():
            for check_type, aircraft_id in checks:
                # Get aircraft status at time of scheduling
                # (Need to reconstruct, since we only have final state)
                # For simplicity, just return 0 here
                # In practice, you'd track this during rollout
                pass
        
        return total_wasted


# =============================================================================
# State Encoding for Neural Networks
# =============================================================================

def encode_state(state: SchedulingState, n_aircraft: int, horizon: int) -> np.ndarray:
    """
    Encode state as feature vector for neural network.
    
    Features:
    - Current day (normalized)
    - For each aircraft: status features including tolerance and grounding
    - Which aircraft scheduled today
    - Remaining capacity today
    """
    features = []
    
    # Time: normalized day
    features.append(state.current_day / horizon)
    
    # Fleet status: for each aircraft (12 features each)
    for i in range(n_aircraft):
        aircraft = state.fleet[i]
        
        features.extend([
            aircraft.dy_a / 120,      # Normalized A-check days
            aircraft.fh_a / 750,      # Normalized A-check FH
            aircraft.fc_a / 750,      # Normalized A-check FC
            aircraft.dy_c / 730,      # Normalized C-check days
            aircraft.fh_c / 7500,     # Normalized C-check FH
            aircraft.fc_c / 5000,     # Normalized C-check FC
            float(aircraft.in_maintenance),  # Binary: in maintenance?
            # Tolerance used (normalized)
            (aircraft.tolerance_used_dy_a + aircraft.tolerance_used_dy_c) / 40,
            (aircraft.tolerance_used_fh_a + aircraft.tolerance_used_fh_c) / 825,
            (aircraft.tolerance_used_fc_a + aircraft.tolerance_used_fc_c) / 575,
            # Check labels (normalized)
            (aircraft.check_label_a - 1) / 3,  # 0-1 range
            (aircraft.check_label_c - 1) / 11,  # 0-1 range
        ])
    
    # Scheduled today: binary indicator per aircraft
    scheduled = [0] * n_aircraft
    if state.current_day in state.schedule:
        for check_type, aircraft_id in state.schedule[state.current_day]:
            scheduled[aircraft_id] = 1
    features.extend(scheduled)
    
    # Remaining capacity today (normalized)
    c_capacity_total = max(1, state.c_checks_today + 1)
    a_capacity_total = max(1, state.a_checks_today + 1)
    
    features.append(state.c_checks_today / c_capacity_total)
    features.append(state.a_checks_today / a_capacity_total)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Utility Functions
# =============================================================================

def visualize_schedule(state: SchedulingState, n_aircraft: int):
    """Print schedule for visualization."""
    print("\n" + "="*80)
    print("MAINTENANCE SCHEDULE")
    print("="*80)
    
    for day in sorted(state.schedule.keys()):
        if state.schedule[day]:
            print(f"Day {day:4d}: ", end="")
            for check_type, aircraft_id in state.schedule[day]:
                print(f"{check_type}-check(aircraft {aircraft_id})  ", end="")
            print()
    
    print("="*80)


def compute_schedule_statistics(state: SchedulingState, n_aircraft: int):
    """Compute statistics for completed schedule."""
    
    total_c_checks = 0
    total_a_checks = 0
    
    for day, checks in state.schedule.items():
        for check_type, aircraft_id in checks:
            if check_type == 'C':
                total_c_checks += 1
            else:
                total_a_checks += 1
    
    return {
        'total_c_checks': total_c_checks,
        'total_a_checks': total_a_checks,
        'total_checks': total_c_checks + total_a_checks,
        'days_with_maintenance': len(state.schedule)
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create environment
    env = SequentialSchedulingEnv(
        n_aircraft=5,
        horizon=30,  # 1 month for quick testing
        seed=42
    )
    
    # Reset
    state = env.reset()
    print(f"Initial state: day {state.current_day}")
    print(f"State dimension: {env.state_dim}")
    print(f"Action space: {env.n_actions}")
    
    # Take some random actions
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 100:
        # Get valid actions
        valid_actions = env.get_valid_actions(state)
        
        # Sample random valid action
        valid_indices = np.where(valid_actions)[0]
        action = np.random.choice(valid_indices)
        
        # Step
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: Day {state.current_day}, Reward {reward:.2f}")
        
        state = next_state
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final day: {state.current_day}")
    
    # Visualize
    visualize_schedule(state, env.n_aircraft)
    
    # Statistics
    stats = compute_schedule_statistics(state, env.n_aircraft)
    print(f"\nSchedule Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


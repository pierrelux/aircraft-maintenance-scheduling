"""
Simplified AMCS instance representation for Maintainer.

This provides a clean interface that Maintainer needs, while the bridge
module handles conversion to/from the full FleetState format.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random


@dataclass
class AircraftConfig:
    """Configuration for a single aircraft."""
    
    id: int                     # Aircraft identifier (0, 1, 2, ...)
    
    # A-check parameters
    A_interval: int             # Days between A-checks
    A_duration: int             # A-check duration in days
    A_tolerance: int            # Scheduling flexibility (days)
    
    # C-check parameters  
    C_interval: int             # Days between C-checks
    C_duration: int             # C-check duration in days
    C_tolerance: int            # Scheduling flexibility (days)
    
    # Initial state (days since last check at planning start)
    initial_days_since_A: int = 0
    initial_days_since_C: int = 0
    
    def max_A_checks(self, horizon: int) -> int:
        """Estimate maximum A-checks needed over horizon."""
        return (horizon + self.initial_days_since_A) // self.A_interval + 1
    
    def max_C_checks(self, horizon: int) -> int:
        """Estimate maximum C-checks needed over horizon."""
        return (horizon + self.initial_days_since_C) // self.C_interval + 1
    
    def A_earliest_day(self, check_index: int) -> int:
        """Earliest feasible day for the k-th A-check (0-indexed)."""
        if check_index == 0:
            # First check: interval - initial_days - tolerance
            return max(0, self.A_interval - self.initial_days_since_A - self.A_tolerance)
        else:
            # Subsequent: (k+1)*interval - initial_days - tolerance
            return max(0, (check_index + 1) * self.A_interval - self.initial_days_since_A - self.A_tolerance)
    
    def A_latest_day(self, check_index: int, horizon: int) -> int:
        """Latest feasible day for the k-th A-check (0-indexed)."""
        if check_index == 0:
            latest = self.A_interval - self.initial_days_since_A + self.A_tolerance
        else:
            latest = (check_index + 1) * self.A_interval - self.initial_days_since_A + self.A_tolerance
        return min(latest, horizon - self.A_duration)
    
    def C_earliest_day(self, check_index: int) -> int:
        """Earliest feasible day for the k-th C-check (0-indexed)."""
        if check_index == 0:
            return max(0, self.C_interval - self.initial_days_since_C - self.C_tolerance)
        else:
            return max(0, (check_index + 1) * self.C_interval - self.initial_days_since_C - self.C_tolerance)
    
    def C_latest_day(self, check_index: int, horizon: int) -> int:
        """Latest feasible day for the k-th C-check (0-indexed)."""
        if check_index == 0:
            latest = self.C_interval - self.initial_days_since_C + self.C_tolerance
        else:
            latest = (check_index + 1) * self.C_interval - self.initial_days_since_C + self.C_tolerance
        return min(latest, horizon - self.C_duration)


@dataclass
class AMCSInstance:
    """
    Simplified AMCS problem instance.
    
    This is the clean interface Maintainer uses. The bridge module
    handles conversion to/from the full MaintenanceData format.
    """
    
    n_aircraft: int                     # Fleet size
    horizon: int                        # Planning horizon in days
    capacity: int                       # Daily hangar capacity (parallel checks)
    aircraft: List[AircraftConfig]      # Per-aircraft configuration
    
    # Optional metadata for bridge
    _source_data: Optional[object] = field(default=None, repr=False)
    
    def __post_init__(self):
        assert len(self.aircraft) == self.n_aircraft, \
            f"Expected {self.n_aircraft} aircraft configs, got {len(self.aircraft)}"
    
    def get_aircraft(self, aircraft_id: int) -> AircraftConfig:
        """Get configuration for an aircraft by ID."""
        return self.aircraft[aircraft_id]
    
    def max_tasks_for_aircraft(self, aircraft_id: int) -> int:
        """Maximum total tasks (A + C checks) for an aircraft."""
        ac = self.aircraft[aircraft_id]
        return ac.max_A_checks(self.horizon) + ac.max_C_checks(self.horizon)
    
    @classmethod
    def generate_random(
        cls,
        n_aircraft: int = 5,
        horizon: int = 365,
        capacity: int = 2,
        seed: Optional[int] = None,
    ) -> "AMCSInstance":
        """
        Generate a random AMCS instance for testing.
        
        Uses realistic parameter ranges based on typical maintenance schedules.
        """
        if seed is not None:
            random.seed(seed)
        
        aircraft = []
        for i in range(n_aircraft):
            # A-checks: typically every 3-4 weeks, 1-2 day duration
            A_interval = random.randint(21, 35)
            A_duration = random.randint(1, 2)
            A_tolerance = random.randint(3, 7)
            
            # C-checks: typically every 12-18 months, 2-4 week duration
            C_interval = random.randint(300, 500)
            C_duration = random.randint(14, 28)
            C_tolerance = random.randint(14, 30)
            
            # Random initial state
            initial_days_since_A = random.randint(0, A_interval - 1)
            initial_days_since_C = random.randint(0, C_interval - 1)
            
            aircraft.append(AircraftConfig(
                id=i,
                A_interval=A_interval,
                A_duration=A_duration,
                A_tolerance=A_tolerance,
                C_interval=C_interval,
                C_duration=C_duration,
                C_tolerance=C_tolerance,
                initial_days_since_A=initial_days_since_A,
                initial_days_since_C=initial_days_since_C,
            ))
        
        return cls(
            n_aircraft=n_aircraft,
            horizon=horizon,
            capacity=capacity,
            aircraft=aircraft,
        )


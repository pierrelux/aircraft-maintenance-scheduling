"""
Schedule representation for AMCS.

A schedule is a 2D array [horizon × n_aircraft] where each cell indicates
what maintenance activity (if any) is happening on that day for that aircraft.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import numpy as np


class CheckType(Enum):
    """Type of maintenance check."""
    NONE = 0
    A = 1
    C = 2


@dataclass
class ScheduledCheck:
    """A scheduled maintenance check."""
    aircraft_id: int
    check_type: str  # 'A' or 'C'
    start_day: int
    duration: int
    
    @property
    def end_day(self) -> int:
        """Last day of the check (inclusive)."""
        return self.start_day + self.duration - 1


class Schedule:
    """
    A maintenance schedule represented as a 2D grid.
    
    grid[day, aircraft_id] = CheckType indicating activity on that day.
    
    For multi-day checks, the check type is recorded for each day
    the aircraft is in maintenance.
    """
    
    def __init__(self, horizon: int, n_aircraft: int):
        self.horizon = horizon
        self.n_aircraft = n_aircraft
        
        # Grid: 0 = no check, 1 = A-check, 2 = C-check
        self._grid = np.zeros((horizon, n_aircraft), dtype=np.int8)
        
        # Track scheduled checks for easy iteration
        self._checks: List[ScheduledCheck] = []
    
    def __getitem__(self, key: Tuple[int, int]) -> Optional[str]:
        """Get the check type at (day, aircraft_id)."""
        day, aircraft_id = key
        val = self._grid[day, aircraft_id]
        if val == 0:
            return None
        elif val == 1:
            return 'A'
        else:
            return 'C'
    
    def assign(
        self,
        start_day: int,
        aircraft_id: int,
        check_type: str,
        duration: int,
    ) -> bool:
        """
        Assign a maintenance check to the schedule.
        
        Returns True if successful, False if there's a conflict.
        """
        if start_day < 0 or start_day + duration > self.horizon:
            return False
        
        # Check for conflicts (aircraft already in maintenance)
        for d in range(start_day, start_day + duration):
            if self._grid[d, aircraft_id] != 0:
                return False
        
        # Assign the check
        check_val = 1 if check_type == 'A' else 2
        for d in range(start_day, start_day + duration):
            self._grid[d, aircraft_id] = check_val
        
        self._checks.append(ScheduledCheck(
            aircraft_id=aircraft_id,
            check_type=check_type,
            start_day=start_day,
            duration=duration,
        ))
        
        return True
    
    def get_capacity_usage(self, day: int) -> int:
        """Get number of aircraft in maintenance on a given day."""
        return int(np.sum(self._grid[day, :] > 0))
    
    def is_capacity_available(self, day: int, capacity: int) -> bool:
        """Check if there's capacity available on a given day."""
        return self.get_capacity_usage(day) < capacity
    
    def get_checks_for_aircraft(self, aircraft_id: int) -> List[ScheduledCheck]:
        """Get all checks scheduled for a specific aircraft."""
        return [c for c in self._checks if c.aircraft_id == aircraft_id]
    
    def get_all_checks(self) -> List[ScheduledCheck]:
        """Get all scheduled checks."""
        return list(self._checks)
    
    def get_checks_by_day(self) -> Dict[int, List[ScheduledCheck]]:
        """Get checks grouped by start day."""
        by_day: Dict[int, List[ScheduledCheck]] = {}
        for check in self._checks:
            if check.start_day not in by_day:
                by_day[check.start_day] = []
            by_day[check.start_day].append(check)
        return by_day
    
    def copy(self) -> "Schedule":
        """Create a deep copy of this schedule."""
        new_schedule = Schedule(self.horizon, self.n_aircraft)
        new_schedule._grid = self._grid.copy()
        new_schedule._checks = list(self._checks)
        return new_schedule
    
    def to_array(self) -> np.ndarray:
        """Return the underlying grid array."""
        return self._grid.copy()
    
    @classmethod
    def from_array(cls, grid: np.ndarray) -> "Schedule":
        """Create a Schedule from a numpy array."""
        horizon, n_aircraft = grid.shape
        schedule = cls(horizon, n_aircraft)
        schedule._grid = grid.copy()
        
        # Reconstruct checks from grid
        # This is approximate - we detect contiguous blocks
        for ac in range(n_aircraft):
            in_check = False
            check_start = 0
            check_type = None
            
            for day in range(horizon):
                val = grid[day, ac]
                
                if val > 0 and not in_check:
                    # Starting a new check
                    in_check = True
                    check_start = day
                    check_type = 'A' if val == 1 else 'C'
                elif val == 0 and in_check:
                    # Ending a check
                    schedule._checks.append(ScheduledCheck(
                        aircraft_id=ac,
                        check_type=check_type,
                        start_day=check_start,
                        duration=day - check_start,
                    ))
                    in_check = False
            
            # Handle check that extends to end of horizon
            if in_check:
                schedule._checks.append(ScheduledCheck(
                    aircraft_id=ac,
                    check_type=check_type,
                    start_day=check_start,
                    duration=horizon - check_start,
                ))
        
        return schedule
    
    def summary(self) -> str:
        """Generate a text summary of the schedule."""
        a_checks = [c for c in self._checks if c.check_type == 'A']
        c_checks = [c for c in self._checks if c.check_type == 'C']
        
        lines = [
            f"Schedule Summary",
            f"  Horizon: {self.horizon} days",
            f"  Aircraft: {self.n_aircraft}",
            f"  A-checks: {len(a_checks)}",
            f"  C-checks: {len(c_checks)}",
            f"  Total grounded days: {int(np.sum(self._grid > 0))}",
        ]
        return "\n".join(lines)


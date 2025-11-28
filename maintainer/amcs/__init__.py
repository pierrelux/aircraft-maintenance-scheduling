"""
AMCS (Aircraft Maintenance Check Scheduling) module.

Provides simplified problem formulation and bridge to the existing Deng solver.
"""

from .instance import AMCSInstance, AircraftConfig
from .schedule import Schedule
from .bridge import solve_with_deng, evaluate_schedule

__all__ = [
    "AMCSInstance",
    "AircraftConfig", 
    "Schedule",
    "solve_with_deng",
    "evaluate_schedule",
]


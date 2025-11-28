"""
Action space management for CPD construction.

Handles legal action enumeration and action masking.
"""

from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np

from .state import CPDState, Action, ActionType


@dataclass
class ActionMask:
    """
    Mask indicating which actions are legal.
    
    Attributes:
        aircraft_mask: Boolean array [n_aircraft] - which aircraft can have tasks added
        type_mask: Boolean array [3] - [A, C, Stop] - which types are legal
        can_stop: Whether Stop is legal
    """
    aircraft_mask: np.ndarray  # [n_aircraft]
    type_mask: np.ndarray  # [3] for A, C, Stop
    can_stop: bool = True
    
    @property
    def any_add_task_legal(self) -> bool:
        """Check if any AddTask action is legal."""
        return bool(np.any(self.aircraft_mask) and np.any(self.type_mask[:2]))


def get_max_checks_for_aircraft(state: CPDState, aircraft_id: int) -> Dict[str, int]:
    """
    Get maximum number of A and C checks allowed for an aircraft.
    
    Based on horizon and check intervals.
    """
    ac = state.instance.get_aircraft(aircraft_id)
    horizon = state.instance.horizon
    
    return {
        'A': ac.max_A_checks(horizon),
        'C': ac.max_C_checks(horizon),
    }


def is_add_task_legal(
    state: CPDState,
    aircraft_id: int,
    check_type: str,
) -> bool:
    """
    Check if AddTask(aircraft_id, check_type) is legal.
    
    A task is illegal if:
    - The aircraft has reached max tasks of that type
    - The task window would be entirely outside the horizon
    """
    if state.is_terminal:
        return False
    
    max_checks = get_max_checks_for_aircraft(state, aircraft_id)
    
    if check_type == 'A':
        current = state.get_a_check_count(aircraft_id)
        if current >= max_checks['A']:
            return False
        
        # Check if next A-check would fit in horizon
        ac = state.instance.get_aircraft(aircraft_id)
        earliest = ac.A_earliest_day(current)
        if earliest >= state.instance.horizon:
            return False
            
    elif check_type == 'C':
        current = state.get_c_check_count(aircraft_id)
        if current >= max_checks['C']:
            return False
        
        # Check if next C-check would fit in horizon
        ac = state.instance.get_aircraft(aircraft_id)
        earliest = ac.C_earliest_day(current)
        if earliest >= state.instance.horizon:
            return False
    
    return True


def get_legal_actions(state: CPDState) -> List[Action]:
    """
    Get all legal actions from a state.
    
    Returns:
        List of legal Action objects
    """
    if state.is_terminal:
        return []
    
    actions = []
    
    # AddTask actions for each aircraft and check type
    for aircraft_id in range(state.instance.n_aircraft):
        for check_type in ['A', 'C']:
            if is_add_task_legal(state, aircraft_id, check_type):
                actions.append(Action.add_task(aircraft_id, check_type))
    
    # Stop is always legal (though may result in incomplete schedule)
    actions.append(Action.stop())
    
    return actions


def get_action_mask(state: CPDState) -> ActionMask:
    """
    Get action mask for a state.
    
    This is more efficient than enumerating actions when you need
    to mask logits in neural network output.
    
    Returns:
        ActionMask with boolean arrays for aircraft and type
    """
    n_aircraft = state.instance.n_aircraft
    
    aircraft_mask = np.zeros(n_aircraft, dtype=bool)
    type_mask = np.array([True, True, True], dtype=bool)  # A, C, Stop
    
    if state.is_terminal:
        aircraft_mask[:] = False
        type_mask[:] = False
        return ActionMask(aircraft_mask, type_mask, can_stop=False)
    
    # Check each aircraft
    any_a_legal = False
    any_c_legal = False
    
    for aircraft_id in range(n_aircraft):
        a_legal = is_add_task_legal(state, aircraft_id, 'A')
        c_legal = is_add_task_legal(state, aircraft_id, 'C')
        
        aircraft_mask[aircraft_id] = a_legal or c_legal
        
        if a_legal:
            any_a_legal = True
        if c_legal:
            any_c_legal = True
    
    type_mask[0] = any_a_legal  # A
    type_mask[1] = any_c_legal  # C
    type_mask[2] = True  # Stop always legal
    
    return ActionMask(aircraft_mask, type_mask, can_stop=True)


def action_to_index(
    action: Action,
    n_aircraft: int,
) -> int:
    """
    Convert an action to a flat index.
    
    Index layout:
    - 0 to n_aircraft-1: AddTask(i, 'A') for aircraft i
    - n_aircraft to 2*n_aircraft-1: AddTask(i, 'C') for aircraft i
    - 2*n_aircraft: Stop
    
    Returns:
        Flat action index
    """
    if action.type == ActionType.STOP:
        return 2 * n_aircraft
    
    aircraft_id = action.aircraft_id
    if action.check_type == 'A':
        return aircraft_id
    else:  # 'C'
        return n_aircraft + aircraft_id


def index_to_action(index: int, n_aircraft: int) -> Action:
    """
    Convert a flat index to an action.
    
    Inverse of action_to_index.
    """
    if index == 2 * n_aircraft:
        return Action.stop()
    elif index < n_aircraft:
        return Action.add_task(index, 'A')
    else:
        return Action.add_task(index - n_aircraft, 'C')


def get_flat_action_mask(state: CPDState) -> np.ndarray:
    """
    Get a flat action mask for all possible actions.
    
    Returns:
        Boolean array of shape [2 * n_aircraft + 1]
        where True indicates a legal action
    """
    n_aircraft = state.instance.n_aircraft
    mask = np.zeros(2 * n_aircraft + 1, dtype=bool)
    
    if state.is_terminal:
        return mask
    
    for aircraft_id in range(n_aircraft):
        if is_add_task_legal(state, aircraft_id, 'A'):
            mask[aircraft_id] = True
        if is_add_task_legal(state, aircraft_id, 'C'):
            mask[n_aircraft + aircraft_id] = True
    
    mask[2 * n_aircraft] = True  # Stop
    
    return mask


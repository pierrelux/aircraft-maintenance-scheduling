"""
CPD State representation for Maintainer.

A CPDState represents a partially constructed Critical Path Diagram (DAG)
of maintenance tasks. Construction is sequential: we add tasks one at a time,
and edges are automatically created to maintain per-aircraft precedence.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import copy

# Import from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maintainer.amcs.instance import AMCSInstance


class ActionType(Enum):
    """Type of action in CPD construction."""
    ADD_TASK = "add_task"
    STOP = "stop"


@dataclass
class TaskNode:
    """
    A node in the CPD representing a maintenance task.
    
    Attributes:
        aircraft_id: Which aircraft this task belongs to
        check_type: 'A' or 'C'
        sequence_index: k-th check of this type for this aircraft (0-indexed)
        duration: How long the check takes
        earliest_day: Earliest feasible start day
        latest_day: Latest feasible start day
    """
    aircraft_id: int
    check_type: str  # 'A' or 'C'
    sequence_index: int  # k-th check for this aircraft (0, 1, 2, ...)
    
    # Computed from instance and rules
    duration: int = 1
    earliest_day: int = 0
    latest_day: int = 365
    
    def __hash__(self):
        return hash((self.aircraft_id, self.check_type, self.sequence_index))
    
    def __eq__(self, other):
        if not isinstance(other, TaskNode):
            return False
        return (self.aircraft_id == other.aircraft_id and
                self.check_type == other.check_type and
                self.sequence_index == other.sequence_index)
    
    def __repr__(self):
        return f"Task({self.aircraft_id}, {self.check_type}, {self.sequence_index})"


@dataclass
class Action:
    """
    An action in CPD construction.
    
    Either AddTask(aircraft_id, check_type) or Stop.
    """
    type: ActionType
    aircraft_id: Optional[int] = None
    check_type: Optional[str] = None  # 'A' or 'C'
    
    def __hash__(self):
        return hash((self.type, self.aircraft_id, self.check_type))
    
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.type == other.type and
                self.aircraft_id == other.aircraft_id and
                self.check_type == other.check_type)
    
    def __repr__(self):
        if self.type == ActionType.STOP:
            return "Stop"
        return f"AddTask({self.aircraft_id}, {self.check_type})"
    
    @classmethod
    def add_task(cls, aircraft_id: int, check_type: str) -> "Action":
        """Create an AddTask action."""
        return cls(ActionType.ADD_TASK, aircraft_id, check_type)
    
    @classmethod
    def stop(cls) -> "Action":
        """Create a Stop action."""
        return cls(ActionType.STOP)


@dataclass
class CPDState:
    """
    State of a partially constructed CPD.
    
    The CPD is built sequentially by adding tasks. The construction order
    defines a topological ordering of the DAG by design (we only add edges
    from older nodes to newer nodes).
    
    Attributes:
        instance: The problem instance (fixed)
        nodes: Tasks added so far, in creation order
        edges: Precedence edges (from_idx, to_idx) in the nodes list
        aircraft_task_counts: How many tasks each aircraft has
        is_terminal: True if Stop was chosen
    """
    instance: AMCSInstance
    nodes: List[TaskNode] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    aircraft_task_counts: Dict[int, int] = field(default_factory=dict)
    is_terminal: bool = False
    
    def __post_init__(self):
        # Initialize task counts if not provided
        if not self.aircraft_task_counts:
            self.aircraft_task_counts = {i: 0 for i in range(self.instance.n_aircraft)}
    
    @classmethod
    def empty(cls, instance: AMCSInstance) -> "CPDState":
        """Create an empty CPD state for an instance."""
        return cls(
            instance=instance,
            nodes=[],
            edges=[],
            aircraft_task_counts={i: 0 for i in range(instance.n_aircraft)},
            is_terminal=False,
        )
    
    def copy(self) -> "CPDState":
        """Create a deep copy of this state."""
        return CPDState(
            instance=self.instance,  # Instance is immutable, share reference
            nodes=list(self.nodes),
            edges=list(self.edges),
            aircraft_task_counts=dict(self.aircraft_task_counts),
            is_terminal=self.is_terminal,
        )
    
    def get_task_count(self, aircraft_id: int) -> int:
        """Get number of tasks for an aircraft."""
        return self.aircraft_task_counts.get(aircraft_id, 0)
    
    def get_a_check_count(self, aircraft_id: int) -> int:
        """Get number of A-checks for an aircraft."""
        return sum(1 for n in self.nodes 
                   if n.aircraft_id == aircraft_id and n.check_type == 'A')
    
    def get_c_check_count(self, aircraft_id: int) -> int:
        """Get number of C-checks for an aircraft."""
        return sum(1 for n in self.nodes 
                   if n.aircraft_id == aircraft_id and n.check_type == 'C')
    
    def get_last_task_for_aircraft(self, aircraft_id: int) -> Optional[int]:
        """Get index of last task for an aircraft, or None if none exist."""
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].aircraft_id == aircraft_id:
                return i
        return None
    
    def get_predecessors(self, node_idx: int) -> List[int]:
        """Get indices of predecessor nodes."""
        return [from_idx for from_idx, to_idx in self.edges if to_idx == node_idx]
    
    def get_successors(self, node_idx: int) -> List[int]:
        """Get indices of successor nodes."""
        return [to_idx for from_idx, to_idx in self.edges if from_idx == node_idx]
    
    def num_nodes(self) -> int:
        """Number of task nodes in the CPD."""
        return len(self.nodes)
    
    def num_edges(self) -> int:
        """Number of edges in the CPD."""
        return len(self.edges)
    
    def __repr__(self):
        return f"CPDState(nodes={len(self.nodes)}, edges={len(self.edges)}, terminal={self.is_terminal})"


def apply_action(state: CPDState, action: Action) -> CPDState:
    """
    Apply an action to a CPD state, returning a new state.
    
    This is the core state transition function for CPD construction.
    
    For AddTask(aircraft_id, check_type):
    1. Create a new TaskNode
    2. Add precedence edge from previous task of same aircraft (if any)
    3. Append to nodes list
    
    For Stop:
    1. Mark state as terminal
    
    Args:
        state: Current CPD state
        action: Action to apply
    
    Returns:
        New CPD state (original is unchanged)
    """
    if state.is_terminal:
        raise ValueError("Cannot apply action to terminal state")
    
    new_state = state.copy()
    
    if action.type == ActionType.STOP:
        new_state.is_terminal = True
        return new_state
    
    # AddTask action
    aircraft_id = action.aircraft_id
    check_type = action.check_type
    
    # Determine sequence index for this check type
    if check_type == 'A':
        seq_idx = state.get_a_check_count(aircraft_id)
    else:
        seq_idx = state.get_c_check_count(aircraft_id)
    
    # Get timing info from instance
    ac = state.instance.get_aircraft(aircraft_id)
    if check_type == 'A':
        duration = ac.A_duration
        earliest = ac.A_earliest_day(seq_idx)
        latest = ac.A_latest_day(seq_idx, state.instance.horizon)
    else:
        duration = ac.C_duration
        earliest = ac.C_earliest_day(seq_idx)
        latest = ac.C_latest_day(seq_idx, state.instance.horizon)
    
    # Create new task node
    new_node = TaskNode(
        aircraft_id=aircraft_id,
        check_type=check_type,
        sequence_index=seq_idx,
        duration=duration,
        earliest_day=earliest,
        latest_day=latest,
    )
    
    # Add precedence edge from previous task of same aircraft
    prev_task_idx = state.get_last_task_for_aircraft(aircraft_id)
    new_node_idx = len(state.nodes)
    
    if prev_task_idx is not None:
        new_state.edges.append((prev_task_idx, new_node_idx))
    
    # Append node
    new_state.nodes.append(new_node)
    new_state.aircraft_task_counts[aircraft_id] += 1
    
    return new_state


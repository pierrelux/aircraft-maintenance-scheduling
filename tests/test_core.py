"""
Unit tests for Maintainer core components.

Tests:
- CPD state transitions
- Action masking
- Decoder correctness
- Round-trip consistency
"""

import pytest
import numpy as np

from maintainer.amcs.instance import AMCSInstance
from maintainer.core.state import CPDState, Action, ActionType, apply_action
from maintainer.core.actions import (
    get_legal_actions,
    get_action_mask,
    is_add_task_legal,
    action_to_index,
    index_to_action,
)
from maintainer.core.decoder import decode_cpd_to_schedule, validate_schedule


@pytest.fixture
def simple_instance():
    """Create a simple test instance."""
    return AMCSInstance.generate_random(
        n_aircraft=3,
        horizon=180,
        capacity=2,
        seed=42,
    )


class TestCPDState:
    """Tests for CPDState and state transitions."""
    
    def test_empty_state(self, simple_instance):
        """Test creating an empty state."""
        state = CPDState.empty(simple_instance)
        
        assert state.num_nodes() == 0
        assert state.num_edges() == 0
        assert not state.is_terminal
        assert len(state.nodes) == 0
    
    def test_add_task_action(self, simple_instance):
        """Test adding a task."""
        state = CPDState.empty(simple_instance)
        
        action = Action.add_task(0, 'A')
        new_state = apply_action(state, action)
        
        assert new_state.num_nodes() == 1
        assert new_state.nodes[0].aircraft_id == 0
        assert new_state.nodes[0].check_type == 'A'
        assert new_state.nodes[0].sequence_index == 0
    
    def test_add_multiple_tasks_same_aircraft(self, simple_instance):
        """Test that edges are created for same-aircraft tasks."""
        state = CPDState.empty(simple_instance)
        
        # Add first task
        state = apply_action(state, Action.add_task(0, 'A'))
        assert state.num_edges() == 0
        
        # Add second task to same aircraft
        state = apply_action(state, Action.add_task(0, 'C'))
        assert state.num_edges() == 1
        assert state.edges[0] == (0, 1)
        
        # Add third task to same aircraft
        state = apply_action(state, Action.add_task(0, 'A'))
        assert state.num_edges() == 2
    
    def test_add_tasks_different_aircraft(self, simple_instance):
        """Test that no edges are created between different aircraft."""
        state = CPDState.empty(simple_instance)
        
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.add_task(1, 'A'))
        state = apply_action(state, Action.add_task(2, 'A'))
        
        assert state.num_nodes() == 3
        assert state.num_edges() == 0
    
    def test_stop_action(self, simple_instance):
        """Test the Stop action."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        
        stop_state = apply_action(state, Action.stop())
        
        assert stop_state.is_terminal
        assert stop_state.num_nodes() == 1
    
    def test_cannot_apply_action_to_terminal(self, simple_instance):
        """Test that actions cannot be applied to terminal states."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.stop())
        
        with pytest.raises(ValueError):
            apply_action(state, Action.add_task(0, 'A'))
    
    def test_state_copy_independence(self, simple_instance):
        """Test that copied states are independent."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        
        copy = state.copy()
        copy = apply_action(copy, Action.add_task(0, 'C'))
        
        assert state.num_nodes() == 1
        assert copy.num_nodes() == 2


class TestActionMasking:
    """Tests for action masking."""
    
    def test_legal_actions_empty_state(self, simple_instance):
        """Test legal actions from empty state."""
        state = CPDState.empty(simple_instance)
        actions = get_legal_actions(state)
        
        # Should have AddTask for each (aircraft, type) + Stop
        expected = simple_instance.n_aircraft * 2 + 1
        assert len(actions) == expected
    
    def test_stop_always_legal(self, simple_instance):
        """Test that Stop is always legal."""
        state = CPDState.empty(simple_instance)
        actions = get_legal_actions(state)
        
        stop_actions = [a for a in actions if a.type == ActionType.STOP]
        assert len(stop_actions) == 1
    
    def test_action_mask_shape(self, simple_instance):
        """Test action mask has correct shape."""
        state = CPDState.empty(simple_instance)
        mask = get_action_mask(state)
        
        assert len(mask.aircraft_mask) == simple_instance.n_aircraft
        assert len(mask.type_mask) == 3
    
    def test_terminal_state_no_actions(self, simple_instance):
        """Test that terminal states have no legal actions."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.stop())
        
        actions = get_legal_actions(state)
        assert len(actions) == 0


class TestActionIndexing:
    """Tests for action index conversion."""
    
    def test_action_to_index_roundtrip(self, simple_instance):
        """Test action <-> index conversion."""
        n_aircraft = simple_instance.n_aircraft
        
        # Test AddTask(i, A)
        for i in range(n_aircraft):
            action = Action.add_task(i, 'A')
            idx = action_to_index(action, n_aircraft)
            recovered = index_to_action(idx, n_aircraft)
            assert action == recovered
        
        # Test AddTask(i, C)
        for i in range(n_aircraft):
            action = Action.add_task(i, 'C')
            idx = action_to_index(action, n_aircraft)
            recovered = index_to_action(idx, n_aircraft)
            assert action == recovered
        
        # Test Stop
        action = Action.stop()
        idx = action_to_index(action, n_aircraft)
        recovered = index_to_action(idx, n_aircraft)
        assert action == recovered


class TestDecoder:
    """Tests for CPD decoder."""
    
    def test_decode_empty_cpd(self, simple_instance):
        """Test decoding an empty (terminal) CPD."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.stop())
        
        schedule = decode_cpd_to_schedule(state)
        
        assert schedule.horizon == simple_instance.horizon
        assert schedule.n_aircraft == simple_instance.n_aircraft
        assert len(schedule.get_all_checks()) == 0
    
    def test_decode_single_task(self, simple_instance):
        """Test decoding a CPD with single task."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.stop())
        
        schedule = decode_cpd_to_schedule(state)
        checks = schedule.get_all_checks()
        
        assert len(checks) == 1
        assert checks[0].aircraft_id == 0
        assert checks[0].check_type == 'A'
    
    def test_decode_respects_precedence(self, simple_instance):
        """Test that decoder respects task precedence."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.add_task(0, 'C'))
        state = apply_action(state, Action.stop())
        
        schedule = decode_cpd_to_schedule(state)
        checks = schedule.get_checks_for_aircraft(0)
        checks.sort(key=lambda c: c.start_day)
        
        assert len(checks) == 2
        # First check should end before second starts
        assert checks[0].start_day + checks[0].duration <= checks[1].start_day
    
    def test_decode_respects_capacity(self, simple_instance):
        """Test that decoder respects capacity constraints."""
        state = CPDState.empty(simple_instance)
        
        # Add one task per aircraft
        for i in range(simple_instance.n_aircraft):
            state = apply_action(state, Action.add_task(i, 'C'))
        state = apply_action(state, Action.stop())
        
        schedule = decode_cpd_to_schedule(state)
        
        # Check capacity on each day
        for day in range(schedule.horizon):
            usage = schedule.get_capacity_usage(day)
            assert usage <= simple_instance.capacity
    
    def test_validate_schedule(self, simple_instance):
        """Test schedule validation."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.stop())
        
        schedule = decode_cpd_to_schedule(state)
        violations = validate_schedule(schedule, simple_instance)
        
        assert len(violations) == 0


class TestRoundTrip:
    """Tests for round-trip consistency."""
    
    def test_cpd_trajectory_roundtrip(self, simple_instance):
        """Test that CPD -> trajectory -> CPD produces same structure."""
        from maintainer.amcs.extraction import cpd_to_trajectory
        
        # Build a CPD
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.add_task(1, 'C'))
        state = apply_action(state, Action.add_task(0, 'C'))
        state = apply_action(state, Action.stop())
        
        # Convert to trajectory
        trajectory = cpd_to_trajectory(state, 0.0)
        
        # Replay trajectory
        replayed = CPDState.empty(simple_instance)
        for step in trajectory:
            replayed = apply_action(replayed, step.action)
        
        # Compare
        assert replayed.num_nodes() == state.num_nodes()
        assert replayed.num_edges() == state.num_edges()
        
        for i, (orig, repl) in enumerate(zip(state.nodes, replayed.nodes)):
            assert orig.aircraft_id == repl.aircraft_id
            assert orig.check_type == repl.check_type
            assert orig.sequence_index == repl.sequence_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


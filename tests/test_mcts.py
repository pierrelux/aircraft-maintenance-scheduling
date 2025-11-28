"""
Unit tests for MCTS.

Tests:
- MCTS search produces valid actions
- Policy-only mode (n_simulations=0)
- Terminal handling
"""

import pytest
import torch

from maintainer.amcs.instance import AMCSInstance
from maintainer.core.state import CPDState, Action, ActionType, apply_action
from maintainer.core.model import MaintainerNet, ModelConfig
from maintainer.core.mcts import MCTS, MCTSConfig, MCTSNode
from maintainer.core.actions import get_legal_actions


@pytest.fixture
def simple_instance():
    """Create a simple test instance."""
    return AMCSInstance.generate_random(
        n_aircraft=3,
        horizon=180,
        capacity=2,
        seed=42,
    )


@pytest.fixture
def model():
    """Create a test model."""
    config = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_aircraft=10,
        n_aircraft=3,
    )
    return MaintainerNet(config)


class TestMCTSNode:
    """Tests for MCTSNode."""
    
    def test_node_creation(self, simple_instance):
        """Test node creation."""
        state = CPDState.empty(simple_instance)
        node = MCTSNode(state)
        
        assert node.state == state
        assert not node.is_terminal
        assert not node.is_expanded()
        assert node.total_visits() == 0


class TestMCTS:
    """Tests for MCTS search."""
    
    def test_search_returns_valid_action(self, simple_instance, model):
        """Test that MCTS search returns a valid action."""
        config = MCTSConfig(n_simulations=10)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        action = mcts.search(state)
        
        assert isinstance(action, Action)
        legal = get_legal_actions(state)
        assert action in legal
    
    def test_policy_only_mode(self, simple_instance, model):
        """Test that n_simulations=0 uses greedy policy only."""
        config = MCTSConfig(n_simulations=0)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        action = mcts.search(state)
        
        # Should match greedy action
        greedy_action = model.greedy_action(state)
        assert action == greedy_action
    
    def test_multiple_simulations(self, simple_instance, model):
        """Test that more simulations don't crash."""
        config = MCTSConfig(n_simulations=50)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        action = mcts.search(state)
        
        assert isinstance(action, Action)
    
    def test_search_from_partial_state(self, simple_instance, model):
        """Test search from a partially constructed CPD."""
        config = MCTSConfig(n_simulations=10)
        mcts = MCTS(model, config)
        
        # Build partial CPD
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.add_task(1, 'C'))
        
        action = mcts.search(state)
        assert isinstance(action, Action)
    
    def test_cannot_search_from_terminal(self, simple_instance, model):
        """Test that search fails on terminal states."""
        config = MCTSConfig(n_simulations=10)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.stop())
        
        with pytest.raises(ValueError):
            mcts.search(state)
    
    def test_get_policy(self, simple_instance, model):
        """Test getting visit-count-based policy."""
        config = MCTSConfig(n_simulations=20)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        policy = mcts.get_policy(state)
        
        # Policy should be a distribution over actions
        assert len(policy) > 0
        total = sum(policy.values())
        assert abs(total - 1.0) < 1e-5
    
    def test_get_root_value(self, simple_instance, model):
        """Test getting value estimate at root."""
        config = MCTSConfig(n_simulations=20)
        mcts = MCTS(model, config)
        
        state = CPDState.empty(simple_instance)
        mcts.search(state)
        
        value = mcts.get_root_value()
        assert isinstance(value, float)


class TestMCTSPlanning:
    """Tests for full planning with MCTS."""
    
    def test_plan_produces_valid_schedule(self, simple_instance, model):
        """Test that MCTS planning produces a valid schedule."""
        from maintainer.core.mcts import plan_with_mcts
        from maintainer.core.decoder import validate_schedule
        
        config = MCTSConfig(n_simulations=5)
        
        schedule, trajectory, cost = plan_with_mcts(
            simple_instance, model, config
        )
        
        violations = validate_schedule(schedule, simple_instance)
        assert len(violations) == 0
    
    def test_plan_terminates(self, simple_instance, model):
        """Test that planning terminates."""
        from maintainer.core.mcts import plan_with_mcts
        
        config = MCTSConfig(n_simulations=5)
        
        # Should not hang
        schedule, trajectory, cost = plan_with_mcts(
            simple_instance, model, config
        )
        
        # Trajectory should end with Stop
        assert trajectory[-1][1].type == ActionType.STOP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


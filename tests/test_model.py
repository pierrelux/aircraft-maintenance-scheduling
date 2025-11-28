"""
Unit tests for MaintainerNet model.

Tests:
- Forward pass shapes
- Action sampling
- Batch processing
"""

import pytest
import torch

from maintainer.amcs.instance import AMCSInstance
from maintainer.core.state import CPDState, Action, apply_action
from maintainer.core.model import MaintainerNet, ModelConfig
from maintainer.core.encoder import TokenEncoder, EncoderConfig
from maintainer.core.actions import get_action_mask


@pytest.fixture
def simple_instance():
    """Create a simple test instance."""
    return AMCSInstance.generate_random(
        n_aircraft=5,
        horizon=180,
        capacity=2,
        seed=42,
    )


@pytest.fixture
def model():
    """Create a test model."""
    config = ModelConfig(
        d_model=64,  # Small for testing
        n_layers=2,
        n_heads=4,
        max_aircraft=10,
        n_aircraft=5,
    )
    return MaintainerNet(config)


class TestEncoder:
    """Tests for TokenEncoder."""
    
    def test_encode_empty_state(self, simple_instance):
        """Test encoding an empty state."""
        config = EncoderConfig(d_model=64, max_aircraft=10)
        encoder = TokenEncoder(config)
        
        state = CPDState.empty(simple_instance)
        embeddings, mask = encoder(state)
        
        # Should have 1 global + n_aircraft tokens
        expected_len = 1 + simple_instance.n_aircraft
        assert embeddings.shape == (expected_len, 64)
        assert mask.shape == (expected_len,)
        assert mask.all()
    
    def test_encode_with_tasks(self, simple_instance):
        """Test encoding a state with tasks."""
        config = EncoderConfig(d_model=64, max_aircraft=10)
        encoder = TokenEncoder(config)
        
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        state = apply_action(state, Action.add_task(1, 'C'))
        
        embeddings, mask = encoder(state)
        
        # Should have 1 global + n_aircraft + 2 task tokens
        expected_len = 1 + simple_instance.n_aircraft + 2
        assert embeddings.shape == (expected_len, 64)
    
    def test_encode_batch(self, simple_instance):
        """Test batch encoding."""
        config = EncoderConfig(d_model=64, max_aircraft=10)
        encoder = TokenEncoder(config)
        
        states = [
            CPDState.empty(simple_instance),
            apply_action(CPDState.empty(simple_instance), Action.add_task(0, 'A')),
        ]
        
        embeddings, masks = encoder.encode_batch(states)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[2] == 64
        assert masks.shape[0] == 2


class TestMaintainerNet:
    """Tests for MaintainerNet."""
    
    def test_forward_empty_state(self, simple_instance, model):
        """Test forward pass on empty state."""
        state = CPDState.empty(simple_instance)
        mask = get_action_mask(state)
        
        policy, value = model(state, mask)
        
        assert policy.aircraft.shape == (simple_instance.n_aircraft,)
        assert policy.type.shape == (3,)
        assert value.shape == ()
    
    def test_forward_with_tasks(self, simple_instance, model):
        """Test forward pass on state with tasks."""
        state = CPDState.empty(simple_instance)
        state = apply_action(state, Action.add_task(0, 'A'))
        mask = get_action_mask(state)
        
        policy, value = model(state, mask)
        
        assert policy.aircraft.shape == (simple_instance.n_aircraft,)
        assert policy.type.shape == (3,)
    
    def test_value_in_range(self, simple_instance, model):
        """Test that value output is in [-1, 1]."""
        state = CPDState.empty(simple_instance)
        mask = get_action_mask(state)
        
        _, value = model(state, mask)
        
        assert -1 <= value.item() <= 1
    
    def test_sample_action(self, simple_instance, model):
        """Test action sampling."""
        state = CPDState.empty(simple_instance)
        
        action = model.sample_action(state, temperature=1.0)
        
        assert isinstance(action, Action)
    
    def test_greedy_action(self, simple_instance, model):
        """Test greedy action selection."""
        state = CPDState.empty(simple_instance)
        
        action = model.greedy_action(state)
        
        assert isinstance(action, Action)
    
    def test_action_probs_sum_to_one(self, simple_instance, model):
        """Test that action probabilities are valid."""
        state = CPDState.empty(simple_instance)
        
        aircraft_probs, type_probs = model.get_action_probs(state)
        
        assert abs(aircraft_probs.sum().item() - 1.0) < 1e-5
        assert abs(type_probs.sum().item() - 1.0) < 1e-5
    
    def test_masking_applied(self, simple_instance, model):
        """Test that action masking is applied."""
        state = CPDState.empty(simple_instance)
        mask = get_action_mask(state)
        
        # Manually mask out aircraft 0
        mask.aircraft_mask[0] = False
        
        policy, _ = model(state, mask)
        
        # Aircraft 0 should have very low probability after softmax
        probs = torch.softmax(policy.aircraft, dim=-1)
        assert probs[0].item() < 1e-6


class TestGradients:
    """Tests for gradient flow."""
    
    def test_gradients_flow(self, simple_instance, model):
        """Test that gradients flow through the model."""
        state = CPDState.empty(simple_instance)
        mask = get_action_mask(state)
        
        policy, value = model(state, mask)
        
        # Compute dummy loss
        loss = policy.aircraft.sum() + policy.type.sum() + value
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


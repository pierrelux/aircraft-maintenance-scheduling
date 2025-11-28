"""
MaintainerNet: Policy-Value Transformer for CPD Construction.

A decoder-only transformer that outputs:
- Policy: distribution over actions (AddTask(i, c) and Stop)
- Value: scalar estimate of final reward
"""

from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .encoder import TokenEncoder, EncoderConfig
from .state import CPDState, Action, ActionType
from .actions import get_action_mask, ActionMask


@dataclass
class ModelConfig:
    """Configuration for MaintainerNet."""
    # Encoder config
    d_model: int = 256
    max_aircraft: int = 50
    max_tasks: int = 500
    max_seq_len: int = 600
    
    # Transformer config
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # For the actual number of aircraft (set at runtime)
    n_aircraft: int = 10


class PolicyLogits(NamedTuple):
    """Policy output from the model."""
    aircraft: torch.Tensor  # [batch, n_aircraft] or [n_aircraft]
    type: torch.Tensor      # [batch, 3] or [3] for A, C, Stop


class TransformerBlock(nn.Module):
    """A single transformer block with causal self-attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        
        self.attn = nn.MultiheadAttention(
            d, config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d)
        
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(config.dropout),
        )
        self.norm2 = nn.LayerNorm(d)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] or [seq_len, d_model]
            attention_mask: [seq_len, seq_len] causal mask
        
        Returns:
            Output tensor same shape as x
        """
        # Handle unbatched input
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)
        
        seq_len = x.shape[1]
        device = x.device
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1,
            )
        
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if unbatched:
            x = x.squeeze(0)
        
        return x


class MaintainerNet(nn.Module):
    """
    Policy-Value network for CPD construction.
    
    Architecture:
    - TokenEncoder: CPDState -> token embeddings
    - Transformer: sequence of decoder blocks
    - Policy head: predicts aircraft and action type
    - Value head: predicts scalar value
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        
        # Token encoder
        encoder_config = EncoderConfig(
            d_model=d,
            max_aircraft=config.max_aircraft,
            max_tasks=config.max_tasks,
            max_seq_len=config.max_seq_len,
        )
        self.encoder = TokenEncoder(encoder_config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Policy heads
        self.aircraft_head = nn.Linear(d, config.max_aircraft)
        self.type_head = nn.Linear(d, 3)  # A, C, Stop
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
            nn.Tanh(),  # Normalized to [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        state: CPDState,
        action_mask: Optional[ActionMask] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[PolicyLogits, torch.Tensor]:
        """
        Forward pass for a single state.
        
        Args:
            state: CPD state to evaluate
            action_mask: Optional mask for illegal actions
            device: Target device
        
        Returns:
            (policy_logits, value)
            policy_logits: PolicyLogits with aircraft and type logits
            value: Scalar value estimate
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Encode state to token sequence
        embeddings, attn_mask = self.encoder(state, device)  # [seq_len, d_model]
        
        # Run through transformer blocks
        x = embeddings
        for block in self.blocks:
            x = block(x)
        
        # Pool: use last token (like GPT)
        h = x[-1]  # [d_model]
        
        # Policy heads
        n_aircraft = state.instance.n_aircraft
        aircraft_logits = self.aircraft_head(h)[:n_aircraft]  # [n_aircraft]
        type_logits = self.type_head(h)  # [3]
        
        # Apply action mask if provided
        if action_mask is not None:
            aircraft_logits = aircraft_logits.masked_fill(
                ~torch.tensor(action_mask.aircraft_mask, device=device),
                float('-inf'),
            )
            type_logits = type_logits.masked_fill(
                ~torch.tensor(action_mask.type_mask, device=device),
                float('-inf'),
            )
        
        policy = PolicyLogits(aircraft=aircraft_logits, type=type_logits)
        
        # Value head
        value = self.value_head(h).squeeze(-1)  # Scalar
        
        return policy, value
    
    def forward_batch(
        self,
        states: list,
        action_masks: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[PolicyLogits, torch.Tensor]:
        """
        Forward pass for a batch of states.
        
        Args:
            states: List of CPDState objects
            action_masks: Optional list of ActionMask objects
            device: Target device
        
        Returns:
            (policy_logits, values)
            policy_logits: PolicyLogits with batched tensors
            values: [batch] tensor of values
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Encode all states
        embeddings, attn_masks = self.encoder.encode_batch(states, device)
        # embeddings: [batch, max_seq_len, d_model]
        
        batch_size = embeddings.shape[0]
        
        # Run through transformer blocks
        x = embeddings
        for block in self.blocks:
            x = block(x)
        
        # Pool: use last valid token for each sequence
        # For simplicity, we use mean pooling over valid tokens
        seq_lens = attn_masks.sum(dim=1)  # [batch]
        h = torch.zeros(batch_size, self.config.d_model, device=device)
        for i in range(batch_size):
            valid_len = int(seq_lens[i].item())
            h[i] = x[i, :valid_len].mean(dim=0)
        
        # Policy heads
        max_aircraft = max(s.instance.n_aircraft for s in states)
        aircraft_logits = self.aircraft_head(h)[:, :max_aircraft]  # [batch, max_aircraft]
        type_logits = self.type_head(h)  # [batch, 3]
        
        # Apply action masks if provided
        if action_masks is not None:
            for i, mask in enumerate(action_masks):
                n_ac = states[i].instance.n_aircraft
                aircraft_logits[i, :n_ac] = aircraft_logits[i, :n_ac].masked_fill(
                    ~torch.tensor(mask.aircraft_mask, device=device),
                    float('-inf'),
                )
                type_logits[i] = type_logits[i].masked_fill(
                    ~torch.tensor(mask.type_mask, device=device),
                    float('-inf'),
                )
        
        policy = PolicyLogits(aircraft=aircraft_logits, type=type_logits)
        
        # Value head
        values = self.value_head(h).squeeze(-1)  # [batch]
        
        return policy, values
    
    def sample_action(
        self,
        state: CPDState,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Action:
        """
        Sample an action from the policy.
        
        Args:
            state: Current state
            temperature: Sampling temperature (1.0 = normal, lower = more greedy)
            device: Target device
        
        Returns:
            Sampled action
        """
        if state.is_terminal:
            raise ValueError("Cannot sample action from terminal state")
        
        mask = get_action_mask(state)
        policy, _ = self.forward(state, mask, device)
        
        # Sample action type
        type_probs = F.softmax(policy.type / temperature, dim=-1)
        type_idx = torch.multinomial(type_probs, 1).item()
        
        if type_idx == 2:  # Stop
            return Action.stop()
        
        # Sample aircraft
        aircraft_probs = F.softmax(policy.aircraft / temperature, dim=-1)
        aircraft_id = torch.multinomial(aircraft_probs, 1).item()
        
        check_type = 'A' if type_idx == 0 else 'C'
        return Action.add_task(aircraft_id, check_type)
    
    def get_action_probs(
        self,
        state: CPDState,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities (for MCTS).
        
        Returns:
            (aircraft_probs, type_probs)
            aircraft_probs: [n_aircraft] probability of each aircraft
            type_probs: [3] probability of A, C, Stop
        """
        if state.is_terminal:
            n_aircraft = state.instance.n_aircraft
            return (
                torch.zeros(n_aircraft),
                torch.zeros(3),
            )
        
        mask = get_action_mask(state)
        policy, _ = self.forward(state, mask, device)
        
        aircraft_probs = F.softmax(policy.aircraft, dim=-1)
        type_probs = F.softmax(policy.type, dim=-1)
        
        return aircraft_probs, type_probs
    
    def greedy_action(
        self,
        state: CPDState,
        device: Optional[torch.device] = None,
    ) -> Action:
        """
        Get the greedy (argmax) action from policy.
        
        Useful for policy-only baseline (no MCTS).
        """
        if state.is_terminal:
            raise ValueError("Cannot get action from terminal state")
        
        mask = get_action_mask(state)
        policy, _ = self.forward(state, mask, device)
        
        # Greedy action type
        type_idx = policy.type.argmax().item()
        
        if type_idx == 2:  # Stop
            return Action.stop()
        
        # Greedy aircraft
        aircraft_id = policy.aircraft.argmax().item()
        check_type = 'A' if type_idx == 0 else 'C'
        
        return Action.add_task(aircraft_id, check_type)


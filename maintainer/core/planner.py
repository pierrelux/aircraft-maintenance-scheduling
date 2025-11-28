"""
High-level planner interface for Maintainer.

Provides clean API for planning with Maintainer+MCTS.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch

from .state import CPDState, Action, apply_action
from .decoder import decode_cpd_to_schedule
from .model import MaintainerNet, ModelConfig
from .mcts import MCTS, MCTSConfig, plan_with_mcts
from maintainer.amcs.instance import AMCSInstance
from maintainer.amcs.schedule import Schedule
from maintainer.amcs.bridge import evaluate_schedule


@dataclass
class PlanResult:
    """Result of planning with Maintainer."""
    schedule: Schedule
    trajectory: List[Tuple[CPDState, Action]]
    cost: float
    cpd: CPDState
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Planning Result",
            "=" * 40,
            f"Cost: {self.cost:.2f}",
            f"Steps: {len(self.trajectory)}",
            f"CPD nodes: {len(self.cpd.nodes)}",
            self.schedule.summary(),
        ]
        return "\n".join(lines)


def plan_with_maintainer(
    instance: AMCSInstance,
    network: MaintainerNet,
    mcts_config: Optional[MCTSConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> PlanResult:
    """
    Plan a maintenance schedule using Maintainer+MCTS.
    
    Args:
        instance: Problem instance
        network: Trained MaintainerNet
        mcts_config: MCTS configuration (None = use defaults)
        device: Torch device
        verbose: Print progress
    
    Returns:
        PlanResult with schedule, trajectory, and cost
    """
    if mcts_config is None:
        mcts_config = MCTSConfig()
    
    if device is None:
        device = next(network.parameters()).device
    
    # Evaluation function for terminal states
    def evaluate_fn(state: CPDState) -> float:
        schedule = decode_cpd_to_schedule(state)
        cost = evaluate_schedule(state.instance, schedule)
        return -cost  # Reward = -cost
    
    # Run MCTS planning
    state = CPDState.empty(instance)
    mcts = MCTS(network, mcts_config, evaluate_fn, device)
    
    trajectory = []
    step = 0
    
    while not state.is_terminal:
        action = mcts.search(state)
        
        if verbose:
            print(f"Step {step}: {action}")
        
        trajectory.append((state.copy(), action))
        state = apply_action(state, action)
        step += 1
    
    # Decode final CPD
    schedule = decode_cpd_to_schedule(state)
    cost = evaluate_schedule(instance, schedule)
    
    return PlanResult(
        schedule=schedule,
        trajectory=trajectory,
        cost=cost,
        cpd=state,
    )


def plan_with_greedy_policy(
    instance: AMCSInstance,
    network: MaintainerNet,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> PlanResult:
    """
    Plan using greedy policy only (no MCTS).
    
    Useful as a baseline to verify MCTS is contributing.
    """
    mcts_config = MCTSConfig(n_simulations=0)
    return plan_with_maintainer(
        instance, network, mcts_config, device, verbose
    )


def create_default_network(
    n_aircraft: int = 10,
    device: Optional[torch.device] = None,
) -> MaintainerNet:
    """
    Create a MaintainerNet with default configuration.
    
    Useful for testing and initialization.
    """
    config = ModelConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_aircraft=50,
        max_tasks=500,
        n_aircraft=n_aircraft,
    )
    
    network = MaintainerNet(config)
    
    if device is not None:
        network = network.to(device)
    
    return network


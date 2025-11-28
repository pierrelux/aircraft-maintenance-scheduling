"""
Evaluation utilities for Maintainer.

Includes:
- Policy accuracy metrics
- Schedule cost comparison vs Deng
- Decoder loss diagnostic
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

from maintainer.amcs.instance import AMCSInstance
from maintainer.amcs.schedule import Schedule
from maintainer.amcs.bridge import solve_with_deng, evaluate_schedule, DENG_AVAILABLE
from maintainer.amcs.extraction import extract_cpd_from_schedule, Trajectory
from maintainer.core.state import CPDState, Action, apply_action
from maintainer.core.decoder import decode_cpd_to_schedule
from maintainer.core.model import MaintainerNet
from maintainer.core.mcts import MCTSConfig
from maintainer.core.planner import plan_with_maintainer, plan_with_greedy_policy


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    # Cost comparison
    deng_costs: List[float]
    maintainer_costs: List[float]
    
    # Derived metrics
    @property
    def avg_deng_cost(self) -> float:
        return np.mean(self.deng_costs) if self.deng_costs else 0.0
    
    @property
    def avg_maintainer_cost(self) -> float:
        return np.mean(self.maintainer_costs) if self.maintainer_costs else 0.0
    
    @property
    def cost_ratio(self) -> float:
        """Maintainer / Deng cost ratio. <1 means Maintainer is better."""
        if self.avg_deng_cost == 0:
            return 1.0
        return self.avg_maintainer_cost / self.avg_deng_cost
    
    @property
    def median_cost_ratio(self) -> float:
        """Median per-instance cost ratio."""
        if not self.deng_costs:
            return 1.0
        ratios = [m / d if d > 0 else 1.0 
                  for m, d in zip(self.maintainer_costs, self.deng_costs)]
        return np.median(ratios)
    
    @property
    def n_wins(self) -> int:
        """Number of instances where Maintainer beats Deng."""
        return sum(1 for m, d in zip(self.maintainer_costs, self.deng_costs) if m < d)
    
    @property
    def n_ties(self) -> int:
        """Number of instances where costs are equal."""
        return sum(1 for m, d in zip(self.maintainer_costs, self.deng_costs) 
                   if abs(m - d) < 1e-6)
    
    @property
    def n_losses(self) -> int:
        """Number of instances where Deng beats Maintainer."""
        return sum(1 for m, d in zip(self.maintainer_costs, self.deng_costs) if m > d)
    
    def summary(self) -> str:
        n = len(self.deng_costs)
        lines = [
            "Evaluation Summary",
            "=" * 40,
            f"Instances evaluated: {n}",
            f"",
            f"Average cost:",
            f"  Deng: {self.avg_deng_cost:.2f}",
            f"  Maintainer: {self.avg_maintainer_cost:.2f}",
            f"  Ratio: {self.cost_ratio:.3f}",
            f"",
            f"Median cost ratio: {self.median_cost_ratio:.3f}",
            f"",
            f"Win/Tie/Loss: {self.n_wins}/{self.n_ties}/{self.n_losses}",
            f"Win rate: {self.n_wins / n * 100:.1f}%" if n > 0 else "",
        ]
        return "\n".join(lines)


def evaluate_on_instances(
    instances: List[AMCSInstance],
    network: MaintainerNet,
    mcts_config: Optional[MCTSConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate Maintainer+MCTS against Deng on a set of instances.
    
    Args:
        instances: List of instances to evaluate
        network: Trained MaintainerNet
        mcts_config: MCTS configuration
        device: Torch device
        verbose: Show progress
    
    Returns:
        EvaluationResult with cost comparisons
    """
    from maintainer.training.data_gen import generate_synthetic_schedule
    
    deng_costs = []
    maintainer_costs = []
    
    iterator = instances
    if verbose:
        iterator = tqdm(instances, desc="Evaluating")
    
    for instance in iterator:
        # Deng baseline
        try:
            if DENG_AVAILABLE and instance._source_data is not None:
                deng_schedule = solve_with_deng(instance)
            else:
                deng_schedule = generate_synthetic_schedule(instance)
        except Exception:
            deng_schedule = generate_synthetic_schedule(instance)
        
        deng_cost = evaluate_schedule(instance, deng_schedule)
        deng_costs.append(deng_cost)
        
        # Maintainer + MCTS
        try:
            result = plan_with_maintainer(
                instance, network, mcts_config, device, verbose=False
            )
            maintainer_costs.append(result.cost)
        except Exception as e:
            # On error, use a high cost
            if verbose:
                print(f"Warning: Maintainer failed: {e}")
            maintainer_costs.append(deng_cost * 2)
    
    return EvaluationResult(
        deng_costs=deng_costs,
        maintainer_costs=maintainer_costs,
    )


def evaluate_policy_accuracy(
    network: MaintainerNet,
    trajectories: List[Trajectory],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate policy accuracy on expert trajectories.
    
    Metrics:
    - top1_accuracy: Fraction where greedy action matches expert
    - top3_accuracy: Fraction where expert action is in top-3
    - avg_log_prob: Average log probability of expert action
    
    Returns:
        Dictionary of metrics
    """
    from maintainer.core.actions import get_action_mask, action_to_index
    
    network.eval()
    
    correct_top1 = 0
    correct_top3 = 0
    total_log_prob = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for trajectory in trajectories:
            for step in trajectory:
                state = step.state
                expert_action = step.action
                
                if state.is_terminal:
                    continue
                
                mask = get_action_mask(state)
                policy, _ = network(state, mask, device)
                
                n_aircraft = state.instance.n_aircraft
                expert_idx = action_to_index(expert_action, n_aircraft)
                
                # Compute action probabilities
                import torch.nn.functional as F
                type_probs = F.softmax(policy.type, dim=-1)
                aircraft_probs = F.softmax(policy.aircraft, dim=-1)
                
                # Compute log prob of expert action
                if expert_action.type.value == "stop":
                    log_prob = torch.log(type_probs[2] + 1e-8)
                elif expert_action.check_type == 'A':
                    log_prob = torch.log(type_probs[0] * aircraft_probs[expert_action.aircraft_id] + 1e-8)
                else:
                    log_prob = torch.log(type_probs[1] * aircraft_probs[expert_action.aircraft_id] + 1e-8)
                
                total_log_prob += log_prob.item()
                
                # Check top-1 accuracy
                greedy_action = network.greedy_action(state, device)
                if greedy_action == expert_action:
                    correct_top1 += 1
                    correct_top3 += 1
                else:
                    # Check top-3 (simplified: check if in top types/aircraft)
                    top_types = type_probs.topk(min(3, 3)).indices.tolist()
                    top_aircraft = aircraft_probs.topk(min(3, n_aircraft)).indices.tolist()
                    
                    if expert_action.type.value == "stop":
                        if 2 in top_types:
                            correct_top3 += 1
                    elif expert_action.check_type == 'A':
                        if 0 in top_types and expert_action.aircraft_id in top_aircraft:
                            correct_top3 += 1
                    else:
                        if 1 in top_types and expert_action.aircraft_id in top_aircraft:
                            correct_top3 += 1
                
                n_samples += 1
    
    return {
        'top1_accuracy': correct_top1 / n_samples if n_samples > 0 else 0.0,
        'top3_accuracy': correct_top3 / n_samples if n_samples > 0 else 0.0,
        'avg_log_prob': total_log_prob / n_samples if n_samples > 0 else 0.0,
        'n_samples': n_samples,
    }


def evaluate_decoder_loss(instance: AMCSInstance) -> Tuple[float, float]:
    """
    Diagnostic: Compare Deng schedule cost with greedy-decoded CPD cost.
    
    This isolates how much cost difference comes from the greedy decoder
    vs. the CPD construction itself.
    
    Algorithm:
    1. Get Deng schedule
    2. Extract CPD from Deng schedule
    3. Re-decode CPD with greedy decoder
    4. Compare costs
    
    Returns:
        (deng_cost, greedy_decoded_cost)
    """
    from maintainer.training.data_gen import generate_synthetic_schedule
    
    # Get Deng schedule
    try:
        if DENG_AVAILABLE and instance._source_data is not None:
            deng_schedule = solve_with_deng(instance)
        else:
            deng_schedule = generate_synthetic_schedule(instance)
    except Exception:
        deng_schedule = generate_synthetic_schedule(instance)
    
    deng_cost = evaluate_schedule(instance, deng_schedule)
    
    # Extract CPD and re-decode
    cpd = extract_cpd_from_schedule(instance, deng_schedule)
    
    # Remove terminal flag to decode
    cpd_for_decode = CPDState(
        instance=cpd.instance,
        nodes=cpd.nodes,
        edges=cpd.edges,
        aircraft_task_counts=cpd.aircraft_task_counts,
        is_terminal=True,  # Keep terminal for decode
    )
    
    greedy_schedule = decode_cpd_to_schedule(cpd_for_decode)
    greedy_cost = evaluate_schedule(instance, greedy_schedule)
    
    return deng_cost, greedy_cost


def batch_evaluate_decoder_loss(
    instances: List[AMCSInstance],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate decoder loss on multiple instances.
    
    Returns:
        Dictionary with average costs and loss statistics
    """
    deng_costs = []
    greedy_costs = []
    
    iterator = instances
    if verbose:
        iterator = tqdm(instances, desc="Evaluating decoder loss")
    
    for instance in iterator:
        deng_cost, greedy_cost = evaluate_decoder_loss(instance)
        deng_costs.append(deng_cost)
        greedy_costs.append(greedy_cost)
    
    # Compute statistics
    deng_costs = np.array(deng_costs)
    greedy_costs = np.array(greedy_costs)
    ratios = greedy_costs / (deng_costs + 1e-8)
    
    return {
        'avg_deng_cost': float(np.mean(deng_costs)),
        'avg_greedy_cost': float(np.mean(greedy_costs)),
        'avg_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'max_ratio': float(np.max(ratios)),
        'n_worse': int(np.sum(greedy_costs > deng_costs)),
        'n_better': int(np.sum(greedy_costs < deng_costs)),
        'n_equal': int(np.sum(np.abs(greedy_costs - deng_costs) < 1e-6)),
    }


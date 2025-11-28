"""
Monte Carlo Tree Search for Maintainer.

Implements PUCT-based MCTS (AlphaZero-style) with the MaintainerNet
providing policy priors and value estimates.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math
import torch
import numpy as np

from .state import CPDState, Action, ActionType, apply_action
from .actions import get_legal_actions, get_action_mask
from .decoder import decode_cpd_to_schedule
from .model import MaintainerNet


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    n_simulations: int = 100        # Number of MCTS simulations per action
    c_puct: float = 1.0             # Exploration constant
    temperature: float = 1.0        # Temperature for action selection (1.0 = proportional to visits)
    dirichlet_alpha: float = 0.3    # Dirichlet noise alpha (0 = no noise)
    dirichlet_epsilon: float = 0.25 # Fraction of noise to add to root


class MCTSNode:
    """
    A node in the MCTS tree.
    
    Each node corresponds to a CPD state and maintains:
    - Statistics for each child action (N, Q, P)
    - Links to child nodes
    """
    
    def __init__(
        self,
        state: CPDState,
        parent: Optional["MCTSNode"] = None,
        action_from_parent: Optional[Action] = None,
    ):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        
        # MCTS statistics per action
        self.children: Dict[Action, MCTSNode] = {}
        self.N: Dict[Action, int] = {}      # Visit counts
        self.Q: Dict[Action, float] = {}    # Action values (mean)
        self.P: Dict[Action, float] = {}    # Prior probabilities
        
        # Terminal state handling
        self.is_terminal = state.is_terminal
        self.terminal_value: Optional[float] = None
        
        # Track if node has been expanded
        self._expanded = False
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return self._expanded
    
    def total_visits(self) -> int:
        """Total visits to this node."""
        return sum(self.N.values())
    
    def __repr__(self):
        return f"MCTSNode(state={self.state}, visits={self.total_visits()}, terminal={self.is_terminal})"


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    
    Uses PUCT selection formula (AlphaZero-style):
    UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_N) / (1 + N(s,a))
    
    Key design decisions:
    1. No rollouts - uses value head directly (AlphaZero-style)
    2. Terminal states evaluated with actual cost (decode + evaluate)
    3. Supports n_simulations=0 for greedy policy-only baseline
    """
    
    def __init__(
        self,
        network: MaintainerNet,
        config: MCTSConfig,
        evaluate_fn=None,  # Optional: (CPDState) -> float for terminal evaluation
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.device = device or torch.device('cpu')
        
        self.root: Optional[MCTSNode] = None
    
    def search(self, root_state: CPDState) -> Action:
        """
        Run MCTS from root_state and return best action.
        
        If n_simulations=0, returns greedy action from policy (no search).
        
        Args:
            root_state: Starting state
        
        Returns:
            Best action based on visit counts
        """
        if root_state.is_terminal:
            raise ValueError("Cannot search from terminal state")
        
        # Special case: n_simulations=0 means greedy policy only
        if self.config.n_simulations == 0:
            return self.network.greedy_action(root_state, self.device)
        
        # Initialize root
        self.root = MCTSNode(state=root_state)
        
        # Run simulations
        for _ in range(self.config.n_simulations):
            self._simulate(self.root)
        
        # Select action based on visit counts
        return self._select_action_from_root()
    
    def _simulate(self, root: MCTSNode):
        """
        Run a single MCTS simulation.
        
        1. Selection: traverse tree using PUCT until leaf
        2. Expansion & Evaluation: expand leaf and get value
        3. Backup: propagate value up the tree
        """
        node = root
        path: List[Tuple[MCTSNode, Action]] = []
        
        # Selection: traverse tree
        while node.is_expanded() and not node.is_terminal:
            action = self._select_action(node)
            path.append((node, action))
            
            if action not in node.children:
                # Need to create child
                break
            node = node.children[action]
        
        # Expansion & Evaluation
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand_and_evaluate(node)
        
        # Backup
        for parent, action in reversed(path):
            parent.N[action] = parent.N.get(action, 0) + 1
            
            # Incremental mean update
            n = parent.N[action]
            old_q = parent.Q.get(action, 0.0)
            parent.Q[action] = old_q + (value - old_q) / n
    
    def _select_action(self, node: MCTSNode) -> Action:
        """
        Select action using PUCT formula.
        
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_N) / (1 + N(s,a))
        """
        sqrt_sum_n = math.sqrt(node.total_visits())
        
        def puct_score(action: Action) -> float:
            q = node.Q.get(action, 0.0)
            p = node.P.get(action, 0.0)
            n = node.N.get(action, 0)
            
            u = self.config.c_puct * p * sqrt_sum_n / (1 + n)
            return q + u
        
        # Select action with highest PUCT score
        legal_actions = list(node.P.keys())
        if not legal_actions:
            # Fallback: get legal actions from state
            legal_actions = get_legal_actions(node.state)
        
        return max(legal_actions, key=puct_score)
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand a leaf node and evaluate it.
        
        For terminal states (Stop action was taken), we:
        - Decode CPD to schedule
        - Evaluate with actual cost function
        
        For non-terminal states, we:
        - Get policy and value from network
        - Initialize prior probabilities
        - Return value estimate
        """
        state = node.state
        
        # If this is a terminal state after Stop, evaluate with actual cost
        if state.is_terminal:
            if node.terminal_value is None:
                if self.evaluate_fn is not None:
                    node.terminal_value = self.evaluate_fn(state)
                else:
                    # Default: decode and use simple cost
                    from maintainer.amcs.bridge import evaluate_schedule
                    schedule = decode_cpd_to_schedule(state)
                    cost = evaluate_schedule(state.instance, schedule)
                    node.terminal_value = -cost  # Reward = -cost
            return node.terminal_value
        
        # Get policy and value from network
        mask = get_action_mask(state)
        with torch.no_grad():
            policy, value = self.network(state, mask, self.device)
        
        # Convert to probabilities
        aircraft_probs = torch.softmax(policy.aircraft, dim=-1).cpu().numpy()
        type_probs = torch.softmax(policy.type, dim=-1).cpu().numpy()
        
        # Initialize priors for all legal actions
        legal_actions = get_legal_actions(state)
        n_aircraft = state.instance.n_aircraft
        
        for action in legal_actions:
            if action.type == ActionType.STOP:
                prob = float(type_probs[2])
            elif action.check_type == 'A':
                prob = float(type_probs[0] * aircraft_probs[action.aircraft_id])
            else:  # 'C'
                prob = float(type_probs[1] * aircraft_probs[action.aircraft_id])
            
            node.P[action] = prob
            node.N[action] = 0
            node.Q[action] = 0.0
        
        # Add Dirichlet noise at root
        if node.parent is None and self.config.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_actions))
            eps = self.config.dirichlet_epsilon
            for i, action in enumerate(legal_actions):
                node.P[action] = (1 - eps) * node.P[action] + eps * noise[i]
        
        # Create child nodes
        for action in legal_actions:
            child_state = apply_action(state, action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                action_from_parent=action,
            )
            node.children[action] = child
            
            # If Stop action, mark child as terminal
            if action.type == ActionType.STOP:
                child.is_terminal = True
        
        node._expanded = True
        
        return value.item()
    
    def _select_action_from_root(self) -> Action:
        """
        Select action from root based on visit counts.
        
        Uses temperature-based sampling:
        - temperature = 0: argmax
        - temperature = 1: proportional to visits
        """
        root = self.root
        
        if not root.N:
            # Fallback: return first legal action
            return get_legal_actions(root.state)[0]
        
        actions = list(root.N.keys())
        visits = [root.N[a] for a in actions]
        
        if self.config.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visits)
            return actions[best_idx]
        
        # Temperature-based sampling
        visits = np.array(visits, dtype=np.float64)
        visits = visits ** (1.0 / self.config.temperature)
        probs = visits / visits.sum()
        
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]
    
    def get_policy(self, root_state: CPDState) -> Dict[Action, float]:
        """
        Run MCTS and return visit-count-based policy.
        
        Useful for training data generation (if doing self-play later).
        """
        self.search(root_state)
        
        total = self.root.total_visits()
        if total == 0:
            return {}
        
        return {
            action: count / total
            for action, count in self.root.N.items()
        }
    
    def get_root_value(self) -> float:
        """Get value estimate at root after search."""
        if self.root is None or not self.root.N:
            return 0.0
        
        # Return average Q value weighted by visits
        total = self.root.total_visits()
        if total == 0:
            return 0.0
        
        return sum(
            self.root.Q[a] * self.root.N[a] / total
            for a in self.root.N
        )


def plan_with_mcts(
    instance,
    network: MaintainerNet,
    config: MCTSConfig,
    evaluate_fn=None,
    device=None,
    verbose: bool = False,
) -> Tuple:
    """
    Full planning loop using MCTS.
    
    Constructs a complete CPD by running MCTS at each step.
    
    Args:
        instance: AMCS instance
        network: MaintainerNet model
        config: MCTS configuration
        evaluate_fn: Optional evaluation function
        device: Torch device
        verbose: Print progress
    
    Returns:
        (schedule, trajectory, final_cost)
    """
    from maintainer.amcs.bridge import evaluate_schedule
    
    state = CPDState.empty(instance)
    mcts = MCTS(network, config, evaluate_fn, device)
    
    trajectory = []
    
    step = 0
    while not state.is_terminal:
        action = mcts.search(state)
        
        if verbose:
            print(f"Step {step}: {action}")
        
        trajectory.append((state.copy(), action))
        state = apply_action(state, action)
        step += 1
    
    # Decode final CPD to schedule
    schedule = decode_cpd_to_schedule(state)
    cost = evaluate_schedule(instance, schedule)
    
    return schedule, trajectory, cost


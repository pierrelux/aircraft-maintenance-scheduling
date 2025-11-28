# Maintainer v1: Implementation Specification

## Executive Summary

**Maintainer** is an aircraft maintenance scheduling system that combines:
- A **transformer-based policy-value network** for CPD (Critical Path Diagram) construction
- **MCTS (Monte Carlo Tree Search)** for optimization
- **Behavior cloning from Deng** (existing solver) for training

**Architecture style**: AlphaZero (policy-value network + MCTS), trained via supervised learning instead of self-play.

**Key simplifications for v1**:
- Standard transformer (no DAG-specific attention)
- Sequential CPD construction (DAG validity by build order)
- No rollout policy (value head only)
- Pure behavior cloning (no RL/self-play)

---

## 1. Core Concepts

### 1.1 Problem Domain: Aircraft Maintenance Scheduling

**Input (AMCS Instance)**:
```python
{
    'n_aircraft': int,              # Fleet size
    'horizon': int,                 # Planning horizon in days
    'capacity': int,                # Daily hangar capacity
    'aircraft': [                   # Per-aircraft data
        {
            'id': int,
            'A_interval': int,      # Days between A-checks
            'C_interval': int,      # Days between C-checks
            'A_duration': int,      # A-check duration
            'C_duration': int,      # C-check duration
            'A_tolerance': int,     # Scheduling flexibility
            'C_tolerance': int,
        }
    ]
}
```

**Output**: Schedule matrix `[T × N_aircraft]` where each cell is `{None, 'A', 'C'}`.

**Objective**: Minimize cost (grounded days + violations + lost flight hours).

### 1.2 CPD (Critical Path Diagram) Representation

**CPD** = Directed Acyclic Graph of maintenance tasks.

**Node**: A maintenance task `(aircraft_id, check_type, sequence_index)`
```python
class TaskNode:
    aircraft_id: int        # Which aircraft
    check_type: str         # 'A' or 'C'
    sequence_index: int     # k-th check for this aircraft (0, 1, 2, ...)
    
    # Attributes (computed from instance + rules)
    duration: int
    earliest_day: int       # Earliest feasible start
    latest_day: int         # Latest feasible start
```

**Edges**: Precedence constraints (per-aircraft chains)
- Edge `(i, c₁, k) → (i, c₂, k+1)` means "aircraft i's k-th check must precede its (k+1)-th check"

**Example CPD** for 2 aircraft:
```
Aircraft 0: A₀ → C₁ → A₂
Aircraft 1: C₀ → A₁
```

### 1.3 Relationship to AlphaGo/AlphaZero

| Component | AlphaGo (2016) | AlphaZero (2017) | Maintainer v1 |
|-----------|----------------|------------------|---------------|
| Policy net | ✅ (supervised from humans) | ✅ (self-play head) | ✅ (BC from Deng) |
| Value net | ✅ | ✅ | ✅ |
| Rollout policy | ✅ (fast playouts) | ❌ | ❌ |
| MCTS | ✅ PUCT | ✅ PUCT | ✅ PUCT |
| Training | Supervised + RL | Pure self-play | Pure BC |

**Key insight**: Maintainer is AlphaZero-style architecture with supervised training.

---

## 2. Sequential CPD Construction (MDP Formulation)

### 2.1 State Space

**State S** = Partially constructed CPD
```python
class CPDState:
    instance: AMCSInstance              # Problem instance (fixed)
    nodes: List[TaskNode]               # Tasks added so far (in creation order)
    edges: List[Tuple[int, int]]        # (from_idx, to_idx) in nodes list
    aircraft_task_counts: Dict[int, int] # How many tasks per aircraft
    is_terminal: bool                   # True if Stop was chosen
```

**Initial state**: Empty CPD (no nodes, no edges).

### 2.2 Action Space

Two action types:

1. **AddTask(aircraft_id, check_type)**:
   - Creates new node `(aircraft_id, check_type, k)` where `k = aircraft_task_counts[aircraft_id]`
   - Auto-adds edge from previous task of same aircraft (if exists)
   - Appends to `nodes` list

2. **Stop**:
   - Marks CPD as complete
   - Triggers terminal evaluation

**Action masking**:
- Illegal if aircraft already has max tasks (computed from horizon and intervals)
- Stop always legal (but may result in incomplete/poor schedule)

### 2.3 State Transitions

```python
def apply_action(state: CPDState, action: Action) -> CPDState:
    if action.type == 'stop':
        return state._replace(is_terminal=True)
    
    # AddTask(aircraft_id, check_type)
    k = state.aircraft_task_counts[action.aircraft_id]
    new_node = TaskNode(action.aircraft_id, action.check_type, k)
    
    new_state = state.copy()
    new_state.nodes.append(new_node)
    new_state.aircraft_task_counts[action.aircraft_id] += 1
    
    # Add precedence edge from previous task (if exists)
    if k > 0:
        prev_idx = _find_prev_task(state, action.aircraft_id, k-1)
        new_state.edges.append((prev_idx, len(state.nodes)))
    
    return new_state
```

**Key property**: Construction order is a valid topological ordering (DAG guaranteed by design).

### 2.4 Terminal Evaluation

When Stop is chosen:

```python
def evaluate_terminal_state(state: CPDState) -> float:
    schedule = decode_cpd_to_schedule(state)  # §2.5
    cost = evaluate_schedule(state.instance, schedule)  # Existing Deng evaluator
    return -cost  # Negative cost = reward
```

### 2.5 CPD Decoding (Fixed Greedy Scheduling Policy)

Given a CPD, we produce a concrete schedule by applying a **fixed, deterministic scheduling policy**. There is no search or optimization inside this step; it is a single forward pass.

**Input**: `CPDState` (nodes in creation order + per-aircraft precedence edges)  
**Output**: Schedule `[T × N_aircraft]` with entries `{None, 'A', 'C'}`

Algorithm sketch:

```python
def decode_cpd_to_schedule(cpd_state: CPDState) -> Schedule:
    # 1. Use construction order as topological order
    topo_order = cpd_state.nodes  # Already a valid topological ordering
    
    schedule = Schedule(
        horizon=cpd_state.instance.horizon,
        n_aircraft=cpd_state.instance.n_aircraft,
    )
    task_start_days = {}  # task_idx -> start_day
    
    for idx, task in enumerate(topo_order):
        # Earliest day respecting:
        # - task window [earliest_day, latest_day]
        # - completion of predecessors
        earliest_possible = task.earliest_day
        
        # Precedence: previous checks on same aircraft
        for (from_idx, to_idx) in cpd_state.edges:
            if to_idx == idx:
                pred_end = task_start_days[from_idx] + topo_order[from_idx].duration
                earliest_possible = max(earliest_possible, pred_end)
        
        # Capacity: scan forward until first non-violating day
        start_day = find_first_feasible_day(
            schedule,
            task,
            earliest_possible,
            task.latest_day,
        )
        
        # Assign and never move this task again
        task_start_days[idx] = start_day
        schedule.assign(start_day, task.aircraft_id, task.check_type, task.duration)
    
    return schedule
```

The helper `find_first_feasible_day` is deliberately simple:

```python
def find_first_feasible_day(schedule, task, earliest, latest):
    """
    Return the first day d ∈ [earliest, latest] such that:
      - assigning `task` starting at day d does not exceed daily capacity
      - duration and horizon bounds are respected

    If no such day exists, we place the task at the latest feasible day
    (or horizon - duration) and accept whatever penalty the evaluator assigns.
    """
```

**Greedy nature.**

The decoder is **greedy** in the sense that, for each task in topological order, it:

1. picks the earliest feasible start day, and
2. never revisits previous assignments.

There is no backtracking or local re-optimization of earlier tasks. All *combinatorial* optimization happens in Maintainer's MCTS over CPD construction; the decoder is a fixed policy that maps a CPD to a schedule.

### 2.6 Decoder Rationale and Optimality Trade-off

This greedy forward decoder is intentionally simple and should be read as part of the **problem definition**, not as a second optimizer:

- Maintainer + MCTS searches over **CPDs** (which tasks to include and in what relative order).
- Given a CPD, the decoder applies a **fixed, deterministic scheduling rule** (earliest-feasible-day), with no internal search.
- The effective objective is therefore:
  \[
    \text{minimize } \text{cost}\big(\pi_{\text{sched}}(\text{CPD})\big),
  \]
  where \(\pi_{\text{sched}}\) is this greedy scheduler.

This mirrors how many human and heuristic planners work in practice: given a precedence network (or Gantt-like view), they scan tasks in some priority order and place each at the earliest day that respects windows and capacity, without solving a nested optimization problem each time.

**From the project scheduling literature**: Serial/parallel schedule generation schemes combined with simple priority rules form the standard class of heuristics for the **Resource-Constrained Project Scheduling Problem (RCPSP)**. These constructive schedulers are widely used in practice and are effective when the task ordering (priority) is good.

**From aircraft maintenance practice**: Heuristics based on graphical procedures and local rules are reported as being used operationally by Air Force organizations and maintenance planners to build flight and maintenance plans, before more sophisticated optimization is applied.

**Optimality trade-off:**

From an optimality standpoint, this means:

- For a fixed CPD, the greedy decoder may not produce the best possible schedule consistent with that CPD.
- Maintainer + MCTS can still adapt to this rule by learning CPDs that work well **under** this decoder.
- In v1 we accept this approximation to keep the action space and search manageable. If needed, later versions can:
  - enrich actions with coarse timing/phase decisions, or
  - replace the greedy step with a slightly stronger local scheduler (e.g., small MILPs or local swap heuristics).

**Diagnostic for future work:**

A simple diagnostic plan to isolate decoder loss:

1. Take a Deng schedule, extract its CPD, and re-decode with the greedy rule.
2. Compare the resulting cost to Deng's original cost.

This isolates the loss, if any, purely due to the decoder.

---

## 3. Maintainer Network Architecture

### 3.1 High-Level Interface

```python
class MaintainerNet(nn.Module):
    """
    Policy-value transformer for CPD construction.
    
    Input: CPDState
    Output: (policy_logits, value_estimate)
    """
    
    def forward(self, state: CPDState) -> Tuple[PolicyLogits, float]:
        """
        Args:
            state: Current partial CPD
            
        Returns:
            policy_logits: ActionLogits(
                aircraft_logits: [n_aircraft],  # P(choose aircraft i)
                type_logits: [3],               # P('A'), P('C'), P('Stop')
            )
            value: Scalar estimate of final reward
        """
```

### 3.2 Input Encoding: CPDState → Token Sequence

**Sequence structure**:
```
[GLOBAL] [AC_0] [AC_1] ... [AC_N] [TASK_0] [TASK_1] ... [TASK_M] [PAD]...
```

**Token types**:

1. **GLOBAL token** (position 0):
```python
global_embed = learned_global_embed + MLP([
    horizon,
    capacity,
    n_aircraft,
    current_num_tasks / expected_total_tasks,  # Progress indicator
])
```

2. **AIRCRAFT token** (one per aircraft):
```python
ac_embed = learned_ac_embed[aircraft_id] + MLP([
    A_interval,
    C_interval,
    A_duration,
    C_duration,
    A_tolerance,
    C_tolerance,
    current_task_count[aircraft_id],
    max_tasks[aircraft_id],
])
```

3. **TASK token** (one per node in CPD, in creation order):
```python
task_embed = (
    learned_ac_embed[task.aircraft_id] +
    learned_type_embed[task.check_type] +  # 'A' or 'C'
    learned_seq_embed[task.sequence_index] +  # 0, 1, 2, ...
    MLP([
        task.duration,
        task.earliest_day / horizon,
        task.latest_day / horizon,
        (task.latest_day - task.earliest_day) / horizon,  # flexibility
    ])
)
```

**Positional encoding**: Standard learned/sinusoidal positional embeddings (position in sequence).

### 3.3 Transformer Backbone

```python
class MaintainerNet(nn.Module):
    def __init__(self, config):
        self.token_encoder = TokenEncoder(config)  # §3.2
        
        # Standard decoder-only transformer (e.g., GPT2)
        self.transformer = GPT2Model(
            n_embd=config.d_model,      # e.g., 256
            n_layer=config.n_layers,    # e.g., 6
            n_head=config.n_heads,      # e.g., 8
            n_positions=config.max_seq_len,  # e.g., 512
        )
        
        # Policy heads
        self.aircraft_head = nn.Linear(config.d_model, config.n_aircraft)
        self.type_head = nn.Linear(config.d_model, 3)  # A, C, Stop
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Tanh(),  # Normalized reward in [-1, 1]
        )
    
    def forward(self, state: CPDState, action_mask: ActionMask):
        # 1. Encode state to token sequence
        token_embeds = self.token_encoder(state)  # [seq_len, d_model]
        
        # 2. Run transformer
        hidden_states = self.transformer(inputs_embeds=token_embeds)
        
        # 3. Pool for policy/value (use last token or mean)
        h = hidden_states[-1]  # Last token
        # OR: h = hidden_states.mean(dim=0)
        
        # 4. Policy heads
        aircraft_logits = self.aircraft_head(h)  # [n_aircraft]
        type_logits = self.type_head(h)          # [3]
        
        # Apply action mask
        aircraft_logits = aircraft_logits.masked_fill(~action_mask.aircraft, -1e9)
        
        policy_logits = ActionLogits(
            aircraft=aircraft_logits,
            type=type_logits,
        )
        
        # 5. Value head
        value = self.value_head(h).squeeze()  # Scalar
        
        return policy_logits, value
```

**Key points**:
- Standard causal attention (no DAG-specific masking)
- Policy factorized as P(aircraft) × P(type | aircraft) or joint
- Value is tanh-normalized scalar

### 3.4 Policy Decoding

```python
def sample_action(policy_logits: ActionLogits, action_mask: ActionMask, temperature=1.0):
    """
    Sample action from policy distribution.
    
    Two modes:
    1. Joint: P(aircraft, type) = softmax over all (i, c) pairs + Stop
    2. Factored: P(aircraft) × P(type)
    """
    
    # Factored approach (simpler for v1):
    type_probs = F.softmax(policy_logits.type / temperature, dim=-1)
    stop_prob = type_probs[2]  # Index 2 = Stop
    
    if random.random() < stop_prob:
        return Action(type='stop')
    
    # Sample check type ('A' or 'C')
    check_type = ['A', 'C'][torch.multinomial(type_probs[:2], 1).item()]
    
    # Sample aircraft
    aircraft_probs = F.softmax(policy_logits.aircraft / temperature, dim=-1)
    aircraft_id = torch.multinomial(aircraft_probs, 1).item()
    
    return Action(type='add_task', aircraft_id=aircraft_id, check_type=check_type)
```

---

## 4. MCTS Implementation

### 4.1 MCTS Node Structure

```python
class MCTSNode:
    state: CPDState                          # CPD state at this node
    parent: Optional[MCTSNode]               # Parent node
    action_from_parent: Optional[Action]     # Action that led here
    
    # MCTS statistics (per child action)
    children: Dict[Action, MCTSNode]         # Expanded children
    N: Dict[Action, int]                     # Visit counts N(s, a)
    Q: Dict[Action, float]                   # Action values Q(s, a)
    P: Dict[Action, float]                   # Prior probabilities P(s, a)
    
    is_terminal: bool
    terminal_value: Optional[float]          # Only set if terminal
```

### 4.2 MCTS Algorithm

```python
class MCTS:
    def __init__(self, network: MaintainerNet, c_puct=1.0, n_simulations=100):
        self.network = network
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root = None
    
    def search(self, root_state: CPDState) -> Action:
        """
        Run MCTS from root_state, return best action.
        """
        self.root = MCTSNode(state=root_state)
        
        for _ in range(self.n_simulations):
            self._simulate(self.root)
        
        # Choose action with highest visit count
        return max(self.root.N.items(), key=lambda x: x[1])[0]
    
    def _simulate(self, node: MCTSNode):
        """Single MCTS iteration: Select → Expand → Backup."""
        
        # 1. Selection: traverse tree using PUCT
        path = []
        current = node
        
        while not current.is_terminal and current.is_expanded():
            action = self._select_action(current)
            path.append((current, action))
            current = current.children[action]
        
        # 2. Expansion & Evaluation
        if current.is_terminal:
            value = current.terminal_value
        else:
            value = self._expand_and_evaluate(current)
        
        # 3. Backup
        for node, action in reversed(path):
            node.N[action] += 1
            node.Q[action] += (value - node.Q[action]) / node.N[action]
    
    def _select_action(self, node: MCTSNode) -> Action:
        """PUCT selection formula."""
        sqrt_sum_N = math.sqrt(sum(node.N.values()))
        
        def puct_score(action):
            q = node.Q[action]
            p = node.P[action]
            n = node.N[action]
            u = self.c_puct * p * sqrt_sum_N / (1 + n)
            return q + u
        
        return max(node.children.keys(), key=puct_score)
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node and evaluate with network.
        
        Key point: NO ROLLOUTS. Use value head directly.
        """
        state = node.state
        
        # Get policy and value from network
        policy_logits, value = self.network(state, action_mask=get_action_mask(state))
        
        # Initialize prior probabilities
        action_probs = compute_action_probs(policy_logits)  # Normalized distribution
        
        for action, prob in action_probs.items():
            node.P[action] = prob
            node.Q[action] = 0.0
            node.N[action] = 0
            
            # Create child node (lazy expansion)
            child_state = apply_action(state, action)
            child_node = MCTSNode(state=child_state, parent=node, action_from_parent=action)
            
            if action.type == 'stop':
                # Terminal state: evaluate with simulator
                child_node.is_terminal = True
                child_node.terminal_value = evaluate_terminal_state(child_state)
            
            node.children[action] = child_node
        
        # Return value estimate for backup
        return value
```

**Critical note**: We do NOT perform rollouts. The value head `v_θ(S)` provides the leaf estimate directly (AlphaZero-style, not original AlphaGo).

### 4.3 MCTS Integration

```python
def plan_with_maintainer(instance: AMCSInstance, 
                        network: MaintainerNet,
                        mcts_config: dict) -> Schedule:
    """
    Full planning loop: use MCTS to construct CPD, then decode to schedule.
    """
    state = CPDState.empty(instance)
    mcts = MCTS(network, **mcts_config)
    
    trajectory = []
    
    while not state.is_terminal:
        action = mcts.search(state)
        trajectory.append((state, action))
        state = apply_action(state, action)
    
    # Decode final CPD to schedule
    schedule = decode_cpd_to_schedule(state)
    return schedule, trajectory
```

---

## 5. Training: Behavior Cloning from Deng

### 5.1 Training Data Generation

**Goal**: Extract (state, action, value) tuples from Deng solutions.

```python
def generate_training_data(n_instances=1000) -> List[Trajectory]:
    """
    Generate training data from Deng solver.
    
    Returns: List of trajectories, each = [(S_t, a_t, z)]
    """
    trajectories = []
    
    for _ in range(n_instances):
        # 1. Generate random instance
        instance = generate_amcs_instance()
        
        # 2. Solve with Deng
        deng_schedule = solve_with_deng(instance)
        deng_cost = evaluate_schedule(instance, deng_schedule)
        z = -deng_cost  # Reward
        
        # 3. Extract CPD from Deng's schedule
        cpd = extract_cpd_from_schedule(instance, deng_schedule)
        
        # 4. Construct canonical build sequence
        trajectory = cpd_to_trajectory(cpd, z)
        trajectories.append(trajectory)
    
    return trajectories
```

**Key function**: `extract_cpd_from_schedule`
```python
def extract_cpd_from_schedule(instance: AMCSInstance, schedule: Schedule) -> CPDState:
    """
    Convert Deng's day-by-day schedule to CPD.
    
    Algorithm:
    1. For each aircraft, extract chronological sequence of checks
    2. Create TaskNode for each check
    3. Add precedence edges along aircraft chains
    4. Sort all tasks by scheduled day (tie-break by aircraft ID)
       → This gives canonical build order
    """
    tasks_by_aircraft = {}
    
    for day in range(schedule.horizon):
        for aircraft_id in range(schedule.n_aircraft):
            check = schedule[day, aircraft_id]
            if check in ['A', 'C']:
                if aircraft_id not in tasks_by_aircraft:
                    tasks_by_aircraft[aircraft_id] = []
                tasks_by_aircraft[aircraft_id].append({
                    'check_type': check,
                    'start_day': day,
                    'aircraft_id': aircraft_id,
                })
    
    # Build CPD in canonical order (sorted by scheduled day)
    all_tasks = []
    for ac_tasks in tasks_by_aircraft.values():
        all_tasks.extend(ac_tasks)
    all_tasks.sort(key=lambda t: (t['start_day'], t['aircraft_id']))
    
    # Create CPD state by sequentially adding tasks
    cpd_state = CPDState.empty(instance)
    for task in all_tasks:
        action = Action('add_task', task['aircraft_id'], task['check_type'])
        cpd_state = apply_action(cpd_state, action)
    
    return cpd_state
```

**Key function**: `cpd_to_trajectory`
```python
def cpd_to_trajectory(cpd: CPDState, final_reward: float) -> Trajectory:
    """
    Convert final CPD to training trajectory (S_t, a_t, z).
    
    Replay the CPD construction sequence.
    """
    trajectory = []
    state = CPDState.empty(cpd.instance)
    
    for task in cpd.nodes:
        action = Action('add_task', task.aircraft_id, task.check_type)
        trajectory.append((state.copy(), action, final_reward))
        state = apply_action(state, action)
    
    # Final Stop action
    stop_action = Action('stop')
    trajectory.append((state.copy(), stop_action, final_reward))
    
    return trajectory
```

### 5.2 Training Loop

```python
def train_maintainer(network: MaintainerNet, 
                    training_data: List[Trajectory],
                    config: TrainingConfig):
    """
    Supervised training: minimize policy cross-entropy + value MSE.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
    
    for epoch in range(config.n_epochs):
        epoch_losses = []
        
        for trajectory in training_data:
            for (state, action, reward) in trajectory:
                # Forward pass
                policy_logits, value = network(state, get_action_mask(state))
                
                # Policy loss: cross-entropy with expert action
                policy_loss = compute_policy_loss(policy_logits, action)
                
                # Value loss: MSE with final reward
                value_loss = (value - reward) ** 2
                
                # Combined loss
                loss = policy_loss + config.value_weight * value_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
        
        print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")
        
        # Validation
        if epoch % config.eval_interval == 0:
            evaluate_on_validation_set(network, config)
```

**Policy loss** (cross-entropy):
```python
def compute_policy_loss(policy_logits: ActionLogits, target_action: Action) -> torch.Tensor:
    """
    Cross-entropy between predicted policy and expert action.
    """
    if target_action.type == 'stop':
        # Target: Stop (index 2 in type_logits)
        return F.cross_entropy(policy_logits.type.unsqueeze(0), 
                              torch.tensor([2]))
    else:
        # Target: AddTask(aircraft_id, check_type)
        aircraft_loss = F.cross_entropy(
            policy_logits.aircraft.unsqueeze(0),
            torch.tensor([target_action.aircraft_id])
        )
        
        type_idx = 0 if target_action.check_type == 'A' else 1
        type_loss = F.cross_entropy(
            policy_logits.type.unsqueeze(0),
            torch.tensor([type_idx])
        )
        
        return aircraft_loss + type_loss
```

### 5.3 Evaluation Metrics

```python
def evaluate_on_validation_set(network: MaintainerNet, config):
    """
    Evaluate Maintainer+MCTS against Deng baseline.
    """
    val_instances = [generate_amcs_instance() for _ in range(config.n_val)]
    
    deng_costs = []
    maintainer_costs = []
    
    for instance in val_instances:
        # Deng baseline
        deng_schedule = solve_with_deng(instance)
        deng_cost = evaluate_schedule(instance, deng_schedule)
        deng_costs.append(deng_cost)
        
        # Maintainer + MCTS
        maintainer_schedule, _ = plan_with_maintainer(
            instance, network, config.mcts_config
        )
        maintainer_cost = evaluate_schedule(instance, maintainer_schedule)
        maintainer_costs.append(maintainer_cost)
    
    print(f"Deng avg cost: {np.mean(deng_costs):.2f}")
    print(f"Maintainer avg cost: {np.mean(maintainer_costs):.2f}")
    print(f"Improvement: {100 * (1 - np.mean(maintainer_costs)/np.mean(deng_costs)):.1f}%")
```

---

## 6. Implementation Plan

### 6.1 Module Structure

```
maintainer/
├── amcs/                           # Problem domain
│   ├── __init__.py
│   ├── instance.py                 # AMCSInstance class, generation
│   ├── deng_solver.py              # Existing Deng solver integration
│   ├── evaluator.py                # evaluate_schedule()
│   ├── cpd.py                      # CPD representation & decoder
│   └── extraction.py               # extract_cpd_from_schedule()
│
├── core/                           # Maintainer system
│   ├── __init__.py
│   ├── state.py                    # CPDState, Action, apply_action()
│   ├── encoder.py                  # TokenEncoder (state → embeddings)
│   ├── model.py                    # MaintainerNet (transformer + heads)
│   └── mcts.py                     # MCTS class
│
├── training/                       # Training pipeline
│   ├── __init__.py
│   ├── data_gen.py                 # generate_training_data()
│   ├── dataset.py                  # PyTorch Dataset wrapper
│   ├── trainer.py                  # train_maintainer()
│   └── evaluation.py               # Evaluation metrics
│
└── scripts/                        # Entry points
    ├── generate_data.py            # Generate BC training data
    ├── train.py                    # Train Maintainer network
    └── evaluate.py                 # Compare vs Deng baseline
```

### 6.2 Implementation Order

**Phase 1: AMCS Foundation** (use existing Deng code)
1. ✅ Already have: `aircraft_scheduler/` with Deng solver
2. Define `AMCSInstance` data class
3. Wrap Deng solver: `solve_with_deng(instance) -> schedule`
4. Wrap evaluator: `evaluate_schedule(instance, schedule) -> cost`

**Phase 2: CPD Representation**
1. Implement `CPDState` class
2. Implement `Action` (AddTask, Stop)
3. Implement `apply_action(state, action) -> state'`
4. Implement CPD decoder: `decode_cpd_to_schedule(cpd_state) -> schedule`
5. Test: CPD → schedule → evaluate

**Phase 3: Extraction from Deng**
1. Implement `extract_cpd_from_schedule(instance, schedule) -> cpd`
2. Implement `cpd_to_trajectory(cpd, reward) -> [(S, a, z)]`
3. Generate dataset: run Deng on 100 instances, extract trajectories
4. Verify: replay trajectories produce valid CPDs

**Phase 4: Maintainer Network**
1. Implement `TokenEncoder` (state → token embeddings)
2. Implement `MaintainerNet` (transformer + policy/value heads)
3. Test forward pass: random state → policy logits + value
4. Verify shapes and masking

**Phase 5: Training**
1. Implement training loop with BC loss
2. Train on Deng trajectories (start with small network)
3. Monitor: policy accuracy (% matching Deng actions), value MSE
4. Sanity check: network learns to imitate Deng's first few actions

**Phase 6: MCTS**
1. Implement `MCTSNode` class
2. Implement `MCTS.search()` (PUCT selection)
3. Test: MCTS with random network → produces valid CPD
4. Test: MCTS with trained network → produces CPD

**Phase 7: End-to-End Evaluation**
1. Implement `plan_with_maintainer(instance, network) -> schedule`
2. Compare Maintainer+MCTS vs Deng on validation set
3. Measure: cost gap, solve time, schedule quality
4. Iterate: tune MCTS hyperparameters (c_puct, n_simulations)

**Phase 8: Polish & Visualization**
1. Add logging and checkpointing
2. Add visualization (Gantt charts, CPD graphs)
3. Add hyperparameter sweep tools
4. Write documentation and usage examples

### 6.3 Testing Strategy

**Unit tests**:
- CPD state transitions (add task, edges correct)
- Action masking (illegal actions filtered)
- CPD decoder (valid schedules produced)
- Trajectory extraction (matches Deng CPD)

**Integration tests**:
- End-to-end: instance → Deng → CPD → trajectory → valid
- Training: can overfit on single trajectory
- MCTS: produces diverse valid CPDs

**Validation tests**:
- Maintainer matches Deng policy on test set (>80% action accuracy)
- Maintainer+MCTS finds schedules ≤ Deng cost on easy instances

---

## 7. Configuration & Hyperparameters

### 7.1 Network Config

```python
network_config = {
    'd_model': 256,           # Transformer hidden size
    'n_layers': 6,            # Transformer depth
    'n_heads': 8,             # Attention heads
    'max_seq_len': 512,       # Max sequence length
    'dropout': 0.1,
}
```

### 7.2 Training Config

```python
training_config = {
    'lr': 1e-4,               # Learning rate
    'batch_size': 32,         # (Or 1 if varying sequence lengths)
    'n_epochs': 100,
    'value_weight': 0.5,      # λ in loss = policy_loss + λ * value_loss
    'grad_clip': 1.0,         # Gradient clipping
}
```

### 7.3 MCTS Config

```python
mcts_config = {
    'n_simulations': 100,     # MCTS iterations per action
    'c_puct': 1.0,            # Exploration constant
    'temperature': 1.0,       # For action sampling (1.0 = proportional to counts)
}
```

### 7.4 Data Generation Config

```python
data_config = {
    'n_train_instances': 1000,
    'n_val_instances': 100,
    'instance_params': {
        'n_aircraft': (5, 10),     # Range
        'horizon': (180, 365),
        'capacity': (1, 3),
    }
}
```

---

## 8. Future Work (Explicitly Out of Scope for v1)

These are NOT implemented in v1 but noted for future iterations:

1. **DAG-aware attention**:
   - Reachability masks in attention
   - Depth-based or ancestor-based positional encodings

2. **Explicit phase/timing**:
   - Add "target phase" to actions
   - Let network predict coarse timing directly

3. **AlphaZero self-play**:
   - Use Maintainer+MCTS to generate training data
   - Iteratively improve beyond Deng

4. **Rollout policy**:
   - Fast CPD completion heuristic for leaf evaluation
   - Could supplement value head in early training

5. **Hybrid search**:
   - Incorporate Deng's heuristics as priors or constraints
   - Multi-objective optimization

6. **Larger-scale experiments**:
   - Real-world instances (100+ aircraft)
   - Multi-fleet, heterogeneous aircraft
   - Stochastic maintenance durations

---

## 9. Dependencies

```python
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.1        # For CPD visualization
tqdm>=4.65.0
pyyaml>=6.0
```

**Existing code**: `aircraft_scheduler/` (Deng solver, evaluator)

---

## 10. Success Criteria for v1

**Minimum viable v1**:
1. ✅ Maintainer trains on Deng data (policy accuracy >80% on validation)
2. ✅ Maintainer+MCTS produces valid schedules (no rule violations)
3. ✅ Maintainer+MCTS matches Deng cost (within 10%) on test instances

**Stretch goals**:
1. Maintainer+MCTS beats Deng on >20% of instances
2. Training is stable and reproducible
3. System scales to 20+ aircraft, 365-day horizon

---

## Summary Checklist

**Architecture decisions**:
- [x] Standard transformer (no DAG attention)
- [x] Sequential CPD construction (build-order validity)
- [x] Policy-value network (factored policy)
- [x] PUCT MCTS (no rollouts)
- [x] Pure behavior cloning (no self-play)

**Implementation scope**:
- [x] CPD representation and decoder
- [x] Extraction from Deng schedules
- [x] Token encoder for CPD states
- [x] Transformer with policy/value heads
- [x] MCTS with network priors
- [x] BC training loop
- [x] Evaluation against Deng baseline

**Ready to implement**: This spec is complete and actionable.

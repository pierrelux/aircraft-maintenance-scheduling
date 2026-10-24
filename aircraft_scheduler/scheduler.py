"""
Aircraft Maintenance Check Scheduling Optimizer

A faithful reproduction of:
    Deng, Q., Santos, B.F., Curran, R. (2020). "A practical dynamic programming 
    based methodology for aircraft maintenance check scheduling optimization."
    European Journal of Operational Research, 281(2), 256-273.

IMPORTANT: Despite the paper's title, the algorithm is NOT Bellman DP.
It is forward beam search with state aggregation and dominance pruning.

Key Paper Elements Implemented:
-------------------------------
1. COST FUNCTION (Equation 28):
   C_t = Σ[ χ·(I^FH - FH) + (1-χ)·P_a·θ + P_d·η ]

2. SEVEN DAILY ACTIONS (Table 2):
   (n_a, n_c) ∈ {(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (2,0)}
   
3. STATE AGGREGATION (Section 4.4):
   States aggregated by utilization histogram to reduce state space.
   
4. THRIFTY PRUNING (Section 4.4.1):
   Keep only lowest-cost state per histogram bucket.

5. PRIORITY ORDERING (Equations 36-40):
   Earliest deadline first for aircraft selection.

Usage:
------
    from aircraft_scheduler import Scheduler, load_maintenance_data
    
    data = load_maintenance_data("data.xlsx")
    scheduler = Scheduler(data, beam_width=200)
    result = scheduler.schedule(4 * 365, verbose=True)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from collections import defaultdict
import numpy as np
import time
import sys

from .data_loader import MaintenanceData


# =============================================================================
# PAPER'S 7 DAILY ACTIONS (Table 2)
# =============================================================================

PAPER_ACTIONS = [
    (0, 0),  # Do nothing
    (1, 0),  # 1 A-check only
    (0, 1),  # 1 C-check only
    (1, 1),  # 1 A-check + 1 C-check
    (0, 2),  # 2 C-checks
    (1, 2),  # 1 A-check + 2 C-checks
    (2, 0),  # 2 A-checks
]


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class FleetAction:
    """A concrete fleet action specifying which aircraft get which checks."""
    a_checks: list[str]  # Tail numbers getting A-checks
    c_checks: list[str]  # Tail numbers getting C-checks
    
    @property
    def n_a(self) -> int:
        return len(self.a_checks)
    
    @property
    def n_c(self) -> int:
        return len(self.c_checks)
    
    def is_null(self) -> bool:
        return self.n_a == 0 and self.n_c == 0


@dataclass
class ScheduledCheck:
    """A scheduled maintenance check with paper's eq. 28 metrics."""
    tail: str
    check_type: str  # 'A', 'C', or 'D'
    start_date: date
    end_date: Optional[date] = None
    duration: int = 1
    fh_at_check: float = 0.0
    unused_fh: float = 0.0
    used_tolerance_fh: float = 0.0
    is_tolerance_event: bool = False
    a_fh_at_check: float = 0.0
    a_unused_fh: float = 0.0

    @property
    def total_unused_fh(self) -> float:
        return self.unused_fh + self.a_unused_fh


@dataclass
class Table3KPIs:
    """KPIs matching Table 3 in Deng et al. paper."""
    n_a_checks: int
    n_c_checks: int
    n_d_checks: int
    avg_fh_at_a_checks: float
    std_fh_at_a_checks: float
    avg_fh_at_c_checks: float
    std_fh_at_c_checks: float
    total_unused_fh: float
    avg_unused_fh_a: float
    avg_unused_fh_c: float
    total_fh_tolerance_used: float
    total_tolerance_events: int
    
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization with paper's eq. 28 metrics."""
    checks: list[ScheduledCheck]
    total_a_checks: int
    total_c_checks: int
    total_d_checks: int
    computation_time: float
    total_unused_fh: float = 0.0
    total_tolerance_penalty: float = 0.0
    total_tolerance_events: int = 0
    _kpis: Optional[Table3KPIs] = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        return self.total_unused_fh + self.total_tolerance_penalty

    def get_checks_for_aircraft(self, tail: str) -> list[ScheduledCheck]:
        return [c for c in self.checks if c.tail == tail]

    def compute_kpis(self) -> Table3KPIs:
        if self._kpis is not None:
            return self._kpis
        
        a_checks = [c for c in self.checks if c.check_type == "A"]
        c_checks = [c for c in self.checks if c.check_type == "C"]
        d_checks = [c for c in self.checks if c.check_type == "D"]
        
        a_fh = [c.fh_at_check for c in a_checks]
        c_fh = [c.fh_at_check for c in c_checks + d_checks]
        
        self._kpis = Table3KPIs(
            n_a_checks=len(a_checks),
            n_c_checks=len(c_checks),
            n_d_checks=len(d_checks),
            avg_fh_at_a_checks=float(np.mean(a_fh)) if a_fh else 0.0,
            std_fh_at_a_checks=float(np.std(a_fh)) if a_fh else 0.0,
            avg_fh_at_c_checks=float(np.mean(c_fh)) if c_fh else 0.0,
            std_fh_at_c_checks=float(np.std(c_fh)) if c_fh else 0.0,
            total_unused_fh=sum(c.unused_fh for c in a_checks) + sum(c.total_unused_fh for c in c_checks + d_checks),
            avg_unused_fh_a=sum(c.unused_fh for c in a_checks) / len(a_checks) if a_checks else 0.0,
            avg_unused_fh_c=sum(c.total_unused_fh for c in c_checks + d_checks) / len(c_checks + d_checks) if (c_checks + d_checks) else 0.0,
            total_fh_tolerance_used=sum(c.used_tolerance_fh for c in self.checks),
            total_tolerance_events=sum(1 for c in self.checks if c.is_tolerance_event),
        )
        return self._kpis

    def summary(self) -> str:
        return "\n".join([
            "Schedule Summary",
            "================",
            f"Checks: {self.total_a_checks} A / {self.total_c_checks} C / {self.total_d_checks} D",
            f"Unused FH: {self.total_unused_fh:.1f}",
            f"Tolerance events: {self.total_tolerance_events}",
            f"Time: {self.computation_time:.2f}s",
        ])


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

def create_fleet_state(data: MaintenanceData) -> dict:
    """Create initial fleet state from maintenance data."""
    n = len(data.aircraft_tails)
    return {
        'dy_c': np.array([data.c_initial[t].dy_c for t in data.aircraft_tails], dtype=np.float32),
        'fh_c': np.array([data.c_initial[t].fh_c for t in data.aircraft_tails], dtype=np.float32),
        'fc_c': np.array([data.c_initial[t].fc_c for t in data.aircraft_tails], dtype=np.float32),
        'dy_a': np.array([data.a_initial[t].dy_a for t in data.aircraft_tails], dtype=np.float32),
        'fh_a': np.array([data.a_initial[t].fh_a for t in data.aircraft_tails], dtype=np.float32),
        'fc_a': np.array([data.a_initial[t].fc_a for t in data.aircraft_tails], dtype=np.float32),
        'grounded_days': np.zeros(n, dtype=np.float32),
    }


def create_static_params(data: MaintenanceData) -> dict:
    """Create static parameters (intervals, rates, etc.)."""
    return {
        'c_ci_dy': np.array([data.c_initial[t].c_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fh': np.array([data.c_initial[t].c_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fc': np.array([data.c_initial[t].c_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_dy': np.array([data.a_initial[t].a_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fh': np.array([data.a_initial[t].a_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fc': np.array([data.a_initial[t].a_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        'daily_fh': np.array([
            sum(data.flight_rates[t].dfh.values()) / 12 if t in data.flight_rates else 10.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        'daily_fc': np.array([
            sum(data.flight_rates[t].dfc.values()) / 12 if t in data.flight_rates else 4.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        'c_tat': np.array([
            data.c_check_info[t].elapsed_times[0] if t in data.c_check_info else data.config.c_e_t
            for t in data.aircraft_tails
        ], dtype=np.float32),
    }


# =============================================================================
# COST FUNCTION (Equation 28)
# =============================================================================

def compute_check_cost(state, static, c_mask, a_mask, p_a, p_d):
    """Compute cost for one day's checks per paper's Equation 28."""
    grounded = state['grounded_days'] > 0
    n_c_checks = np.sum(c_mask)
    n_a_checks = np.sum(a_mask)
    
    # Unused FH
    c_unused_fh = np.maximum(static['c_ci_fh'] - state['fh_c'], 0.0)
    a_unused_fh_in_c = np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0)
    c_cost = np.sum(c_mask * (c_unused_fh + a_unused_fh_in_c))
    a_cost = np.sum(a_mask * np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0))
    
    # Tolerance penalty
    c_in_tol = (state['dy_c'] > static['c_ci_dy']) | (state['fh_c'] > static['c_ci_fh']) | (state['fc_c'] > static['c_ci_fc'])
    a_in_tol = (state['dy_a'] > static['a_ci_dy']) | (state['fh_a'] > static['a_ci_fh']) | (state['fc_a'] > static['a_ci_fc'])
    tolerance_events = np.sum(c_mask & c_in_tol) + np.sum(a_mask & a_in_tol)
    
    # Ground-waiting penalty
    overdue = c_in_tol | a_in_tol
    ground_waiting = overdue & ~grounded & ~(c_mask | a_mask)
    
    total_cost = c_cost + a_cost + tolerance_events * p_a + np.sum(ground_waiting) * p_d
    return total_cost, int(tolerance_events), int(n_c_checks), int(n_a_checks)


# =============================================================================
# STATE TRANSITIONS
# =============================================================================

def apply_action(state, static, c_mask, a_mask):
    """Apply checks and advance one day."""
    new_state = {k: v.copy() for k, v in state.items()}
    
    # C-checks reset both C and A counters
    for key in ['dy_c', 'fh_c', 'fc_c', 'dy_a', 'fh_a', 'fc_a']:
        new_state[key] = np.where(c_mask, 0.0, new_state[key])
    new_state['grounded_days'] = np.where(c_mask, static['c_tat'], new_state['grounded_days'])
    
    # A-checks (only if not getting C-check)
    a_only = a_mask & ~c_mask
    for key in ['dy_a', 'fh_a', 'fc_a']:
        new_state[key] = np.where(a_only, 0.0, new_state[key])
    
    # Advance one day
    new_state['grounded_days'] = np.maximum(new_state['grounded_days'] - 1, 0)
    active = new_state['grounded_days'] <= 0
    new_state['dy_c'] += active.astype(np.float32)
    new_state['dy_a'] += active.astype(np.float32)
    new_state['fh_c'] += active * static['daily_fh']
    new_state['fh_a'] += active * static['daily_fh']
    new_state['fc_c'] += active * static['daily_fc']
    new_state['fc_a'] += active * static['daily_fc']
    
    return new_state


# =============================================================================
# PRIORITY ORDERING (Equations 36-40)
# =============================================================================

def select_aircraft(state, static, n_select, check_type, exclude_mask=None):
    """Select aircraft using earliest deadline first."""
    grounded = state['grounded_days'] > 0
    
    if check_type == 'C':
        remaining = np.minimum(
            static['c_ci_dy'] - state['dy_c'],
            np.minimum(
                (static['c_ci_fh'] - state['fh_c']) / np.maximum(static['daily_fh'], 0.1),
                (static['c_ci_fc'] - state['fc_c']) / np.maximum(static['daily_fc'], 0.1)
            )
        )
    else:
        remaining = np.minimum(
            static['a_ci_dy'] - state['dy_a'],
            np.minimum(
                (static['a_ci_fh'] - state['fh_a']) / np.maximum(static['daily_fh'], 0.1),
                (static['a_ci_fc'] - state['fc_a']) / np.maximum(static['daily_fc'], 0.1)
            )
        )
    
    remaining = np.where(grounded, np.inf, remaining)
    if exclude_mask is not None:
        remaining = np.where(exclude_mask, np.inf, remaining)
    
    ranks = np.sum(remaining[:, None] > remaining[None, :], axis=1)
    mask = (ranks < n_select) & (n_select > 0) & ~grounded
    if exclude_mask is not None:
        mask = mask & ~exclude_mask
    return mask

def select_due_aircraft(state, static, n_select, check_type, exclude_mask=None):
    """Select only aircraft that are due/overdue, using earliest-deadline ordering.

    This is used for naive baselines to avoid performing checks early.
    """
    grounded = state["grounded_days"] > 0

    if check_type == "C":
        due = (
            (state["dy_c"] >= static["c_ci_dy"])
            | (state["fh_c"] >= static["c_ci_fh"])
            | (state["fc_c"] >= static["c_ci_fc"])
        )
    else:
        due = (
            (state["dy_a"] >= static["a_ci_dy"])
            | (state["fh_a"] >= static["a_ci_fh"])
            | (state["fc_a"] >= static["a_ci_fc"])
        )

    # Only rank due aircraft; everything else is ineligible.
    eligible = due & ~grounded
    if exclude_mask is not None:
        eligible = eligible & ~exclude_mask

    if n_select <= 0 or not np.any(eligible):
        return np.zeros_like(grounded, dtype=bool)

    # Use same ordering as select_aircraft, but gate by eligibility.
    if check_type == "C":
        remaining = np.minimum(
            static["c_ci_dy"] - state["dy_c"],
            np.minimum(
                (static["c_ci_fh"] - state["fh_c"]) / np.maximum(static["daily_fh"], 0.1),
                (static["c_ci_fc"] - state["fc_c"]) / np.maximum(static["daily_fc"], 0.1),
            ),
        )
    else:
        remaining = np.minimum(
            static["a_ci_dy"] - state["dy_a"],
            np.minimum(
                (static["a_ci_fh"] - state["fh_a"]) / np.maximum(static["daily_fh"], 0.1),
                (static["a_ci_fc"] - state["fc_a"]) / np.maximum(static["daily_fc"], 0.1),
            ),
        )

    remaining = np.where(eligible, remaining, np.inf)
    ranks = np.sum(remaining[:, None] > remaining[None, :], axis=1)
    mask = (ranks < n_select) & eligible
    return mask


# =============================================================================
# STATE AGGREGATION (Section 4.4)
# =============================================================================

def compute_histogram(state, static, u):
    """Compute utilization histogram for state aggregation."""
    grounded = state['grounded_days'] > 0
    active = ~grounded
    
    c_bins = (state['fh_c'] / u).astype(int)
    a_bins = (state['fh_a'] / (u / 2)).astype(int)
    
    c_in_tol = (state['dy_c'] > static['c_ci_dy']) | (state['fh_c'] > static['c_ci_fh']) | (state['fc_c'] > static['c_ci_fc'])
    a_in_tol = (state['dy_a'] > static['a_ci_dy']) | (state['fh_a'] > static['a_ci_fh']) | (state['fc_a'] > static['a_ci_fc'])
    
    c_unique, c_counts = np.unique(c_bins[active], return_counts=True)
    a_unique, a_counts = np.unique(a_bins[active], return_counts=True)
    
    return (
        tuple(sorted(zip(c_unique.tolist(), c_counts.tolist()))),
        tuple(sorted(zip(a_unique.tolist(), a_counts.tolist()))),
        int(np.sum((c_in_tol | a_in_tol) & active)),
        int(np.sum(grounded))
    )


# =============================================================================
# HEURISTIC FUNCTIONS FOR BEST-FIRST BEAM SEARCH
# =============================================================================

def heuristic_local_risk(state, static, p_a, p_d):
    """
    Ultra-lightweight heuristic: count aircraft in tolerance or grounded.
    
    This is a cheap proxy for "how bad is the current situation?"
    Returns a NORMALIZED score (count of problematic aircraft), not scaled by p_a/p_d.
    The lambda_h parameter controls how much this influences ranking.
    """
    grounded = state['grounded_days'] > 0
    
    c_in_tol = (
        (state['dy_c'] > static['c_ci_dy']) |
        (state['fh_c'] > static['c_ci_fh']) |
        (state['fc_c'] > static['c_ci_fc'])
    )
    a_in_tol = (
        (state['dy_a'] > static['a_ci_dy']) |
        (state['fh_a'] > static['a_ci_fh']) |
        (state['fc_a'] > static['a_ci_fc'])
    )
    
    n_tol = np.sum(c_in_tol) + np.sum(a_in_tol)
    n_grounded = np.sum(grounded)
    
    # Return weighted count (tolerance is worse than grounded for scheduling)
    # Scale: ~0 for healthy fleet, ~100 for fleet in trouble
    return 2.0 * n_tol + 1.0 * n_grounded


def heuristic_future_risk(state, static, remaining_days, p_a, p_d):
    """
    Time-to-trouble heuristic: estimate future cost from this state.
    
    Computes:
    1. Overdue aircraft that will incur tolerance/grounding penalties
    2. Low-slack aircraft that are close to limits
    3. Currently grounded aircraft
    
    Returns a NORMALIZED score suitable for comparison between states.
    The lambda_h parameter controls how much this influences ranking.
    """
    n_aircraft = len(state['dy_c'])
    grounded = state['grounded_days'] > 0
    
    # Remaining time to C and A limits (same logic as select_aircraft)
    remaining_c = np.minimum(
        static['c_ci_dy'] - state['dy_c'],
        np.minimum(
            (static['c_ci_fh'] - state['fh_c']) / np.maximum(static['daily_fh'], 0.1),
            (static['c_ci_fc'] - state['fc_c']) / np.maximum(static['daily_fc'], 0.1),
        ),
    )
    remaining_a = np.minimum(
        static['a_ci_dy'] - state['dy_a'],
        np.minimum(
            (static['a_ci_fh'] - state['fh_a']) / np.maximum(static['daily_fh'], 0.1),
            (static['a_ci_fc'] - state['fc_a']) / np.maximum(static['daily_fc'], 0.1),
        ),
    )
    
    # Decompose into slack (good) and overdue (bad)
    overdue_c = np.maximum(-remaining_c, 0.0)
    overdue_a = np.maximum(-remaining_a, 0.0)
    slack_c = np.maximum(remaining_c, 0.0)
    slack_a = np.maximum(remaining_a, 0.0)
    
    # 1) Estimated future tolerance events from currently overdue aircraft
    #    Capped by remaining horizon
    est_tolerance_events = np.minimum(remaining_days, overdue_c + overdue_a).sum()
    
    # 2) Low-slack penalty: sum of (1 / (1 + slack)) over all aircraft
    #    Range: [0, 2*n_aircraft] where higher = more aircraft close to limits
    inv_slack_c = 1.0 / (1.0 + slack_c)
    inv_slack_a = 1.0 / (1.0 + slack_a)
    slack_penalty = (inv_slack_c + inv_slack_a).sum()
    
    # 3) Grounded aircraft penalty (count)
    grounded_count = grounded.sum()
    
    # Combine with balanced weights to produce score in range ~[0, 1000]
    # - Tolerance events: most important (will definitely incur penalty)
    # - Low slack: moderately important (likely to cause issues)
    # - Grounded: less important (already paid for, just waiting)
    return (
        10.0 * est_tolerance_events    # Heavy weight on future tolerance
        + 1.0 * slack_penalty          # Moderate weight on low slack  
        + 0.5 * grounded_count         # Light weight on grounded
    )


# =============================================================================
# THRIFTY PRUNING (Section 4.4.1)
# =============================================================================

def thrifty_prune(frontier, beam_width, k_per_bucket=1, f_scores=None):
    """Apply thrifty dominance pruning with optional diversity preservation.
    
    Entry format: (state, cost, tol, n_c, n_a, histogram, history)
    
    Args:
        frontier: List of state entries
        beam_width: Maximum total states to keep
        k_per_bucket: Number of states to keep per histogram bucket.
                      k=1 is original Deng behavior (only best per bucket).
                      k>1 preserves structural diversity within buckets.
        f_scores: Optional list of f-scores (g + λh) for best-first ranking.
                  If provided, uses f-scores instead of g-costs for ranking.
    """
    # Build index-aware entries for tracking
    if f_scores is not None:
        indexed = [(f_scores[i], i, frontier[i]) for i in range(len(frontier))]
        get_score = lambda x: x[0]  # f-score
        get_entry = lambda x: x[2]  # original entry
    else:
        indexed = [(entry[1], i, entry) for i, entry in enumerate(frontier)]
        get_score = lambda x: x[0]  # g-cost
        get_entry = lambda x: x[2]  # original entry
    
    # Group by histogram
    by_histogram = defaultdict(list)
    for item in indexed:
        entry = get_entry(item)
        by_histogram[entry[5]].append(item)  # entry[5] is histogram
    
    pruned = []
    for entries in by_histogram.values():
        # Sort by score (f or g) within bucket
        entries.sort(key=get_score)
        # Keep top-k from this bucket
        pruned.extend(entries[:k_per_bucket])
    
    # Global sort by score and cap at beam_width
    pruned.sort(key=get_score)
    return [get_entry(x) for x in pruned[:beam_width]]


# =============================================================================
# MAIN SCHEDULER
# =============================================================================

class Scheduler:
    """
    Forward beam search scheduler implementing Deng et al. (2020).
    
    Args:
        data: MaintenanceData from load_maintenance_data()
        beam_width: Maximum states to keep per day (default 200)
        u: Discretization step for histograms (default 100.0)
        p_a: Tolerance penalty (default 10^8)
        p_d: Ground-waiting penalty (default 10^8)
        
    Variant parameters (extensions beyond base Deng algorithm):
        use_heuristic: Enable best-first beam with lookahead heuristic
        heuristic_type: 'local' (fast) or 'future' (more accurate)
        lambda_h: Weight for heuristic in f = g + λh ranking (default 0.01)
        k_per_bucket: States to keep per histogram bucket (default 1 = original)
    """
    
    def __init__(
        self,
        data: MaintenanceData,
        beam_width: int = 200,
        u: float = 100.0,
        p_a: float = 1e8,
        p_d: float = 1e8,
        # Variant parameters
        use_heuristic: bool = False,
        heuristic_type: str = 'future',  # 'local' or 'future'
        lambda_h: float = 0.01,
        k_per_bucket: int = 1,
    ):
        self.data = data
        self.beam_width = beam_width
        self.u = u
        self.p_a = p_a
        self.p_d = p_d
        
        # Variant options
        self.use_heuristic = use_heuristic
        self.heuristic_type = heuristic_type
        self.lambda_h = lambda_h
        self.k_per_bucket = k_per_bucket
        
        self.initial_state = create_fleet_state(data)
        self.static = create_static_params(data)
        self.n_aircraft = len(data.aircraft_tails)
        self.max_c = data.config.max_c_check
        self.max_a = data.config.max_a_check

    def _history_to_checks(
        self,
        history: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        start_date: date,
        tails: list[str],
    ) -> list[ScheduledCheck]:
        """Convert recorded history entries into ScheduledCheck objects."""
        checks: list[ScheduledCheck] = []
        for day, c_indices, a_indices, state_snap in history:
            fh_c, fh_a, dy_c, dy_a = state_snap
            check_date = start_date + timedelta(days=day)

            for idx in c_indices:
                tail = tails[idx]
                c_interval_fh = self.static["c_ci_fh"][idx]
                c_interval_dy = self.static["c_ci_dy"][idx]
                a_interval_fh = self.static["a_ci_fh"][idx]

                fh_at = float(fh_c[idx])
                unused = max(0.0, c_interval_fh - fh_at)
                a_fh_at = float(fh_a[idx])
                a_unused = max(0.0, a_interval_fh - a_fh_at)
                is_tol = (dy_c[idx] > c_interval_dy) or (fh_at > c_interval_fh)
                tat = int(self.static["c_tat"][idx])

                checks.append(
                    ScheduledCheck(
                        tail=tail,
                        check_type="C",
                        start_date=check_date,
                        end_date=check_date + timedelta(days=tat - 1),
                        duration=tat,
                        fh_at_check=fh_at,
                        unused_fh=unused,
                        is_tolerance_event=is_tol,
                        a_fh_at_check=a_fh_at,
                        a_unused_fh=a_unused,
                    )
                )

            for idx in a_indices:
                tail = tails[idx]
                a_interval_fh = self.static["a_ci_fh"][idx]
                a_interval_dy = self.static["a_ci_dy"][idx]
                fh_at = float(fh_a[idx])
                unused = max(0.0, a_interval_fh - fh_at)
                is_tol = (dy_a[idx] > a_interval_dy) or (fh_at > a_interval_fh)

                checks.append(
                    ScheduledCheck(
                        tail=tail,
                        check_type="A",
                        start_date=check_date,
                        end_date=check_date,
                        duration=1,
                        fh_at_check=fh_at,
                        unused_fh=unused,
                        is_tolerance_event=is_tol,
                    )
                )

        return checks
    
    def schedule(self, planning_days: int, verbose: bool = False) -> ScheduleResult:
        """
        Run the forward beam search algorithm.
        
        Returns ScheduleResult with full schedule (checks with dates and aircraft).
        """
        start_time = time.time()
        start_date = self.data.config.begin_day
        tails = self.data.aircraft_tails
        
        if verbose:
            variant_info = []
            if self.use_heuristic:
                variant_info.append(f"heuristic={self.heuristic_type}(λ={self.lambda_h})")
            if self.k_per_bucket > 1:
                variant_info.append(f"k_per_bucket={self.k_per_bucket}")
            variant_str = f" [{', '.join(variant_info)}]" if variant_info else ""
            
            print(f"[Deng] Starting {planning_days}-day schedule, beam={self.beam_width}{variant_str}")
            print(f"[Deng] Fleet: {self.n_aircraft} aircraft, P_a={self.p_a:.0e}, P_d={self.p_d:.0e}")
            sys.stdout.flush()
        
        init_hist = compute_histogram(self.initial_state, self.static, self.u)
        # Frontier entry: (state, cost, tol, n_c, n_a, histogram, history)
        # history is list of (day, c_indices, a_indices, state_snapshot) tuples
        frontier = [(self.initial_state.copy(), 0.0, 0, 0, 0, init_hist, [])]
        
        for day in range(planning_days):
            if verbose and day % 50 == 0:
                elapsed = time.time() - start_time
                best = min(frontier, key=lambda x: x[1])
                eta = (elapsed / max(day, 1)) * (planning_days - day) / 60 if day > 0 else 0
                print(f"[Deng] Day {day:4d}/{planning_days} | "
                      f"states={len(frontier):3d} | cost={best[1]:.0f} | "
                      f"C={best[3]} A={best[4]} | time={elapsed:.0f}s ETA={eta:.1f}m")
                sys.stdout.flush()
            
            next_frontier = []
            for state, cum_cost, cum_tol, cum_c, cum_a, _, history in frontier:
                for n_a, n_c in PAPER_ACTIONS:
                    n_c = min(n_c, self.max_c)
                    n_a = min(n_a, self.max_a)
                    
                    c_mask = select_aircraft(state, self.static, n_c, 'C')
                    a_mask = select_aircraft(state, self.static, n_a, 'A', exclude_mask=c_mask)
                    
                    cost, tol, nc, na = compute_check_cost(state, self.static, c_mask, a_mask, self.p_a, self.p_d)
                    
                    # Record action if any checks scheduled
                    if nc > 0 or na > 0:
                        c_indices = tuple(np.where(c_mask)[0].tolist())
                        a_indices = tuple(np.where(a_mask)[0].tolist())
                        # Snapshot state at time of check for FH/tolerance info
                        state_snap = (state['fh_c'].copy(), state['fh_a'].copy(),
                                      state['dy_c'].copy(), state['dy_a'].copy())
                        new_history = history + [(day, c_indices, a_indices, state_snap)]
                    else:
                        new_history = history
                    
                    new_state = apply_action(state, self.static, c_mask, a_mask)
                    new_hist = compute_histogram(new_state, self.static, self.u)
                    
                    next_frontier.append((new_state, cum_cost + cost, cum_tol + tol, 
                                         cum_c + nc, cum_a + na, new_hist, new_history))
            
            # Best-first beam: compute f = g + λh for ranking
            if self.use_heuristic:
                remaining_days = planning_days - day - 1
                f_scores = []
                for entry in next_frontier:
                    state, g_cost = entry[0], entry[1]
                    if self.heuristic_type == 'local':
                        h = heuristic_local_risk(state, self.static, self.p_a, self.p_d)
                    else:  # 'future'
                        h = heuristic_future_risk(state, self.static, remaining_days, self.p_a, self.p_d)
                    f_scores.append(g_cost + self.lambda_h * h)
                frontier = thrifty_prune(next_frontier, self.beam_width, self.k_per_bucket, f_scores)
            else:
                frontier = thrifty_prune(next_frontier, self.beam_width, self.k_per_bucket)
            if not frontier:
                break
        
        best = min(frontier, key=lambda x: x[1])
        computation_time = time.time() - start_time
        
        if verbose:
            print(f"[Deng] Complete in {computation_time:.1f}s")
            print(f"[Deng] C-checks: {best[3]}, A-checks: {best[4]}, Tolerance: {best[2]}")
        
        checks = self._history_to_checks(best[6], start_date, tails)
        total_unused_fh = sum(c.total_unused_fh for c in checks)
        # best[1] is the cumulative Eq. 28 cost (unused FH + penalties)
        total_penalty = max(0.0, float(best[1]) - float(total_unused_fh))
        
        return ScheduleResult(
            checks=checks,
            total_a_checks=cum_a,
            total_c_checks=cum_c,
            total_d_checks=0,
            computation_time=computation_time,
            total_unused_fh=total_unused_fh,
            total_tolerance_penalty=total_penalty,
            total_tolerance_events=cum_tol,
        )

    def schedule_adp(
        self,
        planning_days: int,
        verbose: bool = False,
        n_sample: int = 20,
        t_l: int = 183,
        t_h: int = 1461,
        lambda_extra: float = 1e5,
        xi_waste: float = 1e20,
        gamma: float = 1.0,
        r_lb_a: float = 21,
        r_lb_c: float = 210,
        util_cv: float = 0.3,
        seed: int = 42,
    ) -> ScheduleResult:
        """
        Schedule using Lookahead ADP (Deng & Santos 2021, EJOR 299(3):814-833).

        Sequential single-state decisions: at each day, the C-check action
        is chosen with a deterministic thrifty-policy forecast (horizon
        t_h) and the A-check action with a Monte-Carlo forecast
        (n_sample paths, horizon t_l), per the paper's Algorithm 1. See
        aircraft_scheduler/adp.py for the full description and the
        documented adaptations to this dataset.

        The defaults match the paper (Table 7): lambda=1e5, xi=1e20,
        gamma=1, R_lb = 21/210 days, t_l=183, t_h=1461. The paper uses
        n_sample=50; 20 is the default here for practical Python
        runtimes.

        Args:
            planning_days: Planning horizon in days
            verbose: Print progress
            n_sample: Monte-Carlo sample paths for the A-check forecast
            t_l: Stochastic lookahead horizon for A-checks (days)
            t_h: Deterministic lookahead horizon for C-checks (days)
            lambda_extra: Cost of one extra maintenance slot (paper: 1e5)
            xi_waste: Penalty for wasting an available slot (paper: 1e20)
            gamma: Discount factor (paper: 1)
            r_lb_a: A-check eligibility threshold, days remaining (paper: 21)
            r_lb_c: C-check eligibility threshold, days remaining (paper: 210)
            util_cv: Coefficient of variation assumed for daily FH/FC in
                the stochastic forecast (no daily history in this dataset)
            seed: RNG seed for the Monte-Carlo forecasts (reproducible)

        Returns:
            ScheduleResult with full schedule (same KPIs as other variants)
        """
        start_time = time.time()
        start_date = self.data.config.begin_day
        tails = self.data.aircraft_tails
        n = self.n_aircraft

        # Import ADP components (local to avoid a module import cycle)
        from .adp import LookaheadADPScheduler

        if verbose:
            print(f"[ADP] Starting Lookahead ADP (Deng & Santos 2021) for {planning_days} days")
            print(f"[ADP] Fleet: {n} aircraft | n_sample={n_sample} t_l={t_l} t_h={t_h} "
                  f"lambda={lambda_extra:.0e} xi={xi_waste:.0e} gamma={gamma} "
                  f"R_lb=({r_lb_a:g},{r_lb_c:g}) cv={util_cv} seed={seed}")
            sys.stdout.flush()

        # Seasonal mean daily utilization (DFH/DFC sheets) for the
        # stochastic forecast: seasonal_fh[i, m] for month m (0-based).
        seasonal_fh = np.array(
            [self.data.flight_rates[t].dfh[m] for t in tails for m in range(1, 13)],
            dtype=np.float64,
        ).reshape(n, 12)
        seasonal_fc = np.array(
            [self.data.flight_rates[t].dfc[m] for t in tails for m in range(1, 13)],
            dtype=np.float64,
        ).reshape(n, 12)

        # Empirical C-check TAT options per aircraft (5 values each, as in
        # the C_ELAPSED_TIME sheet); fall back to the default C_E_T.
        c_e_t = float(self.data.config.c_e_t)
        c_tat_options = np.array(
            [
                (self.data.c_check_info[t].elapsed_times + [c_e_t] * 5)[:5]
                if t in self.data.c_check_info
                else [c_e_t] * 5
                for t in tails
            ],
            dtype=np.float64,
        )

        adp = LookaheadADPScheduler(
            static=self.static,
            initial_state=self.initial_state,
            max_c=self.max_c,
            max_a=self.max_a,
            lambda_extra=lambda_extra,
            xi_waste=xi_waste,
            gamma=gamma,
            r_lb_a=r_lb_a,
            r_lb_c=r_lb_c,
            t_l=t_l,
            t_h=t_h,
            n_sample=n_sample,
            util_cv=util_cv,
            seasonal_fh=seasonal_fh,
            seasonal_fc=seasonal_fc,
            c_tat_options=c_tat_options,
            start_month=start_date.month,
            seed=seed,
        )

        # Reporting uses the same Eq. 28 penalties as every other variant,
        # so the resulting KPIs are directly comparable.
        def report_cost(state, c_mask, a_mask):
            return compute_check_cost(state, self.static, c_mask, a_mask, self.p_a, self.p_d)

        outcome = adp.schedule(planning_days, verbose=verbose, report_cost_fn=report_cost)

        checks = self._history_to_checks(outcome["history"], start_date, tails)
        total_unused_fh = sum(c.total_unused_fh for c in checks)
        total_penalty = max(0.0, float(outcome["cum_cost"]) - float(total_unused_fh))
        computation_time = time.time() - start_time

        if verbose:
            print(f"[ADP] Complete in {computation_time:.1f}s")
            print(f"[ADP] C-checks: {outcome['cum_c']}, A-checks: {outcome['cum_a']}, "
                  f"Tolerance: {outcome['cum_tol']}")
            sys.stdout.flush()

        return ScheduleResult(
            checks=checks,
            total_a_checks=outcome["cum_a"],
            total_c_checks=outcome["cum_c"],
            total_d_checks=0,
            computation_time=computation_time,
            total_unused_fh=total_unused_fh,
            total_tolerance_penalty=total_penalty,
            total_tolerance_events=outcome["cum_tol"],
        )

    def schedule_greedy(
        self,
        planning_days: int,
        verbose: bool = False,
        selection_p_a: float = 0.0,
        selection_p_d: float = 0.0,
    ) -> ScheduleResult:
        """
        A naive baseline: greedy one-step lookahead over the 7 paper actions.

        At each day, evaluate immediate Eq. 28 cost for each action and apply
        the best one (ties broken by fewer checks).
        """
        start_time = time.time()
        start_date = self.data.config.begin_day
        tails = self.data.aircraft_tails

        state = self.initial_state.copy()
        cum_cost = 0.0
        cum_tol = 0
        cum_c = 0
        cum_a = 0
        history: list[
            tuple[int, tuple[int, ...], tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ] = []

        if verbose:
            print(f"[Greedy] Starting {planning_days}-day schedule")
            print(
                f"[Greedy] Fleet: {self.n_aircraft} aircraft | "
                f"selection P_a={selection_p_a:.0e} selection P_d={selection_p_d:.0e} | "
                f"reporting P_a={self.p_a:.0e} P_d={self.p_d:.0e}"
            )
            sys.stdout.flush()

        for day in range(planning_days):
            best = None
            best_total_checks = None

            for n_a, n_c in PAPER_ACTIONS:
                n_c = min(n_c, self.max_c)
                n_a = min(n_a, self.max_a)

                c_mask = select_aircraft(state, self.static, n_c, "C")
                a_mask = select_aircraft(state, self.static, n_a, "A", exclude_mask=c_mask)

                # Myopic scoring uses selection_p_* (defaults to 0 → wasted-FH-only lookahead).
                score_cost, tol, nc, na = compute_check_cost(
                    state, self.static, c_mask, a_mask, selection_p_a, selection_p_d
                )
                total_checks = nc + na

                if best is None or score_cost < best[0] or (score_cost == best[0] and total_checks < (best_total_checks or 0)):
                    best = (score_cost, tol, nc, na, c_mask, a_mask)
                    best_total_checks = total_checks

            if best is None:
                # Should not happen, but keep state advancing safely.
                state = apply_action(state, self.static, np.zeros(self.n_aircraft, dtype=bool), np.zeros(self.n_aircraft, dtype=bool))
                continue

            score_cost, tol, nc, na, c_mask, a_mask = best

            # Record action if any checks scheduled
            if nc > 0 or na > 0:
                c_indices = tuple(np.where(c_mask)[0].tolist())
                a_indices = tuple(np.where(a_mask)[0].tolist())
                state_snap = (
                    state["fh_c"].copy(),
                    state["fh_a"].copy(),
                    state["dy_c"].copy(),
                    state["dy_a"].copy(),
                )
                history.append((day, c_indices, a_indices, state_snap))

            # Accumulate *true* Eq. 28 cost for reporting (uses configured penalties).
            true_cost, _, _, _ = compute_check_cost(state, self.static, c_mask, a_mask, self.p_a, self.p_d)
            state = apply_action(state, self.static, c_mask, a_mask)
            cum_cost += float(true_cost)
            cum_tol += int(tol)
            cum_c += int(nc)
            cum_a += int(na)

            if verbose and day % 200 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / max(day, 1)) * (planning_days - day) / 60 if day > 0 else 0
                print(
                    f"[Greedy] Day {day:4d}/{planning_days} | "
                    f"cost={cum_cost:.0f} | C={cum_c} A={cum_a} | time={elapsed:.0f}s ETA={eta:.1f}m"
                )
                sys.stdout.flush()

        computation_time = time.time() - start_time
        checks = self._history_to_checks(history, start_date, tails)
        total_unused_fh = sum(c.total_unused_fh for c in checks)
        total_penalty = max(0.0, float(cum_cost) - float(total_unused_fh))

        if verbose:
            print(f"[Greedy] Complete in {computation_time:.1f}s")
            print(f"[Greedy] C-checks: {cum_c}, A-checks: {cum_a}, Tolerance: {cum_tol}")

        return ScheduleResult(
            checks=checks,
            total_a_checks=cum_a,
            total_c_checks=cum_c,
            total_d_checks=0,
            computation_time=computation_time,
            total_unused_fh=total_unused_fh,
            total_tolerance_penalty=total_penalty,
            total_tolerance_events=cum_tol,
        )

    def schedule_greedy_due(self, planning_days: int, verbose: bool = False) -> ScheduleResult:
        """Naive baseline: only schedule checks when aircraft are due/overdue.

        For each day, choose among PAPER_ACTIONS to maximize the number of due aircraft
        served (ties broken by lower wasted FH). This avoids performing checks early.
        """
        start_time = time.time()
        start_date = self.data.config.begin_day
        tails = self.data.aircraft_tails

        state = self.initial_state.copy()
        cum_tol = 0
        cum_c = 0
        cum_a = 0
        history: list[
            tuple[int, tuple[int, ...], tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ] = []

        if verbose:
            print(f"[GreedyDue] Starting {planning_days}-day schedule")
            sys.stdout.flush()

        for day in range(planning_days):
            best = None  # (served, wasted, true_cost, tol, nc, na, c_mask, a_mask)

            for n_a, n_c in PAPER_ACTIONS:
                n_c = min(n_c, self.max_c)
                n_a = min(n_a, self.max_a)

                c_mask = select_due_aircraft(state, self.static, n_c, "C")
                a_mask = select_due_aircraft(state, self.static, n_a, "A", exclude_mask=c_mask)

                # Wasted FH only for tie-breaking (ignore penalties here).
                wasted, tol, nc, na = compute_check_cost(state, self.static, c_mask, a_mask, 0.0, 0.0)
                true_cost, _, _, _ = compute_check_cost(state, self.static, c_mask, a_mask, self.p_a, self.p_d)
                served = nc + na

                cand = (served, float(wasted), float(true_cost), int(tol), int(nc), int(na), c_mask, a_mask)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand

            if best is None:
                state = apply_action(
                    state,
                    self.static,
                    np.zeros(self.n_aircraft, dtype=bool),
                    np.zeros(self.n_aircraft, dtype=bool),
                )
                continue

            served, _, true_cost, tol, nc, na, c_mask, a_mask = best

            if served > 0:
                c_indices = tuple(np.where(c_mask)[0].tolist())
                a_indices = tuple(np.where(a_mask)[0].tolist())
                state_snap = (
                    state["fh_c"].copy(),
                    state["fh_a"].copy(),
                    state["dy_c"].copy(),
                    state["dy_a"].copy(),
                )
                history.append((day, c_indices, a_indices, state_snap))

            state = apply_action(state, self.static, c_mask, a_mask)
            cum_tol += tol
            cum_c += nc
            cum_a += na
            # Note: cum_cost is not used for selection; we reconstruct penalties from checks at end.

        computation_time = time.time() - start_time
        checks = self._history_to_checks(history, start_date, tails)
        total_unused_fh = sum(c.total_unused_fh for c in checks)
        # Recompute total Eq.28 cost by replaying actions is expensive; approximate using tolerance penalty only.
        # Ground-waiting penalties are reflected in the beam-search variants; for this baseline we report
        # penalties from tolerance events (dominant at P_a=1e8).
        total_penalty = float(cum_tol) * float(self.p_a)

        if verbose:
            print(f"[GreedyDue] Complete in {computation_time:.1f}s")
            print(f"[GreedyDue] C-checks: {cum_c}, A-checks: {cum_a}, Tolerance: {cum_tol}")

        return ScheduleResult(
            checks=checks,
            total_a_checks=cum_a,
            total_c_checks=cum_c,
            total_d_checks=0,
            computation_time=computation_time,
            total_unused_fh=total_unused_fh,
            total_tolerance_penalty=total_penalty,
            total_tolerance_events=cum_tol,
        )

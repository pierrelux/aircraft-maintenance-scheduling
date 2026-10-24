"""
Lookahead Approximate Dynamic Programming (ADP) for Aircraft Maintenance
Check Scheduling.

Faithful implementation of:
    Deng, Q., Santos, B.F. (2022). "Lookahead Approximate Dynamic
    Programming for Stochastic Aircraft Maintenance Check Scheduling
    Optimization." European Journal of Operational Research, 299(3),
    814-833.  https://doi.org/10.1016/j.ejor.2021.09.019

Algorithm (paper's Algorithm 1) — a *sequential, single-state* policy.
There is no beam search and no learned value function: at each day t the
one-step optimal action is found by looking ahead with a reference
(thrifty) policy, combining a deterministic forecast (for C-checks) and a
stochastic Monte-Carlo forecast (for A-checks).

  Step 1 — C-check decision (deterministic forecast, eqs. 37-40):
      For each candidate daily action x_t:
        * apply x_t to get the post-decision state s_hat_t
        * run the thrifty reference policy from s_hat_t over
          [t+1, t+t_h] using the *mean* daily utilization, counting the
          extra maintenance slots g_k that would be created
        * V^(1) = sum_k [ lambda * g_k + xi * waste_k ]        (eq. 38)
        * score = C_t(s_t, x_t) + gamma * V^(1)
      Keep the winning C-check action x*_C.

  Step 2 — A-check decision (stochastic forecast, eqs. 42-46):
      With x*_C fixed, for each candidate A-count n_a:
        * apply (n_a, x*_C) to get s_hat_t
        * run n_sample Monte-Carlo rollouts over [t+1, t+t_l]; each
          rollout samples daily FH/FC from N(mu_i, (cv*mu_i)^2) (with
          seasonal mu_i from the DFH/DFC sheets) and C-check elapsed
          times from the aircraft's empirical TAT list, and runs the
          thrifty policy, counting extra slots
        * G_k = mean extra slots of type k over the sample paths (eq. 44)
        * V^(2) = sum_k [ lambda * G_k + xi * waste_k ]         (eq. 45)
        * score = C_t + gamma * V^(2)
      Keep the winning A-check action; apply (n_a*, x*_C).

  Then the *actual* state advances one day (deterministic mean-rate
  transition, identical to every other variant in this repo) and the
  loop repeats.

Immediate cost C_t (eq. 6):
    sum_i [ unused FH of checks started on i ] + lambda * (extra slots),
where a C-check pays only its own unused C-FH (the merged A-FH is not
charged, since the A-mask can never include the C-checked aircraft).

The xi "waste" term (eqs. 38/45) penalizes leaving a type-k slot idle on
day t when (a) some aircraft is eligible for type k (expected remaining
utilization R_k <= R^k_lb), (b) the lookahead predicts extra type-k
slots in the future, and (c) no type-k check is done today.

Reference (thrifty) policy (paper sec. 4.5.2, rules 4.4 i-iv):
    Every day, for each check type k, check the earliest-deadline-first
    eligible aircraft (expected remaining utilization R_k <= R^k_lb) up
    to hangar capacity. C-checks occupy a C hangar for the full TAT, so
    in-flight C-checks reduce same-day C capacity; A-checks take one
    day. A check performed while the aircraft is already past a usage
    limit counts as one extra slot.

Adaptations to this codebase (documented deviations from the paper):
  * The dataset has no per-day historical utilization series, so the
    stochastic forecast assumes sigma = util_cv * mu (default 0.3) around
    the *seasonal* monthly mean mu_i (DFH/DFC sheets). The paper
    estimated mu/sigma from historical daily data.
  * The paper's extra slot (eq. 16) is a hangar-capacity shortfall over
    the check's TAT window; here it is counted as a check performed while
    the aircraft is past at least one usage limit (this repo's
    "tolerance event"). The reference simulation enforces in-flight C
    capacity, so capacity-induced waits surface as extra slots.
  * The actual day-to-day rollout uses the deterministic mean-rate
    transition (apply_action) exactly like the other variants, so the
    produced schedule is directly comparable in run_variant_experiments.py.
    Monte-Carlo sampling is used only inside the lookahead (forecasts).
  * The paper uses n_sample = 50; the default here is 20 to keep a full
    4-year run practical in pure Python. Pass n_sample=50 for the paper's
    exact setting.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from .scheduler import apply_action

# =============================================================================
# PAPER DEFAULTS (Table 7 and section 5.3)
# =============================================================================

LAMBDA_EXTRA_SLOT = 1e5      # lambda: cost of creating one extra slot
XI_SLOT_WASTE = 1e20         # xi: penalty for wasting an available slot
GAMMA = 1.0                  # discount factor (paper: 1)
R_LB_A = 21                  # A-check eligibility threshold (days remaining)
R_LB_C = 210                 # C-check eligibility threshold (days remaining)
T_LOOKAHEAD_A = 183          # t_l: stochastic lookahead for A-checks (days)
T_LOOKAHEAD_C = 1461         # t_h: deterministic lookahead for C-checks (days)
N_SAMPLE_DEFAULT = 20        # paper uses 50; 20 keeps Python runtimes sane
UTIL_CV = 0.3                # assumed coefficient of variation for daily FH/FC
_SEED_DEFAULT = 42

_INF = float("inf")


def _sgn(x: float) -> float:
    """Sign function (paper eq. 39)."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


# =============================================================================
# EXPECTED REMAINING UTILIZATION (paper eqs. 31-34)
# =============================================================================

def remaining_utilization_days(
    dy_c: np.ndarray, fh_c: np.ndarray, fc_c: np.ndarray,
    dy_a: np.ndarray, fh_a: np.ndarray, fc_a: np.ndarray,
    ci_c: dict, ci_a: dict, daily_fh: np.ndarray, daily_fc: np.ndarray,
    active: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expected remaining utilization R (days) per aircraft, per check type.

    R = min(days to DY limit, days to FH limit, days to FC limit), using
    mean daily utilization (paper eqs. 31-34). Grounded aircraft return
    +inf (not schedulable).
    """
    rem_c = np.minimum(
        ci_c["dy"] - dy_c,
        np.minimum(
            (ci_c["fh"] - fh_c) / np.maximum(daily_fh, 0.1),
            (ci_c["fc"] - fc_c) / np.maximum(daily_fc, 0.1),
        ),
    )
    rem_a = np.minimum(
        ci_a["dy"] - dy_a,
        np.minimum(
            (ci_a["fh"] - fh_a) / np.maximum(daily_fh, 0.1),
            (ci_a["fc"] - fc_a) / np.maximum(daily_fc, 0.1),
        ),
    )
    rem_c = np.where(active, rem_c, _INF)
    rem_a = np.where(active, rem_a, _INF)
    return rem_c, rem_a


# =============================================================================
# THRIFTY REFERENCE POLICY ROLLOUT (paper sec. 4.5.2)
# =============================================================================

def thrifty_rollout(
    state: dict,
    ci: dict,
    daily_fh: np.ndarray,
    daily_fc: np.ndarray,
    c_tat: np.ndarray,
    days: int,
    max_c: int,
    max_a: int,
    r_lb_c: float,
    r_lb_a: float,
    rng: Optional[np.random.Generator] = None,
    seasonal_fh: Optional[np.ndarray] = None,
    seasonal_fc: Optional[np.ndarray] = None,
    util_cv: float = UTIL_CV,
    start_month: int = 9,
    c_tat_options: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Run the thrifty reference policy for `days` days from `state`.

    Returns (extra_c, extra_a): extra slots (checks performed while the
    aircraft was past at least one usage limit) of each type in the window.

    With rng=None the *deterministic* forecast is used (mean daily
    utilization, mean TAT — paper step 1). With an rng, daily FH/FC are
    sampled from N(mu_i(month), (util_cv*mu_i(month))^2) and C-check TATs
    are sampled from each aircraft's empirical list (paper step 2).
    """
    if days <= 0:
        return 0, 0

    n = len(state["dy_c"])
    dy_c = state["dy_c"].astype(np.float64).copy()
    fh_c = state["fh_c"].astype(np.float64).copy()
    fc_c = state["fc_c"].astype(np.float64).copy()
    dy_a = state["dy_a"].astype(np.float64).copy()
    fh_a = state["fh_a"].astype(np.float64).copy()
    fc_a = state["fc_a"].astype(np.float64).copy()
    grounded = state["grounded_days"].astype(np.float64).copy()

    extra_c = 0
    extra_a = 0

    for t in range(days):
        active = grounded <= 0

        rem_c, rem_a = remaining_utilization_days(
            dy_c, fh_c, fc_c, dy_a, fh_a, fc_a,
            ci["c"], ci["a"], daily_fh, daily_fc, active,
        )

        # -- thrifty selection: earliest deadline first among eligible ----
        # C capacity: hangars minus C-checks currently in flight (TAT days)
        in_flight_c = int(np.sum(grounded > 0))
        cap_c = max(0, max_c - in_flight_c)

        c_rank = np.argsort(rem_c, kind="stable")
        k_c = min(cap_c, int(np.sum(rem_c <= r_lb_c)))
        c_mask = np.zeros(n, dtype=bool)
        if k_c > 0:
            c_mask[c_rank[:k_c]] = True

        a_rank = np.argsort(np.where(c_mask, _INF, rem_a), kind="stable")
        a_elig = (rem_a <= r_lb_a) & ~c_mask
        k_a = min(max_a, int(np.sum(a_elig)))
        a_mask = np.zeros(n, dtype=bool)
        if k_a > 0:
            a_mask[a_rank[:k_a]] = True

        # -- extra slots: check performed while already past a limit -----
        c_over = (dy_c > ci["c"]["dy"]) | (fh_c > ci["c"]["fh"]) | (fc_c > ci["c"]["fc"])
        a_over = (dy_a > ci["a"]["dy"]) | (fh_a > ci["a"]["fh"]) | (fc_a > ci["a"]["fc"])
        extra_c += int(np.sum(c_mask & c_over))
        extra_a += int(np.sum(a_mask & a_over))

        # -- apply checks --------------------------------------------------
        if k_c > 0 or k_a > 0:
            fh_c = np.where(c_mask, 0.0, fh_c)
            fc_c = np.where(c_mask, 0.0, fc_c)
            dy_c = np.where(c_mask, 0.0, dy_c)
            fh_a = np.where(c_mask, 0.0, fh_a)
            fc_a = np.where(c_mask, 0.0, fc_a)
            dy_a = np.where(c_mask, 0.0, dy_a)
            a_only = a_mask & ~c_mask
            if k_a > 0:
                fh_a = np.where(a_only, 0.0, fh_a)
                fc_a = np.where(a_only, 0.0, fc_a)
                dy_a = np.where(a_only, 0.0, dy_a)
            if k_c > 0:
                if c_tat_options is not None and rng is not None:
                    sel_idx = np.flatnonzero(c_mask)
                    drawn = rng.integers(0, c_tat_options.shape[1], size=len(sel_idx))
                    tats = c_tat_options[sel_idx, drawn]
                    grounded = grounded.copy()
                    grounded[sel_idx] = tats
                else:
                    grounded = np.where(c_mask, c_tat, grounded)

        # -- advance one day ------------------------------------------------
        grounded = np.maximum(grounded - 1, 0)
        active = grounded <= 0
        if rng is not None and seasonal_fh is not None:
            month = (start_month - 1 + t) % 12
            mu_f = seasonal_fh[:, month]
            mu_c = seasonal_fc[:, month]
            fh_c = fh_c + active * rng.normal(mu_f, util_cv * mu_f).clip(0.0)
            fc_c = fc_c + active * rng.normal(mu_c, util_cv * mu_c).clip(0.0)
            fh_a = fh_a + active * rng.normal(mu_f, util_cv * mu_f).clip(0.0)
            fc_a = fc_a + active * rng.normal(mu_c, util_cv * mu_c).clip(0.0)
        else:
            fh_c = fh_c + active * daily_fh
            fc_c = fc_c + active * daily_fc
            fh_a = fh_a + active * daily_fh
            fc_a = fc_a + active * daily_fc
        dy_c = dy_c + active
        dy_a = dy_a + active

        # -- early exit (deterministic only): nothing can reach a limit in
        #    the remaining window. In stochastic mode FH is sampled and can
        #    spike above the mean, so the deterministic rem estimate is not a
        #    safe lower bound; always run the full (short) stochastic window.
        if rng is None and t < days - 1:
            rem_c2, rem_a2 = remaining_utilization_days(
                dy_c, fh_c, fc_c, dy_a, fh_a, fc_a,
                ci["c"], ci["a"], daily_fh, daily_fc, active,
            )
            left = days - t - 1
            min_rem = np.min(np.minimum(rem_c2, rem_a2))
            if min_rem > left:
                break

    return extra_c, extra_a


# =============================================================================
# LOOKAHEAD ADP SCHEDULER (paper Algorithm 1)
# =============================================================================

class LookaheadADPScheduler:
    """Sequential lookahead ADP policy (Deng & Santos 2021).

    One state per day; at each day the one-step optimal action is chosen
    by a hybrid lookahead (deterministic forecast for C-checks, Monte
    Carlo forecast for A-checks) as described in the module docstring.
    """

    def __init__(
        self,
        static: dict,
        initial_state: dict,
        max_c: int,
        max_a: int,
        lambda_extra: float = LAMBDA_EXTRA_SLOT,
        xi_waste: float = XI_SLOT_WASTE,
        gamma: float = GAMMA,
        r_lb_a: float = R_LB_A,
        r_lb_c: float = R_LB_C,
        t_l: int = T_LOOKAHEAD_A,
        t_h: int = T_LOOKAHEAD_C,
        n_sample: int = N_SAMPLE_DEFAULT,
        util_cv: float = UTIL_CV,
        seasonal_fh: Optional[np.ndarray] = None,
        seasonal_fc: Optional[np.ndarray] = None,
        c_tat_options: Optional[np.ndarray] = None,
        start_month: int = 9,
        seed: int = _SEED_DEFAULT,
    ):
        self.static = static
        self.initial_state = initial_state
        self.max_c = max_c
        self.max_a = max_a
        self.lambda_extra = lambda_extra
        self.xi_waste = xi_waste
        self.gamma = gamma
        self.r_lb_a = r_lb_a
        self.r_lb_c = r_lb_c
        self.t_l = t_l
        self.t_h = t_h
        self.n_sample = n_sample
        self.util_cv = util_cv
        self.seasonal_fh = seasonal_fh
        self.seasonal_fc = seasonal_fc
        self.c_tat_options = c_tat_options
        self.start_month = start_month
        self.rng = np.random.default_rng(seed)

        n = len(static["daily_fh"])
        self._ci = {
            "c": {
                "dy": static["c_ci_dy"].astype(np.float64),
                "fh": static["c_ci_fh"].astype(np.float64),
                "fc": static["c_ci_fc"].astype(np.float64),
            },
            "a": {
                "dy": static["a_ci_dy"].astype(np.float64),
                "fh": static["a_ci_fh"].astype(np.float64),
                "fc": static["a_ci_fc"].astype(np.float64),
            },
        }
        self._daily_fh = static["daily_fh"].astype(np.float64)
        self._daily_fc = static["daily_fc"].astype(np.float64)
        self._c_tat = static["c_tat"].astype(np.float64)
        self._n = n

    # ------------------------------------------------------------------
    # Action construction (priority ordering, paper sec. 4.3, rule 4.4 i)
    # ------------------------------------------------------------------

    def _action_masks(
        self,
        state: dict,
        n_c: int,
        n_a: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Masks for a candidate (n_a, n_c) action under the priority rule.

        Only aircraft with expected remaining utilization <= R_lb are
        eligible (basic scheduling rule 4.4 i). Returns
        (c_mask, a_mask, rem_c_min, rem_a_min).
        """
        active = state["grounded_days"] <= 0
        rem_c, rem_a = remaining_utilization_days(
            state["dy_c"], state["fh_c"], state["fc_c"],
            state["dy_a"], state["fh_a"], state["fc_a"],
            self._ci["c"], self._ci["a"],
            self._daily_fh, self._daily_fc, active,
        )
        c_rank = np.argsort(rem_c, kind="stable")
        k_c = min(n_c, self.max_c, int(np.sum(rem_c <= self.r_lb_c)))
        c_mask = np.zeros(self._n, dtype=bool)
        if k_c > 0:
            c_mask[c_rank[:k_c]] = True

        a_rank = np.argsort(np.where(c_mask, _INF, rem_a), kind="stable")
        k_a = min(n_a, self.max_a, int(np.sum((rem_a <= self.r_lb_a) & ~c_mask)))
        a_mask = np.zeros(self._n, dtype=bool)
        if k_a > 0:
            a_mask[a_rank[:k_a]] = True

        rem_c_min = float(np.min(rem_c))
        rem_a_min = float(np.min(rem_a))
        return c_mask, a_mask, rem_c_min, rem_a_min

    def _candidate_actions(self, state: dict) -> list:
        """Distinct (c_mask, a_mask) candidate actions for the day.

        Enumerates n_c in 0..max_c and n_a in 0..max_a, deduplicated by
        mask identity (counts beyond the number of eligible aircraft
        collapse to the same action).
        """
        seen = {}
        candidates = []
        for n_c in range(self.max_c + 1):
            for n_a in range(self.max_a + 1):
                c_mask, a_mask, r_c, r_a = self._action_masks(state, n_c, n_a)
                key = (c_mask.tobytes(), a_mask.tobytes())
                if key in seen:
                    continue
                seen[key] = True
                candidates.append((c_mask, a_mask, r_c, r_a))
        return candidates

    # ------------------------------------------------------------------
    # Costs (eqs. 6 and 38/45)
    # ------------------------------------------------------------------

    def _immediate_cost(self, state: dict, c_mask: np.ndarray, a_mask: np.ndarray) -> float:
        """C_t(s_t, x_t): unused FH of checks started today + lambda * extras (eq. 6).

        A C-check pays only its own unused C-FH: the unused A-FH of the
        aircraft whose A-check is merged into the C-check is not charged,
        because the A-mask can never include the C-checked aircraft --
        charging it would systematically defer C-checks past their interval.
        """
        ci_c, ci_a = self._ci["c"], self._ci["a"]
        unused_c = np.maximum(ci_c["fh"] - state["fh_c"], 0.0)
        unused_a = np.maximum(ci_a["fh"] - state["fh_a"], 0.0)
        unused = float(np.sum(c_mask * unused_c) + np.sum(a_mask * unused_a))

        c_over = (
            (state["dy_c"] > ci_c["dy"]) | (state["fh_c"] > ci_c["fh"]) | (state["fc_c"] > ci_c["fc"])
        )
        a_over = (
            (state["dy_a"] > ci_a["dy"]) | (state["fh_a"] > ci_a["fh"]) | (state["fc_a"] > ci_a["fc"])
        )
        extras = int(np.sum(c_mask & c_over)) + int(np.sum(a_mask & a_over))
        return unused + self.lambda_extra * extras

    def _waste_penalty(
        self,
        n_c: int,
        n_a: int,
        rem_c_min: float,
        rem_a_min: float,
        g_c: float,
        g_a: float,
    ) -> float:
        """xi slot-waste term of eqs. 38/45 (capacity sgn factors included)."""
        waste = 0.0
        # C: eligible aircraft exists, future extras predicted, no C-check today
        if rem_c_min <= self.r_lb_c and g_c > 0 and n_c == 0:
            # sgn( sum_{t..t+L} M^C / L - chi ) with M^C = max_c per day
            if _sgn(self.max_c - 0.0) > 0:
                waste += self.xi_waste
        # A: with M^A = max_a per day and L = 1, the capacity sgn is
        # sgn(max_a - chi); for max_a == 1 it is 1 exactly when chi == 0.
        if rem_a_min <= self.r_lb_a and g_a > 0 and n_a == 0:
            if _sgn(self.max_a - 0.0) > 0:
                waste += self.xi_waste
        return waste

    # ------------------------------------------------------------------
    # Lookaheads
    # ------------------------------------------------------------------

    def _deterministic_forecast(self, post_state: dict, horizon: int) -> Tuple[float, float]:
        """g_k: extra slots under the thrifty policy, mean utilization (eq. 37)."""
        return thrifty_rollout(
            post_state, self._ci, self._daily_fh, self._daily_fc, self._c_tat,
            horizon, self.max_c, self.max_a, self.r_lb_c, self.r_lb_a,
        )

    def _stochastic_forecast(self, post_state: dict, horizon: int, start_month: int) -> Tuple[float, float]:
        """G_k: mean extra slots over n_sample Monte-Carlo paths (eqs. 42-44).

        `start_month` is the 1-based calendar month of the first day of the
        window, so the seasonal daily means line up with the calendar.
        """
        if self.n_sample <= 0 or horizon <= 0:
            return 0.0, 0.0
        total_c = 0
        total_a = 0
        for _ in range(self.n_sample):
            ec, ea = thrifty_rollout(
                post_state, self._ci, self._daily_fh, self._daily_fc, self._c_tat,
                horizon, self.max_c, self.max_a, self.r_lb_c, self.r_lb_a,
                rng=self.rng,
                seasonal_fh=self.seasonal_fh,
                seasonal_fc=self.seasonal_fc,
                util_cv=self.util_cv,
                start_month=start_month,
                c_tat_options=self.c_tat_options,
            )
            total_c += ec
            total_a += ea
        return total_c / self.n_sample, total_a / self.n_sample

    # ------------------------------------------------------------------
    # Main loop (Algorithm 1)
    # ------------------------------------------------------------------

    def schedule(
        self,
        planning_days: int,
        verbose: bool = False,
        report_cost_fn=None,
    ) -> dict:
        """Run the sequential lookahead ADP over `planning_days` days.

        Returns dict with keys: history, cum_cost, cum_tol, cum_c, cum_a,
        where cum_cost is the actual (deterministic) Eq. 28 cost using the
        caller's reporting penalties (passed via report_cost_fn).
        """
        static = self.static
        state = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.initial_state.items()}
        history = []
        cum_cost = 0.0
        cum_tol = 0
        cum_c = 0
        cum_a = 0

        start_time = time.time()
        for day in range(planning_days):
            if verbose and day % 100 == 0:
                print(
                    f"[ADP] Day {day:4d}/{planning_days} | "
                    f"C={cum_c} A={cum_a} tol={cum_tol} | "
                    f"t={time.time() - start_time:.0f}s"
                )

            candidates = self._candidate_actions(state)

            # ---- Step 1: best C-check action (deterministic forecast) ----
            h_c = min(self.t_h, planning_days - day - 1)
            best_score = _INF
            best_c = None
            for c_mask, a_mask, r_c, r_a in candidates:
                post = apply_action(state, static, c_mask, a_mask)
                g_c, g_a = self._deterministic_forecast(post, h_c)
                waste = self._waste_penalty(
                    int(np.sum(c_mask)), int(np.sum(a_mask)), r_c, r_a, g_c, g_a
                )
                v1 = self.lambda_extra * (g_c + g_a) + waste
                score = self._immediate_cost(state, c_mask, a_mask) + self.gamma * v1
                if score < best_score:
                    best_score = score
                    best_c = (c_mask, a_mask, r_c, r_a)

            c_mask_c, a_mask_c, r_c, r_a = best_c

            # ---- Step 2: best A-check action (stochastic forecast) ------
            # C action fixed; re-score A candidates (n_a in 0..max_a),
            # deduplicated by mask identity.
            h_a = min(self.t_l, planning_days - day - 1)
            best_score_a = _INF
            best_a = None
            seen_a = set()
            n_c_fixed = int(np.sum(c_mask_c))
            # Calendar month of the first simulated day (day + 1), 1-based.
            sim_start_month = (self.start_month - 1 + day + 1) % 12 + 1
            for n_a in range(self.max_a + 1):
                c_mask, a_mask, r_c2, r_a2 = self._action_masks(state, n_c_fixed, n_a)
                key = a_mask.tobytes()
                if key in seen_a:
                    continue
                seen_a.add(key)
                post = apply_action(state, static, c_mask, a_mask)
                g_c, g_a = self._stochastic_forecast(post, h_a, sim_start_month)
                waste = self._waste_penalty(
                    int(np.sum(c_mask)), int(np.sum(a_mask)), r_c2, r_a2, g_c, g_a
                )
                v2 = self.lambda_extra * (g_c + g_a) + waste
                score = self._immediate_cost(state, c_mask, a_mask) + self.gamma * v2
                if score < best_score_a:
                    best_score_a = score
                    best_a = (c_mask, a_mask)

            c_mask, a_mask = best_a

            # ---- Apply the winning action in the actual rollout ---------
            if report_cost_fn is not None:
                cost, tol, _, _ = report_cost_fn(state, c_mask, a_mask)
                cum_cost += cost
                cum_tol += tol
            nc = int(np.sum(c_mask))
            na = int(np.sum(a_mask))
            cum_c += nc
            cum_a += na

            if nc > 0 or na > 0:
                c_indices = tuple(np.where(c_mask)[0].tolist())
                a_indices = tuple(np.where(a_mask)[0].tolist())
                state_snap = (
                    state["fh_c"].copy(), state["fh_a"].copy(),
                    state["dy_c"].copy(), state["dy_a"].copy(),
                )
                history.append((day, c_indices, a_indices, state_snap))

            state = apply_action(state, static, c_mask, a_mask)

        return {
            "history": history,
            "cum_cost": cum_cost,
            "cum_tol": cum_tol,
            "cum_c": cum_c,
            "cum_a": cum_a,
        }

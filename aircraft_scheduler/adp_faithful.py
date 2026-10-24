"""
Faithful reproduction of the Table 8 KPIs from Deng & Santos (2021).

Paper
-----
    Deng, Q., & Santos, B. F. (2022). "Lookahead approximate dynamic
    programming for stochastic aircraft maintenance check scheduling
    optimization." European Journal of Operational Research, 299(3),
    814-833.  https://doi.org/10.1016/j.ejor.2021.09.019
    (submitted / in press 2021; the file in this repo is the accepted
    2021 manuscript, S0377-2217(21)00794-3).

What Table 8 reports
--------------------
Case study 1: 25 Sep 2017 - 31 Dec 2020 (1194 days), 100 Monte-Carlo test
sample paths. For each check type (A and C) it reports, averaged over the
100 runs:

    * Mean Average FH    -- mean (over runs) of the mean FH at which the
                            fleet's checks of that type are performed
    * Mean Extra Slots   -- mean (over runs) of the number of extra slots
                            created for that check type
    * Mean Total Checks  -- mean (over runs) of the total number of checks

plus, for the fleet as a whole:
    * Mean Total Extra Slots (A + C), with its 95% confidence interval
    * Computation time per day (seconds)

...for six methods: **Airline**, **DP-based**, **Myopic**, **Thrifty**,
**ADP-D**, **ADP-DS**.

What this module implements
---------------------------
A faithful, self-contained implementation of the lookahead ADP of
Algorithm 1 of the paper:

  * expected remaining utilization R_k (eqs. 31-34),
  * priority ordering by earliest deadline (sec. 4.3),
  * the two-step one-step-optimal decision (sec. 4.6):
        Step 1 -- C-check via *deterministic* thrifty forecast (eqs. 37-40)
        Step 2 -- A-check via *stochastic* Monte-Carlo thrifty forecast
                  (eqs. 42-46)
  * the value approximations V^(1) (eq. 38) and V^(2) (eq. 45) with
    lambda (cost of one extra slot), xi (slot-waste penalty), gamma
    (discount) and the sign() factors, plus a per-check tolerance penalty
    (TOL_PENALTY) so checks are not deferred into the tolerance band
    (fidelity note 6),
  * the xi slot-waste penalty conditions (C.1)-(C.3),
  * Monte-Carlo sampling of daily FH/FC and of the C-check elapsed time
    (sec. 4.2),
  * the basic scheduling rules (4.4), including the R_k_lb eligibility
    threshold and the d_c spacing between C-check starts.

It also implements the two reference policies of sec. 4.5:

  * **Myopic** (4.5.1)  -- greedy, only serves aircraft that are due.
  * **Thrifty** (4.5.2) -- conservative, serves as soon as a slot is free.

Run it
------
    python -m aircraft_scheduler.adp_faithful                 # quick study
    python -m aircraft_scheduler.adp_faithful --table8        # paper setting
                                                            # (SLOW: 100 runs,
                                                            #  1194 days, n=50)
    python -m aircraft_scheduler.adp_faithful --methods adp_ds,thrifty --runs 20

Fidelity notes / documented deviations
--------------------------------------
1. Daily utilization. The paper samples daily FH/FC from the airline's
   *historical daily* series (per-aircraft mean mu_i and variance
   sigma_i^2). This dataset only provides the *monthly* DFH/DFC means, so
   we sample N(monthly_mean, (cv * monthly_mean)^2) with a default
   cv = 0.3 and use the monthly mean for the deterministic forecast and
   for R. Change it with --util-cv. This is the single biggest driver of
   the gap between our absolute numbers and the paper's.

2. "Extra slot". In the paper (eq. 16) an extra slot is a hangar-capacity
   shortfall over a check's elapsed-time window (rule 4.4 iv: a check forced
   beyond the normal capacity). We approximate it faithfully as a check
   performed while the aircraft is pushed *past its interval + tolerance*
   (i.e. truly overdue, so it must be grounded on an extra slot). Using the
   interval+tolerance band (A: +75 FH, C: +500 FH) means a check done right
   at the limit -- the optimum for minimizing unused FH -- is NOT counted;
   only a capacity-forced delay beyond the allowed tolerance is. This gives
   the paper's qualitative Table-8 behaviour (Thrifty ~ 0 extra, Myopic
   large, ADP-DS small).

3. "Airline" and "DP-based" columns are shown as the paper's *published*
   reference values. The deterministic DP-based schedule is the Deng et
   al. (2020) beam search (see scheduler.py); re-running it and its
   stochastic re-evaluation is out of scope for this module, so those two
   columns are labelled "paper (ref)".

4. The exact decimals in Table 8 depend on the airline's private daily
   utilization / TAT history. With the monthly means available here the
   *numbers* will differ, but the *ranking and scale* of the KPIs should
   match: ADP-DS has the fewest extra slots among the high-utilization
   methods, Thrifty has ~0 extra slots but the lowest average FH, Myopic
   has the most extra slots.

5. A-check capacity. This dataset sets MAX A CHECK = 1, while the paper's
   case study uses 2 A-check slots per workday. We default to the data
   value; pass --cap-a 2 (and --cap-c 3) to match the paper's operational
   capacities. A tighter A capacity inflates the A-extra-slot KPIs for
   every method (most of all the ones that do not schedule ahead).

6. Decision-cost deviations in C_t (needed to recover the paper's
   qualitative behaviour on this dataset):
   (a) A C-check does NOT pay the unused A-FH of the aircraft whose A-check
       it merges/resets. The candidate action space allows at most one
       A-check per day and never on the C-checked aircraft, so charging
       that unused A-FH to the C-action made "defer the C-check"
       systematically cheaper: C-checks drifted into (and past) the
       tolerance band until the lambda cliff at interval + tolerance.
   (b) A per-check tolerance penalty TOL_PENALTY is added to C_t and to the
       forecast value terms V^(1)/V^(2) whenever a check is performed past
       its interval (but inside the tolerance band). Without it, flying
       from the interval to interval + tolerance costs nothing, so the
       optimizer defers until the lambda cliff. The penalty (1e4) sits
       above the per-day unused-FH slope (~11 FH) and far below lambda
       (1e5), so a true extra slot remains the worst outcome. Reported
       "extra slot" KPIs keep the interval + tolerance definition.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# =============================================================================
# Paper defaults (Table 7)
# =============================================================================
R_LB_A = 21          # A-check eligibility threshold (days remaining)
R_LB_C = 210         # C-check eligibility threshold (days remaining)
LAMBDA_EXTRA = 1e5   # cost of creating one extra slot
XI_WASTE = 1e20      # penalty for wasting an available slot
TOL_PENALTY = 1e4    # decision-cost penalty per tolerance event (check done
                     # past its interval but inside the tolerance band);
                     # > per-day unused-FH slope, < LAMBDA_EXTRA
GAMMA = 1.0          # discount factor
N_SAMPLE = 50        # Monte-Carlo sample paths for the A-check forecast
T_L = 183            # stochastic lookahead for A-checks (days)
T_H = 1461           # deterministic lookahead for C-checks (days)
UTIL_CV = 0.3        # assumed coefficient of variation for daily FH/FC
N_RUNS = 100         # number of test sample paths
HORIZON_CASE1 = 1194  # 25 Sep 2017 - 31 Dec 2020
_SEED = 42

_INF = float("inf")

# Published Table 8 (case study 1: 2017-2020, 1194 days, 100 runs).
# Used only as the reference columns of the printed comparison.
TABLE8_REFERENCE = {
    "airline": {"c_avg_fh": 6646.8, "c_extra": 15.0, "c_checks": 88.0,
                "a_avg_fh": 695.0, "a_extra": 15.0, "a_checks": 750.0,
                "total_extra": 30.0, "comp_day": None,
                "ci": None},
    "dp":      {"c_avg_fh": 6785.4, "c_extra": 90.4, "c_checks": 77.0,
                "a_avg_fh": 713.3, "a_extra": 20.4, "a_checks": 727.0,
                "total_extra": 110.8, "comp_day": 0.02, "ci": "[109.41, 112.19]"},
    "myopic":  {"c_avg_fh": 7142.1, "c_extra": 368.4, "c_checks": 75.3,
                "a_avg_fh": 744.6, "a_extra": 367.3, "a_checks": 698.4,
                "total_extra": 735.7, "comp_day": 0.09, "ci": "[732.23, 739.17]"},
    "thrifty": {"c_avg_fh": 6200.6, "c_extra": 0.0, "c_checks": 83.1,
                "a_avg_fh": 573.6, "a_extra": 0.0, "a_checks": 893.6,
                "total_extra": 0.0, "comp_day": 0.05, "ci": "[0, 0]"},
    "adp_d":   {"c_avg_fh": 6849.2, "c_extra": 6.5, "c_checks": 79.2,
                "a_avg_fh": 705.9, "a_extra": 1.6, "a_checks": 733.6,
                "total_extra": 8.1, "comp_day": 0.35, "ci": "[7.32, 8.88]"},
    "adp_ds":  {"c_avg_fh": 6838.4, "c_extra": 5.7, "c_checks": 79.4,
                "a_avg_fh": 703.5, "a_extra": 0.8, "a_checks": 735.9,
                "total_extra": 6.5, "comp_day": 2.63, "ci": "[5.88, 7.12]"},
}


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class Params:
    """Static fleet parameters extracted from the dataset."""
    n: int
    ci_c_dy: np.ndarray
    ci_c_fh: np.ndarray
    ci_c_fc: np.ndarray
    ci_a_dy: np.ndarray
    ci_a_fh: np.ndarray
    ci_a_fc: np.ndarray
    tol_c_dy: np.ndarray
    tol_c_fh: np.ndarray
    tol_c_fc: np.ndarray
    tol_a_dy: np.ndarray
    tol_a_fh: np.ndarray
    tol_a_fc: np.ndarray
    mean_fh: np.ndarray          # annual-mean daily flight hours
    mean_fc: np.ndarray          # annual-mean daily flight cycles
    seasonal_fh: np.ndarray      # (n, 12) monthly daily FH
    seasonal_fc: np.ndarray      # (n, 12) monthly daily FC
    c_tat_mean: np.ndarray       # (n,) mean C-check elapsed time (days)
    c_tat_options: np.ndarray    # (n, 5) empirical C-check TAT options
    cap_c: int                   # C-check hangar capacity per day
    cap_a: int                   # A-check slots per day
    d_c: int                     # min days between C-check starts
    no_c: set                     # absolute day indices where C-checks are banned
    op_day: np.ndarray           # (n,) first operational day index
    init: Dict[str, np.ndarray]  # initial state arrays
    start_month: int              # 1-based month of day 0
    # paper hyper-parameters
    lambda_extra: float = LAMBDA_EXTRA
    xi_waste: float = XI_WASTE
    gamma: float = GAMMA
    r_lb_c: float = R_LB_C
    r_lb_a: float = R_LB_A
    util_cv: float = UTIL_CV
    tol_penalty: float = TOL_PENALTY


def build_params(data, util_cv: float = UTIL_CV, start_month: Optional[int] = None) -> Params:
    """Extract the static fleet arrays the faithful solver needs."""
    tails = data.aircraft_tails
    n = len(tails)
    begin = data.config.begin_day

    ci_c_dy = np.array([data.c_initial[t].c_ci_dy for t in tails], dtype=float)
    ci_c_fh = np.array([data.c_initial[t].c_ci_fh for t in tails], dtype=float)
    ci_c_fc = np.array([data.c_initial[t].c_ci_fc for t in tails], dtype=float)
    ci_a_dy = np.array([data.a_initial[t].a_ci_dy for t in tails], dtype=float)
    ci_a_fh = np.array([data.a_initial[t].a_ci_fh for t in tails], dtype=float)
    ci_a_fc = np.array([data.a_initial[t].a_ci_fc for t in tails], dtype=float)
    tol_c_dy = np.array([data.c_initial[t].c_tol_dy for t in tails], dtype=float)
    tol_c_fh = np.array([data.c_initial[t].c_tol_fh for t in tails], dtype=float)
    tol_c_fc = np.array([data.c_initial[t].c_tol_fc for t in tails], dtype=float)
    tol_a_dy = np.array([data.a_initial[t].a_tol_dy for t in tails], dtype=float)
    tol_a_fh = np.array([data.a_initial[t].a_tol_fh for t in tails], dtype=float)
    tol_a_fc = np.array([data.a_initial[t].a_tol_fc for t in tails], dtype=float)

    mean_fh = np.array([float(np.mean(list(data.flight_rates[t].dfh.values()))) for t in tails])
    mean_fc = np.array([float(np.mean(list(data.flight_rates[t].dfc.values()))) for t in tails])
    seasonal_fh = np.array([[data.flight_rates[t].dfh[m] for m in range(1, 13)] for t in tails], dtype=float)
    seasonal_fc = np.array([[data.flight_rates[t].dfc[m] for m in range(1, 13)] for t in tails], dtype=float)

    c_e_t = data.config.c_e_t
    c_tat_options = np.array(
        [ (data.c_check_info[t].elapsed_times + [c_e_t] * 5)[:5]
          if t in data.c_check_info else [c_e_t] * 5 for t in tails ],
        dtype=float,
    )
    c_tat_mean = c_tat_options.mean(axis=1)

    no_c: set = set()
    for r in data.c_not_allowed:
        a = (r.begin - begin).days
        b = (r.end - begin).days
        for dd in range(max(a, 0), b + 1):
            no_c.add(dd)

    op_day = np.array(
        [ max(0, (data.d_initial[t].operation - begin).days)
          if data.d_initial[t].operation else 0 for t in tails ],
        dtype=int,
    )

    init = {
        "dy_c": np.array([data.c_initial[t].dy_c for t in tails], dtype=float),
        "fh_c": np.array([data.c_initial[t].fh_c for t in tails], dtype=float),
        "fc_c": np.array([data.c_initial[t].fc_c for t in tails], dtype=float),
        "dy_a": np.array([data.a_initial[t].dy_a for t in tails], dtype=float),
        "fh_a": np.array([data.a_initial[t].fh_a for t in tails], dtype=float),
        "fc_a": np.array([data.a_initial[t].fc_a for t in tails], dtype=float),
        "c_ground": np.zeros(n, dtype=float),
    }

    return Params(
        n=n, ci_c_dy=ci_c_dy, ci_c_fh=ci_c_fh, ci_c_fc=ci_c_fc,
        ci_a_dy=ci_a_dy, ci_a_fh=ci_a_fh, ci_a_fc=ci_a_fc,
        tol_c_dy=tol_c_dy, tol_c_fh=tol_c_fh, tol_c_fc=tol_c_fc,
        tol_a_dy=tol_a_dy, tol_a_fh=tol_a_fh, tol_a_fc=tol_a_fc,
        mean_fh=mean_fh, mean_fc=mean_fc,
        seasonal_fh=seasonal_fh, seasonal_fc=seasonal_fc,
        c_tat_mean=c_tat_mean, c_tat_options=c_tat_options,
        cap_c=data.config.max_c_check, cap_a=data.config.max_a_check,
        d_c=data.config.start_day_interval, no_c=no_c, op_day=op_day,
        init=init,
        start_month=start_month or begin.month,
        util_cv=util_cv,
    )


# =============================================================================
# Low-level helpers
# =============================================================================
def _copy_state(st: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in st.items()}


def _remaining(st, P: Params, active: np.ndarray):
    """Expected remaining utilization R (eqs. 31-34), in days, per type."""
    fh = np.maximum(P.mean_fh, 0.1)
    fc = np.maximum(P.mean_fc, 0.1)
    rem_c = np.minimum(
        P.ci_c_dy - st["dy_c"],
        np.minimum((P.ci_c_fh - st["fh_c"]) / fh, (P.ci_c_fc - st["fc_c"]) / fc),
    )
    rem_a = np.minimum(
        P.ci_a_dy - st["dy_a"],
        np.minimum((P.ci_a_fh - st["fh_a"]) / fh, (P.ci_a_fc - st["fc_a"]) / fc),
    )
    rem_c = np.where(active, rem_c, _INF)
    rem_a = np.where(active, rem_a, _INF)
    return rem_c, rem_a


def _select(rem: np.ndarray, eligible: np.ndarray, k: int) -> List[int]:
    """Earliest-deadline-first: the k smallest R among the eligible."""
    if k <= 0:
        return []
    m = eligible & np.isfinite(rem)
    if not m.any():
        return []
    order = np.argsort(np.where(m, rem, _INF), kind="stable")
    out: List[int] = []
    for i in order:
        if not m[i]:
            continue
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def _rates(st, P: Params, day: int, mode: str, rng):
    """Daily utilization for `day`: deterministic mean or a sampled draw."""
    month = (P.start_month - 1 + day) % 12
    mu_f = P.seasonal_fh[:, month]
    mu_c = P.seasonal_fc[:, month]
    if mode == "stoch" and rng is not None:
        rf = np.clip(rng.normal(mu_f, P.util_cv * mu_f), 0.0, None)
        rc = np.clip(rng.normal(mu_c, P.util_cv * mu_c), 0.0, None)
    else:
        rf = mu_f
        rc = mu_c
    return rf, rc


def _tats(P: Params, c_idx, mode: str, rng) -> np.ndarray:
    """Elapsed time (TAT) for each C-check in c_idx."""
    if not c_idx:
        return np.zeros(0, dtype=float)
    ci = np.array(c_idx)
    if mode == "stoch" and rng is not None:
        j = rng.integers(0, P.c_tat_options.shape[1], size=len(ci))
        return P.c_tat_options[ci, j]
    return P.c_tat_mean[ci]


def _apply(st, P: Params, c_idx, a_idx, tatts):
    """Apply checks (reset counters, ground for C), count extra slots.

    An "extra slot" (repo convention) = a check performed while the
    aircraft is already past interval + tolerance. A "tolerance event"
    = a check performed past the interval but inside the tolerance band
    (feeds the ADP decision cost only, not the reported KPIs).
    Returns (fh_at_c_list, fh_at_a_list, extra_c, extra_a, tol_c, tol_a).
    """
    extra_c = extra_a = 0
    tol_c = tol_a = 0
    fh_c_list: List[float] = []
    fh_a_list: List[float] = []

    for i in c_idx:
        if (st["dy_c"][i] > P.ci_c_dy[i]
                or st["fh_c"][i] > P.ci_c_fh[i]
                or st["fc_c"][i] > P.ci_c_fc[i]):
            tol_c += 1
        if (st["dy_c"][i] > P.ci_c_dy[i] + P.tol_c_dy[i]
                or st["fh_c"][i] > P.ci_c_fh[i] + P.tol_c_fh[i]
                or st["fc_c"][i] > P.ci_c_fc[i] + P.tol_c_fc[i]):
            extra_c += 1
        fh_c_list.append(float(st["fh_c"][i]))

    for i in c_idx:
        st["fh_c"][i] = 0.0
        st["fc_c"][i] = 0.0
        st["dy_c"][i] = 0.0
        st["fh_a"][i] = 0.0          # A-check merged into the C-check
        st["fc_a"][i] = 0.0
        st["dy_a"][i] = 0.0

    if c_idx:
        st["c_ground"] = st["c_ground"].copy()
        st["c_ground"][np.array(c_idx)] = np.asarray(tatts, dtype=float)

    for i in a_idx:
        if (st["dy_a"][i] > P.ci_a_dy[i]
                or st["fh_a"][i] > P.ci_a_fh[i]
                or st["fc_a"][i] > P.ci_a_fc[i]):
            tol_a += 1
        if (st["dy_a"][i] > P.ci_a_dy[i] + P.tol_a_dy[i]
                or st["fh_a"][i] > P.ci_a_fh[i] + P.tol_a_fh[i]
                or st["fc_a"][i] > P.ci_a_fc[i] + P.tol_a_fc[i]):
            extra_a += 1
        fh_a_list.append(float(st["fh_a"][i]))
        st["fh_a"][i] = 0.0
        st["fc_a"][i] = 0.0
        st["dy_a"][i] = 0.0

    return fh_c_list, fh_a_list, extra_c, extra_a, tol_c, tol_a


def _advance(st, P: Params, day: int, rates_fh: np.ndarray, rates_fc: np.ndarray):
    """Advance one day: release finished C-checks, age the active aircraft."""
    st["c_ground"] = np.maximum(st["c_ground"] - 1, 0)
    active = (st["c_ground"] <= 0) & (day >= P.op_day)
    if not active.any():
        return
    st["fh_c"][active] += rates_fh[active]
    st["fc_c"][active] += rates_fc[active]
    st["fh_a"][active] += rates_fh[active]
    st["fc_a"][active] += rates_fc[active]
    st["dy_c"][active] += 1
    st["dy_a"][active] += 1


# =============================================================================
# Thrifty reference-policy rollout (the paper's lookahead / benchmarks)
# =============================================================================
def thrifty_run(st0, P: Params, days: int, mode: str, rng, day0: int,
                last_c0: int = -10 ** 9) -> tuple:
    """Run the thrifty policy for `days` days; return
    (extra_c, extra_a, tol_c, tol_a).

    mode 'det' uses mean daily utilization and mean TAT; mode 'stoch'
    samples both. This is the inner engine of the ADP forecasts (sec. 4.6)
    and of the Thrifty benchmark (4.5.2).
    """
    if days <= 0:
        return 0, 0, 0, 0
    st = _copy_state(st0)
    extra_c = extra_a = 0
    tol_c = tol_a = 0
    last_c = last_c0

    for t in range(days):
        day = day0 + t
        inprog_c = int(np.sum(st["c_ground"] > 0))
        active = (st["c_ground"] <= 0) & (day >= P.op_day)
        rem_c, rem_a = _remaining(st, P, active)

        can_c = (inprog_c < P.cap_c) and ((day - last_c) >= P.d_c) and (day not in P.no_c)
        elig_c = active & (rem_c <= P.r_lb_c)
        c_idx = _select(rem_c, elig_c, 1) if can_c else []

        elig_a = active & (rem_a <= P.r_lb_a)
        for i in c_idx:
            elig_a[i] = False
        a_idx = _select(rem_a, elig_a, P.cap_a)

        tatts = _tats(P, c_idx, mode, rng)
        _, _, ec, ea, tc, ta = _apply(st, P, c_idx, a_idx, tatts)
        extra_c += ec
        extra_a += ea
        tol_c += tc
        tol_a += ta
        if c_idx:
            last_c = day

        # early exit: nothing can become eligible within the window
        if t < days - 1:
            rem_c2, rem_a2 = _remaining(st, P, st["c_ground"] <= 0)
            if np.min(np.minimum(rem_c2, rem_a2)) > (days - t - 1):
                break

        rf, rc = _rates(st, P, day, mode, rng)
        _advance(st, P, day, rf, rc)

    return extra_c, extra_a, tol_c, tol_a


# =============================================================================
# ADP scoring terms (eqs. 6, 38, 45)
# =============================================================================
# When two candidate actions score within this many FH of each other, prefer
# the one that performs more checks. This stops the greedy per-day policy from
# "deferring on a tie" and drifting an aircraft past its tolerance band (which
# would be counted as an extra slot). With the paper's lambda (1e5 = one extra
# slot), a real future-extra-slot difference is ~1e5, far above this epsilon,
# so the lookahead still wins whenever it actually predicts a future shortfall.
TIE_EPS = 100.0


def _pick_best(best, score, n_checks, payload):
    """argmin on score with a near-tie broken in favour of more checks."""
    if best is None or score < best[0] - TIE_EPS:
        return (score, payload, n_checks)
    if abs(score - best[0]) <= TIE_EPS and n_checks > best[2]:
        return (score, payload, n_checks)
    return best


def _immediate_cost(st, P: Params, c_idx, a_idx) -> float:
    """C_t(s_t, x_t) (eq. 6): unused FH + tolerance penalty + lambda * extras.

    A C-check pays only its own (C) unused FH: the unused A-FH of the
    aircraft whose A-check is merged into the C-check is deliberately not
    charged, because the action space can never do an A-check on the same
    aircraft the same day -- charging it made "defer the C-check"
    systematically cheaper and pushed C-checks past their interval
    (fidelity note 6a). The tolerance term (note 6b) penalizes checks
    done past the interval, so deferral is bounded at the interval rather
    than running all the way to the lambda cliff at interval + tolerance.
    """
    unused = 0.0
    tol = 0
    for i in c_idx:
        unused += max(0.0, P.ci_c_fh[i] - st["fh_c"][i])
        if (st["dy_c"][i] > P.ci_c_dy[i]
                or st["fh_c"][i] > P.ci_c_fh[i]
                or st["fc_c"][i] > P.ci_c_fc[i]):
            tol += 1
    for i in a_idx:
        unused += max(0.0, P.ci_a_fh[i] - st["fh_a"][i])
        if (st["dy_a"][i] > P.ci_a_dy[i]
                or st["fh_a"][i] > P.ci_a_fh[i]
                or st["fc_a"][i] > P.ci_a_fc[i]):
            tol += 1

    extras = 0
    for i in c_idx:
        if (st["dy_c"][i] > P.ci_c_dy[i] + P.tol_c_dy[i]
                or st["fh_c"][i] > P.ci_c_fh[i] + P.tol_c_fh[i]
                or st["fc_c"][i] > P.ci_c_fc[i] + P.tol_c_fc[i]):
            extras += 1
    for i in a_idx:
        if (st["dy_a"][i] > P.ci_a_dy[i] + P.tol_a_dy[i]
                or st["fh_a"][i] > P.ci_a_fh[i] + P.tol_a_fh[i]
                or st["fc_a"][i] > P.ci_a_fc[i] + P.tol_a_fc[i]):
            extras += 1

    return unused + P.tol_penalty * tol + P.lambda_extra * extras


def _waste_penalty(n_c: int, n_a: int, rem_c: np.ndarray, rem_a: np.ndarray,
                   g_c: float, g_a: float, P: Params) -> float:
    """xi slot-waste term of eqs. 38/45 (conditions C.1-C.3).

    The paper multiplies by sign() of the remaining hangar capacity; with a
    positive daily capacity that factor is +1, so it is dropped here.
    """
    waste = 0.0
    if n_c == 0 and g_c > 0 and np.min(rem_c) <= P.r_lb_c:
        waste += P.xi_waste
    if n_a == 0 and g_a > 0 and np.min(rem_a) <= P.r_lb_a:
        waste += P.xi_waste
    return waste


def _post_dec(st, P: Params, c_idx, a_idx) -> Dict[str, np.ndarray]:
    """Post-decision state s_hat_t = S_X(s_t, x_t) (no aging yet)."""
    s = _copy_state(st)
    tatts = P.c_tat_mean[np.array(c_idx)] if c_idx else np.zeros(0)
    _apply(s, P, list(c_idx), list(a_idx), tatts)
    return s


# =============================================================================
# Decision policies
# =============================================================================
def _make_thrifty_decide(P: Params, threshold: str) -> Callable:
    """thrifty (threshold='lb') or myopic (threshold='due') daily decision."""
    def decide(st, day, last_c, inprog_c, horizon) -> tuple:
        active = (st["c_ground"] <= 0) & (day >= P.op_day)
        rem_c, rem_a = _remaining(st, P, active)
        cutoff_c = P.r_lb_c if threshold == "lb" else 0.0
        cutoff_a = P.r_lb_a if threshold == "lb" else 0.0

        can_c = (inprog_c < P.cap_c) and ((day - last_c) >= P.d_c) and (day not in P.no_c)
        elig_c = active & (rem_c <= cutoff_c)
        c_idx = _select(rem_c, elig_c, 1) if can_c else []

        elig_a = active & (rem_a <= cutoff_a)
        for i in c_idx:
            elig_a[i] = False
        a_idx = _select(rem_a, elig_a, P.cap_a)
        return c_idx, a_idx
    return decide


def _make_adp_decide(P: Params, stochastic: bool, n_sample: int, t_l: int, t_h: int,
                     f_rng: np.random.Generator) -> Callable:
    """The two-step lookahead ADP decision (Algorithm 1, sec. 4.6)."""

    def _actions(st, day, inprog_c, last_c, active, rem_c, rem_a):
        can_c = (inprog_c < P.cap_c) and ((day - last_c) >= P.d_c) and (day not in P.no_c)
        elig_c = active & (rem_c <= P.r_lb_c)
        best_c = _select(rem_c, elig_c, 1) if can_c else []

        c_opts = [()]
        if best_c:
            c_opts.append(tuple(best_c))

        def a_opts(c_tuple):
            e = active & (rem_a <= P.r_lb_a)
            for i in c_tuple:
                e[i] = False
            ba = _select(rem_a, e, 1)
            o = [()]
            if ba:
                o.append(tuple(ba))
            return o

        seen = set()
        actions = []
        for c in c_opts:
            for a in a_opts(c):
                key = (c, a)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(key)
        return actions

    def decide(st, day, last_c, inprog_c, horizon) -> tuple:
        active = (st["c_ground"] <= 0) & (day >= P.op_day)
        rem_c, rem_a = _remaining(st, P, active)
        actions = _actions(st, day, inprog_c, last_c, active, rem_c, rem_a)

        # ---- Step 1: C-check via deterministic forecast (eqs. 37-40) ----
        h_c = min(t_h, horizon - 1 - day)
        best = None
        for (c, a) in actions:
            post = _post_dec(st, P, c, a)
            gc, ga, tc, ta = thrifty_run(post, P, h_c, "det", None, day + 1,
                                 last_c0=day if c else last_c) if h_c > 0 else (0, 0, 0, 0)
            waste = _waste_penalty(len(c), len(a), rem_c, rem_a, gc, ga, P)
            v1 = P.lambda_extra * (gc + ga) + P.tol_penalty * (tc + ta) + waste
            score = _immediate_cost(st, P, c, a) + P.gamma * v1
            best = _pick_best(best, score, len(c) + len(a), (c, a))
        c_commit, _ = best[1]

        # ---- Step 2: A-check (stochastic forecast, eqs. 42-46) ----
        if stochastic:
            h_a = min(t_l, horizon - 1 - day)
            e = active & (rem_a <= P.r_lb_a)
            for i in c_commit:
                e[i] = False
            ba = _select(rem_a, e, 1)
            a_opts = [()] + ([tuple(ba)] if ba else [])

            best2 = None
            for a in a_opts:
                post = _post_dec(st, P, c_commit, a)
                if h_a > 0:
                    tot_c = tot_a = 0
                    tot_tc = tot_ta = 0
                    for _ in range(n_sample):
                        gc, ga, tc, ta = thrifty_run(post, P, h_a, "stoch", f_rng, day + 1,
                                                     last_c0=day if c_commit else last_c)
                        tot_c += gc
                        tot_a += ga
                        tot_tc += tc
                        tot_ta += ta
                    Gc, Ga = tot_c / n_sample, tot_a / n_sample
                    Tc, Ta = tot_tc / n_sample, tot_ta / n_sample
                else:
                    Gc = Ga = Tc = Ta = 0.0
                waste = _waste_penalty(len(c_commit), len(a), rem_c, rem_a, Gc, Ga, P)
                v2 = P.lambda_extra * (Gc + Ga) + P.tol_penalty * (Tc + Ta) + waste
                score = _immediate_cost(st, P, c_commit, a) + P.gamma * v2
                best2 = _pick_best(best2, score, len(c_commit) + len(a), a)
            a_commit = best2[1]
        else:
            a_commit = best[1][1]  # ADP-D: deterministic forecast only

        return list(c_commit), list(a_commit)

    return decide


def make_policy(kind: str, P: Params, *, n_sample: int, t_l: int, t_h: int,
                f_seed: int) -> Callable:
    if kind == "myopic":
        return _make_thrifty_decide(P, "due")
    if kind == "thrifty":
        return _make_thrifty_decide(P, "lb")
    if kind == "adp_d":
        return _make_adp_decide(P, stochastic=False, n_sample=n_sample, t_l=t_l,
                                t_h=t_h, f_rng=np.random.default_rng(f_seed))
    if kind == "adp_ds":
        return _make_adp_decide(P, stochastic=True, n_sample=n_sample, t_l=t_l,
                                t_h=t_h, f_rng=np.random.default_rng(f_seed))
    raise ValueError(f"unknown method: {kind}")


# =============================================================================
# Single test sample path
# =============================================================================
def run_path(policy, P: Params, horizon: int, path_rng: np.random.Generator) -> Dict:
    """Execute one Monte-Carlo sample path (a full run of the horizon)."""
    st = _copy_state(P.init)
    fh_c: List[float] = []
    fh_a: List[float] = []
    n_c = n_a = 0
    extra_c = extra_a = 0
    last_c = -10 ** 9
    t0 = time.time()

    for day in range(horizon):
        inprog_c = int(np.sum(st["c_ground"] > 0))
        c_idx, a_idx = policy(st, day, last_c, inprog_c, horizon)

        tatts = _tats(P, c_idx, "stoch", path_rng)  # actual TAT is stochastic
        fc, fa, ec, ea, _tc, _ta = _apply(st, P, c_idx, a_idx, tatts)
        fh_c.extend(fc)
        fh_a.extend(fa)
        extra_c += ec
        extra_a += ea
        n_c += len(c_idx)
        n_a += len(a_idx)
        if c_idx:
            last_c = day

        rf, rc = _rates(st, P, day, "stoch", path_rng)
        _advance(st, P, day, rf, rc)

    dt = time.time() - t0
    return {
        "avg_fh_c": float(np.mean(fh_c)) if fh_c else float("nan"),
        "avg_fh_a": float(np.mean(fh_a)) if fh_a else float("nan"),
        "n_c": n_c,
        "n_a": n_a,
        "extra_c": extra_c,
        "extra_a": extra_a,
        "total_extra": extra_c + extra_a,
        "time": dt,
    }


# =============================================================================
# KPI aggregation (the Table 8 quantities)
# =============================================================================
def _ci95(values: List[float]) -> Optional[str]:
    if len(values) < 2:
        return None
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return f"[{m - 1.96 * se:.2f}, {m + 1.96 * se:.2f}]"


def aggregate(results: List[Dict], horizon: int) -> Dict:
    """Collapse a list of per-run results into the Table 8 KPIs."""
    if not results:
        return {}

    def _mean(key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    times = [r["time"] / horizon for r in results]
    return {
        "c_avg_fh": _mean("avg_fh_c"),
        "a_avg_fh": _mean("avg_fh_a"),
        "c_checks": _mean("n_c"),
        "a_checks": _mean("n_a"),
        "c_extra": _mean("extra_c"),
        "a_extra": _mean("extra_a"),
        "total_extra": _mean("total_extra"),
        "total_extra_ci": _ci95([r["total_extra"] for r in results]),
        "comp_day": float(np.mean(times)),
    }


# =============================================================================
# Study driver
# =============================================================================
LIVE_METHODS = ["myopic", "thrifty", "adp_d", "adp_ds"]


def run_study(P: Params, horizon: int, methods: List[str], runs: int, *,
              n_sample: int, t_l: int, t_h: int, seed: int = _SEED,
              verbose: bool = True) -> Dict[str, Dict]:
    """Run `runs` sample paths for each requested method; return KPIs."""
    out: Dict[str, Dict] = {}
    for mi, method in enumerate(methods):
        if verbose:
            print(f"  [{method}] {runs} runs x {horizon} days "
                  f"(n_sample={n_sample}, t_l={t_l}, t_h={t_h}) ...", flush=True)
        results = []
        for r in range(runs):
            path_rng = np.random.default_rng(seed + 1000 * r + mi)
            f_seed = seed + 7777 * (r + 1) + 13 * mi
            policy = make_policy(method, P, n_sample=n_sample, t_l=t_l,
                                 t_h=t_h, f_seed=f_seed)
            res = run_path(policy, P, horizon, path_rng)
            results.append(res)
            if verbose:
                print(f"    run {r + 1}/{runs}: C={res['n_c']} A={res['n_a']} "
                      f"extra={res['total_extra']} avg_fh_c={res['avg_fh_c']:.0f} "
                      f"in {res['time']:.1f}s", flush=True)
        out[method] = aggregate(results, horizon)
    return out


# =============================================================================
# Table 8 printing
# =============================================================================
def _fmt(x, spec):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if spec == "i":
        return f"{x:.1f}" if x != int(x) else str(int(x))
    if spec == "f1":
        return f"{x:.1f}"
    if spec == "f2":
        return f"{x:.2f}"
    return str(x)


def print_table8(computed: Dict[str, Dict], methods: List[str]) -> None:
    ref = TABLE8_REFERENCE
    cols = methods + [m for m in ("airline", "dp") if m in ref]
    # order: live methods first (as given), then references
    header = ["KPI"] + [
        m if m in computed else f"{m}*" for m in cols
    ]

    rows = [
        ("C  Mean Average FH", "c_avg_fh", "f1"),
        ("C  Mean Extra Slots", "c_extra", "i"),
        ("C  Mean Total Checks", "c_checks", "f1"),
        ("A  Mean Average FH", "a_avg_fh", "f1"),
        ("A  Mean Extra Slots", "a_extra", "i"),
        ("A  Mean Total Checks", "a_checks", "f1"),
        ("Mean Total Extra Slots", "total_extra", "i"),
        ("95% CI (total extra)", None, None),
        ("Computation time/day (s)", "comp_day", "f2"),
    ]

    def cell(method, key, spec):
        if method in computed:
            v = computed[method].get(key)
            return _fmt(v, spec) if spec else (computed[method].get("total_extra_ci") or "-")
        rv = ref.get(method, {})
        if key is None:
            return rv.get("ci") or "-"
        return _fmt(rv.get(key), spec)

    widths = [24] + [max(len(h), 12) for h in header[1:]]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))

    print()
    print("=" * (sum(widths) + 2 * len(widths)))
    print("Comparison of KPIs  (Deng & Santos 2021, Table 8 format)")
    print("  * = paper's published reference value (not recomputed here)")
    print("=" * (sum(widths) + 2 * len(widths)))
    print(line(header))
    print("-" * (sum(widths) + 2 * len(widths)))
    for label, key, spec in rows:
        cells = [label] + [cell(m, key, spec) for m in cols]
        print(line(cells))
    print("=" * (sum(widths) + 2 * len(widths)))
    print()


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    from .data_loader import load_maintenance_data

    p = argparse.ArgumentParser(
        description="Reproduce the Table 8 KPIs of Deng & Santos (2021).")
    p.add_argument("--data", default="Sep 25 2017 (Case Study 2018-2021).xlsx")
    p.add_argument("--horizon-days", type=int, default=240,
                   help="planning horizon in days (paper case 1 = 1194)")
    p.add_argument("--runs", type=int, default=2,
                   help="number of Monte-Carlo test sample paths (paper: 100)")
    p.add_argument("--n-sample", type=int, default=5,
                   help="MC paths for the A-check forecast (paper: 50)")
    p.add_argument("--t-l", type=int, default=40,
                   help="stochastic lookahead for A-checks, days (paper: 183)")
    p.add_argument("--t-h", type=int, default=100,
                   help="deterministic lookahead for C-checks, days (paper: 1461)")
    p.add_argument("--util-cv", type=float, default=UTIL_CV)
    p.add_argument("--cap-a", type=int, default=None,
                   help="A-check slots/day (default: from data; paper case 1 = 2)")
    p.add_argument("--cap-c", type=int, default=None,
                   help="C-check hangars/day (default: from data; paper case 1 = 3)")
    p.add_argument("--seed", type=int, default=_SEED)
    p.add_argument("--methods", default=",".join(LIVE_METHODS),
                   help="comma list of: " + ",".join(LIVE_METHODS))
    p.add_argument("--table8", action="store_true",
                   help="use the paper's exact setting (SLOW)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.table8:
        args.horizon_days = HORIZON_CASE1
        args.runs = N_RUNS
        args.n_sample = N_SAMPLE
        args.t_l = T_L
        args.t_h = T_H
        print("!!  --table8: paper-faithful setting (1194 days, 100 runs, "
              "n_sample=50). This is SLOW in pure Python and may take hours.")
        print("!!  Use a reduced setting (default) for a quick sanity check.\n")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in LIVE_METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}; choose from {LIVE_METHODS}")

    print(f"Loading {args.data} ...")
    data = load_maintenance_data(args.data)
    P = build_params(data, util_cv=args.util_cv)
    if args.cap_a is not None:
        P.cap_a = args.cap_a
    if args.cap_c is not None:
        P.cap_c = args.cap_c
    print(f"Fleet: {P.n} aircraft | cap_c={P.cap_c} cap_a={P.cap_a} d_c={P.d_c} "
          f"| C interval {int(P.ci_c_fh[0])} FH | A interval {int(P.ci_a_fh[0])} FH")
    print(f"Horizon {args.horizon_days} days | {args.runs} runs | methods={methods}\n")

    t_start = time.time()
    computed = run_study(
        P, args.horizon_days, methods, args.runs,
        n_sample=args.n_sample, t_l=args.t_l, t_h=args.t_h,
        seed=args.seed, verbose=not args.quiet,
    )
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")

    print_table8(computed, methods)


if __name__ == "__main__":
    main()

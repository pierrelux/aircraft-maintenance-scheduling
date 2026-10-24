#!/usr/bin/env python3
"""Run algorithm variants and write results for poster plots.

Writes: output/variant_results.json

Designed to be runnable per-variant (to keep runs short).

Examples:
  source .venv/bin/activate
  python run_variant_experiments.py --variant greedy
  python run_variant_experiments.py --variant future
  python run_variant_experiments.py --variant full

You can also run everything (slow):
  python run_variant_experiments.py --variant all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_scheduler import Scheduler, load_maintenance_data


DEFAULT_OUT = Path("output/variant_results.json")
DEFAULT_PKL = Path("output/schedule_deng.pkl")
DEFAULT_LAMBDA_FUTURE = 100.0

def _jsonable(x):
    """Convert numpy scalars / nested containers into JSON-serializable types."""
    try:
        import numpy as np

        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def summarize_result(result) -> dict:
    kpis = _jsonable(result.compute_kpis().to_dict())
    return {
        "total_unused_fh": float(result.total_unused_fh),
        "objective": float(result.objective),
        "total_tolerance_events": int(result.total_tolerance_events),
        "total_a_checks": int(result.total_a_checks),
        "total_c_checks": int(result.total_c_checks),
        "computation_time_s": float(result.computation_time),
        "kpis": kpis,
    }


def run_variant(variant: str, years: int, beam_width: int, data_path: str | None) -> tuple[str, dict]:
    data_file = data_path or "Sep 25 2017 (Case Study 2018-2021).xlsx"
    data = load_maintenance_data(data_file)
    planning_days = years * 365

    if variant == "greedy":
        sched = Scheduler(data, beam_width=beam_width, u=100.0)
        result = sched.schedule_greedy_due(planning_days, verbose=True)
        return "Greedy baseline (due-only)", summarize_result(result)

    if variant == "deng":
        sched = Scheduler(data, beam_width=beam_width, u=100.0)
        result = sched.schedule(planning_days, verbose=True)
        return "Base Deng (beam)", summarize_result(result)

    if variant == "local":
        sched = Scheduler(
            data,
            beam_width=beam_width,
            u=100.0,
            use_heuristic=True,
            heuristic_type="local",
            lambda_h=0.1,
            k_per_bucket=1,
        )
        result = sched.schedule(planning_days, verbose=True)
        return "+ Local heuristic", summarize_result(result)

    if variant == "future":
        sched = Scheduler(
            data,
            beam_width=beam_width,
            u=100.0,
            use_heuristic=True,
            heuristic_type="future",
            lambda_h=DEFAULT_LAMBDA_FUTURE,
            k_per_bucket=1,
        )
        result = sched.schedule(planning_days, verbose=True)
        return f"+ Future heuristic (λ={DEFAULT_LAMBDA_FUTURE:g})", summarize_result(result)

    if variant == "diversity":
        sched = Scheduler(
            data,
            beam_width=beam_width,
            u=100.0,
            use_heuristic=False,
            k_per_bucket=2,
        )
        result = sched.schedule(planning_days, verbose=True)
        return "+ Diversity (k=2)", summarize_result(result)

    if variant == "full":
        sched = Scheduler(
            data,
            beam_width=beam_width,
            u=100.0,
            use_heuristic=True,
            heuristic_type="future",
            lambda_h=DEFAULT_LAMBDA_FUTURE,
            k_per_bucket=2,
        )
        result = sched.schedule(planning_days, verbose=True)
        return f"Full variant (λ={DEFAULT_LAMBDA_FUTURE:g})", summarize_result(result)

    if variant == "adp":
        sched = Scheduler(data, beam_width=beam_width, u=100.0)
        result = sched.schedule_adp(
            planning_days,
            verbose=True,
            n_sample=20,       # paper uses 50; 20 keeps the Python runtime sane
            t_l=183,           # paper Table 7
            t_h=1461,          # paper Table 7
            lambda_extra=1e5,  # paper Table 7
            xi_waste=1e20,     # paper Table 7
            gamma=1.0,         # paper Table 7
            r_lb_a=21,         # paper Table 7
            r_lb_c=210,        # paper Table 7
            util_cv=0.3,
            seed=42,
        )
        return "ADP (Deng & Santos 2021)", summarize_result(result)

    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["greedy", "deng", "local", "future", "diversity", "full", "adp", "all"],
    )
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    payload = load_existing(out_path)

    order = ["greedy", "deng", "local", "future", "diversity", "full", "adp"]
    targets = order if args.variant == "all" else [args.variant]

    for v in targets:
        label, res = run_variant(v, args.years, args.beam_width, args.data)
        payload[v] = {"label": label, "result": res}
        save(out_path, payload)
        print(f"Saved: {v} -> {label}")


if __name__ == "__main__":
    main()


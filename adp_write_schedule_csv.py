#!/usr/bin/env python3
"""
Helper script to write ADP ScheduleResult to CSV format for poster visualizations.

Based on Deng & Santos (2021) - Forward Look-ahead ADP.

Usage:
    python adp_write_schedule_csv.py --run --years 4 --output output/schedule_adp.csv
    OR
    python adp_write_schedule_csv.py --result <pickle_file> --output output/schedule_adp.csv
"""

import argparse
import sys
import pickle
import pandas as pd
from pathlib import Path

from aircraft_scheduler import Scheduler, load_maintenance_data, ScheduleResult


def result_to_csv(result: ScheduleResult, output_path: str):
    """Convert ScheduleResult to CSV format matching poster_visualizations expectations."""
    rows = []
    for check in result.checks:
        rows.append({
            'Aircraft': check.tail,
            'Fleet': '',  # Would need to get from data
            'Check Type': check.check_type,
            'Start Date': check.start_date.strftime('%Y-%m-%d'),
            'End Date': check.end_date.strftime('%Y-%m-%d') if check.end_date else check.start_date.strftime('%Y-%m-%d'),
            'Duration (days)': check.duration,
            'Unused FH (Eq. 28)': check.total_unused_fh,
        })
    
    # Ensure columns are always present, even with empty results
    columns = ['Aircraft', 'Fleet', 'Check Type', 'Start Date', 
               'End Date', 'Duration (days)', 'Unused FH (Eq. 28)']
    df = pd.DataFrame(rows, columns=columns)
    
    if len(df) == 0:
        print(f"Wrote 0 checks to {output_path} (empty schedule)")
    else:
        print(f"Wrote {len(rows)} checks to {output_path}")
        print(f"Date range: {df['Start Date'].min()} to {df['Start Date'].max()}")
    
    df.to_csv(output_path, index=False)


def run_and_save(
    years: int,
    output_path: str,
    data_file: str = None,
    beam_width: int = 200,
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
):
    """Run the Lookahead ADP scheduler (Deng & Santos 2021) and save to CSV."""
    if data_file is None:
        data_file = "Sep 25 2017 (Case Study 2018-2021).xlsx"
    
    print(f"Loading data from {data_file}...")
    data = load_maintenance_data(data_file)
    
    planning_days = years * 365
    print(f"Running Lookahead ADP for {years} years ({planning_days} days)...")
    print(f"ADP params: n_sample={n_sample}, t_l={t_l}, t_h={t_h}, "
          f"lambda={lambda_extra:g}, xi={xi_waste:g}, gamma={gamma}, "
          f"R_lb=({r_lb_a:g},{r_lb_c:g}), cv={util_cv}, seed={seed}")
    
    scheduler = Scheduler(data, beam_width=beam_width, u=100.0)
    result = scheduler.schedule_adp(
        planning_days,
        verbose=True,
        n_sample=n_sample,
        t_l=t_l,
        t_h=t_h,
        lambda_extra=lambda_extra,
        xi_waste=xi_waste,
        gamma=gamma,
        r_lb_a=r_lb_a,
        r_lb_c=r_lb_c,
        util_cv=util_cv,
        seed=seed,
    )
    
    print(f"\nResults:")
    print(f"  C-checks: {result.total_c_checks}")
    print(f"  A-checks: {result.total_a_checks}")
    print(f"  Tolerance events: {result.total_tolerance_events}")
    print(f"  Total unused FH: {result.total_unused_fh:.1f}")
    print(f"  Computation time: {result.computation_time:.1f}s")
    
    # Save to CSV
    result_to_csv(result, output_path)
    
    # Also save result object for future use
    result_path = Path(output_path).with_suffix('.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved result object to {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Write ADP ScheduleResult to CSV for poster visualizations"
    )
    parser.add_argument(
        '--result',
        type=str,
        help='Path to pickled ScheduleResult file'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run ADP scheduler instead of loading result'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=4,
        help='Planning horizon in years (default: 4)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to Excel data file'
    )
    parser.add_argument(
        '--beam-width',
        type=int,
        default=200,
        help='Beam width for scheduler (default: 200)'
    )
    parser.add_argument(
        '--n-sample',
        type=int,
        default=20,
        help='Monte-Carlo sample paths for the A-check forecast (paper: 50; default: 20)'
    )
    parser.add_argument(
        '--t-l',
        type=int,
        default=183,
        help='Stochastic lookahead horizon for A-checks in days (paper: 183)'
    )
    parser.add_argument(
        '--t-h',
        type=int,
        default=1461,
        help='Deterministic lookahead horizon for C-checks in days (paper: 1461)'
    )
    parser.add_argument(
        '--lambda-extra',
        type=float,
        default=1e5,
        help='Cost of one extra maintenance slot (paper: 1e5)'
    )
    parser.add_argument(
        '--xi-waste',
        type=float,
        default=1e20,
        help='Penalty for wasting an available slot (paper: 1e20)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Discount factor (paper: 1)'
    )
    parser.add_argument(
        '--r-lb-a',
        type=float,
        default=21,
        help='A-check eligibility threshold in days remaining (paper: 21)'
    )
    parser.add_argument(
        '--r-lb-c',
        type=float,
        default=210,
        help='C-check eligibility threshold in days remaining (paper: 210)'
    )
    parser.add_argument(
        '--util-cv',
        type=float,
        default=0.3,
        help='Assumed coefficient of variation for daily FH/FC in the stochastic forecast'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='RNG seed for the Monte-Carlo forecasts'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/schedule_adp.csv',
        help='Output CSV path (default: output/schedule_adp.csv)'
    )
    
    args = parser.parse_args()
    
    if args.run:
        run_and_save(
            args.years,
            args.output,
            args.data,
            args.beam_width,
            n_sample=args.n_sample,
            t_l=args.t_l,
            t_h=args.t_h,
            lambda_extra=args.lambda_extra,
            xi_waste=args.xi_waste,
            gamma=args.gamma,
            r_lb_a=args.r_lb_a,
            r_lb_c=args.r_lb_c,
            util_cv=args.util_cv,
            seed=args.seed,
        )
    elif args.result:
        print(f"Loading result from {args.result}...")
        with open(args.result, 'rb') as f:
            result = pickle.load(f)
        result_to_csv(result, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

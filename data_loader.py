"""
Data loader for Aircraft Maintenance Check Scheduling.

Loads Excel data and creates the sequential scheduling environment.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np

from sequential_env import (
    SequentialSchedulingEnv,
    SchedulingState,
    AircraftStatus,
    MaintenanceConfig
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AircraftDInitial:
    """D-check initial state for an aircraft."""
    fleet: str
    tail: str
    d_cycle: int
    dy_d: int
    d_ci_dy: int
    d_tol_dy: int
    d_tolu_dy: int
    d_max: int
    d_end: Optional[date]
    operation: date


@dataclass
class AircraftCInitial:
    """C-check initial state for an aircraft."""
    fleet: str
    tail: str
    c_sn: float
    c_max: int
    dy_c: int
    fh_c: float
    fc_c: float
    c_ci_dy: int
    c_ci_fh: float
    c_ci_fc: float
    c_tol_dy: int
    c_tol_fh: float
    c_tol_fc: float
    c_tolu_dy: int
    c_tolu_fh: float
    c_tolu_fc: float
    c_start: Optional[date]
    c_end: Optional[date]


@dataclass
class AircraftAInitial:
    """A-check initial state for an aircraft."""
    fleet: str
    tail: str
    a_sn: float
    a_max: int
    dy_a: int
    fh_a: float
    fc_a: float
    a_ci_dy: int
    a_ci_fh: float
    a_ci_fc: float
    a_tol_dy: int
    a_tol_fh: float
    a_tol_fc: float
    a_tolu_dy: int
    a_tolu_fh: float
    a_tolu_fc: float
    a_end: Optional[date]


@dataclass
class FlightRates:
    """Daily flight hours and cycles for an aircraft by month."""
    fleet: str
    tail: str
    dfh: dict[int, float]
    dfc: dict[int, float]


@dataclass
class CCheckInfo:
    """Future C-check codes and elapsed times for an aircraft."""
    fleet: str
    tail: str
    check_codes: list[float]
    elapsed_times: list[int]


@dataclass
class DateRange:
    """A date range with start and end dates."""
    begin: date
    end: date

    def contains(self, d: date) -> bool:
        return self.begin <= d <= self.end


@dataclass
class PeakSeason:
    """Commercial peak season when C-checks should be avoided."""
    year: int
    begin: date
    end: date


@dataclass
class ReservedMaintenance:
    """Reserved maintenance activity."""
    fleet: str
    tail: str
    reserved_type: str
    begin: date
    end: date
    time: int
    slot: str


@dataclass
class AdditionalSlots:
    """Additional maintenance slots."""
    begin: date
    end: date
    total_slots: int


@dataclass
class Config:
    """Configuration parameters for optimization."""
    begin_year: int
    total_years: int
    begin_day: date
    m_cost: float
    c_cost: float
    max_c_check: int
    max_a_check: int
    start_day_interval: int
    c_e_t: int
    max_tol_dy_c: int
    max_tol_fh_c: float
    max_tol_fc_c: float
    max_tol_dy_a: int
    max_tol_fh_a: float
    max_tol_fc_a: float
    max_tol_dy_d: int
    dtc_min: int
    dta_min: int
    c_task_types: dict[str, int]
    a_task_types: dict[str, int]


@dataclass
class MaintenanceData:
    """Complete maintenance data from the Excel file."""
    d_initial: dict[str, AircraftDInitial]
    c_initial: dict[str, AircraftCInitial]
    a_initial: dict[str, AircraftAInitial]
    flight_rates: dict[str, FlightRates]
    c_check_info: dict[str, CCheckInfo]
    config: Config
    reserved: list[ReservedMaintenance]
    c_peak: list[PeakSeason]
    c_not_allowed: list[DateRange]
    a_not_allowed: list[date]
    more_c_slots: list[AdditionalSlots]
    more_a_slots: dict[date, int]
    public_holidays: list[date]
    aircraft_tails: list[str]


# =============================================================================
# Excel Parsing
# =============================================================================

def _parse_date(val) -> Optional[date]:
    """Parse a date value from Excel, handling -1 as None."""
    if pd.isna(val) or val == -1:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return datetime.strptime(val, "%Y-%m-%d").date()
    return None


def load_maintenance_data(excel_path: str) -> MaintenanceData:
    """Load all maintenance data from the Excel file."""
    xl = pd.ExcelFile(excel_path)

    # Load D_INITIAL
    df_d = pd.read_excel(xl, "D_INITIAL")
    d_initial = {}
    for _, row in df_d.iterrows():
        tail = row["A/C TAIL"]
        d_initial[tail] = AircraftDInitial(
            fleet=row["FLEET"],
            tail=tail,
            d_cycle=int(row["D-CYCLE"]),
            dy_d=int(row["DY-D"]),
            d_ci_dy=int(row["D-CI-DY"]),
            d_tol_dy=int(row["D-TOL-DY"]),
            d_tolu_dy=int(row["D-TOLU-DY"]),
            d_max=int(row["D-MAX"]),
            d_end=_parse_date(row["D-END"]),
            operation=_parse_date(row["OPERATION"]) or date(2017, 9, 25),
        )

    # Load C_INITIAL
    df_c = pd.read_excel(xl, "C_INITIAL")
    c_initial = {}
    for _, row in df_c.iterrows():
        tail = row["A/C TAIL"]
        c_initial[tail] = AircraftCInitial(
            fleet=row["FLEET"],
            tail=tail,
            c_sn=float(row["C-SN"]),
            c_max=int(row["C-MAX"]),
            dy_c=int(row["DY-C"]),
            fh_c=float(row["FH-C"]),
            fc_c=float(row["FC-C"]),
            c_ci_dy=int(row["C-CI-DY"]),
            c_ci_fh=float(row["C-CI-FH"]),
            c_ci_fc=float(row["C-CI-FC"]),
            c_tol_dy=int(row["C-TOL-DY"]),
            c_tol_fh=float(row["C-TOL-FH"]),
            c_tol_fc=float(row["C-TOL-FC"]),
            c_tolu_dy=int(row["C-TOLU-DY"]),
            c_tolu_fh=float(row["C-TOLU-FH"]),
            c_tolu_fc=float(row["C-TOLU-FC"]),
            c_start=_parse_date(row["C-START"]),
            c_end=_parse_date(row["C-END"]),
        )

    # Load A_INITIAL
    df_a = pd.read_excel(xl, "A_INITIAL")
    a_initial = {}
    for _, row in df_a.iterrows():
        tail = row["A/C TAIL"]
        a_initial[tail] = AircraftAInitial(
            fleet=row["FLEET"],
            tail=tail,
            a_sn=float(row["A-SN"]),
            a_max=int(row["A-MAX"]),
            dy_a=int(row["DY-A"]),
            fh_a=float(row["FH-A"]),
            fc_a=float(row["FC-A"]),
            a_ci_dy=int(row["A-CI-DY"]),
            a_ci_fh=float(row["A-CI-FH"]),
            a_ci_fc=float(row["A-CI-FC"]),
            a_tol_dy=int(row["A-TOL-DY"]),
            a_tol_fh=float(row["A-TOL-FH"]),
            a_tol_fc=float(row["A-TOL-FC"]),
            a_tolu_dy=int(row["A-TOLU-DY"]),
            a_tolu_fh=float(row["A-TOLU-FH"]),
            a_tolu_fc=float(row["A-TOLU-FC"]),
            a_end=_parse_date(row["A-END"]),
        )

    # Load DFH and DFC
    df_dfh = pd.read_excel(xl, "DFH")
    df_dfc = pd.read_excel(xl, "DFC")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    flight_rates = {}
    for _, row in df_dfh.iterrows():
        tail = row["A/C TAIL"]
        dfh = {i + 1: float(row[m]) for i, m in enumerate(months)}
        flight_rates[tail] = FlightRates(fleet=row["FLEET"], tail=tail, dfh=dfh, dfc={})

    for _, row in df_dfc.iterrows():
        tail = row["A/C TAIL"]
        dfc = {i + 1: float(row[m]) for i, m in enumerate(months)}
        if tail in flight_rates:
            flight_rates[tail].dfc = dfc

    # Load C_CHECK_CODE and C_ELAPSED_TIME
    df_code = pd.read_excel(xl, "C_CHECK_CODE")
    df_time = pd.read_excel(xl, "C_ELAPSED_TIME")

    c_check_info = {}
    code_cols = [c for c in df_code.columns if c.startswith("C CODE")]
    time_cols = [c for c in df_time.columns if c.startswith("TAT")]

    for idx, row in df_code.iterrows():
        tail = row["A/C TAIL"]
        codes = [float(row[c]) for c in code_cols]
        times_row = df_time.iloc[idx]
        times = [int(times_row[c]) for c in time_cols]
        c_check_info[tail] = CCheckInfo(fleet=row["FLEET"], tail=tail, check_codes=codes, elapsed_times=times)

    # Load ADDITIONAL config
    df_add = pd.read_excel(xl, "ADDITIONAL")
    config_dict = {}
    c_task_types = {}
    a_task_types = {}

    for _, row in df_add.iterrows():
        key = str(row.iloc[0]).strip()
        val = row.iloc[1]
        if key == "C TASK TYPE":
            c_task_types[str(val)] = int(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
        elif key == "A TASK TYPE":
            a_task_types[str(val)] = int(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
        else:
            config_dict[key] = val

    begin_day = _parse_date(config_dict.get("BEGIN DAY")) or date(2017, 9, 25)

    config = Config(
        begin_year=int(config_dict.get("BEGIN YEAR", 2017)),
        total_years=int(config_dict.get("TOTAL YEARS", 4)),
        begin_day=begin_day,
        m_cost=float(config_dict.get("M COST", 100000000)),
        c_cost=float(config_dict.get("C COST", 100000)),
        max_c_check=int(config_dict.get("MAX C CHECK", 3)),
        max_a_check=int(config_dict.get("MAX A CHECK", 1)),
        start_day_interval=int(config_dict.get("START DAY INTERVAL", 3)),
        c_e_t=int(config_dict.get("C_E_T", 23)),
        max_tol_dy_c=int(config_dict.get("MAX TOL DY C", 0)),
        max_tol_fh_c=float(config_dict.get("MAX TOL FH C", 0)),
        max_tol_fc_c=float(config_dict.get("MAX TOL FC C", 0)),
        max_tol_dy_a=int(config_dict.get("MAX TOL DY A", 0)),
        max_tol_fh_a=float(config_dict.get("MAX TOL FH A", 0)),
        max_tol_fc_a=float(config_dict.get("MAX TOL FC A", 0)),
        max_tol_dy_d=int(config_dict.get("MAX TOL DY D", 0)),
        dtc_min=int(config_dict.get("DTC MIN", 365)),
        dta_min=int(config_dict.get("DTA MIN", 21)),
        c_task_types=c_task_types,
        a_task_types=a_task_types,
    )

    # Load RESERVED
    df_res = pd.read_excel(xl, "RESERVED")
    reserved = []
    for _, row in df_res.iterrows():
        if pd.notna(row["A/C TAIL"]):
            reserved.append(ReservedMaintenance(
                fleet=row["FLEET"],
                tail=row["A/C TAIL"],
                reserved_type=row["RESERVED TYPE"],
                begin=_parse_date(row["RESERVED BEGIN"]),
                end=_parse_date(row["RESERVED END"]),
                time=int(row["RESERVED TIME"]),
                slot=row["RESERVED SLOT"],
            ))

    # Load C_PEAK
    df_peak = pd.read_excel(xl, "C_PEAK")
    c_peak = [PeakSeason(year=int(row["YEAR"]), begin=_parse_date(row["PEAK BEGIN"]), end=_parse_date(row["PEAK END"]))
              for _, row in df_peak.iterrows()]

    # Load C_NOT_ALLOWED
    df_cna = pd.read_excel(xl, "C_NOT_ALLOWED")
    c_not_allowed = [DateRange(begin=_parse_date(row["BEGIN"]), end=_parse_date(row["END"]))
                     for _, row in df_cna.iterrows()]

    # Load A_NOT_ALLOWED
    df_ana = pd.read_excel(xl, "A_NOT_ALLOWED")
    a_not_allowed = [_parse_date(row["DATE"]) for _, row in df_ana.iterrows()]

    # Load MORE_C_SLOTS
    df_mcs = pd.read_excel(xl, "MORE_C_SLOTS")
    more_c_slots = [AdditionalSlots(begin=_parse_date(row["BEGIN"]), end=_parse_date(row["END"]),
                                     total_slots=int(row["TOTAL SLOTS"])) for _, row in df_mcs.iterrows()]

    # Load MORE_A_SLOTS
    df_mas = pd.read_excel(xl, "MORE_A_SLOTS")
    more_a_slots = {}
    for _, row in df_mas.iterrows():
        d = _parse_date(row["DATE"])
        if d:
            more_a_slots[d] = int(row["TOTAL SLOTS"])

    # Load PUBLIC_HOLIDAYS
    df_ph = pd.read_excel(xl, "PUBLIC_HOLIDAYS")
    public_holidays = [_parse_date(row["DATE"]) for _, row in df_ph.iterrows()]

    return MaintenanceData(
        d_initial=d_initial,
        c_initial=c_initial,
        a_initial=a_initial,
        flight_rates=flight_rates,
        c_check_info=c_check_info,
        config=config,
        reserved=reserved,
        c_peak=c_peak,
        c_not_allowed=c_not_allowed,
        a_not_allowed=a_not_allowed,
        more_c_slots=more_c_slots,
        more_a_slots=more_a_slots,
        public_holidays=public_holidays,
        aircraft_tails=list(d_initial.keys()),
    )


# =============================================================================
# Environment with Loaded Data
# =============================================================================

class MaintenanceSchedulingEnv(SequentialSchedulingEnv):
    """Sequential scheduling environment initialized from Excel data."""
    
    def __init__(self, data: MaintenanceData, tail_to_id: Optional[Dict[str, int]] = None):
        self.data = data
        
        if tail_to_id is None:
            self.tail_to_id = {tail: i for i, tail in enumerate(data.aircraft_tails)}
        else:
            self.tail_to_id = tail_to_id
        self.id_to_tail = {v: k for k, v in self.tail_to_id.items()}
        
        config = self._build_config(data)
        start_date = data.config.begin_day
        horizon_days = data.config.total_years * 365
        self.start_date = start_date
        
        c_capacity, a_capacity = self._build_capacity(data, start_date, horizon_days)
        
        super().__init__(
            n_aircraft=len(data.aircraft_tails),
            horizon=horizon_days,
            config=config,
            c_capacity=c_capacity,
            a_capacity=a_capacity,
            seed=42
        )
        self.flight_rates = data.flight_rates
    
    def _build_config(self, data: MaintenanceData) -> MaintenanceConfig:
        first_tail = data.aircraft_tails[0]
        c_init = data.c_initial[first_tail]
        a_init = data.a_initial[first_tail]
        
        return MaintenanceConfig(
            a_interval_dy=float(a_init.a_ci_dy),
            a_interval_fh=float(a_init.a_ci_fh),
            a_interval_fc=float(a_init.a_ci_fc),
            c_interval_dy=float(c_init.c_ci_dy),
            c_interval_fh=float(c_init.c_ci_fh),
            c_interval_fc=float(c_init.c_ci_fc),
            a_min_interval=float(data.config.dta_min),
            c_min_interval=float(data.config.dtc_min),
            a_check_duration=1,
            c_check_duration=data.config.c_e_t,
            a_eligibility_threshold=float(a_init.a_tol_dy + a_init.a_ci_dy * 0.25),
            c_eligibility_threshold=float(c_init.c_tol_dy + c_init.c_ci_dy * 0.15),
        )
    
    def _build_capacity(self, data: MaintenanceData, start_date: date, horizon_days: int) -> tuple:
        c_capacity = {}
        a_capacity = {}
        
        c_blocked_dates = set()
        a_blocked_dates = set(data.a_not_allowed)
        a_blocked_dates.update(data.public_holidays)
        
        for peak in data.c_peak:
            d = peak.begin
            while d <= peak.end:
                c_blocked_dates.add(d)
                d += timedelta(days=1)
        
        for not_allowed in data.c_not_allowed:
            d = not_allowed.begin
            while d <= not_allowed.end:
                c_blocked_dates.add(d)
                d += timedelta(days=1)
        
        more_c_lookup = {}
        for slot in data.more_c_slots:
            d = slot.begin
            while d <= slot.end:
                more_c_lookup[d] = slot.total_slots
                d += timedelta(days=1)
        
        for day_idx in range(horizon_days):
            day_date = start_date + timedelta(days=day_idx)
            day_of_week = day_date.weekday()
            
            if day_date in c_blocked_dates:
                c_cap = 0
            elif day_of_week >= 5:
                c_cap = 0
            else:
                c_cap = 1
                if day_date in more_c_lookup:
                    c_cap = more_c_lookup[day_date]
            c_capacity[day_idx] = c_cap
            
            if day_date in a_blocked_dates:
                a_cap = 0
            elif day_of_week >= 5:
                a_cap = 0
            else:
                a_cap = data.config.max_a_check
                if day_date in data.more_a_slots:
                    a_cap += data.more_a_slots[day_date]
            a_capacity[day_idx] = a_cap
        
        return c_capacity, a_capacity
    
    def reset(self) -> SchedulingState:
        fleet = {}
        for i, tail in enumerate(self.data.aircraft_tails):
            c_init = self.data.c_initial[tail]
            a_init = self.data.a_initial[tail]
            flight_rate = self.data.flight_rates[tail]
            
            avg_dfh = np.mean(list(flight_rate.dfh.values()))
            avg_dfc = np.mean(list(flight_rate.dfc.values()))
            
            aircraft = AircraftStatus(
                aircraft_id=i,
                dy_a=float(a_init.dy_a),
                fh_a=float(a_init.fh_a),
                fc_a=float(a_init.fc_a),
                dy_c=float(c_init.dy_c),
                fh_c=float(c_init.fh_c),
                fc_c=float(c_init.fc_c),
                daily_fh=float(avg_dfh),
                daily_fc=float(avg_dfc),
                in_maintenance=False,
                maintenance_days_remaining=0,
                current_check_type=None
            )
            fleet[i] = aircraft
        
        self.state = SchedulingState(
            current_day=0,
            fleet=fleet,
            schedule={},
            c_checks_today=0,
            a_checks_today=0
        )
        return self.state
    
    def get_aircraft_tail(self, aircraft_id: int) -> str:
        return self.id_to_tail[aircraft_id]
    
    def get_aircraft_id(self, tail: str) -> int:
        return self.tail_to_id[tail]


# =============================================================================
# Loading Functions
# =============================================================================

def load_env_from_excel(filepath: str) -> MaintenanceSchedulingEnv:
    """Load environment from Excel file."""
    print(f"Loading data from: {filepath}")
    data = load_maintenance_data(filepath)
    
    print(f"  - Loaded {len(data.aircraft_tails)} aircraft")
    print(f"  - Planning horizon: {data.config.total_years} years from {data.config.begin_day}")
    print(f"  - Peak seasons: {len(data.c_peak)}")
    
    env = MaintenanceSchedulingEnv(data)
    
    print(f"\nEnvironment created:")
    print(f"  - Aircraft: {env.n_aircraft}")
    print(f"  - Horizon: {env.horizon} days (~{env.horizon/365:.1f} years)")
    print(f"  - State dim: {env.state_dim}")
    print(f"  - Action space: {env.n_actions}")
    
    return env


# Alias for backward compatibility
load_env = load_env_from_excel


def create_synthetic_env(n_aircraft: int = 45, horizon_days: int = 1461) -> SequentialSchedulingEnv:
    """Create synthetic environment matching paper's statistics."""
    print("Creating synthetic environment...")
    
    config = MaintenanceConfig(
        a_interval_dy=120, a_interval_fh=750, a_interval_fc=750,
        c_interval_dy=730, c_interval_fh=7500, c_interval_fc=5000,
        a_min_interval=21, c_min_interval=365,
        a_check_duration=1, c_check_duration=23,
        a_eligibility_threshold=30, c_eligibility_threshold=90
    )
    
    start_date = date(2017, 9, 25)
    peak_periods = [
        (date(2018, 6, 1), date(2018, 9, 30)), (date(2018, 12, 15), date(2019, 1, 5)),
        (date(2019, 6, 1), date(2019, 9, 30)), (date(2019, 12, 15), date(2020, 1, 5)),
        (date(2020, 6, 1), date(2020, 9, 30)), (date(2020, 12, 15), date(2021, 1, 5)),
        (date(2021, 6, 1), date(2021, 9, 30)),
    ]
    
    c_capacity, a_capacity = {}, {}
    for day in range(horizon_days):
        day_date = start_date + timedelta(days=day)
        in_peak = any(start <= day_date <= end for start, end in peak_periods)
        blocked = day_date.weekday() >= 5 or in_peak
        c_capacity[day] = 0 if blocked else 1
        a_capacity[day] = 0 if blocked else 1
    
    env = SequentialSchedulingEnv(
        n_aircraft=n_aircraft, horizon=horizon_days, config=config,
        c_capacity=c_capacity, a_capacity=a_capacity, seed=42
    )
    print(f"  - Aircraft: {env.n_aircraft}, Horizon: {env.horizon} days")
    return env


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    try:
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
        state = env.reset()
        print(f"\nFirst 3 aircraft:")
        for i in range(min(3, env.n_aircraft)):
            tail = env.get_aircraft_tail(i)
            ac = state.fleet[i]
            print(f"  {tail}: A({ac.dy_a:.0f}d), C({ac.dy_c:.0f}d), {ac.daily_fh:.1f} FH/day")
    except Exception as e:
        print(f"Could not load Excel: {e}")
        env = create_synthetic_env()

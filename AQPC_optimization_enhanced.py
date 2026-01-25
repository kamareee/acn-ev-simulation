"""
AQPC/AQPS: Adaptive Queuing-based Priority Charging Optimization Module
========================================================================
A standalone Python module for solving EV fleet charging problems using:
- Model Predictive Control (MPC) via AQPC
- Heuristic Two-Tier Queue via AQPS (no solver dependency)

Includes ACN-compatible scenario generator and comprehensive simulation
scenarios (S1-S6) benchmarking against baseline algorithms.

Dependencies:
    - numpy
    - pandas
    - cvxpy (optional: only for AQPC MPC solver)
    - matplotlib
    - seaborn

Author: Research Implementation for AQPC Paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union
from enum import Enum
import time
import copy
import os
import warnings
from abc import ABC, abstractmethod

# Optional CVXPY import for MPC-based AQPC
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. AQPC (MPC) scheduler will be disabled.")

# Set global plot style for publication-quality output
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 10

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


class TariffType(Enum):
    """Time-of-Use tariff types."""
    FLAT = "flat"
    TOU_DEFAULT = "tou_default"
    TOU_AGGRESSIVE = "tou_aggressive"  # Higher peak/off-peak ratio
    REAL_TIME = "real_time"  # Dynamic pricing


class WeatherCondition(Enum):
    """Weather conditions for PV generation."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    VARIABLE = "variable"
    OVERCAST = "overcast"


@dataclass
class EVSession:
    """Represents a single EV charging session (ACN-compatible)."""
    
    session_id: str
    arrival_time: int          # Time step index
    departure_time: int        # Time step index
    energy_requested: float    # kWh
    max_rate: float           # kW (or Amps if working in current domain)
    min_rate: float           # kW - minimum charging rate (0 for non-priority)
    is_priority: bool         # True = Hard Constraint, False = Soft Penalty
    
    # State tracking
    energy_delivered: float = 0.0
    is_active: bool = True    # Used for preemption tracking
    station_id: str = ""      # EVSE station assignment
    
    # ACN-specific fields
    user_id: str = ""         # For fairness tracking
    vehicle_type: str = "generic"  # sedan, suv, truck, etc.
    
    @property
    def remaining_energy(self) -> float:
        """Energy still needed."""
        return max(0, self.energy_requested - self.energy_delivered)
    
    @property
    def is_complete(self) -> bool:
        """Check if charging is complete (within tolerance)."""
        return self.remaining_energy < 0.01


@dataclass
class TOUSchedule:
    """Time-of-Use electricity price schedule."""
    
    peak_hours: List[Tuple[int, int]]       # List of (start_hour, end_hour)
    mid_peak_hours: List[Tuple[int, int]]
    off_peak_hours: List[Tuple[int, int]]
    peak_price: float           # $/kWh
    mid_peak_price: float
    off_peak_price: float
    
    def get_price(self, hour: float) -> float:
        """Get electricity price for given hour of day."""
        for start, end in self.peak_hours:
            if start <= hour < end:
                return self.peak_price
        for start, end in self.mid_peak_hours:
            if start <= hour < end:
                return self.mid_peak_price
        return self.off_peak_price
    
    def get_period_type(self, hour: float) -> str:
        """Get period type (peak/mid-peak/off-peak) for given hour."""
        for start, end in self.peak_hours:
            if start <= hour < end:
                return "peak"
        for start, end in self.mid_peak_hours:
            if start <= hour < end:
                return "mid_peak"
        return "off_peak"
    
    def next_cheaper_slot(self, current_hour: float) -> Optional[float]:
        """Find next time slot with lower price."""
        current_price = self.get_price(current_hour)
        # Search forward up to 24 hours
        for offset in np.arange(0.25, 24, 0.25):
            future_hour = (current_hour + offset) % 24
            if self.get_price(future_hour) < current_price:
                return current_hour + offset
        return None


@dataclass
class OptimizationParams:
    """Hyperparameters for optimization objective."""
    
    alpha_energy: float = 1.0       # Weight for Energy Cost (H^EC)
    alpha_penalty: float = 100.0    # Weight for Non-Completion Penalty (H^NC)
    alpha_smooth: float = 0.1       # Weight for Peak Demand / Smoothing (H^QC)
    time_step_hours: float = 5/60   # Duration of one time step in hours (5 min default)
    horizon_steps: int = 12         # MPC lookahead horizon (1 hour with 5-min steps)
    min_priority_rate: float = 11.0 # Minimum rate guarantee for priority EVs (Amps/kW)


@dataclass
class InfrastructureConfig:
    """Charging infrastructure configuration (ACN-compatible)."""
    
    num_evses: int = 15
    max_power_per_evse: float = 22.0      # kW
    aggregate_power_limit: float = 150.0   # kW total site limit
    voltage: float = 240.0                 # V (for Amp conversion)
    phases: int = 3                        # Three-phase charging
    discrete_pilot_signals: bool = True    # Quantized pilot signals
    pilot_signal_step: float = 1.0         # Amp increment for discrete signals


@dataclass 
class PreemptionEvent:
    """Record of a preemption event."""
    
    timestamp: int
    priority_session_id: str
    preempted_session_ids: List[str]
    freed_capacity: float
    method: str  # 'highest_laxity' or 'proportional'


@dataclass
class SimulationMetrics:
    """Comprehensive metrics from a simulation run."""
    
    # Fulfillment metrics
    total_energy_requested: float = 0.0
    total_energy_delivered: float = 0.0
    priority_energy_requested: float = 0.0
    priority_energy_delivered: float = 0.0
    non_priority_energy_requested: float = 0.0
    non_priority_energy_delivered: float = 0.0
    
    # Session counts
    total_sessions: int = 0
    priority_sessions: int = 0
    priority_sessions_fulfilled: int = 0  # 100% delivered
    non_priority_sessions: int = 0
    non_priority_sessions_fulfilled: int = 0
    
    # Cost metrics
    total_energy_cost: float = 0.0
    peak_period_cost: float = 0.0
    off_peak_period_cost: float = 0.0
    
    # Fairness metrics
    jains_index: float = 0.0
    min_fulfillment_ratio: float = 0.0
    max_fulfillment_ratio: float = 0.0
    
    # Computational metrics
    total_computation_time: float = 0.0
    avg_step_time: float = 0.0
    max_step_time: float = 0.0
    
    # Preemption metrics
    total_preemptions: int = 0
    preemptions_option_b: int = 0
    preemptions_option_a: int = 0
    
    @property
    def overall_fulfillment_pct(self) -> float:
        if self.total_energy_requested > 0:
            return (self.total_energy_delivered / self.total_energy_requested) * 100
        return 0.0
    
    @property
    def priority_fulfillment_pct(self) -> float:
        if self.priority_energy_requested > 0:
            return (self.priority_energy_delivered / self.priority_energy_requested) * 100
        return 0.0
    
    @property
    def non_priority_fulfillment_pct(self) -> float:
        if self.non_priority_energy_requested > 0:
            return (self.non_priority_energy_delivered / self.non_priority_energy_requested) * 100
        return 0.0


# ==============================================================================
# ACN SCENARIO GENERATOR
# ==============================================================================


class ACNScenarioGenerator:
    """
    Generates realistic EV charging scenarios based on ACN-Data patterns.
    
    Reference: ACN-Data from Caltech/JPL charging networks
    https://ev.caltech.edu/dataset
    """
    
    # ACN-Data derived arrival patterns (workplace charging)
    ARRIVAL_PATTERNS = {
        'workplace_morning': {
            'peak_hour': 8.5,      # 8:30 AM
            'std_dev': 1.5,        # Hours
            'weight': 0.7
        },
        'workplace_midday': {
            'peak_hour': 12.0,
            'std_dev': 1.0,
            'weight': 0.2
        },
        'workplace_afternoon': {
            'peak_hour': 14.0,
            'std_dev': 1.0,
            'weight': 0.1
        }
    }
    
    # Energy demand distributions by vehicle type (kWh)
    ENERGY_DISTRIBUTIONS = {
        'sedan': {'mean': 25.0, 'std': 10.0, 'max': 60.0},
        'suv': {'mean': 35.0, 'std': 15.0, 'max': 80.0},
        'compact': {'mean': 18.0, 'std': 8.0, 'max': 40.0},
        'truck': {'mean': 45.0, 'std': 20.0, 'max': 100.0},
    }
    
    # Vehicle type distribution
    VEHICLE_TYPE_WEIGHTS = {
        'sedan': 0.45,
        'suv': 0.35,
        'compact': 0.15,
        'truck': 0.05
    }
    
    # Parking duration patterns (hours)
    DURATION_PATTERNS = {
        'short': {'mean': 2.0, 'std': 0.5, 'weight': 0.15},
        'medium': {'mean': 4.5, 'std': 1.0, 'weight': 0.35},
        'workday': {'mean': 8.0, 'std': 1.5, 'weight': 0.50}
    }
    
    def __init__(
        self,
        seed: Optional[int] = None,
        step_duration_mins: int = 5
    ):
        """
        Initialize scenario generator.
        
        Args:
            seed: Random seed for reproducibility
            step_duration_mins: Duration of each simulation time step
        """
        self.rng = np.random.default_rng(seed)
        self.step_duration_mins = step_duration_mins
        self.steps_per_hour = 60 / step_duration_mins
    
    def _hour_to_step(self, hour: float) -> int:
        """Convert hour of day to time step index."""
        return int(hour * self.steps_per_hour)
    
    def _step_to_hour(self, step: int) -> float:
        """Convert time step to hour of day."""
        return step / self.steps_per_hour
    
    def _sample_arrival_hour(self) -> float:
        """Sample arrival time using mixture of Gaussians."""
        patterns = self.ARRIVAL_PATTERNS
        weights = [p['weight'] for p in patterns.values()]
        
        # Select pattern
        pattern_idx = self.rng.choice(len(patterns), p=weights)
        pattern = list(patterns.values())[pattern_idx]
        
        # Sample from selected Gaussian
        hour = self.rng.normal(pattern['peak_hour'], pattern['std_dev'])
        return np.clip(hour, 6.0, 18.0)  # Constrain to business hours
    
    def _sample_vehicle_type(self) -> str:
        """Sample vehicle type."""
        types = list(self.VEHICLE_TYPE_WEIGHTS.keys())
        weights = list(self.VEHICLE_TYPE_WEIGHTS.values())
        return self.rng.choice(types, p=weights)
    
    def _sample_energy_demand(self, vehicle_type: str) -> float:
        """Sample energy demand based on vehicle type."""
        dist = self.ENERGY_DISTRIBUTIONS.get(vehicle_type, self.ENERGY_DISTRIBUTIONS['sedan'])
        energy = self.rng.normal(dist['mean'], dist['std'])
        return np.clip(energy, 5.0, dist['max'])
    
    def _sample_duration_hours(self) -> float:
        """Sample parking duration."""
        patterns = self.DURATION_PATTERNS
        weights = [p['weight'] for p in patterns.values()]
        
        pattern_idx = self.rng.choice(len(patterns), p=weights)
        pattern = list(patterns.values())[pattern_idx]
        
        duration = self.rng.normal(pattern['mean'], pattern['std'])
        return np.clip(duration, 1.0, 12.0)
    
    def generate_sessions(
        self,
        num_evs: int,
        simulation_steps: int,
        priority_ratio: float = 0.3,
        priority_selection_method: str = 'random',
        max_rate_kw: float = 22.0,
        scenario_type: str = 'workplace'
    ) -> List[EVSession]:
        """
        Generate EV charging sessions with ACN-realistic patterns.
        
        Args:
            num_evs: Number of EV sessions to generate
            simulation_steps: Total simulation time steps
            priority_ratio: Fraction of EVs designated as priority
            priority_selection_method: 'random', 'moderate_demand', or 'laxity_aware'
            max_rate_kw: Maximum charging rate per EV
            scenario_type: 'workplace', 'depot', or 'public'
            
        Returns:
            List of EVSession objects
        """
        sessions = []
        
        for i in range(num_evs):
            # Sample arrival and duration
            arrival_hour = self._sample_arrival_hour()
            duration_hours = self._sample_duration_hours()
            
            arrival_step = self._hour_to_step(arrival_hour)
            departure_step = min(
                arrival_step + self._hour_to_step(duration_hours),
                simulation_steps - 1
            )
            
            # Ensure minimum charging window
            if departure_step - arrival_step < 6:  # At least 30 minutes
                departure_step = min(arrival_step + 12, simulation_steps - 1)
            
            # Sample vehicle and energy
            vehicle_type = self._sample_vehicle_type()
            energy_requested = self._sample_energy_demand(vehicle_type)
            
            # Create session
            session = EVSession(
                session_id=f"EV_{i:03d}",
                arrival_time=arrival_step,
                departure_time=departure_step,
                energy_requested=energy_requested,
                max_rate=max_rate_kw,
                min_rate=0.0,  # Will be set for priority EVs
                is_priority=False,
                station_id=f"EVSE_{i % 15:02d}",
                user_id=f"user_{i:03d}",
                vehicle_type=vehicle_type
            )
            sessions.append(session)
        
        # Assign priority status
        sessions = self._assign_priority(
            sessions, 
            priority_ratio, 
            priority_selection_method
        )
        
        return sessions
    
    def _assign_priority(
        self,
        sessions: List[EVSession],
        priority_ratio: float,
        method: str
    ) -> List[EVSession]:
        """
        Assign priority status to sessions using specified method.
        
        Methods:
        - random: Random selection
        - moderate_demand: Prefer EVs with moderate energy demand
        - laxity_aware: Prefer EVs with sufficient charging window
        """
        num_priority = int(len(sessions) * priority_ratio)
        
        if method == 'random':
            priority_indices = self.rng.choice(
                len(sessions), 
                size=num_priority, 
                replace=False
            )
        
        elif method == 'moderate_demand':
            # Score: prefer 20-40 kWh range
            scores = []
            for s in sessions:
                if 20 <= s.energy_requested <= 40:
                    score = 1.0
                else:
                    score = 0.5 / (1 + abs(s.energy_requested - 30) / 20)
                scores.append(score)
            
            scores = np.array(scores)
            scores /= scores.sum()
            priority_indices = self.rng.choice(
                len(sessions),
                size=num_priority,
                replace=False,
                p=scores
            )
        
        elif method == 'laxity_aware':
            # Score: prefer EVs with higher laxity (more flexible)
            scores = []
            for s in sessions:
                duration_steps = s.departure_time - s.arrival_time
                time_needed_steps = (s.energy_requested / s.max_rate) * self.steps_per_hour
                laxity = duration_steps - time_needed_steps
                score = max(0.1, laxity / 12)  # Normalize
                scores.append(score)
            
            scores = np.array(scores)
            scores /= scores.sum()
            priority_indices = self.rng.choice(
                len(sessions),
                size=num_priority,
                replace=False,
                p=scores
            )
        else:
            raise ValueError(f"Unknown priority selection method: {method}")
        
        # Set priority status and minimum rate
        for idx in priority_indices:
            sessions[idx].is_priority = True
            sessions[idx].min_rate = 11.0  # Minimum rate guarantee (Amps or kW)
        
        return sessions
    
    def generate_delayed_arrivals(
        self,
        sessions: List[EVSession],
        delay_probability: float = 0.2,
        delay_steps: int = 12
    ) -> List[EVSession]:
        """
        Simulate delayed arrivals (stress test scenario).
        
        Args:
            sessions: Original sessions
            delay_probability: Probability each EV arrives late
            delay_steps: How many steps late
        """
        for s in sessions:
            if self.rng.random() < delay_probability:
                s.arrival_time += delay_steps
                # Keep departure fixed (shrinks window)
                if s.arrival_time >= s.departure_time:
                    s.departure_time = s.arrival_time + 6  # Minimum 30 min
        
        return sessions


# ==============================================================================
# GRID AND PV DATA GENERATION
# ==============================================================================


def generate_tou_schedule(tariff_type: TariffType = TariffType.TOU_DEFAULT) -> TOUSchedule:
    """Generate a TOU schedule based on tariff type."""
    
    if tariff_type == TariffType.FLAT:
        return TOUSchedule(
            peak_hours=[],
            mid_peak_hours=[],
            off_peak_hours=[(0, 24)],
            peak_price=0.20,
            mid_peak_price=0.20,
            off_peak_price=0.20
        )
    
    elif tariff_type == TariffType.TOU_DEFAULT:
        return TOUSchedule(
            peak_hours=[(16, 21)],           # 4pm - 9pm
            mid_peak_hours=[(7, 16), (21, 23)],
            off_peak_hours=[(0, 7), (23, 24)],
            peak_price=0.35,
            mid_peak_price=0.22,
            off_peak_price=0.12
        )
    
    elif tariff_type == TariffType.TOU_AGGRESSIVE:
        return TOUSchedule(
            peak_hours=[(14, 20)],           # 2pm - 8pm (extended peak)
            mid_peak_hours=[(10, 14), (20, 22)],
            off_peak_hours=[(0, 10), (22, 24)],
            peak_price=0.50,                 # Higher peak
            mid_peak_price=0.25,
            off_peak_price=0.08              # Lower off-peak
        )
    
    else:
        return generate_tou_schedule(TariffType.TOU_DEFAULT)


def generate_grid_price_data(
    num_steps: int,
    step_duration_mins: int = 5,
    tou_schedule: Optional[TOUSchedule] = None,
    tariff_type: TariffType = TariffType.TOU_DEFAULT
) -> np.ndarray:
    """
    Generate electricity price profile based on TOU schedule.
    
    Args:
        num_steps: Number of simulation steps
        step_duration_mins: Duration of each step
        tou_schedule: Optional pre-defined TOU schedule
        tariff_type: Tariff type if schedule not provided
    """
    if tou_schedule is None:
        tou_schedule = generate_tou_schedule(tariff_type)
    
    prices = np.zeros(num_steps)
    
    for t in range(num_steps):
        minute_of_day = (t * step_duration_mins) % (24 * 60)
        hour_of_day = minute_of_day / 60.0
        prices[t] = tou_schedule.get_price(hour_of_day)
    
    return prices


def generate_pv_data(
    num_steps: int,
    step_duration_mins: int = 5,
    peak_power_kw: float = 50.0,
    weather: WeatherCondition = WeatherCondition.SUNNY,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic PV generation profile.
    
    Args:
        num_steps: Number of simulation steps
        step_duration_mins: Duration of each step
        peak_power_kw: Peak PV capacity
        weather: Weather condition affecting generation
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    pv_profile = np.zeros(num_steps)
    
    for t in range(num_steps):
        minute_of_day = (t * step_duration_mins) % (24 * 60)
        # Sunrise/Sunset approximately 6am - 6pm
        day_progress = (minute_of_day - 6 * 60) / (12 * 60)
        
        if 0 <= day_progress <= 1:
            angle = day_progress * np.pi
            base_gen = np.sin(angle) * peak_power_kw
            
            if weather == WeatherCondition.SUNNY:
                noise = rng.normal(1.0, 0.05)
            elif weather == WeatherCondition.CLOUDY:
                noise = rng.uniform(0.2, 0.6)
            elif weather == WeatherCondition.VARIABLE:
                # High frequency clouds passing
                noise = rng.uniform(0.1, 1.0)
            elif weather == WeatherCondition.OVERCAST:
                noise = rng.uniform(0.1, 0.3)
            else:
                noise = 1.0
            
            pv_profile[t] = max(0, base_gen * noise)
    
    return pv_profile


# ==============================================================================
# FAIRNESS METRICS
# ==============================================================================


def calculate_jains_index(fulfillment_ratios: List[float]) -> float:
    """
    Calculate Jain's Fairness Index.
    
    J = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Returns value in [0, 1], where 1 is perfectly fair.
    """
    if not fulfillment_ratios:
        return 1.0
    
    n = len(fulfillment_ratios)
    x = np.array(fulfillment_ratios)
    
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x ** 2)
    
    if sum_x_sq == 0:
        return 1.0
    
    return (sum_x ** 2) / (n * sum_x_sq)


def calculate_fulfillment_ratios(sessions: List[EVSession]) -> Dict[str, float]:
    """Calculate fulfillment ratio for each session."""
    ratios = {}
    for s in sessions:
        if s.energy_requested > 0:
            ratios[s.session_id] = s.energy_delivered / s.energy_requested
        else:
            ratios[s.session_id] = 1.0
    return ratios


# ==============================================================================
# SCHEDULING ALGORITHMS
# ==============================================================================


class BaseScheduler(ABC):
    """Abstract base class for charging schedulers."""
    
    def __init__(
        self,
        params: OptimizationParams,
        infrastructure_limit_kw: float,
        tou_schedule: Optional[TOUSchedule] = None
    ):
        self.params = params
        self.infra_limit = infrastructure_limit_kw
        self.tou_schedule = tou_schedule or generate_tou_schedule()
    
    @abstractmethod
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute charging rates for current time step.
        
        Returns:
            Dict mapping session_id -> charging rate (kW)
        """
        pass
    
    def calculate_laxity(self, session: EVSession, current_step: int) -> float:
        """Calculate laxity (slack time) for a session."""
        remaining_energy = session.remaining_energy
        if remaining_energy <= 0.01:
            return float('inf')
        
        time_remaining_hours = (session.departure_time - current_step) * self.params.time_step_hours
        time_needed_hours = remaining_energy / session.max_rate
        
        return time_remaining_hours - time_needed_hours


class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin (RR) Scheduler.
    
    Allocates maximum charging rate to EVs in cyclic order until 
    infrastructure limit is reached.
    
    Reference: IEEE Transactions on Smart Grid, Vol. 3, No. 3, 2012
    """
    
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        if not active_sessions:
            return {}
        
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        available_cap = self.infra_limit + pv_now
        
        # Sort by session ID for deterministic ordering
        sorted_sessions = sorted(active_sessions, key=lambda s: s.session_id)
        
        # Rotate based on step for round-robin effect
        shift = current_step % len(sorted_sessions)
        rotated = sorted_sessions[shift:] + sorted_sessions[:shift]
        
        results = {}
        current_load = 0.0
        
        for session in rotated:
            if session.is_complete:
                results[session.session_id] = 0.0
                continue
            
            rate = min(session.max_rate, available_cap - current_load)
            
            if rate > 0:
                results[session.session_id] = rate
                current_load += rate
            else:
                results[session.session_id] = 0.0
        
        return results


class LeastLaxityFirstScheduler(BaseScheduler):
    """
    Least Laxity First (LLF) Scheduler.
    
    Prioritizes EVs with smallest laxity (most urgent).
    Laxity = (Time Remaining) - (Time Needed to Charge)
    
    Reference: IEEE Transactions on Vehicular Technology, Vol. 65, No. 6, 2016
    """
    
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        if not active_sessions:
            return {}
        
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        available_cap = self.infra_limit + pv_now
        
        # Calculate laxity for each session
        session_laxity = [
            (s, self.calculate_laxity(s, current_step))
            for s in active_sessions
        ]
        
        # Sort by laxity (least first)
        sorted_sessions = sorted(session_laxity, key=lambda x: x[1])
        
        results = {}
        current_load = 0.0
        
        for session, laxity in sorted_sessions:
            if laxity == float('inf'):
                results[session.session_id] = 0.0
                continue
            
            rate = min(session.max_rate, available_cap - current_load)
            
            if rate > 0:
                results[session.session_id] = rate
                current_load += rate
            else:
                results[session.session_id] = 0.0
        
        return results


class UncontrolledScheduler(BaseScheduler):
    """
    Uncontrolled Charging.
    
    All EVs charge at maximum rate (no smart scheduling).
    Infrastructure limit enforced by proportional curtailment.
    """
    
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        if not active_sessions:
            return {}
        
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        available_cap = self.infra_limit + pv_now
        
        results = {}
        total_demand = 0.0
        
        # First pass: calculate total demand
        for session in active_sessions:
            if not session.is_complete:
                total_demand += session.max_rate
        
        # Calculate curtailment ratio if needed
        if total_demand > available_cap and total_demand > 0:
            ratio = available_cap / total_demand
        else:
            ratio = 1.0
        
        # Assign rates
        for session in active_sessions:
            if session.is_complete:
                results[session.session_id] = 0.0
            else:
                results[session.session_id] = session.max_rate * ratio
        
        return results


class AQPSScheduler(BaseScheduler):
    """
    Adaptive Queuing Priority Scheduler (AQPS).
    
    Heuristic two-tier queue scheduler implementing AQPC principles
    without requiring optimization solvers. O(n log n) complexity.
    
    Features:
    - Two-tier priority queue (priority EVs always first)
    - Departure-aware TOU cost optimization
    - Preemption with highest-laxity-first policy
    - No solver dependency
    """
    
    def __init__(
        self,
        params: OptimizationParams,
        infrastructure_limit_kw: float,
        priority_sessions: Optional[Set[str]] = None,
        tou_schedule: Optional[TOUSchedule] = None,
        enable_deferral: bool = True,
        preemption_policy: str = 'highest_laxity'
    ):
        super().__init__(params, infrastructure_limit_kw, tou_schedule)
        self.priority_sessions = priority_sessions or set()
        self.enable_deferral = enable_deferral
        self.preemption_policy = preemption_policy
        self.preemption_events: List[PreemptionEvent] = []
    
    def _partition_sessions(
        self, 
        sessions: List[EVSession]
    ) -> Tuple[List[EVSession], List[EVSession]]:
        """Partition sessions into priority and non-priority queues."""
        priority = []
        non_priority = []
        
        for s in sessions:
            if s.is_priority or s.session_id in self.priority_sessions:
                priority.append(s)
            else:
                non_priority.append(s)
        
        return priority, non_priority
    
    def _can_defer_charging(
        self,
        session: EVSession,
        current_step: int,
        current_hour: float
    ) -> bool:
        """Check if charging can be deferred to a cheaper slot."""
        if not self.enable_deferral:
            return False
        
        cheaper_hour = self.tou_schedule.next_cheaper_slot(current_hour)
        if cheaper_hour is None:
            return False
        
        hours_to_cheaper = cheaper_hour - current_hour
        steps_to_cheaper = int(hours_to_cheaper / self.params.time_step_hours)
        
        remaining_steps = session.departure_time - current_step
        steps_needed = int(
            (session.remaining_energy / session.max_rate) / self.params.time_step_hours
        )
        
        slack = remaining_steps - steps_to_cheaper - steps_needed
        
        return slack > 0
    
    def _execute_preemption(
        self,
        needed_capacity: float,
        non_priority_sessions: List[EVSession],
        schedule: Dict[str, float],
        current_step: int
    ) -> Tuple[Dict[str, float], float]:
        """
        Execute preemption to free capacity for priority EVs.
        
        Option B: Target highest-laxity non-priority EVs first
        Option A (fallback): Proportional reduction of all non-priority
        """
        freed = 0.0
        preempted = []
        
        # Option B: Highest laxity first
        sessions_by_laxity = sorted(
            non_priority_sessions,
            key=lambda s: self.calculate_laxity(s, current_step),
            reverse=True  # Highest laxity first
        )
        
        for session in sessions_by_laxity:
            if freed >= needed_capacity:
                break
            
            sid = session.session_id
            current_rate = schedule.get(sid, 0)
            min_rate = session.min_rate
            
            reducible = current_rate - min_rate
            if reducible > 0:
                reduction = min(reducible, needed_capacity - freed)
                schedule[sid] = current_rate - reduction
                freed += reduction
                preempted.append(sid)
        
        # Option A fallback: Proportional reduction
        if freed < needed_capacity:
            shortfall = needed_capacity - freed
            total_reducible = sum(
                schedule.get(s.session_id, 0) - s.min_rate
                for s in non_priority_sessions
                if schedule.get(s.session_id, 0) > s.min_rate
            )
            
            if total_reducible > 0:
                ratio = shortfall / total_reducible
                for session in non_priority_sessions:
                    sid = session.session_id
                    current = schedule.get(sid, 0)
                    reduction = (current - session.min_rate) * ratio
                    schedule[sid] = current - reduction
                    freed += reduction
        
        # Log preemption event
        if preempted:
            self.preemption_events.append(PreemptionEvent(
                timestamp=current_step,
                priority_session_id="multiple",
                preempted_session_ids=preempted,
                freed_capacity=freed,
                method=self.preemption_policy
            ))
        
        return schedule, freed
    
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        if not active_sessions:
            return {}
        
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        available_cap = self.infra_limit + pv_now
        
        # Partition into priority and non-priority
        priority_queue, non_priority_queue = self._partition_sessions(active_sessions)
        
        # Sort both queues by laxity (least first)
        priority_queue = sorted(
            priority_queue,
            key=lambda s: self.calculate_laxity(s, current_step)
        )
        non_priority_queue = sorted(
            non_priority_queue,
            key=lambda s: self.calculate_laxity(s, current_step)
        )
        
        schedule = {}
        current_load = 0.0
        
        # Step 1: Allocate minimum guaranteed rate to priority EVs
        for session in priority_queue:
            if session.is_complete:
                schedule[session.session_id] = 0.0
                continue
            
            min_rate = max(session.min_rate, self.params.min_priority_rate)
            rate = min(min_rate, session.max_rate, available_cap - current_load)
            
            schedule[session.session_id] = rate
            current_load += rate
        
        # Step 2: Check if preemption needed
        priority_min_demand = sum(
            max(s.min_rate, self.params.min_priority_rate)
            for s in priority_queue if not s.is_complete
        )
        
        if priority_min_demand > available_cap:
            # Need to preempt non-priority EVs
            needed = priority_min_demand - (available_cap - current_load)
            schedule, _ = self._execute_preemption(
                needed, non_priority_queue, schedule, current_step
            )
        
        # Step 3: Allocate remaining capacity to priority EVs (up to max)
        for session in priority_queue:
            if session.is_complete:
                continue
            
            current_rate = schedule.get(session.session_id, 0)
            additional = min(
                session.max_rate - current_rate,
                available_cap - current_load
            )
            
            if additional > 0:
                schedule[session.session_id] = current_rate + additional
                current_load += additional
        
        # Step 4: Allocate to non-priority EVs with TOU awareness
        current_hour = (current_step * self.params.time_step_hours * 60) % (24 * 60) / 60
        remaining_cap = available_cap - current_load
        
        for session in non_priority_queue:
            if session.is_complete:
                schedule[session.session_id] = 0.0
                continue
            
            # TOU-aware deferral
            if self._can_defer_charging(session, current_step, current_hour):
                # Defer: only charge minimum necessary
                desired_rate = session.min_rate
            else:
                # Charge normally
                desired_rate = session.max_rate
            
            rate = min(desired_rate, remaining_cap)
            schedule[session.session_id] = rate
            remaining_cap -= rate
            current_load += rate
        
        return schedule


class AQPCOptimizer(BaseScheduler):
    """
    Adaptive Queuing-based Predictive Control (AQPC).
    
    MPC-based optimizer combining adaptive queuing (preemption) with
    convex optimization for cost minimization.
    
    Requires CVXPY with a compatible solver (OSQP, ECOS, or MOSEK).
    """
    
    def __init__(
        self,
        params: OptimizationParams,
        infrastructure_limit_kw: float,
        tou_schedule: Optional[TOUSchedule] = None,
        solver: str = 'OSQP'
    ):
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for AQPC optimizer")
        
        super().__init__(params, infrastructure_limit_kw, tou_schedule)
        self.solver = solver
    
    def _adjust_queue(
        self,
        active_sessions: List[EVSession],
        current_step: int,
        available_cap: float
    ) -> List[EVSession]:
        """
        Adaptive queue adjustment (Algorithm 1).
        
        Preemptively disconnects low-priority EVs when congestion detected.
        """
        total_max_draw = sum(s.max_rate for s in active_sessions if s.is_active)
        
        if total_max_draw > available_cap:
            # Sort: Non-priority first, then by highest laxity
            sorted_sessions = sorted(
                [s for s in active_sessions if s.is_active],
                key=lambda x: (x.is_priority, -self.calculate_laxity(x, current_step))
            )
            
            # Shed load from non-priority EVs
            current_draw = total_max_draw
            for s in sorted_sessions:
                if current_draw <= available_cap:
                    break
                if not s.is_priority:
                    s.is_active = False
                    current_draw -= s.max_rate
        
        return active_sessions
    
    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        available_cap = self.infra_limit + pv_now
        
        # Run adaptive queue logic
        active_sessions = self._adjust_queue(active_sessions, current_step, available_cap)
        
        # Filter to active sessions only
        solver_sessions = [s for s in active_sessions if s.is_active]
        
        if not solver_sessions:
            return {}
        
        T = self.params.horizon_steps
        dt = self.params.time_step_hours
        
        # Horizon slicing
        horizon_prices = grid_prices[current_step:current_step + T]
        horizon_pv = pv_generation[current_step:current_step + T]
        
        if len(horizon_prices) < T:
            horizon_prices = np.pad(horizon_prices, (0, T - len(horizon_prices)), 'edge')
        if len(horizon_pv) < T:
            horizon_pv = np.pad(horizon_pv, (0, T - len(horizon_pv)), 'edge')
        
        num_evs = len(solver_sessions)
        
        # Decision variables
        x = cp.Variable((num_evs, T), nonneg=True)
        constraints = []
        cost_penalty = 0
        
        sum_x = cp.sum(x, axis=0)
        
        for i, session in enumerate(solver_sessions):
            # Rate limits
            constraints += [x[i, :] <= session.max_rate]
            
            # Time window constraints
            rel_start = max(0, session.arrival_time - current_step)
            rel_end = min(T - 1, session.departure_time - current_step)
            
            for t in range(T):
                if t < rel_start or t > rel_end:
                    constraints += [x[i, t] == 0]
            
            remaining_demand = session.remaining_energy
            projected_delivery = cp.sum(x[i, :]) * dt
            
            if session.is_priority:
                # Hard constraint for priority EVs
                if rel_end < T:
                    max_possible = session.max_rate * (rel_end - rel_start + 1) * dt
                    if remaining_demand <= max_possible:
                        constraints += [projected_delivery >= remaining_demand]
                    else:
                        # Force max rate
                        constraints += [x[i, rel_start:rel_end + 1] == session.max_rate]
            else:
                # Soft constraint for non-priority
                constraints += [projected_delivery <= remaining_demand]
                cost_penalty += cp.square(remaining_demand - projected_delivery)
        
        # Infrastructure constraint
        constraints += [sum_x <= self.infra_limit + horizon_pv]
        
        # Objective function
        net_load = sum_x - horizon_pv
        cost_grid = cp.sum(cp.multiply(net_load, horizon_prices)) * dt
        cost_smooth = cp.sum_squares(sum_x)
        
        objective = cp.Minimize(
            self.params.alpha_energy * cost_grid
            + self.params.alpha_penalty * cost_penalty
            + self.params.alpha_smooth * cost_smooth
        )
        
        prob = cp.Problem(objective, constraints)
        
        try:
            if self.solver == 'MOSEK':
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            return {s.session_id: 0.0 for s in solver_sessions}
        
        if prob.status in ['infeasible', 'unbounded']:
            return {s.session_id: 0.0 for s in solver_sessions}
        
        results = {}
        for i, session in enumerate(solver_sessions):
            val = x[i, 0].value
            if val is None:
                val = 0.0
            results[session.session_id] = max(0.0, min(val, session.max_rate))
        
        return results


# ==============================================================================
# SIMULATION ENGINE
# ==============================================================================


def run_simulation(
    scenario_name: str,
    scheduler: BaseScheduler,
    sessions: List[EVSession],
    grid_prices: np.ndarray,
    pv_generation: np.ndarray,
    num_steps: int,
    tou_schedule: Optional[TOUSchedule] = None
) -> Tuple[SimulationMetrics, pd.DataFrame]:
    """
    Run a complete charging simulation.
    
    Args:
        scenario_name: Name for logging/identification
        scheduler: Scheduling algorithm instance
        sessions: List of EV sessions
        grid_prices: Price profile
        pv_generation: PV generation profile
        num_steps: Number of simulation steps
        tou_schedule: TOU schedule for cost breakdown
    
    Returns:
        Tuple of (SimulationMetrics, history DataFrame)
    """
    params = scheduler.params
    tou = tou_schedule or scheduler.tou_schedule
    
    # Initialize history tracking
    history = {
        'time_step': [],
        'total_load_kw': [],
        'pv_gen_kw': [],
        'grid_price': [],
        'net_grid_load_kw': [],
        'energy_delivered_kwh': [],
        'step_cost': [],
        'active_evs': [],
        'priority_evs_active': [],
    }
    
    # Deep copy sessions to avoid modifying originals
    sim_sessions = copy.deepcopy(sessions)
    step_times = []
    
    start_time = time.time()
    
    for t in range(num_steps):
        step_start = time.time()
        
        # Find active sessions
        active = [
            s for s in sim_sessions
            if s.arrival_time <= t <= s.departure_time
        ]
        
        # Reset active flag for sessions
        for s in active:
            s.is_active = True
        
        # Get charging rates
        rates = scheduler.solve(t, active, grid_prices, pv_generation)
        
        # Apply charging and track metrics
        step_energy = 0.0
        total_load = 0.0
        
        for s in active:
            rate = rates.get(s.session_id, 0.0)
            energy = rate * params.time_step_hours
            s.energy_delivered += energy
            step_energy += energy
            total_load += rate
        
        # Calculate cost
        pv_now = pv_generation[t] if t < len(pv_generation) else 0
        net_load = max(0, total_load - pv_now)
        step_cost = net_load * grid_prices[t] * params.time_step_hours
        
        # Record history
        history['time_step'].append(t)
        history['total_load_kw'].append(total_load)
        history['pv_gen_kw'].append(pv_now)
        history['grid_price'].append(grid_prices[t])
        history['net_grid_load_kw'].append(net_load)
        history['energy_delivered_kwh'].append(step_energy)
        history['step_cost'].append(step_cost)
        history['active_evs'].append(len(active))
        history['priority_evs_active'].append(
            sum(1 for s in active if s.is_priority)
        )
        
        step_times.append(time.time() - step_start)
    
    total_time = time.time() - start_time
    
    # Compute final metrics
    metrics = SimulationMetrics()
    
    # Energy metrics
    for s in sim_sessions:
        metrics.total_energy_requested += s.energy_requested
        metrics.total_energy_delivered += s.energy_delivered
        metrics.total_sessions += 1
        
        if s.is_priority:
            metrics.priority_sessions += 1
            metrics.priority_energy_requested += s.energy_requested
            metrics.priority_energy_delivered += s.energy_delivered
            if s.energy_delivered >= s.energy_requested * 0.99:
                metrics.priority_sessions_fulfilled += 1
        else:
            metrics.non_priority_sessions += 1
            metrics.non_priority_energy_requested += s.energy_requested
            metrics.non_priority_energy_delivered += s.energy_delivered
            if s.energy_delivered >= s.energy_requested * 0.99:
                metrics.non_priority_sessions_fulfilled += 1
    
    # Cost breakdown
    df_hist = pd.DataFrame(history)
    metrics.total_energy_cost = df_hist['step_cost'].sum()
    
    # Compute cost by TOU period
    for idx, row in df_hist.iterrows():
        hour = (row['time_step'] * params.time_step_hours * 60) % (24 * 60) / 60
        period = tou.get_period_type(hour)
        if period == 'peak':
            metrics.peak_period_cost += row['step_cost']
        else:
            metrics.off_peak_period_cost += row['step_cost']
    
    # Fairness metrics
    fulfillment_ratios = calculate_fulfillment_ratios(sim_sessions)
    ratios_list = list(fulfillment_ratios.values())
    metrics.jains_index = calculate_jains_index(ratios_list)
    metrics.min_fulfillment_ratio = min(ratios_list) if ratios_list else 0
    metrics.max_fulfillment_ratio = max(ratios_list) if ratios_list else 0
    
    # Computational metrics
    metrics.total_computation_time = total_time
    metrics.avg_step_time = np.mean(step_times)
    metrics.max_step_time = np.max(step_times)
    
    # Preemption metrics (if AQPS)
    if hasattr(scheduler, 'preemption_events'):
        metrics.total_preemptions = len(scheduler.preemption_events)
    
    return metrics, df_hist


# ==============================================================================
# SCENARIO DEFINITIONS (S1-S6)
# ==============================================================================


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    
    name: str
    description: str
    num_evs: int
    num_steps: int
    priority_ratio: float
    infrastructure_limit_kw: float
    tariff_type: TariffType
    weather: WeatherCondition
    pv_capacity_kw: float
    delayed_arrival_prob: float = 0.0
    seed: int = 42


def get_scenario_configs() -> Dict[str, ScenarioConfig]:
    """
    Define the six comprehensive simulation scenarios.
    
    S1: Baseline Normal Operations
    S2: High Priority Ratio Stress
    S3: Constrained Infrastructure
    S4: TOU Sensitivity Analysis
    S5: Renewable Integration (PV Variability)
    S6: Scalability Test
    """
    return {
        'S1_Baseline': ScenarioConfig(
            name='S1_Baseline',
            description='Normal operations with moderate priority ratio',
            num_evs=30,
            num_steps=288,  # 24 hours at 5-min intervals
            priority_ratio=0.27,
            infrastructure_limit_kw=150.0,
            tariff_type=TariffType.TOU_DEFAULT,
            weather=WeatherCondition.SUNNY,
            pv_capacity_kw=50.0,
            seed=42
        ),
        
        'S2_HighPriority': ScenarioConfig(
            name='S2_HighPriority',
            description='High priority ratio stress test (50% priority)',
            num_evs=30,
            num_steps=288,
            priority_ratio=0.50,
            infrastructure_limit_kw=150.0,
            tariff_type=TariffType.TOU_DEFAULT,
            weather=WeatherCondition.SUNNY,
            pv_capacity_kw=50.0,
            seed=42
        ),
        
        'S3_Constrained': ScenarioConfig(
            name='S3_Constrained',
            description='Infrastructure constrained with delayed arrivals',
            num_evs=40,
            num_steps=288,
            priority_ratio=0.30,
            infrastructure_limit_kw=80.0,  # Reduced capacity
            tariff_type=TariffType.TOU_DEFAULT,
            weather=WeatherCondition.CLOUDY,
            pv_capacity_kw=50.0,
            delayed_arrival_prob=0.25,
            seed=42
        ),
        
        'S4_TOUSensitivity': ScenarioConfig(
            name='S4_TOUSensitivity',
            description='Aggressive TOU tariff for cost optimization testing',
            num_evs=30,
            num_steps=288,
            priority_ratio=0.27,
            infrastructure_limit_kw=150.0,
            tariff_type=TariffType.TOU_AGGRESSIVE,
            weather=WeatherCondition.SUNNY,
            pv_capacity_kw=50.0,
            seed=42
        ),
        
        'S5_Renewable': ScenarioConfig(
            name='S5_Renewable',
            description='Variable PV generation (cloud cover)',
            num_evs=30,
            num_steps=288,
            priority_ratio=0.27,
            infrastructure_limit_kw=150.0,
            tariff_type=TariffType.TOU_DEFAULT,
            weather=WeatherCondition.VARIABLE,
            pv_capacity_kw=80.0,  # Higher PV capacity
            seed=42
        ),
        
        'S6_Scalability': ScenarioConfig(
            name='S6_Scalability',
            description='Large fleet scalability test',
            num_evs=100,
            num_steps=144,  # 12 hours
            priority_ratio=0.25,
            infrastructure_limit_kw=400.0,
            tariff_type=TariffType.TOU_DEFAULT,
            weather=WeatherCondition.SUNNY,
            pv_capacity_kw=100.0,
            seed=42
        ),
    }


def create_scenario_data(
    config: ScenarioConfig,
    generator: Optional[ACNScenarioGenerator] = None
) -> Tuple[List[EVSession], np.ndarray, np.ndarray, TOUSchedule]:
    """
    Create complete scenario data from configuration.
    
    Returns:
        Tuple of (sessions, grid_prices, pv_generation, tou_schedule)
    """
    if generator is None:
        generator = ACNScenarioGenerator(seed=config.seed)
    
    # Generate sessions
    sessions = generator.generate_sessions(
        num_evs=config.num_evs,
        simulation_steps=config.num_steps,
        priority_ratio=config.priority_ratio,
        priority_selection_method='laxity_aware'
    )
    
    # Apply delayed arrivals if configured
    if config.delayed_arrival_prob > 0:
        sessions = generator.generate_delayed_arrivals(
            sessions,
            delay_probability=config.delayed_arrival_prob,
            delay_steps=12
        )
    
    # Generate TOU schedule and prices
    tou_schedule = generate_tou_schedule(config.tariff_type)
    grid_prices = generate_grid_price_data(
        config.num_steps,
        tou_schedule=tou_schedule
    )
    
    # Generate PV data
    pv_generation = generate_pv_data(
        config.num_steps,
        peak_power_kw=config.pv_capacity_kw,
        weather=config.weather,
        seed=config.seed
    )
    
    return sessions, grid_prices, pv_generation, tou_schedule


# ==============================================================================
# ALGORITHM FACTORY
# ==============================================================================


def get_schedulers(
    params: OptimizationParams,
    infra_limit: float,
    tou_schedule: Optional[TOUSchedule] = None,
    include_mpc: bool = True
) -> Dict[str, BaseScheduler]:
    """
    Create all scheduler instances for comparison.
    
    Args:
        params: Optimization parameters
        infra_limit: Infrastructure power limit
        tou_schedule: TOU schedule for cost optimization
        include_mpc: Whether to include MPC-based AQPC (requires CVXPY)
    """
    schedulers = {
        'Round Robin': RoundRobinScheduler(params, infra_limit, tou_schedule),
        'LLF': LeastLaxityFirstScheduler(params, infra_limit, tou_schedule),
        'Uncontrolled': UncontrolledScheduler(params, infra_limit, tou_schedule),
        'AQPS': AQPSScheduler(
            params, infra_limit,
            tou_schedule=tou_schedule,
            enable_deferral=True
        ),
    }
    
    if include_mpc and CVXPY_AVAILABLE:
        schedulers['AQPC'] = AQPCOptimizer(params, infra_limit, tou_schedule)
    
    return schedulers


# ==============================================================================
# COMPREHENSIVE SIMULATION RUNNER
# ==============================================================================


def run_all_scenarios(
    output_dir: str = 'results',
    include_mpc: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all six scenarios with all algorithms.
    
    Returns:
        DataFrame with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = get_scenario_configs()
    params = OptimizationParams()
    results = []
    
    for scenario_name, config in scenarios.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Scenario: {config.name}")
            print(f"Description: {config.description}")
            print(f"{'='*60}")
        
        # Generate scenario data
        sessions, grid_prices, pv_gen, tou_schedule = create_scenario_data(config)
        
        # Get schedulers
        schedulers = get_schedulers(
            params,
            config.infrastructure_limit_kw,
            tou_schedule,
            include_mpc=include_mpc
        )
        
        for alg_name, scheduler in schedulers.items():
            if verbose:
                print(f"  Running {alg_name}...", end=' ')
            
            # Run simulation
            metrics, history = run_simulation(
                f"{scenario_name}_{alg_name}",
                scheduler,
                sessions,
                grid_prices,
                pv_gen,
                config.num_steps,
                tou_schedule
            )
            
            if verbose:
                print(f"Done. Priority: {metrics.priority_fulfillment_pct:.1f}%, "
                      f"Cost: ${metrics.total_energy_cost:.2f}")
            
            # Store results
            results.append({
                'Scenario': config.name,
                'Algorithm': alg_name,
                'Priority Fulfillment (%)': metrics.priority_fulfillment_pct,
                'Non-Priority Fulfillment (%)': metrics.non_priority_fulfillment_pct,
                'Overall Fulfillment (%)': metrics.overall_fulfillment_pct,
                'Total Cost ($)': metrics.total_energy_cost,
                'Peak Cost ($)': metrics.peak_period_cost,
                'Off-Peak Cost ($)': metrics.off_peak_period_cost,
                'Jains Index': metrics.jains_index,
                'Avg Step Time (ms)': metrics.avg_step_time * 1000,
                'Max Step Time (ms)': metrics.max_step_time * 1000,
                'Priority Sessions': metrics.priority_sessions,
                'Priority Fulfilled': metrics.priority_sessions_fulfilled,
                'Preemptions': metrics.total_preemptions,
            })
            
            # Save detailed history
            history.to_csv(
                f"{output_dir}/{scenario_name}_{alg_name}_history.csv",
                index=False
            )
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{output_dir}/all_results.csv", index=False)
    
    return df_results


# ==============================================================================
# VISUALIZATION
# ==============================================================================


def plot_scenario_comparison(
    df: pd.DataFrame,
    output_dir: str = 'results',
    figsize: Tuple[int, int] = (14, 10)
):
    """Generate publication-quality comparison plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Priority Fulfillment by Scenario
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Priority Fulfillment
    ax1 = axes[0, 0]
    pivot_priority = df.pivot(
        index='Scenario', 
        columns='Algorithm', 
        values='Priority Fulfillment (%)'
    )
    pivot_priority.plot(kind='bar', ax=ax1, colormap='viridis')
    ax1.set_ylabel('Priority Fulfillment (%)')
    ax1.set_title('Priority EV Fulfillment by Scenario')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right')
    ax1.tick_params(axis='x', rotation=45)
    
    # Total Cost
    ax2 = axes[0, 1]
    pivot_cost = df.pivot(
        index='Scenario',
        columns='Algorithm',
        values='Total Cost ($)'
    )
    pivot_cost.plot(kind='bar', ax=ax2, colormap='magma')
    ax2.set_ylabel('Total Energy Cost ($)')
    ax2.set_title('Operational Cost by Scenario')
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', rotation=45)
    
    # Fairness (Jain's Index)
    ax3 = axes[1, 0]
    pivot_fairness = df.pivot(
        index='Scenario',
        columns='Algorithm',
        values='Jains Index'
    )
    pivot_fairness.plot(kind='bar', ax=ax3, colormap='coolwarm')
    ax3.set_ylabel("Jain's Fairness Index")
    ax3.set_title('Fairness by Scenario')
    ax3.set_ylim(0, 1.05)
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target')
    ax3.legend(loc='lower right')
    ax3.tick_params(axis='x', rotation=45)
    
    # Computation Time
    ax4 = axes[1, 1]
    pivot_time = df.pivot(
        index='Scenario',
        columns='Algorithm',
        values='Avg Step Time (ms)'
    )
    pivot_time.plot(kind='bar', ax=ax4, colormap='plasma')
    ax4.set_ylabel('Avg Computation Time (ms)')
    ax4.set_title('Computational Performance')
    ax4.set_yscale('log')
    ax4.legend(loc='upper right')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario_comparison.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/scenario_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. Algorithm Performance Summary (Radar/Spider Chart)
    algorithms = df['Algorithm'].unique()
    metrics_to_plot = [
        'Priority Fulfillment (%)',
        'Non-Priority Fulfillment (%)',
        'Jains Index',
    ]
    
    # Normalize metrics for radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg][metrics_to_plot].mean()
        values = alg_data.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=alg)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_title('Algorithm Performance Summary (Averaged Across Scenarios)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_radar.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def plot_priority_sweep(
    output_dir: str = 'results',
    ratios: List[float] = None
):
    """Run and plot priority ratio sweep analysis."""
    
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    os.makedirs(output_dir, exist_ok=True)
    params = OptimizationParams()
    results = []
    
    print("\nRunning Priority Ratio Sweep...")
    
    generator = ACNScenarioGenerator(seed=42)
    
    for ratio in ratios:
        print(f"  Priority Ratio: {ratio*100:.0f}%")
        
        sessions = generator.generate_sessions(
            num_evs=30,
            simulation_steps=288,
            priority_ratio=ratio
        )
        
        grid_prices = generate_grid_price_data(288)
        pv_gen = generate_pv_data(288)
        
        schedulers = get_schedulers(params, 150.0, include_mpc=CVXPY_AVAILABLE)
        
        for alg_name, scheduler in schedulers.items():
            metrics, _ = run_simulation(
                f"sweep_{ratio}_{alg_name}",
                scheduler,
                sessions,
                grid_prices,
                pv_gen,
                288
            )
            
            results.append({
                'Priority Ratio (%)': ratio * 100,
                'Algorithm': alg_name,
                'Priority Fulfillment (%)': metrics.priority_fulfillment_pct,
                'Non-Priority Fulfillment (%)': metrics.non_priority_fulfillment_pct,
                'Total Cost ($)': metrics.total_energy_cost,
            })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Priority Fulfillment vs Ratio
    ax1 = axes[0]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax1.plot(
            alg_data['Priority Ratio (%)'],
            alg_data['Priority Fulfillment (%)'],
            'o-', label=alg, linewidth=2
        )
    ax1.set_xlabel('Priority EV Ratio (%)')
    ax1.set_ylabel('Priority Fulfillment (%)')
    ax1.set_title('Impact of Priority Saturation')
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-Priority Fulfillment
    ax2 = axes[1]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax2.plot(
            alg_data['Priority Ratio (%)'],
            alg_data['Non-Priority Fulfillment (%)'],
            'o-', label=alg, linewidth=2
        )
    ax2.set_xlabel('Priority EV Ratio (%)')
    ax2.set_ylabel('Non-Priority Fulfillment (%)')
    ax2.set_title('Non-Priority EV Service Level')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cost
    ax3 = axes[2]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax3.plot(
            alg_data['Priority Ratio (%)'],
            alg_data['Total Cost ($)'],
            'o-', label=alg, linewidth=2
        )
    ax3.set_xlabel('Priority EV Ratio (%)')
    ax3.set_ylabel('Total Cost ($)')
    ax3.set_title('Operational Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/priority_sweep.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/priority_sweep.pdf", bbox_inches='tight')
    plt.close()
    
    df.to_csv(f"{output_dir}/priority_sweep_results.csv", index=False)
    print(f"Priority sweep results saved to {output_dir}/")


def plot_scalability_analysis(
    output_dir: str = 'results',
    fleet_sizes: List[int] = None
):
    """Run and plot scalability analysis."""
    
    if fleet_sizes is None:
        fleet_sizes = [20, 40, 60, 80, 100, 150, 200]
    
    os.makedirs(output_dir, exist_ok=True)
    params = OptimizationParams()
    results = []
    
    print("\nRunning Scalability Analysis...")
    
    for n in fleet_sizes:
        print(f"  Fleet Size: {n}")
        
        generator = ACNScenarioGenerator(seed=42)
        sessions = generator.generate_sessions(
            num_evs=n,
            simulation_steps=72,  # 6 hours for speed
            priority_ratio=0.27
        )
        
        grid_prices = generate_grid_price_data(72)
        pv_gen = generate_pv_data(72)
        infra_limit = n * 4.0  # Scale infrastructure
        
        schedulers = get_schedulers(params, infra_limit, include_mpc=CVXPY_AVAILABLE)
        
        for alg_name, scheduler in schedulers.items():
            metrics, _ = run_simulation(
                f"scale_{n}_{alg_name}",
                scheduler,
                sessions,
                grid_prices,
                pv_gen,
                72
            )
            
            results.append({
                'Fleet Size': n,
                'Algorithm': alg_name,
                'Avg Step Time (ms)': metrics.avg_step_time * 1000,
                'Max Step Time (ms)': metrics.max_step_time * 1000,
                'Total Time (s)': metrics.total_computation_time,
            })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax1.plot(
            alg_data['Fleet Size'],
            alg_data['Avg Step Time (ms)'],
            'o-', label=alg, linewidth=2
        )
    ax1.axhline(y=1000, color='red', linestyle='--', label='Real-time Limit (1s)')
    ax1.set_xlabel('Fleet Size (EVs)')
    ax1.set_ylabel('Avg Computation Time per Step (ms)')
    ax1.set_title('Computational Scalability')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax2.plot(
            alg_data['Fleet Size'],
            alg_data['Total Time (s)'],
            'o-', label=alg, linewidth=2
        )
    ax2.set_xlabel('Fleet Size (EVs)')
    ax2.set_ylabel('Total Simulation Time (s)')
    ax2.set_title('Total Execution Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scalability_analysis.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/scalability_analysis.pdf", bbox_inches='tight')
    plt.close()
    
    df.to_csv(f"{output_dir}/scalability_results.csv", index=False)
    print(f"Scalability results saved to {output_dir}/")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def main():
    """Run complete simulation study."""
    
    print("="*70)
    print("AQPC/AQPS EV Fleet Charging Optimization - Simulation Study")
    print("="*70)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check CVXPY availability
    if CVXPY_AVAILABLE:
        print("\n✓ CVXPY available - MPC-based AQPC will be included")
    else:
        print("\n✗ CVXPY not available - Only heuristic algorithms will run")
    
    # Run all scenarios
    print("\n" + "-"*70)
    print("PHASE 1: Running All Scenarios (S1-S6)")
    print("-"*70)
    
    df_results = run_all_scenarios(
        output_dir=output_dir,
        include_mpc=CVXPY_AVAILABLE,
        verbose=True
    )
    
    # Generate comparison plots
    print("\n" + "-"*70)
    print("PHASE 2: Generating Comparison Plots")
    print("-"*70)
    
    plot_scenario_comparison(df_results, output_dir)
    
    # Priority ratio sweep
    print("\n" + "-"*70)
    print("PHASE 3: Priority Ratio Sensitivity Analysis")
    print("-"*70)
    
    plot_priority_sweep(output_dir)
    
    # Scalability analysis
    print("\n" + "-"*70)
    print("PHASE 4: Scalability Analysis")
    print("-"*70)
    
    plot_scalability_analysis(output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SIMULATION STUDY COMPLETE")
    print("="*70)
    
    print("\nResults Summary:")
    print(df_results.groupby('Algorithm').agg({
        'Priority Fulfillment (%)': 'mean',
        'Total Cost ($)': 'mean',
        'Jains Index': 'mean',
        'Avg Step Time (ms)': 'mean'
    }).round(2))
    
    print(f"\nAll results saved to '{output_dir}/' directory")
    print("  - all_results.csv: Complete results table")
    print("  - scenario_comparison.png/pdf: Main comparison plots")
    print("  - priority_sweep.png/pdf: Priority ratio analysis")
    print("  - scalability_analysis.png/pdf: Computational scaling")
    print("  - *_history.csv: Detailed simulation histories")


if __name__ == "__main__":
    main()

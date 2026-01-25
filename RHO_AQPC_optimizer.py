"""
RHO-AQPC: Receding Horizon Optimization with Adaptive Queuing Priority Charging
================================================================================

Implementation of the AQPC algorithm with Receding Horizon Optimization (RHO)
based on the refined mathematical formulation from peer review.

Mathematical Formulation:
-------------------------
Objective Function:
    J = min α₁·H^EC + α₂·H^NC + α₃·H^QC

Where:
    H^EC = Σ γ(t)·[Σ x_{j,t} - P_PV(t) - P_BESS(t)]·Δt    (Grid Energy Cost)
    H^NC = Σ_{j∈NP} (e_j - Σ x_{j,t}·Δt)²                  (Non-Completion Penalty)
    H^QC = Σ (Σ x_{j,t})²                                   (Smoothing/Convergence)

Constraints:
    - Priority (j ∈ P):     E_j ≥ e_j  (Hard lower bound - must fulfill)
    - Non-Priority (j ∈ NP): E_j ≤ e_j  (Soft upper bound - can curtail)
    - Infrastructure:        Σ x_{j,t} ≤ P_max + P_PV(t) + P_BESS(t)
    - Rate limits:           0 ≤ x_{j,t} ≤ x̄_j

Algorithm Structure (Two-Layer):
    Layer 1: ADJUST_QUEUE - Adaptive queue management with preemption
    Layer 2: Predictive Optimization - MPC with CVXPY/MOSEK

RHO Implementation:
    - Solve QP (or CONVEX optimization) over horizon [t, t+N] at each time step
    - Apply only first control action x_{j,1}
    - Shift horizon and repeat

Dependencies:
    - numpy
    - cvxpy (with OSQP, ECOS, or MOSEK solver)
    - pandas
    - matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import time
import copy
import warnings

# CVXPY import
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None  # Placeholder

# Plot styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class EVSession:
    """
    EV charging session with all parameters needed for AQPC optimization.

    Notation from paper:
        - x_{j,t}: Charging power (kW) for EV j at time t
        - e_j: Energy demand (kWh)
        - T_j: Set of time steps when EV j is connected
    """

    session_id: str

    # Timing (indices into simulation time steps)
    arrival_time: int  # First time step available
    departure_time: int  # Last time step available

    # Energy parameters
    energy_requested: float  # e_j: Energy demand (kWh)
    max_rate: float  # x̄_j: Maximum charging rate (kW)
    min_rate: float = 0.0  # Minimum charging rate (kW)

    # Priority status
    is_priority: bool = False  # True if j ∈ P, False if j ∈ NP

    # State tracking
    energy_delivered: float = 0.0
    is_active: bool = True  # Can be set False by ADJUST_QUEUE (preemption)

    # Laxity tracking for ADJUST_QUEUE
    laxity: float = float("inf")

    def __post_init__(self):
        """Initialize derived quantities."""
        self.original_departure = self.departure_time

    @property
    def remaining_energy(self) -> float:
        """e_j - E_j: Remaining energy to be delivered (kWh)."""
        return max(0, self.energy_requested - self.energy_delivered)

    @property
    def is_complete(self) -> bool:
        """Check if energy demand is satisfied."""
        return self.remaining_energy < 0.01

    def calculate_laxity(self, current_step: int, dt_hours: float) -> float:
        """
        Calculate laxity (slack time) for ADJUST_QUEUE priority scoring.

        Laxity = (Remaining dwell time) - (Time needed to charge at max rate)

        High laxity = More flexible, can be preempted
        Low laxity = Urgent, should not be preempted
        """
        if self.is_complete:
            return float("inf")

        remaining_steps = self.departure_time - current_step
        remaining_time_hours = remaining_steps * dt_hours
        time_needed_hours = self.remaining_energy / self.max_rate

        self.laxity = remaining_time_hours - time_needed_hours
        return self.laxity

    def is_available_at(self, time_step: int) -> bool:
        """Check if EV is connected at given time step (t ∈ T_j)."""
        return self.arrival_time <= time_step <= self.departure_time


@dataclass
class OptimizationParams:
    """
    Parameters for the AQPC optimization problem.

    Notation from paper:
        - α₁, α₂, α₃: Objective function weights
        - N: Prediction horizon steps
        - Δt: Time step duration (hours)
    """

    # Objective weights
    alpha_energy: float = 1.0  # α₁: Weight for H^EC (grid energy cost)
    alpha_penalty: float = 100.0  # α₂: Weight for H^NC (non-completion penalty)
    alpha_smooth: float = 0.1  # α₃: Weight for H^QC (smoothing/convergence)

    # Time parameters
    horizon_steps: int = 12  # N: Prediction horizon (e.g., 12 steps = 1 hour at 5-min)
    time_step_hours: float = 5 / 60  # Δt: Duration of each time step (hours)

    # Infrastructure limits
    infrastructure_limit_kw: float = 150.0  # P_max: Maximum grid import (kW)

    # Priority settings
    min_priority_rate: float = 11.0  # Minimum guaranteed rate for priority EVs (kW)

    # Solver settings
    solver: str = "OSQP"  # CVXPY solver: 'OSQP', 'ECOS', 'MOSEK'
    solver_verbose: bool = False


@dataclass
class RHOState:
    """
    State container for Receding Horizon Optimization.

    Tracks the algorithm state across time steps for RHO implementation.
    """

    current_step: int = 0

    # Session tracking
    active_set: Dict[str, EVSession] = field(default_factory=dict)  # V_t: Active EV set
    completed_sessions: Dict[str, EVSession] = field(default_factory=dict)
    preempted_sessions: List[str] = field(default_factory=list)

    # Metrics
    solve_times: List[float] = field(default_factory=list)
    preemption_events: List[Dict] = field(default_factory=list)
    infeasibility_events: List[Dict] = field(default_factory=list)


@dataclass
class SimulationMetrics:
    """Comprehensive metrics from simulation run."""

    # Energy fulfillment
    total_energy_requested: float = 0.0
    total_energy_delivered: float = 0.0
    priority_energy_requested: float = 0.0
    priority_energy_delivered: float = 0.0
    non_priority_energy_requested: float = 0.0
    non_priority_energy_delivered: float = 0.0

    # Session counts
    total_sessions: int = 0
    priority_sessions: int = 0
    priority_sessions_fulfilled: int = 0
    non_priority_sessions: int = 0

    # Cost metrics
    total_energy_cost: float = 0.0

    # Fairness
    jains_index: float = 0.0

    # Computational
    total_solve_time: float = 0.0
    avg_solve_time_ms: float = 0.0
    max_solve_time_ms: float = 0.0

    # RHO/AQPC specific
    total_preemptions: int = 0
    total_infeasibilities: int = 0

    @property
    def priority_fulfillment_pct(self) -> float:
        if self.priority_energy_requested > 0:
            return (
                self.priority_energy_delivered / self.priority_energy_requested
            ) * 100
        return 100.0

    @property
    def non_priority_fulfillment_pct(self) -> float:
        if self.non_priority_energy_requested > 0:
            return (
                self.non_priority_energy_delivered / self.non_priority_energy_requested
            ) * 100
        return 100.0

    @property
    def overall_fulfillment_pct(self) -> float:
        if self.total_energy_requested > 0:
            return (self.total_energy_delivered / self.total_energy_requested) * 100
        return 100.0


# ==============================================================================
# RHO-AQPC OPTIMIZER
# ==============================================================================


class RHOAQPCOptimizer:
    """
    Receding Horizon Optimization with Adaptive Queuing Priority Charging.

    Implements the two-layer AQPC algorithm with RHO:

    Layer 1: ADJUST_QUEUE
        - State monitoring: Identify plugged-in EVs
        - Priority scoring: Calculate urgency (laxity)
        - Conflict resolution: Preempt low-priority high-laxity EVs if needed
        - Generate active set V_t for optimizer

    Layer 2: Predictive Optimization (MPC)
        - Build QP problem with objective J = α₁H^EC + α₂H^NC + α₃H^QC
        - Apply constraints (priority hard, non-priority soft, infrastructure)
        - Solve with CVXPY
        - Extract x_{j,1} for immediate application

    RHO Loop:
        - At each time step t:
            1. Run ADJUST_QUEUE to get V_t
            2. Solve QP over horizon [t, t+N]
            3. Apply first control action x_{j,1}
            4. Update states (energy delivered, SoC)
            5. Shift horizon: t ← t + 1
    """

    def __init__(
        self, params: OptimizationParams, priority_sessions: Optional[Set[str]] = None
    ):
        """
        Initialize RHO-AQPC optimizer.

        Args:
            params: Optimization parameters
            priority_sessions: Set of session IDs designated as priority
        """
        self.params = params
        self.priority_sessions = priority_sessions or set()
        self.state = RHOState()

        # Validate CVXPY
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "CVXPY is required for RHO-AQPC optimization. "
                "Install with: pip install cvxpy"
            )

    # ==========================================================================
    # LAYER 1: ADJUST_QUEUE
    # ==========================================================================

    def adjust_queue(
        self,
        all_sessions: List[EVSession],
        current_step: int,
        available_capacity: float,
    ) -> List[EVSession]:
        """
        ADJUST_QUEUE: Adaptive Queue Adjustment (Layer 1).

        Algorithm from Section 2.1 of peer review document:
        1. State Monitoring: Identify current plugged-in EVs
        2. Priority Scoring: Calculate urgency based on laxity
        3. Conflict Resolution: Preempt if needed
        4. Generate active set V_t

        Args:
            all_sessions: All EV sessions in the system
            current_step: Current time step
            available_capacity: Available charging capacity (kW)

        Returns:
            List of active sessions (V_t) for the optimizer
        """
        dt = self.params.time_step_hours

        # Step 1: State Monitoring - Get currently connected EVs
        connected = [
            s
            for s in all_sessions
            if s.is_available_at(current_step) and not s.is_complete
        ]

        # Reset active status
        for s in connected:
            s.is_active = True

        if not connected:
            return []

        # Step 2: Priority Scoring - Calculate laxity for all
        for s in connected:
            s.calculate_laxity(current_step, dt)

        # Separate priority and non-priority
        priority_evs = [
            s
            for s in connected
            if s.is_priority or s.session_id in self.priority_sessions
        ]
        non_priority_evs = [s for s in connected if s not in priority_evs]

        # Calculate total demand
        total_max_demand = sum(s.max_rate for s in connected)
        priority_min_demand = sum(
            max(s.min_rate, self.params.min_priority_rate) for s in priority_evs
        )

        # Step 3: Conflict Resolution - Check if preemption needed
        if priority_min_demand > available_capacity:
            # Critical: Cannot even serve priority EVs
            # Log this as an infeasibility event
            self.state.infeasibility_events.append(
                {
                    "step": current_step,
                    "priority_demand": priority_min_demand,
                    "available_capacity": available_capacity,
                    "shortfall": priority_min_demand - available_capacity,
                }
            )
            # Still try to serve as many priority EVs as possible
            # Sort priority by laxity (lowest first = most urgent)
            priority_evs.sort(key=lambda s: s.laxity)

        elif total_max_demand > available_capacity:
            # Need to preempt some non-priority EVs
            # Sort non-priority by laxity descending (highest laxity = most flexible)
            non_priority_evs.sort(key=lambda s: s.laxity, reverse=True)

            current_demand = total_max_demand
            preempted = []

            for s in non_priority_evs:
                if current_demand <= available_capacity:
                    break
                # Preempt this EV
                s.is_active = False
                current_demand -= s.max_rate
                preempted.append(s.session_id)

            if preempted:
                self.state.preemption_events.append(
                    {
                        "step": current_step,
                        "preempted": preempted,
                        "freed_capacity": total_max_demand - current_demand,
                        "reason": "capacity_constraint",
                    }
                )
                self.state.preempted_sessions.extend(preempted)

        # Step 4: Generate active set V_t
        active_set = [s for s in connected if s.is_active]

        return active_set

    # ==========================================================================
    # LAYER 2: PREDICTIVE OPTIMIZATION (MPC)
    # ==========================================================================

    def build_and_solve_qp(
        self,
        active_sessions: List[EVSession],
        current_step: int,
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
        bess_power: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Build and solve the AQPC Quadratic Program (Layer 2).

        Objective Function (Eq. 1):
            J = min α₁·H^EC + α₂·H^NC + α₃·H^QC

        Where:
            H^EC = Σ γ(t)·[Σ x_{j,t} - P_PV(t) - P_BESS(t)]·Δt  (Eq. 2)
            H^NC = Σ_{j∈NP} (e_j - Σ x_{j,t}·Δt)²               (Eq. 3)
            H^QC = Σ (Σ x_{j,t})²                                (Eq. 4)

        Constraints:
            - Priority (j ∈ P):     Σ x_{j,t}·Δt ≥ e_j  (Hard)
            - Non-Priority (j ∈ NP): Σ x_{j,t}·Δt ≤ e_j  (Soft upper bound)
            - Infrastructure:        Σ x_{j,t} ≤ P_max + P_PV(t) + P_BESS(t)
            - Rate limits:           0 ≤ x_{j,t} ≤ x̄_j
            - Availability:          x_{j,t} = 0 if t ∉ T_j

        Args:
            active_sessions: List of active EVs (V_t from ADJUST_QUEUE)
            current_step: Current simulation time step
            grid_prices: γ(t) - Electricity prices ($/kWh)
            pv_generation: P_PV(t) - Solar power (kW)
            bess_power: P_BESS(t) - Battery storage power (kW), optional

        Returns:
            Dict mapping session_id -> charging rate (kW) for current step
        """
        if not active_sessions:
            return {}

        # Extract parameters
        N = self.params.horizon_steps
        dt = self.params.time_step_hours
        P_max = self.params.infrastructure_limit_kw

        α1 = self.params.alpha_energy
        α2 = self.params.alpha_penalty
        α3 = self.params.alpha_smooth

        # Prepare horizon data
        horizon_prices = self._get_horizon_data(grid_prices, current_step, N)
        horizon_pv = self._get_horizon_data(pv_generation, current_step, N)

        if bess_power is not None:
            horizon_bess = self._get_horizon_data(bess_power, current_step, N)
        else:
            horizon_bess = np.zeros(N)

        # Number of EVs
        num_evs = len(active_sessions)

        # =========================
        # CVXPY Problem Formulation
        # =========================

        # Decision variables: x[j, t] = charging power for EV j at horizon step t
        x = cp.Variable((num_evs, N), nonneg=True)

        constraints = []

        # Track priority vs non-priority for H^NC
        priority_indices = []
        non_priority_indices = []

        for i, session in enumerate(active_sessions):
            # Determine priority status
            if session.is_priority or session.session_id in self.priority_sessions:
                priority_indices.append(i)
            else:
                non_priority_indices.append(i)

            # ----- Rate Limit Constraints -----
            # 0 ≤ x_{j,t} ≤ x̄_j (nonnegativity already in Variable definition)
            constraints.append(x[i, :] <= session.max_rate)

            # ----- Availability Constraints -----
            # x_{j,t} = 0 if t ∉ T_j
            rel_arrival = session.arrival_time - current_step
            rel_departure = session.departure_time - current_step

            for t in range(N):
                if t < rel_arrival or t > rel_departure:
                    constraints.append(x[i, t] == 0)

            # ----- Energy Constraints -----
            # E_j = Σ x_{j,t} · Δt (energy delivered over horizon)
            E_j = cp.sum(x[i, :]) * dt
            e_j = session.remaining_energy  # Remaining demand

            if session.is_priority or session.session_id in self.priority_sessions:
                # Priority: E_j ≥ e_j (Hard constraint - must fulfill)
                # Only enforce if departure is within horizon
                if rel_departure < N and e_j > 0:
                    # Check if constraint is feasible
                    available_steps = max(
                        0, min(rel_departure, N - 1) - max(0, rel_arrival) + 1
                    )
                    max_possible = session.max_rate * available_steps * dt

                    if e_j <= max_possible:
                        constraints.append(E_j >= e_j)
                    else:
                        # Infeasible - charge at max rate
                        for t in range(max(0, rel_arrival), min(N, rel_departure + 1)):
                            constraints.append(x[i, t] == session.max_rate)
            else:
                # Non-Priority: E_j ≤ e_j (Soft upper bound)
                constraints.append(E_j <= e_j)

        # ----- Infrastructure Constraints -----
        # Σ x_{j,t} ≤ P_max + P_PV(t) + P_BESS(t)
        for t in range(N):
            total_capacity = P_max + horizon_pv[t] + horizon_bess[t]
            constraints.append(cp.sum(x[:, t]) <= total_capacity)

        # =========================
        # Objective Function
        # =========================

        # Sum of charging power at each time step
        sum_x = cp.sum(x, axis=0)  # Shape: (N,)

        # ----- H^EC: Grid Energy Cost (Eq. 2) -----
        # H^EC = Σ γ(t) · [Σ x_{j,t} - P_PV(t) - P_BESS(t)] · Δt
        net_grid_load = sum_x - horizon_pv - horizon_bess
        H_EC = cp.sum(cp.multiply(net_grid_load, horizon_prices)) * dt

        # ----- H^NC: Non-Completion Penalty (Eq. 3) -----
        # H^NC = Σ_{j∈NP} (e_j - Σ x_{j,t} · Δt)²
        H_NC = 0
        for i in non_priority_indices:
            session = active_sessions[i]
            e_j = session.remaining_energy
            E_j = cp.sum(x[i, :]) * dt
            H_NC += cp.square(e_j - E_j)

        # ----- H^QC: Smoothing/Convergence (Eq. 4) -----
        # H^QC = Σ (Σ x_{j,t})²
        H_QC = cp.sum_squares(sum_x)

        # ----- Combined Objective -----
        objective = cp.Minimize(α1 * H_EC + α2 * H_NC + α3 * H_QC)

        # =========================
        # Solve Problem
        # =========================

        problem = cp.Problem(objective, constraints)

        solve_start = time.time()
        try:
            if self.params.solver == "MOSEK":
                problem.solve(solver=cp.MOSEK, verbose=self.params.solver_verbose)
            elif self.params.solver == "ECOS":
                problem.solve(solver=cp.ECOS, verbose=self.params.solver_verbose)
            else:
                problem.solve(solver=cp.OSQP, verbose=self.params.solver_verbose)
        except cp.SolverError as e:
            warnings.warn(f"Solver error at step {current_step}: {e}")
            return {s.session_id: 0.0 for s in active_sessions}

        solve_time = time.time() - solve_start
        self.state.solve_times.append(solve_time)

        # Check solution status
        if problem.status in ["infeasible", "unbounded", None]:
            warnings.warn(f"Problem {problem.status} at step {current_step}")
            self.state.infeasibility_events.append(
                {"step": current_step, "status": problem.status, "num_evs": num_evs}
            )
            return {s.session_id: 0.0 for s in active_sessions}

        # =========================
        # Extract First Step Rates (RHO: Apply x_{j,1})
        # =========================

        results = {}
        for i, session in enumerate(active_sessions):
            val = x[i, 0].value
            if val is None:
                val = 0.0
            # Ensure within bounds
            rate = max(0.0, min(float(val), session.max_rate))
            results[session.session_id] = rate

        return results

    def _get_horizon_data(
        self, data: np.ndarray, current_step: int, horizon: int
    ) -> np.ndarray:
        """Extract and pad horizon data from full array."""
        end_idx = current_step + horizon

        if current_step >= len(data):
            return np.full(horizon, data[-1] if len(data) > 0 else 0.0)

        available = data[current_step : min(end_idx, len(data))]

        if len(available) < horizon:
            # Pad with last value
            padding = np.full(
                horizon - len(available), available[-1] if len(available) > 0 else 0.0
            )
            return np.concatenate([available, padding])

        return available

    # ==========================================================================
    # MAIN RHO SOLVE METHOD
    # ==========================================================================

    def solve(
        self,
        current_step: int,
        all_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
        bess_power: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Main RHO-AQPC solve method.

        Implements the complete RHO loop for one time step:
        1. Run ADJUST_QUEUE (Layer 1) to get active set V_t
        2. Build and solve QP (Layer 2) over horizon [t, t+N]
        3. Return first-step rates x_{j,1} for immediate application

        This method is called at each simulation time step.

        Args:
            current_step: Current time step t
            all_sessions: All EV sessions
            grid_prices: γ(t) price profile
            pv_generation: P_PV(t) solar profile
            bess_power: P_BESS(t) battery profile (optional)

        Returns:
            Dict mapping session_id -> charging rate (kW)
        """
        self.state.current_step = current_step

        # Calculate available capacity
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        bess_now = (
            bess_power[current_step]
            if bess_power is not None and current_step < len(bess_power)
            else 0
        )
        available_capacity = self.params.infrastructure_limit_kw + pv_now + bess_now

        # Layer 1: ADJUST_QUEUE
        active_sessions = self.adjust_queue(
            all_sessions, current_step, available_capacity
        )

        if not active_sessions:
            return {}

        # Layer 2: Predictive Optimization
        rates = self.build_and_solve_qp(
            active_sessions, current_step, grid_prices, pv_generation, bess_power
        )

        return rates

    def get_metrics(self) -> Dict:
        """Get solver performance metrics."""
        if not self.state.solve_times:
            return {}

        return {
            "avg_solve_time_ms": np.mean(self.state.solve_times) * 1000,
            "max_solve_time_ms": np.max(self.state.solve_times) * 1000,
            "total_preemptions": len(self.state.preemption_events),
            "total_infeasibilities": len(self.state.infeasibility_events),
        }


# ==============================================================================
# SIMULATION ENGINE
# ==============================================================================


def run_rho_aqpc_simulation(
    scenario_name: str,
    optimizer: RHOAQPCOptimizer,
    sessions: List[EVSession],
    grid_prices: np.ndarray,
    pv_generation: np.ndarray,
    num_steps: int,
    bess_power: np.ndarray = None,
) -> Tuple[SimulationMetrics, pd.DataFrame]:
    """
    Run complete RHO-AQPC simulation.

    At each time step:
    1. Call optimizer.solve() which runs ADJUST_QUEUE + QP
    2. Apply returned rates x_{j,1} to EVs
    3. Update energy delivered
    4. Record metrics
    5. Advance to next step (RHO shift)

    Args:
        scenario_name: Name for identification
        optimizer: RHO-AQPC optimizer instance
        sessions: List of EV sessions
        grid_prices: Price profile
        pv_generation: PV generation profile
        num_steps: Total simulation steps
        bess_power: BESS power profile (optional)

    Returns:
        Tuple of (SimulationMetrics, history DataFrame)
    """
    params = optimizer.params
    dt = params.time_step_hours

    # Deep copy sessions
    sim_sessions = copy.deepcopy(sessions)

    # History tracking
    history = {
        "time_step": [],
        "hour": [],
        "total_load_kw": [],
        "pv_gen_kw": [],
        "grid_price": [],
        "net_grid_load_kw": [],
        "energy_delivered_kwh": [],
        "step_cost": [],
        "active_evs": [],
        "priority_evs": [],
        "solve_time_ms": [],
    }

    total_start = time.time()

    # Main simulation loop
    for t in range(num_steps):
        # Get charging rates from RHO-AQPC
        rates = optimizer.solve(t, sim_sessions, grid_prices, pv_generation, bess_power)

        # Apply rates and track metrics
        step_energy = 0.0
        total_load = 0.0
        active_count = 0
        priority_count = 0

        for session in sim_sessions:
            if session.is_available_at(t) and not session.is_complete:
                active_count += 1
                if session.is_priority:
                    priority_count += 1

                rate = rates.get(session.session_id, 0.0)
                energy = rate * dt
                session.energy_delivered += energy
                step_energy += energy
                total_load += rate

        # Calculate cost
        pv_now = pv_generation[t] if t < len(pv_generation) else 0
        bess_now = (
            bess_power[t] if bess_power is not None and t < len(bess_power) else 0
        )
        net_load = max(0, total_load - pv_now - bess_now)
        price = grid_prices[t] if t < len(grid_prices) else grid_prices[-1]
        step_cost = net_load * price * dt

        # Get solve time
        solve_time_ms = (
            optimizer.state.solve_times[-1] * 1000 if optimizer.state.solve_times else 0
        )

        # Record history
        history["time_step"].append(t)
        history["hour"].append(t * dt)
        history["total_load_kw"].append(total_load)
        history["pv_gen_kw"].append(pv_now)
        history["grid_price"].append(price)
        history["net_grid_load_kw"].append(net_load)
        history["energy_delivered_kwh"].append(step_energy)
        history["step_cost"].append(step_cost)
        history["active_evs"].append(active_count)
        history["priority_evs"].append(priority_count)
        history["solve_time_ms"].append(solve_time_ms)

    total_time = time.time() - total_start

    # Compute final metrics
    metrics = SimulationMetrics()

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

    # Cost
    metrics.total_energy_cost = sum(history["step_cost"])

    # Computational
    metrics.total_solve_time = total_time
    if optimizer.state.solve_times:
        metrics.avg_solve_time_ms = np.mean(optimizer.state.solve_times) * 1000
        metrics.max_solve_time_ms = np.max(optimizer.state.solve_times) * 1000

    # Preemptions/infeasibilities
    metrics.total_preemptions = len(optimizer.state.preemption_events)
    metrics.total_infeasibilities = len(optimizer.state.infeasibility_events)

    # Fairness (Jain's Index)
    ratios = [
        s.energy_delivered / s.energy_requested
        for s in sim_sessions
        if s.energy_requested > 0
    ]
    if ratios:
        n = len(ratios)
        sum_x = sum(ratios)
        sum_x_sq = sum(r**2 for r in ratios)
        metrics.jains_index = (sum_x**2) / (n * sum_x_sq) if sum_x_sq > 0 else 1.0

    return metrics, pd.DataFrame(history)


# ==============================================================================
# SCENARIO GENERATION
# ==============================================================================


def generate_tou_prices(
    num_steps: int,
    dt_hours: float = 5 / 60,
    peak_price: float = 0.35,
    mid_peak_price: float = 0.22,
    off_peak_price: float = 0.12,
) -> np.ndarray:
    """Generate TOU electricity prices."""
    prices = np.zeros(num_steps)

    for t in range(num_steps):
        hour = (t * dt_hours) % 24

        # Peak: 2pm - 8pm (14-20)
        if 14 <= hour < 20:
            prices[t] = peak_price
        # Mid-peak: 7am - 2pm, 8pm - 10pm
        elif (7 <= hour < 14) or (20 <= hour < 22):
            prices[t] = mid_peak_price
        # Off-peak: 10pm - 7am
        else:
            prices[t] = off_peak_price

    return prices


def generate_pv_profile(
    num_steps: int,
    dt_hours: float = 5 / 60,
    peak_power_kw: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic PV generation profile."""
    rng = np.random.default_rng(seed)
    pv = np.zeros(num_steps)

    for t in range(num_steps):
        hour = (t * dt_hours) % 24
        # Daylight: 6am - 6pm
        if 6 <= hour <= 18:
            day_progress = (hour - 6) / 12
            angle = day_progress * np.pi
            base = np.sin(angle) * peak_power_kw
            noise = rng.normal(1.0, 0.1)
            pv[t] = max(0, base * noise)

    return pv


def generate_ev_sessions(
    num_evs: int,
    num_steps: int,
    dt_hours: float = 5 / 60,
    priority_ratio: float = 0.3,
    seed: int = 42,
) -> List[EVSession]:
    """Generate synthetic EV sessions."""
    rng = np.random.default_rng(seed)
    sessions = []
    steps_per_hour = 1 / dt_hours

    for i in range(num_evs):
        # Arrival: 7am - 11am
        arr_hour = rng.uniform(7, 11)
        arr_step = int(arr_hour * steps_per_hour)

        # Duration: 6-12 hours
        duration = rng.uniform(6, 12)
        dep_step = int(arr_step + duration * steps_per_hour)
        dep_step = min(dep_step, num_steps - 1)

        # Energy: 20-50 kWh
        energy = rng.uniform(20, 50)

        # Priority
        is_priority = rng.random() < priority_ratio

        session = EVSession(
            session_id=f"EV_{i:03d}",
            arrival_time=arr_step,
            departure_time=dep_step,
            energy_requested=energy,
            max_rate=6.6,  # Level 2 charger
            min_rate=3.3 if is_priority else 0.0,
            is_priority=is_priority,
        )
        sessions.append(session)

    return sessions


# ==============================================================================
# DEMONSTRATION
# ==============================================================================


def run_demonstration():
    """Run RHO-AQPC demonstration."""

    print("=" * 70)
    print("RHO-AQPC: Receding Horizon Optimization with")
    print("         Adaptive Queuing Priority Charging")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("MATHEMATICAL FORMULATION (from Peer Review Document)")
    print("=" * 70)

    print(
        """
    OBJECTIVE FUNCTION (Eq. 1):
    ──────────────────────────
    J = min α₁·H^EC + α₂·H^NC + α₃·H^QC
    
    WHERE:
    
    H^EC = Grid Energy Cost (Eq. 2):
           Σ γ(t)·[Σ x_{j,t} - P_PV(t) - P_BESS(t)]·Δt
           
    H^NC = Non-Completion Penalty (Eq. 3):
           Σ_{j∈NP} (e_j - Σ x_{j,t}·Δt)²
           
    H^QC = Smoothing/Convergence (Eq. 4):
           Σ (Σ x_{j,t})²
    
    CONSTRAINTS:
    ────────────
    • Priority (j ∈ P):      E_j ≥ e_j   [Hard - MUST fulfill]
    • Non-Priority (j ∈ NP): E_j ≤ e_j   [Soft - upper bound]
    • Infrastructure:        Σ x_{j,t} ≤ P_max + P_PV(t) + P_BESS(t)
    • Rate limits:           0 ≤ x_{j,t} ≤ x̄_j
    • Availability:          x_{j,t} = 0 if t ∉ T_j
    """
    )

    print("\n" + "=" * 70)
    print("ALGORITHM STRUCTURE (Two-Layer)")
    print("=" * 70)

    print(
        """
    LAYER 1: ADJUST_QUEUE (Section 2.1)
    ────────────────────────────────────
    1. State Monitoring:   Identify plugged-in EVs
    2. Priority Scoring:   Calculate laxity = (dwell time) - (charge time needed)
    3. Conflict Resolution: If capacity < demand, preempt low-priority high-laxity EVs
    4. Generate V_t:       Active EV set for optimizer
    
    LAYER 2: PREDICTIVE OPTIMIZATION (Section 2.2)
    ──────────────────────────────────────────────
    1. Build Problem:  Construct J and constraints using CVXPY
    2. Solve:          Execute solver (OSQP/MOSEK) over horizon N
    3. Apply:          Extract x_{j,1} for immediate control
    4. Iterate:        Shift window by Δt (RHO)
    """
    )

    print("\n" + "=" * 70)
    print("RHO IMPLEMENTATION")
    print("=" * 70)

    print(
        """
    At each time step t:
    ┌─────────────────────────────────────────────────────────┐
    │  1. ADJUST_QUEUE(all_sessions, t, capacity)             │
    │     → Returns active set V_t                            │
    │                                                         │
    │  2. BUILD_QP(V_t, prices[t:t+N], pv[t:t+N])             │
    │     → Constructs: min α₁H^EC + α₂H^NC + α₃H^QC          │
    │     → Subject to: Priority hard, Non-priority soft      │
    │                                                         │
    │  3. SOLVE_QP() change to convex                         │
    │     → Returns x*[j, 0:N] for all j in V_t               │
    │                                                         │
    │  4. APPLY x*[j, 0] (first step only - RHO principle)    │
    │                                                         │
    │  5. UPDATE energy_delivered[j] += x*[j,0] · Δt          │
    │                                                         │
    │  6. t ← t + 1 (shift horizon)                           │
    └─────────────────────────────────────────────────────────┘
    """
    )

    # Check CVXPY availability
    if not CVXPY_AVAILABLE:
        print("\n" + "!" * 70)
        print("NOTE: CVXPY not installed - cannot run actual simulation")
        print("      Install with: pip install cvxpy")
        print("!" * 70)

        print("\nShowing implementation structure instead...")

        # Show code structure
        print("\n" + "=" * 70)
        print("KEY CODE COMPONENTS")
        print("=" * 70)

        print(
            """
        class RHOAQPCOptimizer:
            
            def adjust_queue(self, sessions, current_step, capacity):
                '''
                Layer 1: ADJUST_QUEUE
                - Calculates laxity for each EV
                - Preempts non-priority EVs if needed
                - Returns active set V_t
                '''
                
            def build_and_solve_qp(self, active_sessions, step, prices, pv):
                '''
                Layer 2: QP Optimization
                
                # Decision variables
                x = cp.Variable((num_evs, N), nonneg=True)
                
                # H^EC: Grid Energy Cost
                net_load = sum_x - pv - bess
                H_EC = cp.sum(cp.multiply(net_load, prices)) * dt
                
                # H^NC: Non-Completion Penalty  
                H_NC = 0
                for j in non_priority:
                    H_NC += cp.square(e_j - E_j)
                
                # H^QC: Smoothing
                H_QC = cp.sum_squares(sum_x)
                
                # Constraints
                for j in priority:
                    constraints.append(E_j >= e_j)  # Hard
                for j in non_priority:
                    constraints.append(E_j <= e_j)  # Soft upper bound
                
                # Solve
                objective = cp.Minimize(α1*H_EC + α2*H_NC + α3*H_QC)
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.OSQP)
                
                return {session_id: x[j, 0].value for j in V_t}
                '''
                
            def solve(self, step, sessions, prices, pv):
                '''
                Main RHO method - called each time step
                1. active = adjust_queue(...)
                2. rates = build_and_solve_qp(active, ...)
                3. return rates  # Only first-step rates (RHO)
                '''
        """
        )

        return None, None, None

    # If CVXPY is available, run actual simulation
    # Configuration
    num_steps = 288  # 24 hours at 5-min intervals
    num_evs = 30
    dt = 5 / 60  # hours

    print(f"\nConfiguration:")
    print(f"  - Simulation: {num_steps} steps ({num_steps * dt:.1f} hours)")
    print(f"  - EVs: {num_evs}")
    print(f"  - Time step: {dt*60:.0f} minutes")

    # Generate scenario
    print("\n1. Generating scenario...")
    sessions = generate_ev_sessions(num_evs, num_steps, dt, priority_ratio=0.3, seed=42)
    grid_prices = generate_tou_prices(num_steps, dt)
    pv_gen = generate_pv_profile(num_steps, dt, peak_power_kw=50, seed=42)

    n_priority = sum(1 for s in sessions if s.is_priority)
    print(f"   - Priority EVs: {n_priority} ({n_priority/num_evs*100:.0f}%)")
    print(f"   - Non-priority EVs: {num_evs - n_priority}")

    # Create optimizer
    print("\n2. Creating RHO-AQPC optimizer...")
    params = OptimizationParams(
        alpha_energy=1.0,
        alpha_penalty=100.0,
        alpha_smooth=0.1,
        horizon_steps=12,
        time_step_hours=dt,
        infrastructure_limit_kw=150.0,
        solver="OSQP",
    )

    optimizer = RHOAQPCOptimizer(params=params)

    print(
        f"   - Objective: J = {params.alpha_energy}·H^EC + {params.alpha_penalty}·H^NC + {params.alpha_smooth}·H^QC"
    )
    print(
        f"   - Horizon: {params.horizon_steps} steps ({params.horizon_steps * dt * 60:.0f} min)"
    )
    print(f"   - Solver: {params.solver}")

    # Run simulation
    print("\n3. Running RHO-AQPC simulation...")
    metrics, history = run_rho_aqpc_simulation(
        "RHO_AQPC_Demo", optimizer, sessions, grid_prices, pv_gen, num_steps
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nEnergy Fulfillment:")
    print(
        f"  - Priority:     {metrics.priority_fulfillment_pct:.1f}% "
        f"({metrics.priority_sessions_fulfilled}/{metrics.priority_sessions} sessions)"
    )
    print(f"  - Non-Priority: {metrics.non_priority_fulfillment_pct:.1f}%")
    print(f"  - Overall:      {metrics.overall_fulfillment_pct:.1f}%")

    print(f"\nCost & Efficiency:")
    print(f"  - Total Energy Cost: ${metrics.total_energy_cost:.2f}")
    print(f"  - Jain's Fairness:   {metrics.jains_index:.3f}")

    print(f"\nComputational Performance:")
    print(f"  - Avg Solve Time: {metrics.avg_solve_time_ms:.2f} ms")
    print(f"  - Max Solve Time: {metrics.max_solve_time_ms:.2f} ms")
    print(f"  - Total Time:     {metrics.total_solve_time:.2f} s")

    print(f"\nAQPC Events:")
    print(f"  - Preemptions:     {metrics.total_preemptions}")
    print(f"  - Infeasibilities: {metrics.total_infeasibilities}")

    # Verify priority constraint satisfaction
    priority_sessions = [s for s in sessions if s.is_priority]
    priority_fulfilled = sum(
        1 for s in priority_sessions if s.energy_delivered >= s.energy_requested * 0.99
    )
    print(f"\nPriority Constraint Verification:")
    print(
        f"  - Hard constraint (E_j ≥ e_j) satisfied: {priority_fulfilled}/{len(priority_sessions)}"
    )

    return metrics, history, optimizer


if __name__ == "__main__":
    metrics, history, optimizer = run_demonstration()

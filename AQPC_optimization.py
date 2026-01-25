"""
AQPC: Adaptive Queuing-based Predictive Control Optimization Module
-------------------------------------------------------------------
A standalone Python module for solving EV fleet charging problems using
Model Predictive Control (MPC) with priority-aware constraints.

Includes comprehensive simulation scenarios benchmarking AQPC against
baseline algorithms (Round Robin, Least Laxity First).

Dependencies:
    - numpy
    - pandas
    - cvxpy (requires a solver like OSQP, ECOS, or MOSEK)
    - matplotlib
    - seaborn
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import time
import copy
import os

# Set global plot style for high-quality output
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# --- DATA STRUCTURES ---


@dataclass
class EVSession:
    """Represents a single EV charging session."""

    session_id: str
    arrival_time: int  # Time step index
    departure_time: int  # Time step index
    energy_requested: float  # kWh
    max_rate: float  # kW
    is_priority: bool  # True = Hard Constraint, False = Soft Penalty

    # State tracking
    energy_delivered: float = 0.0
    is_active: bool = True  # Used for preemption


@dataclass
class OptimizationParams:
    """Hyperparameters for the optimization objective."""

    alpha_energy: float = 1.0  # Weight for Grid Cost
    alpha_penalty: float = 100.0  # Weight for Non-Completion Penalty
    alpha_smooth: float = 0.1  # Weight for Load Smoothing (Regularization)
    time_step_hours: float = 5 / 60  # Duration of one time step in hours (e.g., 5 min)
    horizon_steps: int = 12  # MPC Lookahead horizon (e.g., 1 hour)


# --- DATA GENERATION FUNCTIONS ---


def generate_grid_price_data(
    num_steps: int, step_duration_mins: int = 5, tariff_type: str = "tou_default"
) -> np.ndarray:
    """
    Generates a Time-of-Use (ToU) electricity price profile.
    """
    prices = np.zeros(num_steps)

    for t in range(num_steps):
        minute_of_day = (t * step_duration_mins) % (24 * 60)
        hour_of_day = minute_of_day / 60.0

        if tariff_type == "flat":
            prices[t] = 0.20
        else:
            if 0 <= hour_of_day < 7:
                prices[t] = 0.15  # Off-peak
            elif 7 <= hour_of_day < 16:
                prices[t] = 0.22  # Shoulder
            elif 16 <= hour_of_day < 21:
                prices[t] = 0.35  # Peak
            else:
                prices[t] = 0.15  # Off-peak

    return prices


def generate_pv_data(
    num_steps: int,
    step_duration_mins: int = 5,
    peak_power_kw: float = 50.0,
    weather: str = "sunny",
) -> np.ndarray:
    """
    Generates a synthetic PV generation profile.
    """
    pv_profile = np.zeros(num_steps)

    for t in range(num_steps):
        minute_of_day = (t * step_duration_mins) % (24 * 60)
        # Sunrise/Sunset approx 6am-6pm
        day_progress = (minute_of_day - (6 * 60)) / (12 * 60)

        if 0 <= day_progress <= 1:
            angle = day_progress * np.pi
            base_gen = np.sin(angle) * peak_power_kw

            if weather == "sunny":
                noise = np.random.normal(1.0, 0.05)
            elif weather == "cloudy":
                # Significant drop and variance
                noise = np.random.uniform(0.2, 0.6)
            elif weather == "variable":
                # High frequency noise
                noise = np.random.uniform(0.1, 1.0)
            else:
                noise = 1.0

            pv_profile[t] = max(0, base_gen * noise)

    return pv_profile


def generate_ev_session_data(
    num_evs: int,
    simulation_steps: int,
    priority_ratio: float = 0.3,
    delayed_arrival_prob: float = 0.0,
    delay_steps: int = 12,  # 1 hour delay
) -> List[EVSession]:
    """
    Generates synthetic EV sessions.
    Args:
        delayed_arrival_prob: Probability an EV arrives late (Stress Test).
        delay_steps: How late the EV arrives (shrinking charging window).
    """
    sessions = []

    for i in range(num_evs):
        # Base Arrival: 07:00 - 11:00
        arr_hour = np.random.normal(9, 2)
        arr_step = int(max(0, arr_hour * 12))

        # Duration: 4 - 10 hours
        duration_hours = np.random.uniform(4, 10)
        dep_step = int(arr_step + (duration_hours * 12))

        # Apply Delay (Stress Test Logic)
        if np.random.random() < delayed_arrival_prob:
            arr_step += delay_steps
            # Ensure departure stays fixed (window shrinks) or pushed slightly
            if arr_step >= dep_step:
                dep_step = arr_step + 6  # Min 30 min charge

        if dep_step >= simulation_steps:
            dep_step = simulation_steps - 1

        energy = np.random.uniform(20, 60)  # kWh
        is_priority = np.random.random() < priority_ratio

        sessions.append(
            EVSession(
                session_id=f"EV_{i:03d}",
                arrival_time=arr_step,
                departure_time=dep_step,
                energy_requested=energy,
                max_rate=22.0,
                is_priority=is_priority,
            )
        )

    return sessions


# --- SCHEDULING ALGORITHMS ---


class Scheduler:
    """Base class for charging schedulers."""

    def __init__(self, params: OptimizationParams, infrastructure_limit_kw: float):
        self.params = params
        self.infra_limit = infrastructure_limit_kw

    def solve(
        self,
        current_step: int,
        active_sessions: List[EVSession],
        grid_prices: np.ndarray,
        pv_generation: np.ndarray,
    ) -> Dict[str, float]:
        raise NotImplementedError


class RoundRobinScheduler(Scheduler):
    """
    Round Robin (RR) Scheduler.
    Allocates maximum charging rate to EVs in a cyclic order until infrastructure limit is reached.
    If capacity is full, subsequent EVs get 0.
    Implementation based on: IEEE Transactions on Smart Grid, Vol. 3, No. 3, 2012.
    """

    def __init__(self, params: OptimizationParams, infrastructure_limit_kw: float):
        super().__init__(params, infrastructure_limit_kw)

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

        # Sort or order sessions to maintain rotation
        sorted_sessions = sorted(active_sessions, key=lambda s: s.session_id)

        # Rotate list based on step to simulate "Round Robin" over time
        shift = current_step % len(sorted_sessions)
        rotated_sessions = sorted_sessions[shift:] + sorted_sessions[:shift]

        results = {}
        current_load = 0.0

        for session in rotated_sessions:
            remaining_demand = max(
                0, session.energy_requested - session.energy_delivered
            )
            if remaining_demand <= 0.01:
                results[session.session_id] = 0.0
                continue

            # Allocate Max Power if capacity exists
            rate = min(session.max_rate, available_cap - current_load)

            if rate > 0:
                results[session.session_id] = rate
                current_load += rate
            else:
                results[session.session_id] = 0.0

        return results


class LeastLaxityFirstScheduler(Scheduler):
    """
    Least Laxity First (LLF) Scheduler.
    Prioritizes EVs with the smallest "laxity".
    Laxity = (Time Remaining) - (Time Needed to Charge).
    Implementation based on: IEEE Transactions on Vehicular Technology, Vol. 65, No. 6, 2016.
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

        session_laxity = []
        for s in active_sessions:
            remaining_energy = max(0, s.energy_requested - s.energy_delivered)
            if remaining_energy <= 0.01:
                laxity = float("inf")  # Done
            else:
                time_remaining_hours = (
                    s.departure_time - current_step
                ) * self.params.time_step_hours
                time_needed_hours = remaining_energy / s.max_rate
                laxity = time_remaining_hours - time_needed_hours

            session_laxity.append((s, laxity))

        # Sort by Least Laxity First
        sorted_by_laxity = sorted(session_laxity, key=lambda x: x[1])

        results = {}
        current_load = 0.0

        for session, laxity in sorted_by_laxity:
            if laxity == float("inf"):
                results[session.session_id] = 0.0
                continue

            rate = min(session.max_rate, available_cap - current_load)

            if rate > 0:
                results[session.session_id] = rate
                current_load += rate
            else:
                results[session.session_id] = 0.0

        return results


class AQPCOptimizer(Scheduler):
    """
    Adaptive Queuing-based Predictive Control (AQPC).
    Combines Adaptive Queuing (Preemption) with MPC Optimization.
    """

    def adjust_queue(
        self, active_sessions: List[EVSession], current_step: int, pv_now: float
    ) -> List[EVSession]:
        """
        Implements the ADJUST_QUEUE logic (Algorithm 1).
        Checks for congestion and preemptively disconnects low-priority EVs.
        """
        total_max_draw = sum(s.max_rate for s in active_sessions if s.is_active)
        available_cap = self.infra_limit + pv_now

        if total_max_draw > available_cap:

            def calculate_laxity(s):
                rem_energy = max(0, s.energy_requested - s.energy_delivered)
                time_steps = max(1, s.departure_time - current_step)
                time_needed = rem_energy / s.max_rate / self.params.time_step_hours
                return (time_steps * self.params.time_step_hours) - time_needed

            # Sort: Non-Priority first, then by highest Laxity (most flexible)
            sorted_sessions = sorted(
                [s for s in active_sessions if s.is_active],
                key=lambda x: (x.is_priority, -calculate_laxity(x)),
            )

            # Shed load
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
        bess_state: Dict = None,
    ) -> Dict[str, float]:
        """Solves MPC for current step."""

        # 1. Run Adaptive Queue Logic first
        pv_now = pv_generation[current_step] if current_step < len(pv_generation) else 0
        active_sessions = self.adjust_queue(active_sessions, current_step, pv_now)

        # Filter only active sessions for the solver
        solver_sessions = [s for s in active_sessions if s.is_active]

        T = self.params.horizon_steps
        dt = self.params.time_step_hours

        # Horizon Slicing
        horizon_prices = grid_prices[current_step : current_step + T]
        horizon_pv = pv_generation[current_step : current_step + T]

        if len(horizon_prices) < T:
            horizon_prices = np.pad(
                horizon_prices, (0, T - len(horizon_prices)), "edge"
            )
        if len(horizon_pv) < T:
            horizon_pv = np.pad(horizon_pv, (0, T - len(horizon_pv)), "edge")

        num_evs = len(solver_sessions)
        if num_evs == 0:
            return {}

        x = cp.Variable((num_evs, T), nonneg=True)
        constraints = []
        cost_penalty = 0

        sum_x = cp.sum(x, axis=0)

        for i, session in enumerate(solver_sessions):
            constraints += [x[i, :] <= session.max_rate]

            rel_start = session.arrival_time - current_step
            rel_end = session.departure_time - current_step

            for t in range(T):
                if t < rel_start or t > rel_end:
                    constraints += [x[i, t] == 0]

            remaining_demand = max(
                0, session.energy_requested - session.energy_delivered
            )
            projected_delivery = cp.sum(x[i, :]) * dt

            if session.is_priority:
                # Hard Constraint for Priority
                if rel_end < T:
                    max_possible = (
                        session.max_rate * (rel_end - max(0, rel_start) + 1) * dt
                    )
                    if remaining_demand <= max_possible:
                        constraints += [projected_delivery >= remaining_demand]
                    else:
                        constraints += [
                            x[i, max(0, rel_start) : rel_end + 1] == session.max_rate
                        ]
            else:
                # Soft Constraint for Non-Priority
                constraints += [projected_delivery <= remaining_demand]
                cost_penalty += cp.square(remaining_demand - projected_delivery)

        # Load <= Grid Limit + PV
        constraints += [sum_x <= self.infra_limit + horizon_pv]

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
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            return {ev.session_id: 0.0 for ev in solver_sessions}

        if prob.status in ["infeasible", "unbounded"]:
            return {ev.session_id: 0.0 for ev in solver_sessions}

        results = {}
        for i, session in enumerate(solver_sessions):
            val = x[i, 0].value
            if val is None:
                val = 0.0
            results[session.session_id] = max(0.0, min(val, session.max_rate))

        return results


# --- SIMULATION RUNNER ---


def run_simulation(
    scenario_name: str,
    optimizer: Scheduler,
    sessions: List[EVSession],
    grid_prices: np.ndarray,
    pv_gen: np.ndarray,
    num_steps: int,
) -> Dict:
    """Runs a full simulation loop and returns metrics."""

    params = optimizer.params
    history = {
        "time_step": [],
        "total_load_kw": [],
        "pv_gen_kw": [],
        "grid_price": [],
        "energy_delivered_step_kwh": [],
        "cost_step": [],
    }

    sim_sessions = copy.deepcopy(sessions)
    start_time = time.time()

    for t in range(num_steps):
        active = []
        for s in sim_sessions:
            if s.arrival_time <= t <= s.departure_time:
                s.is_active = True
                active.append(s)

        rates = optimizer.solve(t, active, grid_prices, pv_gen)

        step_energy = 0
        total_load = 0

        for s in active:
            if s.session_id in rates:
                rate = rates[s.session_id]
                energy = rate * params.time_step_hours
                s.energy_delivered += energy
                step_energy += energy
                total_load += rate

        # Calculate approximate cost for this step
        net_load = max(0, total_load - pv_gen[t])
        cost = net_load * grid_prices[t] * params.time_step_hours

        history["time_step"].append(t)
        history["total_load_kw"].append(total_load)
        history["pv_gen_kw"].append(pv_gen[t])
        history["grid_price"].append(grid_prices[t])
        history["energy_delivered_step_kwh"].append(step_energy)
        history["cost_step"].append(cost)

    sim_time = time.time() - start_time

    total_req = sum(s.energy_requested for s in sim_sessions)
    total_del = sum(s.energy_delivered for s in sim_sessions)
    total_cost = sum(history["cost_step"])

    priority_req = sum(s.energy_requested for s in sim_sessions if s.is_priority)
    priority_del = sum(s.energy_delivered for s in sim_sessions if s.is_priority)

    return {
        "scenario": scenario_name,
        "history": pd.DataFrame(history),
        "metrics": {
            "total_demand_kwh": total_req,
            "total_delivered_kwh": total_del,
            "fulfillment_pct": (total_del / total_req * 100) if total_req > 0 else 0,
            "priority_fulfillment_pct": (
                (priority_del / priority_req * 100) if priority_req > 0 else 0
            ),
            "total_cost": total_cost,
            "sim_time_sec": sim_time,
            "avg_step_time_sec": sim_time / num_steps,
        },
    }


# --- COMPARISON SCENARIOS ---


def get_schedulers(
    params: OptimizationParams, infra_limit: float
) -> List[Tuple[str, Scheduler]]:
    """Helper to get all 3 algorithm instances."""
    return [
        ("AQPC", AQPCOptimizer(params, infra_limit)),
        ("Round Robin", RoundRobinScheduler(params, infra_limit)),
        ("Least Laxity", LeastLaxityFirstScheduler(params, infra_limit)),
    ]


def run_priority_sweep(output_dir="results"):
    """Scenario 1: Parametric Sweep of Priority Ratios (Comparing All Algorithms)."""
    print("Running Priority Sweep (Comparative)...")
    results = []
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_steps = 288

    grid = generate_grid_price_data(num_steps)
    pv = generate_pv_data(num_steps)
    params = OptimizationParams()
    infra_limit = 100.0

    for r in ratios:
        # Generate same sessions for fair comparison
        sessions = generate_ev_session_data(40, num_steps, priority_ratio=r)

        for name, sched in get_schedulers(params, infra_limit):
            res = run_simulation(f"{name}_{r}", sched, sessions, grid, pv, num_steps)
            results.append(
                {
                    "Algorithm": name,
                    "Ratio": r * 100,
                    "Priority Fulfillment": res["metrics"]["priority_fulfillment_pct"],
                    "Total Fulfillment": res["metrics"]["fulfillment_pct"],
                }
            )

    df = pd.DataFrame(results)

    # Plot Priority Fulfillment
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Ratio",
        y="Priority Fulfillment",
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        linewidth=2.5,
    )
    plt.ylim(0, 105)
    plt.xlabel("Percentage of Priority EVs (%)")
    plt.ylabel("Priority Demand Met (%)")
    plt.title("Impact of Priority Saturation on Service Levels")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/priority_sweep_comparison.png", bbox_inches="tight")
    plt.close()

    print("Priority Sweep Completed.")


def run_stress_test(output_dir="results"):
    """Scenario 2: Stress Test (Comparing All Algorithms)."""
    print("Running Stress Test (Comparative)...")
    num_steps = 288
    grid = generate_grid_price_data(num_steps)
    pv = generate_pv_data(num_steps)
    params = OptimizationParams()

    scenarios = [
        {"name": "Normal Ops", "cap": 150.0, "delay_prob": 0.0},
        {"name": "Stress Test", "cap": 80.0, "delay_prob": 0.3},
    ]

    all_metrics = []

    for sc in scenarios:
        sessions = generate_ev_session_data(
            30, num_steps, priority_ratio=0.4, delayed_arrival_prob=sc["delay_prob"]
        )

        for name, sched in get_schedulers(params, sc["cap"]):
            res = run_simulation(sc["name"], sched, sessions, grid, pv, num_steps)
            all_metrics.append(
                {
                    "Scenario": sc["name"],
                    "Algorithm": name,
                    "Priority Fulfillment": res["metrics"]["priority_fulfillment_pct"],
                    "Total Fulfillment": res["metrics"]["fulfillment_pct"],
                }
            )

    df = pd.DataFrame(all_metrics)

    # Plot Comparison
    plt.figure(figsize=(12, 6))
    # Reshape for grouped bar plot
    melted = df.melt(
        id_vars=["Scenario", "Algorithm"],
        value_vars=["Priority Fulfillment", "Total Fulfillment"],
        var_name="Metric",
        value_name="Percentage",
    )

    sns.catplot(
        data=melted,
        x="Scenario",
        y="Percentage",
        hue="Algorithm",
        col="Metric",
        kind="bar",
        palette="viridis",
        height=5,
        aspect=1.2,
    )
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Resilience Under Stress Conditions: Algorithm Comparison")
    plt.savefig(f"{output_dir}/stress_test_comparison.png", bbox_inches="tight")
    plt.close()

    print("Stress Test Completed.")


def run_solar_sensitivity(output_dir="results"):
    """Scenario 3: Solar Sensitivity & Cost (Comparing All Algorithms)."""
    print("Running Cost/Solar Analysis (Comparative)...")
    num_steps = 288
    grid = generate_grid_price_data(num_steps)
    sessions = generate_ev_session_data(30, num_steps)
    params = OptimizationParams()
    infra_limit = 100.0

    weathers = ["sunny", "cloudy"]
    results = []

    for w in weathers:
        pv = generate_pv_data(num_steps, peak_power_kw=60.0, weather=w)

        for name, sched in get_schedulers(params, infra_limit):
            res = run_simulation(f"Weather_{w}", sched, sessions, grid, pv, num_steps)
            results.append(
                {
                    "Weather": w.capitalize(),
                    "Algorithm": name,
                    "Total Operational Cost ($)": res["metrics"]["total_cost"],
                }
            )

    df = pd.DataFrame(results)

    # Plot Cost
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x="Weather",
        y="Total Operational Cost ($)",
        hue="Algorithm",
        palette="magma",
    )
    plt.title("Operational Cost Efficiency Comparison")
    plt.ylabel("Total Cost ($)")
    plt.savefig(f"{output_dir}/cost_comparison.png", bbox_inches="tight")
    plt.close()

    print("Solar/Cost Analysis Completed.")


def run_scalability_test(output_dir="results"):
    """Scenario 4: Scalability (Comparing All Algorithms)."""
    print("Running Scalability Test (Comparative)...")
    num_steps = 50  # Short run
    grid = generate_grid_price_data(num_steps)
    pv = generate_pv_data(num_steps)
    params = OptimizationParams()

    sizes = [20, 50, 100, 200]
    results = []

    for n in sizes:
        sessions = generate_ev_session_data(n, num_steps)
        infra_limit = n * 5.0

        for name, sched in get_schedulers(params, infra_limit):
            res = run_simulation(f"Size_{n}", sched, sessions, grid, pv, num_steps)
            results.append(
                {
                    "Fleet Size": n,
                    "Algorithm": name,
                    "Avg Time/Step (s)": res["metrics"]["avg_step_time_sec"],
                }
            )

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Fleet Size",
        y="Avg Time/Step (s)",
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        linewidth=2,
    )
    plt.axhline(y=1.0, color="red", linestyle="--", label="Real-time Limit (1s)")
    plt.yscale("log")  # Log scale because MPC is much slower than heuristics
    plt.ylabel("Computation Time per Step (s) [Log Scale]")
    plt.title("Computation Scalability Benchmark")
    plt.legend()
    plt.savefig(f"{output_dir}/scalability_comparison.png", bbox_inches="tight")
    plt.close()

    print("Scalability Test Completed.")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    run_priority_sweep()
    run_stress_test()
    run_solar_sensitivity()
    run_scalability_test()

    print("\nAll comparative simulations complete. Check 'results/' folder.")

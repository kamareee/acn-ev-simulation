"""
Priority Scheduler Wrapper for ACN-Sim
=======================================

This module provides a BaseAlgorithm-compatible scheduler that integrates
AdaptiveChargingOptimizationWithPriority into ACN-Sim simulations.

Usage:
    from modified_adacharge.priority_scheduler import AdaptiveSchedulingAlgorithmWithPriority
    from modified_adacharge.modified_adaptive_charging_optimization_with_priority import ObjectiveComponent
    from modified_adacharge.modified_adaptive_charging_optimization import tou_energy_cost_with_pv

    scheduler = AdaptiveSchedulingAlgorithmWithPriority(
        objective=[ObjectiveComponent(tou_energy_cost_with_pv, 1.0)],
        priority_sessions={'session_1', 'session_2'},
        enforce_energy_equality=True,
        solver='MOSEK'
    )
"""

import warnings
from typing import List, Optional, Set, Union
import numpy as np

from acnportal.algorithms import BaseAlgorithm
from modified_adacharge.modified_adaptive_charging_optimization_with_priority import (
    AdaptiveChargingOptimizationWithPriority,
    ObjectiveComponent,
    PriorityConfig,
)
from modified_adacharge.modified_adacharge import (
    enforce_pilot_limit,
    apply_upper_bound_estimate,
    apply_minimum_charging_rate,
    project_into_discrete_feasible_pilots,
    project_into_continuous_feasible_pilots,
    diff_based_reallocation,
)


class AdaptiveSchedulingAlgorithmWithPriority(BaseAlgorithm):
    """
    Model Predictive Control based Adaptive Scheduling Algorithm with Priority EV support.

    This scheduler wraps AdaptiveChargingOptimizationWithPriority to make it compatible
    with ACN-Sim's BaseAlgorithm interface.

    Args:
        objective (List[ObjectiveComponent]): List of ObjectiveComponents for the optimization.
        constraint_type (str): String representing which constraint type to use.
            Options are 'SOC' for Second Order Cone or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy
            requested for each EV. If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current.
            If None, no limit is enforced.
        priority_sessions (Set[str]): Set of session IDs that should be treated as priority.
        priority_config (PriorityConfig): Configuration for priority selection rules.
        auto_select_priority (bool): If True, automatically select priority sessions based on
            session characteristics.
        estimate_max_rate (bool): If true, estimate maximum charging rate using max_rate_estimator.
        max_rate_estimator: Estimator object for predicting maximum charging rates.
        uninterrupted_charging (bool): If true, EVs should charge at least at minimum rate.
        quantize (bool): If true, apply project_into_discrete_feasible_pilots post-processing step.
        reallocate (bool): If true, apply index_based_reallocation post-processing step.
        max_recompute (int): Maximum number of control periods between optimization solves.
        allow_overcharging (bool): Allow the algorithm to exceed the energy request by at most
            the energy delivered at the minimum allowable rate for one period.
        verbose (bool): Solve with verbose logging. Helpful for debugging.
    """

    def __init__(
        self,
        objective: List[ObjectiveComponent],
        constraint_type: str = "SOC",
        enforce_energy_equality: bool = False,
        solver: str = "ECOS",
        peak_limit: Optional[Union[float, List[float], np.ndarray]] = None,
        priority_sessions: Optional[Set[str]] = None,
        priority_config: Optional[PriorityConfig] = None,
        auto_select_priority: bool = False,
        estimate_max_rate: bool = False,
        max_rate_estimator=None,
        uninterrupted_charging: bool = False,
        quantize: bool = True,
        reallocate: bool = False,
        max_recompute: Optional[int] = None,
        allow_overcharging: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.objective = objective
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.peak_limit = peak_limit
        self.priority_sessions = priority_sessions
        self.priority_config = priority_config
        self.auto_select_priority = auto_select_priority
        self.estimate_max_rate = estimate_max_rate
        self.max_rate_estimator = max_rate_estimator
        self.uninterrupted_charging = uninterrupted_charging
        self.quantize = quantize
        self.reallocate = reallocate
        self.verbose = verbose

        if not self.quantize and self.reallocate:
            raise ValueError(
                "reallocate cannot be true without quantize. "
                "Otherwise there is nothing to reallocate :)."
            )
        if self.quantize:
            if max_recompute is not None:
                warnings.warn("Overriding max_recompute to 1 since quantization is on.")
            self.max_recompute = 1
        else:
            self.max_recompute = max_recompute
        self.allow_overcharging = allow_overcharging

    def register_interface(self, interface):
        """Register interface to the simulator/physical system.

        This interface is the only connection between the algorithm and what it
        is controlling. Its purpose is to abstract the underlying network so that
        the same algorithms can run on a simulated environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network whether
                simulated or real.

        Returns:
            None
        """
        self._interface = interface
        if self.max_rate_estimator is not None:
            self.max_rate_estimator.register_interface(interface)

    def schedule(self, active_sessions):
        """Generate a schedule of charging rates for all active sessions.

        Args:
            active_sessions: List of active charging sessions.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping station IDs to charging rate schedules.
        """
        if len(active_sessions) == 0:
            return {}

        infrastructure = self.interface.infrastructure_info()
        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)

        if self.estimate_max_rate:
            active_sessions = apply_upper_bound_estimate(
                self.max_rate_estimator, active_sessions
            )
        if self.uninterrupted_charging:
            active_sessions = apply_minimum_charging_rate(
                active_sessions, infrastructure, self.interface.period
            )

        # Create optimizer with priority support
        optimizer = AdaptiveChargingOptimizationWithPriority(
            self.objective,
            self.interface,
            self.constraint_type,
            self.enforce_energy_equality,
            solver=self.solver,
            priority_sessions=self.priority_sessions,
            priority_config=self.priority_config,
            auto_select_priority=self.auto_select_priority,
        )

        # Handle peak limit
        if self.peak_limit is None or np.isscalar(self.peak_limit):
            trimmed_peak = self.peak_limit
        else:
            t = self.interface.current_time
            optimization_horizon = max(
                s.arrival_offset + s.remaining_time for s in active_sessions
            )
            trimmed_peak = self.peak_limit[t : t + optimization_horizon]

        # Solve optimization problem
        rates_matrix = optimizer.solve(
            active_sessions,
            infrastructure,
            peak_limit=trimmed_peak,
            prev_peak=self.interface.get_prev_peak(),
            verbose=self.verbose,
        )

        # Post-processing
        if self.quantize:
            if self.reallocate:
                rates_matrix = diff_based_reallocation(
                    rates_matrix, active_sessions, infrastructure, self.interface
                )
            else:
                rates_matrix = project_into_discrete_feasible_pilots(
                    rates_matrix, infrastructure
                )
        else:
            rates_matrix = project_into_continuous_feasible_pilots(
                rates_matrix, infrastructure
            )

        rates_matrix = np.maximum(rates_matrix, 0)

        return {
            station_id: rates_matrix[i, :]
            for i, station_id in enumerate(infrastructure.station_ids)
        }

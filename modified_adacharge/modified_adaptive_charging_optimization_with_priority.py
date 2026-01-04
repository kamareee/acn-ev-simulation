"""
Modified Adaptive Charging Optimization with Automated Priority Selection
==========================================================================

This is a drop-in replacement for modified_adaptive_charging_optimization.py
that integrates the PrioritySessionManager for automated priority EV handling.

Changes from original:
1. Added priority_sessions parameter to __init__
2. Modified charging_rate_bounds() to use PrioritySessionManager
3. Modified energy_constraints() to use PrioritySessionManager
4. Added methods for dynamic priority updates

Usage:
    from priority_ev_automation import PrioritySelector, PriorityConfig
    
    # Generate sessions and select priorities
    selector = PrioritySelector(PriorityConfig(max_priority_pct=0.27))
    priority_ids = selector.select_from_sessions(sessions)
    
    # Create optimizer with priorities
    optimizer = AdaptiveChargingOptimizationWithPriority(
        objective=objective,
        interface=interface,
        priority_sessions=priority_ids
    )

Author: Research Team
"""

from typing import List, Union, Optional, Set
from collections import namedtuple
import numpy as np
import cvxpy as cp

# Import from your existing modified_adacharge
from modified_adacharge.modified_interface import (
    Interface,
    SessionInfo,
    InfrastructureInfo,
)

# Import priority automation (should be in same package or PYTHONPATH)
try:
    from .priority_ev_automation import (
        PrioritySessionManager,
        PrioritySelector,
        PriorityConfig
    )
    PRIORITY_AUTOMATION_AVAILABLE = True
except ImportError:
    PRIORITY_AUTOMATION_AVAILABLE = False
    print("Warning: priority_ev_automation not found. Priority features disabled.")


class InfeasibilityException(Exception):
    pass


ObjectiveComponent = namedtuple(
    "ObjectiveComponent", ["function", "coefficient", "kwargs"]
)
ObjectiveComponent.__new__.__defaults__ = (1, {})


class AdaptiveChargingOptimizationWithPriority:
    """
    MPC-based charging algorithm with automated priority EV handling.
    
    This class extends the original AdaptiveChargingOptimization with
    integrated priority selection and management.

    Args:
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        interface (Interface): Interface providing information used by the algorithm.
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        priority_sessions (Set[str]): Set of session IDs that should be treated as priority.
            If None, no sessions are prioritized (unless auto_select_priority=True).
        priority_config (PriorityConfig): Configuration for priority selection rules.
        auto_select_priority (bool): If True, automatically select priority sessions
            based on session characteristics.
    """

    def __init__(
        self,
        objective: List[ObjectiveComponent],
        interface: Interface,
        constraint_type: str = "SOC",
        enforce_energy_equality: bool = False,
        solver: str = "ECOS",
        priority_sessions: Optional[Set[str]] = None,
        priority_config: Optional['PriorityConfig'] = None,
        auto_select_priority: bool = False,
    ):
        self.interface = interface
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.objective_configuration = objective
        
        # Initialize priority management
        self.auto_select_priority = auto_select_priority
        
        if PRIORITY_AUTOMATION_AVAILABLE:
            self.priority_config = priority_config or PriorityConfig()
            self.priority_manager = PrioritySessionManager(
                priority_sessions=priority_sessions,
                config=self.priority_config
            )
            self.priority_selector = PrioritySelector(self.priority_config)
        else:
            self.priority_manager = None
            self.priority_selector = None
            if priority_sessions:
                # Fallback: store as simple set
                self._priority_sessions = priority_sessions
            else:
                self._priority_sessions = set()
    
    def _is_priority(self, session_id: str) -> bool:
        """Check if a session is priority."""
        if self.priority_manager:
            return self.priority_manager.is_priority(session_id)
        return session_id in self._priority_sessions
    
    def _get_priority_min_rate(self) -> float:
        """Get minimum charging rate for priority sessions."""
        if self.priority_manager:
            return self.priority_manager._config.min_charging_rate_priority
        return 11.0  # Default fallback
    
    def set_priority_sessions(self, priority_sessions: Set[str]):
        """Update the set of priority sessions."""
        if self.priority_manager:
            self.priority_manager.priority_sessions = priority_sessions
        else:
            self._priority_sessions = priority_sessions
    
    def get_priority_sessions(self) -> Set[str]:
        """Get current priority session IDs."""
        if self.priority_manager:
            return self.priority_manager.priority_sessions
        return self._priority_sessions.copy()
    
    def auto_update_priorities(self, total_periods: int = 288):
        """
        Automatically update priority sessions based on current active sessions.
        
        Call this at the start of each optimization cycle if auto_select_priority=True.
        """
        if not self.auto_select_priority or not self.priority_selector:
            return
        
        active_sessions = self.interface.active_sessions()
        if not active_sessions:
            return
        
        new_priorities = self.priority_selector.select_from_sessions(
            active_sessions, total_periods
        )
        
        if self.priority_manager:
            self.priority_manager.priority_sessions = new_priorities
        else:
            self._priority_sessions = new_priorities

    def charging_rate_bounds(
        self, 
        rates: cp.Variable, 
        active_sessions: List[SessionInfo], 
        evse_index: List[str]
    ):
        """Get upper and lower bound constraints for each charging rate.
        
        Priority sessions receive a higher minimum charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            evse_index (List[str]): List of IDs for all EVSEs. Index in evse_index represents the row number of that
                EVSE in rates.

        Returns:
            Dict[str, cp.Constraint]: Dictionary of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        priority_min_rate = self._get_priority_min_rate()
        
        for session in active_sessions:
            i = evse_index.index(session.station_id)
            session_slice = slice(
                session.arrival_offset, 
                session.arrival_offset + session.remaining_time
            )

            if self._is_priority(session.session_id):
                # Set a higher minimum charging rate for priority sessions
                # session.min_rates is a numpy array of length remaining_time
                lb[i, session_slice] = np.maximum(session.min_rates, priority_min_rate)
            else:
                lb[i, session_slice] = session.min_rates

            ub[i, session_slice] = session.max_rates
        
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        
        return {
            "charging_rate_bounds.lb": rates >= lb,
            "charging_rate_bounds.ub": rates <= ub,
        }

    def energy_constraints(
        self,
        rates: cp.Variable,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        period: int,
        enforce_energy_equality: bool = False,
    ):
        """Get constraints on the energy delivered for each session.
        
        Priority sessions get >= constraint (must deliver at least requested).
        Non-priority sessions get <= constraint (can deliver less).

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            period (int): Length of each discrete time period. (min)
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.

        Returns:
            Dict[str, cp.Constraint]: Dictionary of energy delivered constraints for each session.
        """
        constraints = {}
        
        for session in active_sessions:
            i = infrastructure.get_station_index(session.station_id)
            planned_energy = cp.sum(
                rates[
                    i,
                    session.arrival_offset : session.arrival_offset + session.remaining_time,
                ]
            )
            planned_energy *= infrastructure.voltages[i] * period / 1e3 / 60
            constraint_name = f"energy_constraints.{session.session_id}"
            
            if enforce_energy_equality:
                constraints[constraint_name] = (
                    planned_energy == session.remaining_demand
                )
            elif self._is_priority(session.session_id):
                # Priority sessions must receive at least their requested energy
                constraints[constraint_name] = (
                    planned_energy >= session.remaining_demand
                )
            else:
                # Non-priority sessions can receive less
                constraints[constraint_name] = (
                    planned_energy <= session.remaining_demand
                )
        
        return constraints

    @staticmethod
    def infrastructure_constraints(
        rates: cp.Variable, infrastructure: InfrastructureInfo, constraint_type="SOC"
    ):
        """Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order
                Cone or 'LINEAR' for linearized constraints.

        Returns:
            Dict[str, cp.Constraint]: Dictionary of constraints, one for each bottleneck in the electrical infrastructure.
        """
        # If constraint_matrix is empty, no need to add infrastructure constraints.
        if (
            infrastructure.constraint_matrix is None
            or infrastructure.constraint_matrix.shape == (0, 0)
        ):
            return {}
        
        constraints = {}
        if constraint_type == "SOC":
            if infrastructure.phases is None:
                raise ValueError(
                    "phases is required when using SOC infrastructure constraints."
                )
            phase_in_rad = np.deg2rad(infrastructure.phases)
            for j, v in enumerate(infrastructure.constraint_matrix):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraint_name = (
                    f"infrastructure_constraints.{infrastructure.constraint_ids[j]}"
                )
                constraints[constraint_name] = (
                    cp.norm(a @ rates, axis=0) <= infrastructure.constraint_limits[j]
                )
        elif constraint_type == "LINEAR":
            for j, v in enumerate(infrastructure.constraint_matrix):
                constraint_name = (
                    f"infrastructure_constraints.{infrastructure.constraint_ids[j]}"
                )
                constraints[constraint_name] = (
                    np.abs(v) @ rates <= infrastructure.constraint_limits[j]
                )
        else:
            raise ValueError(
                f"Invalid infrastructure constraint type: {constraint_type}. Valid options are SOC or LINEAR."
            )
        return constraints

    @staticmethod
    def peak_constraint(
        rates: cp.Variable, peak_limit: Union[float, List[float], np.ndarray]
    ):
        """Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.

        Returns:
            Dict[str, cp.Constraint]: Dictionary of constraints.
        """
        if peak_limit is not None:
            return {"peak_constraint": cp.sum(rates, axis=0) <= peak_limit}
        return {}

    def build_objective(
        self, rates: cp.Variable, infrastructure: InfrastructureInfo, **kwargs
    ):
        def _merge_dicts(*args):
            """Merge dictionaries where later dicts override earlier ones."""
            merged = dict()
            for d in args:
                merged.update(d)
            return merged

        obj = cp.Constant(0)
        for component in self.objective_configuration:
            obj += component.coefficient * component.function(
                rates,
                infrastructure,
                self.interface,
                **_merge_dicts(kwargs, component.kwargs),
            )
        return obj

    def build_problem(
        self,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        peak_limit: Optional[Union[float, List[float], np.ndarray]] = None,
        prev_peak: float = 0,
    ):
        """Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': dict of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        optimization_horizon = max(
            s.arrival_offset + s.remaining_time for s in active_sessions
        )
        num_evses = len(infrastructure.station_ids)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))
        constraints = {}

        # Rate constraints (uses priority logic)
        constraints.update(
            self.charging_rate_bounds(
                rates, active_sessions, infrastructure.station_ids
            )
        )

        # Energy Delivered Constraints (uses priority logic)
        constraints.update(
            self.energy_constraints(
                rates,
                active_sessions,
                infrastructure,
                self.interface.period,
                self.enforce_energy_equality,
            )
        )

        # Infrastructure Constraints
        constraints.update(
            self.infrastructure_constraints(rates, infrastructure, self.constraint_type)
        )

        # Peak Limit
        constraints.update(self.peak_constraint(rates, peak_limit))

        # Objective Function
        objective = cp.Minimize(
            self.build_objective(rates, infrastructure, prev_peak=prev_peak)
        )
        
        return {
            "objective": objective,
            "constraints": constraints,
            "variables": {"rates": rates},
        }

    def solve(
        self,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        peak_limit: Union[float, List[float], np.ndarray] = None,
        prev_peak: float = 0,
        verbose: bool = False,
    ):
        """Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            verbose (bool): See cp.Problem.solve()

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        # Auto-update priorities if enabled
        if self.auto_select_priority:
            self.auto_update_priorities()
        
        if len(active_sessions) == 0:
            return np.zeros((infrastructure.num_stations, 1))
        
        problem_dict = self.build_problem(
            active_sessions, infrastructure, peak_limit, prev_peak
        )
        prob = cp.Problem(
            problem_dict["objective"], list(problem_dict["constraints"].values())
        )
        prob.solve(solver=self.solver, verbose=verbose)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f"Solve failed with status {prob.status}")
        
        return problem_dict["variables"]["rates"].value


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# For backward compatibility, you can use this as a drop-in replacement
AdaptiveChargingOptimization = AdaptiveChargingOptimizationWithPriority


# =============================================================================
# Helper function for migration from hardcoded lists
# =============================================================================

def migrate_from_hardcoded_priority(
    hardcoded_rate_bounds: List[str],
    hardcoded_energy_constraints: List[str]
) -> Set[str]:
    """
    Helper to migrate from hardcoded priority lists to unified set.
    
    Usage:
        # Your old hardcoded lists
        rate_bounds_priority = ["session_15", "session_12", "session_36", ...]
        energy_constraints_priority = ["session_13", "session_10", "session_31", ...]
        
        # Merge them
        priority_sessions = migrate_from_hardcoded_priority(
            rate_bounds_priority, 
            energy_constraints_priority
        )
    """
    # Union of both lists - assuming they should be the same
    combined = set(hardcoded_rate_bounds) | set(hardcoded_energy_constraints)
    
    # Warn if they were different
    if set(hardcoded_rate_bounds) != set(hardcoded_energy_constraints):
        rate_only = set(hardcoded_rate_bounds) - set(hardcoded_energy_constraints)
        energy_only = set(hardcoded_energy_constraints) - set(hardcoded_rate_bounds)
        print(f"Warning: Priority lists were different!")
        print(f"  In rate_bounds only: {rate_only}")
        print(f"  In energy_constraints only: {energy_only}")
        print(f"  Combined: {combined}")
    
    return combined

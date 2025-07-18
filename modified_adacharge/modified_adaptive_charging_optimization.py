from typing import List, Union, Optional
from collections import namedtuple
import numpy as np
import cvxpy as cp

# from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
from modified_adacharge.modified_interface import (
    Interface,
    SessionInfo,
    InfrastructureInfo,
)


class InfeasibilityException(Exception):
    pass


ObjectiveComponent = namedtuple(
    "ObjectiveComponent", ["function", "coefficient", "kwargs"]
)
ObjectiveComponent.__new__.__defaults__ = (1, {})


class AdaptiveChargingOptimization:
    """Base class for all MPC based charging algorithms.

    Args:
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        interface (Interface): Interface providing information used by the algorithm.
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
    """

    def __init__(
        self,
        objective: List[ObjectiveComponent],
        interface: Interface,
        constraint_type="SOC",
        enforce_energy_equality=False,
        solver="ECOS",
    ):
        self.interface = interface
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.objective_configuration = objective

    @staticmethod
    def charging_rate_bounds(
        rates: cp.Variable, active_sessions: List[SessionInfo], evse_index: List[str]
    ):
        """Get upper and lower bound constraints for each charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            evse_index (List[str]): List of IDs for all EVSEs. Index in evse_index represents the row number of that
                EVSE in rates.

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        priority_sessions = [
            "session_14",
            "session_10",
            "session_13",
            "session_12",
            "session_40",
            "session_11",
            "session_29",
            "session_37",
            "session_41",
            "session_4",
            "session_5",
            "session_9",
            "session_35",
        ]
        for session in active_sessions:
            i = evse_index.index(session.station_id)
            session_slice = slice(
                session.arrival_offset, session.arrival_offset + session.remaining_time
            )

            if session.session_id in priority_sessions:
                # Set a higher minimum charging rate for priority sessions
                lb[i, session_slice] = np.maximum(
                    session.min_rates, 8
                )  # Example: minimum 6A for priority EVs
            else:
                lb[i, session_slice] = session.min_rates

            ub[i, session_slice] = session.max_rates
            # lb[
            #     i,
            #     session.arrival_offset : session.arrival_offset
            #     + session.remaining_time,
            # ] = session.min_rates
            # ub[
            #     i,
            #     session.arrival_offset : session.arrival_offset
            #     + session.remaining_time,
            # ] = session.max_rates
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return {
            "charging_rate_bounds.lb": rates >= lb,
            "charging_rate_bounds.ub": rates <= ub,
        }

    @staticmethod
    def energy_constraints(
        rates: cp.Variable,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        period,
        enforce_energy_equality=False,
    ):
        """Get constraints on the energy delivered for each session.

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
            List[cp.Constraint]: List of energy delivered constraints for each session.
        """
        constraints = {}
        for session in active_sessions:
            i = infrastructure.get_station_index(session.station_id)
            planned_energy = cp.sum(
                rates[
                    i,
                    session.arrival_offset : session.arrival_offset
                    + session.remaining_time,
                ]
            )
            planned_energy *= infrastructure.voltages[i] * period / 1e3 / 60
            constraint_name = f"energy_constraints.{session.session_id}"
            if enforce_energy_equality:
                constraints[constraint_name] = (
                    planned_energy == session.remaining_demand
                )
            elif session.session_id in [
                "session_14",
                "session_10",
                "session_13",
                "session_12",
                "session_40",
                "session_11",
                "session_29",
                "session_37",
                "session_41",
                "session_4",
                "session_5",
                "session_9",
                "session_35",
            ]:
                constraints[constraint_name] = (
                    planned_energy
                    >= session.remaining_demand
                    # planned_energy
                    # <= session.remaining_demand * 1.01
                )
            else:
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
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        # If constraint_matrix is empty, no need to add infrastructure
        # constraints.
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
                    f"infrastructure_constraints." f"{infrastructure.constraint_ids[j]}"
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
                "Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.".format(
                    constraint_type
                )
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
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        if peak_limit is not None:
            return {"peak_constraint": cp.sum(rates, axis=0) <= peak_limit}
        return {}

    def build_objective(
        self, rates: cp.Variable, infrastructure: InfrastructureInfo, **kwargs
    ):
        def _merge_dicts(*args):
            """Merge two dictionaries where d2 override d1 when there is a conflict."""
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
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        optimization_horizon = max(
            s.arrival_offset + s.remaining_time for s in active_sessions
        )
        num_evses = len(infrastructure.station_ids)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))
        constraints = {}

        # Rate constraints
        constraints.update(
            self.charging_rate_bounds(
                rates, active_sessions, infrastructure.station_ids
            )
        )

        # Energy Delivered Constraints
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
        # objective = cp.Maximize(
        #     self.build_objective(rates, infrastructure, prev_peak=prev_peak)
        # )
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
        prev_peak=0,
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
        # Here we take in arguments which describe the problem and build a problem instance.
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


# ---------------------------------------------------------------------------------
#  Objective Functions
#
#
#  All objectives should take rates as their first positional argument.
#  All other arguments should be passed as keyword arguments.
#  All functions should except **kwargs as their last argument to avoid errors
#  when unknown arguments are passed.
#
# ---------------------------------------------------------------------------------


def charging_power(rates, infrastructure, **kwargs):
    """Returns a matrix with the same shape as rates but with units kW instead of A."""
    voltage_matrix = np.tile(infrastructure.voltages, (rates.shape[1], 1)).T
    return cp.multiply(rates, voltage_matrix) / 1e3


def aggregate_power(rates, infrastructure, **kwargs):
    """Returns aggregate charging power for each time period."""
    return cp.sum(charging_power(rates, infrastructure=infrastructure), axis=0)


def get_period_energy(rates, infrastructure, period, **kwargs):
    """Return energy delivered in kWh during each time period and each session."""
    power = charging_power(rates, infrastructure=infrastructure)
    period_in_hours = period / 60
    return power * period_in_hours


def aggregate_period_energy(rates, infrastructure, interface, **kwargs):
    """Returns the aggregate energy delivered in kWh during each time period."""
    # get charging rates in kWh per period
    energy_per_period = get_period_energy(
        rates, infrastructure=infrastructure, period=interface.period
    )
    return cp.sum(energy_per_period, axis=0)


def quick_charge(rates, infrastructure, interface, **kwargs):
    optimization_horizon = rates.shape[1]
    c = np.array(
        [
            (optimization_horizon - t) / optimization_horizon
            for t in range(optimization_horizon)
        ]
    )
    return c @ cp.sum(rates, axis=0)


def equal_share(rates, infrastructure, interface, **kwargs):
    return -cp.sum_squares(rates)


def tou_energy_cost(rates, infrastructure, interface, **kwargs):
    current_prices = interface.get_prices(rates.shape[1])  # $/kWh
    return -current_prices @ aggregate_period_energy(rates, infrastructure, interface)


def total_energy(rates, infrastructure, interface, **kwargs):
    return cp.sum(get_period_energy(rates, infrastructure, interface.period))


def peak(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    agg_power = aggregate_power(rates, infrastructure)
    max_power = cp.max(agg_power)
    prev_peak = interface.get_prev_peak() * infrastructure.voltages[0] / 1000
    if baseline_peak > 0:
        return cp.maximum(max_power, baseline_peak, prev_peak)
    else:
        return cp.maximum(max_power, prev_peak)


def demand_charge(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    p = peak(rates, infrastructure, interface, baseline_peak, **kwargs)
    dc = interface.get_demand_charge()
    return -dc * p


def load_flattening(rates, infrastructure, interface, external_signal=None, **kwargs):
    if external_signal is None:
        external_signal = np.zeros(rates.shape[1])
    aggregate_rates_kW = aggregate_power(rates, infrastructure)
    total_aggregate = aggregate_rates_kW + external_signal
    return -cp.sum_squares(total_aggregate)


def tou_energy_cost_with_pv(rates, infrastructure, interface, **kwargs):
    current_prices = interface.get_prices(rates.shape[1])  # $/kWh
    if interface.current_datetime.hour > 8 and interface.current_datetime.hour < 16:
        pv_power = np.array([20 for i in range(rates.shape[1])])
    else:
        pv_power = np.array([5 for i in range(rates.shape[1])])
    # return current_prices @ (
    #     aggregate_period_energy(rates, infrastructure, interface) - pv_power
    # )

    total_energy = aggregate_period_energy(rates, infrastructure, interface)

    # Calculate grid energy
    grid_energy = cp.maximum(total_energy - pv_power, 0)
    return current_prices @ grid_energy


def non_completion_penalty(rates, infrastructure, interface, **kwargs):
    session_requested_energy = np.array(
        [session.remaining_demand for session in interface.active_sessions()]
    )
    requested_energy = cp.sum(session_requested_energy)
    # return -cp.norm(
    #     aggregate_period_energy(rates, infrastructure, interface) - requested_energy,
    #     p=1,
    return cp.norm(
        aggregate_period_energy(rates, infrastructure, interface) - requested_energy,
        p=1,
    )


# This function is for non completion penalty with priority ev charging sessions
def non_completion_penalty_for_priority_ev(rates, infrastructure, interface, **kwargs):
    priority_sessions = [
        session
        for session in interface.active_sessions()
        if session.session_id
        in [
            "session_22",
            "session_5",
            "session_24",
            "session_7",
            "session_2",
            "session_13",
        ]
    ]

    if not priority_sessions:
        return cp.Constant(
            np.random.uniform(0, 0.001)
        )  # No penalty if no priority sessions

    priority_requested_energy = np.array(
        [session.remaining_demand for session in priority_sessions]
    )
    total_priority_requested_energy = cp.sum(priority_requested_energy)

    # Calculate energy delivered to priority sessions
    priority_indices = [
        infrastructure.get_station_index(session.station_id)
        for session in priority_sessions
    ]
    priority_rates = rates[priority_indices, :]

    # Calculate energy delivered to priority sessions
    priority_voltages = infrastructure.voltages[priority_indices]
    priority_power = cp.multiply(priority_rates, priority_voltages[:, np.newaxis]) / 1e3
    period_in_hours = interface.period / 60
    priority_delivered_energy = cp.sum(priority_power * period_in_hours)

    # return cp.norm(priority_delivered_energy - total_priority_requested_energy, p=1)
    return cp.norm2(priority_delivered_energy - total_priority_requested_energy)


def non_completion_penalty_without_priority_ev(
    rates, infrastructure, interface, **kwargs
):
    non_priority_sessions = [
        session
        for session in interface.active_sessions()
        if session.session_id
        not in [
            "session_2",
            "session_4",
            "session_10",
            "session_12",
            "session_16",
            "session_19",
            "session_20",
        ]
    ]

    if not non_priority_sessions:
        return cp.Constant(np.random.uniform(0, 0.001))

    non_priority_session_requested_energy = np.array(
        [session.remaining_demand for session in non_priority_sessions]
    )
    total_non_priority_requested_energy = cp.sum(non_priority_session_requested_energy)
    non_priority_indices = [
        infrastructure.get_station_index(session.station_id)
        for session in non_priority_sessions
    ]

    non_priority_rates = rates[non_priority_indices, :]

    # Calculate energy delivered to priority sessions
    non_priority_voltages = infrastructure.voltages[non_priority_indices]
    non_priority_power = (
        cp.multiply(non_priority_rates, non_priority_voltages[:, np.newaxis]) / 1e3
    )
    period_in_hours = interface.period / 60
    non_priority_sessions_delivered_energy = cp.sum(
        non_priority_power * period_in_hours
    )

    return cp.norm(
        non_priority_sessions_delivered_energy - total_non_priority_requested_energy,
        p=1,
    )


# def tou_energy_cost_with_pv(rates, infrastructure, interface, **kwargs):
#     current_prices = interface.get_prices(rates.shape[1])  # $/kWh
#     all_pv_power = [5, 8, 11]
#     session_pv_power = []
#     for i in range(rates.shape[1]):
#         session_pv_power.extend(all_pv_power)
#     # pv_power = np.array([i for i in range(rates.shape[1])])
#     pv_power = np.array(session_pv_power[0 : rates.shape[1]])
#     return -current_prices @ (
#         aggregate_period_energy(rates, infrastructure, interface) - pv_power
#     )


# def smoothing(rates, active_sessions, infrastructure, previous_rates, normp=1, *args, **kwargs):
#     reg = -cp.norm(cp.diff(rates, axis=1), p=normp)
#     prev_mask = np.logical_not(np.isnan(previous_rates))
#     if np.any(prev_mask):
#         reg -= cp.norm(rates[0, prev_mask] - previous_rates[prev_mask], p=normp)
#     return reg

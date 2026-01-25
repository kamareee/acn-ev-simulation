"""
Modified AdaCharge Package
==========================

This package provides modified versions of the ACN-Sim AdaCharge components
with support for priority EV charging.

Main Components:
- AdaptiveChargingOptimization: Base MPC optimization class
- AdaptiveChargingOptimizationWithPriority: Priority-aware optimization class
- AdaptiveSchedulingAlgorithm: Algorithm wrapper for simulations
- PrioritySelector, PriorityConfig: Priority selection tools
- ACNScenarioGenerator: Scenario generation with priority support
"""

# Core optimization classes
from .modified_adaptive_charging_optimization import (
    AdaptiveChargingOptimization,
    InfeasibilityException,
    ObjectiveComponent,
    # Objective functions
    charging_power,
    aggregate_power,
    get_period_energy,
    aggregate_period_energy,
    quick_charge,
    equal_share,
    tou_energy_cost,
    total_energy,
    peak,
    demand_charge,
    load_flattening,
    tou_energy_cost_with_pv,
    non_completion_penalty,
    non_completion_penalty_for_priority_ev,
    non_completion_penalty_without_priority_ev,
)

# Priority-aware optimization class
from .modified_adaptive_charging_optimization_with_priority import (
    AdaptiveChargingOptimizationWithPriority,
    migrate_from_hardcoded_priority,
)

# Interface classes
from .modified_interface import (
    Interface,
    SessionInfo,
    InfrastructureInfo,
)

# Algorithm wrappers
from .modified_adacharge import (
    AdaptiveSchedulingAlgorithm,
    AdaptiveSchedulingAlgorithmWithPriority,
    AdaptiveChargingAlgorithmOffline,
    get_active_sessions,
)

# Priority automation
from .priority_ev_automation import (
    PrioritySelector,
    PriorityConfig,
    PrioritySessionManager,
    PriorityCandidate,
    PriorityReason,
    create_modified_charging_rate_bounds,
    create_modified_energy_constraints,
    mark_priority_candidates_in_events,
)

# Scenario generation
from .acn_scenario_generator_with_priority import (
    ACNScenarioGenerator,
    SCENARIOS,
    generate_scenario_with_priorities,
    get_pv_factor,
    list_scenarios,
    get_scenario_info,
    SessionParams,
    ScenarioResult,
)

# Postprocessing utilities
from .modified_postprocessing import (
    project_into_continuous_feasible_pilots,
    project_into_discrete_feasible_pilots,
    index_based_reallocation,
    diff_based_reallocation,
)

__all__ = [
    # Core optimization
    "AdaptiveChargingOptimization",
    "AdaptiveChargingOptimizationWithPriority",
    "InfeasibilityException",
    "ObjectiveComponent",
    # Objective functions
    "charging_power",
    "aggregate_power",
    "get_period_energy",
    "aggregate_period_energy",
    "quick_charge",
    "equal_share",
    "tou_energy_cost",
    "total_energy",
    "peak",
    "demand_charge",
    "load_flattening",
    "tou_energy_cost_with_pv",
    "non_completion_penalty",
    "non_completion_penalty_for_priority_ev",
    "non_completion_penalty_without_priority_ev",
    # Interface
    "Interface",
    "SessionInfo",
    "InfrastructureInfo",
    # Algorithm wrappers
    "AdaptiveSchedulingAlgorithm",
    "AdaptiveSchedulingAlgorithmWithPriority",
    "AdaptiveChargingAlgorithmOffline",
    "get_active_sessions",
    # Priority automation
    "PrioritySelector",
    "PriorityConfig",
    "PrioritySessionManager",
    "PriorityCandidate",
    "PriorityReason",
    "create_modified_charging_rate_bounds",
    "create_modified_energy_constraints",
    "mark_priority_candidates_in_events",
    "migrate_from_hardcoded_priority",
    # Scenario generation
    "ACNScenarioGenerator",
    "SCENARIOS",
    "generate_scenario_with_priorities",
    "get_pv_factor",
    "list_scenarios",
    "get_scenario_info",
    "SessionParams",
    "ScenarioResult",
    # Postprocessing
    "project_into_continuous_feasible_pilots",
    "project_into_discrete_feasible_pilots",
    "index_based_reallocation",
    "diff_based_reallocation",
]

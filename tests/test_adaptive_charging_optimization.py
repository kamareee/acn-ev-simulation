import unittest
import numpy as np
import cvxpy as cp

from modified_adacharge.modified_interface import (
    Interface,
    SessionInfo,
    InfrastructureInfo,
)

from modified_adacharge.modified_adaptive_charging_optimization import (
    tou_energy_cost_with_pv,
)


def test_tou_energy_cost_with_pv():
    # Define the input variables for the test
    rates = cp.Variable((5, 24))
    infrastructure = InfrastructureInfo(voltages=np.array([240, 240, 240, 240, 240]))
    interface = Interface(period=60)

    # Generate some random prices for each time period
    current_prices = np.random.uniform(low=0.1, high=0.5, size=(24,))
    interface.get_prices = lambda n: current_prices

    # Generate some random PV power values for each time period
    pv_power = np.random.uniform(low=0, high=10, size=(5, 24))

    # Calculate the expected result
    period_in_hours = interface.period / 60
    energy_per_period = rates * period_in_hours
    aggregate_period_energy = np.sum(energy_per_period, axis=0)
    aggregate_pv_period_energy = np.sum(pv_power * period_in_hours, axis=0)
    expected_result = -np.dot(
        current_prices, aggregate_period_energy - aggregate_pv_period_energy
    )

    # Call the objective function
    result = tou_energy_cost_with_pv(rates, infrastructure, interface)

    # Check if the result matches the expected result
    assert np.isclose(result, expected_result)

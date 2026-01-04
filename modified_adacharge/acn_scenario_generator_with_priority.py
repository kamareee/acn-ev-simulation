"""
ACN-Sim Compatible Scenario Generator with Priority Selection (Corrected)
==========================================================================

This module generates EV charging scenarios compatible with ACN-Sim's event system
AND integrates with the priority selection automation for AQPC algorithm.

KEY DESIGN PRINCIPLE:
- The generator does NOT need station IDs - it generates session characteristics only
- EV objects are created with a placeholder station_id
- The StochasticNetwork.plugin() method handles actual station assignment during simulation
  by finding an available EVSE and calling ev.update_station_id()

This matches the behavior of get_synth_events/GaussianMixtureEvents.generate_events()

Usage:
    from acn_scenario_generator_with_priority import (
        ACNScenarioGenerator,
        SCENARIOS,
        generate_scenario_with_priorities,
        PriorityConfig
    )
    
    # Generate events (no station_ids needed!)
    events, priority_ids, session_params = generate_scenario_with_priorities(
        scenario_key='S1_baseline',
        n_sessions=140,
        period=5,
        voltage=415,
        max_battery_power=21,
        seed=42,
        priority_config=PriorityConfig(max_priority_pct=0.27)
    )
    
    # Create network separately
    network = ev_fleet_level_2_network(transformer_cap=130)
    
    # Use with simulation - network handles station assignment
    sim = Simulator(network, scheduler, events, start, period=5)
    
    # Pass priority_ids to optimizer
    optimizer = AdaptiveChargingOptimizationWithPriority(
        ..., 
        priority_sessions=priority_ids
    )

Author: Research Team
Compatible with: acnportal, modified_adacharge, priority_ev_automation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

# Try to import ACN-Sim components
try:
    from acnportal.acnsim.models.ev import EV
    from acnportal.acnsim.models.battery import Battery
    from acnportal.acnsim.events import PluginEvent, EventQueue
    ACN_AVAILABLE = True
except ImportError:
    ACN_AVAILABLE = False
    print("Warning: acnportal not found. Running in standalone mode for testing.")


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    'S1_baseline': {
        'name': 'S1: Baseline',
        'description': 'Standard operational scenario with 27% priority EVs',
        'priority_pct': 0.27,
        'arrival_pattern': 'uniform',
        'pv_factor': 1.0,
    },
    'S2_low_priority': {
        'name': 'S2: Low Priority',
        'description': 'Few urgent vehicles requiring priority charging',
        'priority_pct': 0.10,
        'arrival_pattern': 'uniform',
        'pv_factor': 1.0,
    },
    'S3_high_priority': {
        'name': 'S3: High Priority',
        'description': 'Many urgent vehicles requiring priority charging',
        'priority_pct': 0.50,
        'arrival_pattern': 'uniform',
        'pv_factor': 1.0,
    },
    'S4_morning_rush': {
        'name': 'S4: Morning Rush',
        'description': 'Clustered morning arrivals (6AM-9AM)',
        'priority_pct': 0.27,
        'arrival_pattern': 'clustered_am',
        'pv_factor': 1.0,
    },
    'S5_cloudy_day': {
        'name': 'S5: Cloudy Day',
        'description': 'Reduced PV generation (50% of normal)',
        'priority_pct': 0.27,
        'arrival_pattern': 'uniform',
        'pv_factor': 0.5,
    },
    'S6_peak_stress': {
        'name': 'S6: Peak Stress',
        'description': 'High demand + reduced PV + afternoon clustering',
        'priority_pct': 0.50,
        'arrival_pattern': 'clustered_pm',
        'pv_factor': 0.3,
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriorityConfig:
    """Configuration for priority EV selection."""
    max_priority_pct: float = 0.27
    min_energy_kwh: float = 10.0
    max_energy_kwh: float = 30.0
    min_duration_hours: float = 2.0
    high_energy_threshold: float = 25.0
    high_energy_min_duration: float = 3.0
    max_high_energy_pct: float = 0.06


@dataclass
class SessionParams:
    """Parameters for a single charging session."""
    session_id: str
    arrival_period: int
    departure_period: int
    energy_requested: float  # kWh
    battery_capacity: float  # kWh
    
    @property
    def duration_periods(self) -> int:
        return self.departure_period - self.arrival_period
    
    def duration_hours(self, period_minutes: float) -> float:
        return self.duration_periods * period_minutes / 60.0


@dataclass
class ScenarioResult:
    """Result of scenario generation with priority selection."""
    events: 'EventQueue'
    priority_ids: Set[str]
    session_params: List[SessionParams]
    scenario_config: Dict
    pv_factor: float
    n_sessions: int
    n_priority: int
    
    def get_session_summary(self) -> Dict:
        """Get summary statistics of generated sessions."""
        energies = [s.energy_requested for s in self.session_params]
        durations = [s.duration_periods for s in self.session_params]
        priority_energies = [s.energy_requested for s in self.session_params 
                           if s.session_id in self.priority_ids]
        
        return {
            'total_sessions': self.n_sessions,
            'priority_sessions': self.n_priority,
            'priority_pct': self.n_priority / self.n_sessions * 100 if self.n_sessions > 0 else 0,
            'avg_energy_kwh': np.mean(energies) if energies else 0,
            'total_energy_kwh': sum(energies),
            'avg_duration_periods': np.mean(durations) if durations else 0,
            'priority_avg_energy_kwh': np.mean(priority_energies) if priority_energies else 0,
        }


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class ACNScenarioGenerator:
    """
    Generates ACN-Sim compatible EV charging events with integrated priority selection.
    
    This generator creates PluginEvent objects that work directly with the
    acnportal Simulator class. 
    
    IMPORTANT: This generator does NOT assign station IDs. The EV objects are 
    created with a placeholder station_id, and the actual assignment is handled 
    by the ChargingNetwork (specifically StochasticNetwork) during simulation 
    via the plugin() method which calls ev.update_station_id().
    
    This matches the behavior of get_synth_events/GaussianMixtureEvents.
    """
    
    def __init__(
        self,
        period: float = 5,
        voltage: float = 220,
        max_battery_power: float = 21,
        battery_capacity_range: Tuple[float, float] = (60, 80),
        priority_config: Optional[PriorityConfig] = None
    ):
        """
        Initialize the scenario generator.
        
        Args:
            period: Length of each time interval in minutes
            voltage: Voltage of the network in volts
            max_battery_power: Maximum charging power for batteries in kW
            battery_capacity_range: (min, max) battery capacity in kWh
            priority_config: Configuration for priority selection
        """
        self.period = period
        self.voltage = voltage
        self.max_battery_power = max_battery_power
        self.battery_capacity_range = battery_capacity_range
        self.priority_config = priority_config or PriorityConfig()
        
        # Derived parameters
        self.periods_per_hour = 60 / period
        self.periods_per_day = int(24 * self.periods_per_hour)
        self.max_rate_A = (max_battery_power * 1000) / voltage
    
    def generate_scenario(
        self,
        scenario_key: str,
        n_sessions: int,
        seed: Optional[int] = None
    ) -> ScenarioResult:
        """
        Generate a complete scenario with events and priority selection.
        
        Args:
            scenario_key: Key from SCENARIOS dict
            n_sessions: Number of EV sessions to generate
            seed: Random seed for reproducibility
            
        Returns:
            ScenarioResult with events, priority_ids, and session_params
        """
        if scenario_key not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}. "
                           f"Available: {list(SCENARIOS.keys())}")
        
        if seed is not None:
            np.random.seed(seed)
        
        config = SCENARIOS[scenario_key]
        
        # Generate session parameters
        sessions = self._generate_session_params(
            n_sessions=n_sessions,
            arrival_pattern=config['arrival_pattern']
        )
        
        # Select priority sessions
        priority_ids = self._select_priorities(
            sessions=sessions,
            target_priority_pct=config['priority_pct']
        )
        
        # Create events (without station assignment - handled by network)
        events = self._create_event_queue(sessions)
        
        return ScenarioResult(
            events=events,
            priority_ids=priority_ids,
            session_params=sessions,
            scenario_config=config,
            pv_factor=config['pv_factor'],
            n_sessions=len(sessions),
            n_priority=len(priority_ids)
        )
    
    def _generate_session_params(
        self,
        n_sessions: int,
        arrival_pattern: str,
        min_dwell_hours: float = 2.0,
        max_dwell_hours: float = 8.0,
        min_energy_kwh: float = 5.0,
        max_energy_kwh: float = 40.0,
        arrival_start_hour: float = 6.0,
        arrival_end_hour: float = 18.0
    ) -> List[SessionParams]:
        """Generate session parameters for all EVs."""
        
        sessions = []
        
        for i in range(n_sessions):
            session_id = f"session_{i}"
            
            # Generate arrival time based on pattern
            arrival_hour = self._sample_arrival_hour(
                arrival_pattern, arrival_start_hour, arrival_end_hour
            )
            
            # Generate dwell time
            dwell_hours = np.random.uniform(min_dwell_hours, max_dwell_hours)
            departure_hour = arrival_hour + dwell_hours
            
            # Cap departure at end of day
            departure_hour = min(departure_hour, 23.99)
            
            # Convert to periods
            arrival_period = int(arrival_hour * self.periods_per_hour)
            departure_period = int(departure_hour * self.periods_per_hour)
            
            # Ensure minimum duration of 1 hour
            if departure_period - arrival_period < self.periods_per_hour:
                departure_period = arrival_period + int(self.periods_per_hour)
            
            # Cap at end of simulation day
            departure_period = min(departure_period, self.periods_per_day - 1)
            
            # Generate energy request
            energy_requested = np.random.uniform(min_energy_kwh, max_energy_kwh)
            
            # Generate battery capacity
            battery_capacity = np.random.uniform(*self.battery_capacity_range)
            
            # Ensure energy requested doesn't exceed battery capacity
            energy_requested = min(energy_requested, battery_capacity * 0.9)
            
            sessions.append(SessionParams(
                session_id=session_id,
                arrival_period=arrival_period,
                departure_period=departure_period,
                energy_requested=round(energy_requested, 2),
                battery_capacity=round(battery_capacity, 2)
            ))
        
        return sessions
    
    def _sample_arrival_hour(
        self,
        pattern: str,
        start_hour: float,
        end_hour: float
    ) -> float:
        """Sample an arrival hour based on the specified pattern."""
        
        if pattern == 'uniform':
            return np.random.uniform(start_hour, end_hour)
        
        elif pattern == 'clustered_am':
            # Morning rush: 70% arrive between 6-9 AM
            if np.random.random() < 0.7:
                return np.random.uniform(6.0, 9.0)
            else:
                return np.random.uniform(9.0, end_hour)
        
        elif pattern == 'clustered_pm':
            # Afternoon rush: 70% arrive between 2-6 PM
            if np.random.random() < 0.7:
                return np.random.uniform(14.0, 18.0)
            else:
                return np.random.uniform(start_hour, 14.0)
        
        elif pattern == 'bimodal':
            # Two peaks: morning and afternoon
            if np.random.random() < 0.5:
                return np.random.normal(8.0, 1.0)
            else:
                return np.random.normal(16.0, 1.0)
        
        else:
            return np.random.uniform(start_hour, end_hour)
    
    def _select_priorities(
        self,
        sessions: List[SessionParams],
        target_priority_pct: float
    ) -> Set[str]:
        """Select priority sessions based on configuration rules."""
        
        config = self.priority_config
        n_target = int(len(sessions) * target_priority_pct)
        
        # Score each session for priority suitability
        scored_sessions = []
        for session in sessions:
            score = self._calculate_priority_score(session)
            scored_sessions.append((session.session_id, score, session))
        
        # Sort by score (higher is better) and select top candidates
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        
        priority_ids = set()
        high_energy_count = 0
        max_high_energy = max(1, int(n_target * config.max_high_energy_pct / config.max_priority_pct))
        
        for session_id, score, session in scored_sessions:
            if len(priority_ids) >= n_target:
                break
            
            # Check high-energy constraint
            if session.energy_requested >= config.high_energy_threshold:
                if high_energy_count >= max_high_energy:
                    continue
                # Check duration requirement for high-energy
                duration_hours = session.duration_hours(self.period)
                if duration_hours < config.high_energy_min_duration:
                    continue
                high_energy_count += 1
            
            priority_ids.add(session_id)
        
        return priority_ids
    
    def _calculate_priority_score(self, session: SessionParams) -> float:
        """Calculate priority suitability score for a session."""
        
        config = self.priority_config
        score = 0.0
        
        # Prefer energy in sweet spot (10-30 kWh)
        if config.min_energy_kwh <= session.energy_requested <= config.max_energy_kwh:
            score += 10.0
        elif session.energy_requested < config.min_energy_kwh:
            score += 5.0  # Low energy is okay but not ideal
        else:
            score += 2.0  # High energy needs more scrutiny
        
        # Prefer longer durations (more flexibility)
        duration_hours = session.duration_hours(self.period)
        if duration_hours >= config.min_duration_hours:
            score += 5.0
        if duration_hours >= config.high_energy_min_duration:
            score += 3.0
        
        # Add some randomness to avoid deterministic selection
        score += np.random.uniform(0, 2)
        
        return score
    
    def _create_event_queue(
        self,
        sessions: List[SessionParams]
    ) -> 'EventQueue':
        """
        Create an EventQueue compatible with ACN-Sim.
        
        NOTE: EV objects are created with a placeholder station_id.
        The actual station assignment is handled by the ChargingNetwork
        (StochasticNetwork) during simulation via plugin() method.
        """
        
        if not ACN_AVAILABLE:
            raise RuntimeError("acnportal not available - cannot create EventQueue")
        
        events = []
        
        for session in sessions:
            # Create battery
            battery = Battery(
                capacity=session.battery_capacity,
                init_charge=0,
                max_power=self.max_battery_power
            )
            
            # Create EV with placeholder station_id
            # The StochasticNetwork.plugin() will update this to an available EVSE
            ev = EV(
                arrival=session.arrival_period,
                departure=session.departure_period,
                requested_energy=session.energy_requested,
                station_id="waiting",  # Placeholder - will be updated by network.plugin()
                session_id=session.session_id,
                battery=battery,
                estimated_departure=session.departure_period
            )
            
            # Create PluginEvent
            event = PluginEvent(session.arrival_period, ev)
            events.append(event)
        
        return EventQueue(events)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_scenario_with_priorities(
    scenario_key: str,
    n_sessions: int = 140,
    period: float = 5,
    voltage: float = 220,
    max_battery_power: float = 21,
    battery_capacity_range: Tuple[float, float] = (60, 80),
    seed: Optional[int] = None,
    priority_config: Optional[PriorityConfig] = None
) -> Tuple['EventQueue', Set[str], List[SessionParams]]:
    """
    Convenience function to generate a scenario with priorities.
    
    Args:
        scenario_key: Key from SCENARIOS dict
        n_sessions: Number of EV sessions to generate
        period: Simulation period in minutes
        voltage: Network voltage (for battery calculations)
        max_battery_power: Max charging power in kW
        battery_capacity_range: (min, max) battery capacity in kWh
        seed: Random seed for reproducibility
        priority_config: Priority selection configuration
    
    Returns:
        Tuple of (EventQueue, priority_ids, session_params)
    """
    generator = ACNScenarioGenerator(
        period=period,
        voltage=voltage,
        max_battery_power=max_battery_power,
        battery_capacity_range=battery_capacity_range,
        priority_config=priority_config
    )
    
    result = generator.generate_scenario(scenario_key, n_sessions, seed)
    return result.events, result.priority_ids, result.session_params


def get_pv_factor(scenario_key: str) -> float:
    """Get PV factor for a scenario."""
    if scenario_key in SCENARIOS:
        return SCENARIOS[scenario_key].get('pv_factor', 1.0)
    return 1.0


def list_scenarios() -> List[str]:
    """List all available scenarios."""
    return list(SCENARIOS.keys())


def get_scenario_info(scenario_key: str) -> Dict:
    """Get information about a scenario."""
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}")
    return SCENARIOS[scenario_key].copy()

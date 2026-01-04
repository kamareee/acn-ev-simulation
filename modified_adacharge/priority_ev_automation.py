"""
Priority EV Selection and Automation Module for AQPC Algorithm
================================================================

This module provides automated priority EV selection for the Adaptive Queuing
Priority Charging (AQPC) algorithm. It implements a hybrid approach (Option C):
- Scenario generator marks *candidate* priority sessions based on rules
- Optimizer validates/filters for feasibility at runtime

Key Features:
- Rule-based priority candidate selection
- Feasibility-aware filtering to avoid solver infeasibility
- Seamless integration with modified_adacharge and ACN-Sim
- Support for dynamic priority updates during simulation

Integration Points:
1. ACNScenarioGenerator - marks candidate priorities at event generation
2. PrioritySelector - filters candidates based on feasibility rules
3. Modified AdaptiveChargingOptimization - uses priority_sessions set

Author: Research Team
Compatible with: acnportal, modified_adacharge, modified_acnsim
"""

from typing import List, Set, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

# Try to import ACN-Sim components
try:
    from acnportal.acnsim.interface import SessionInfo
    ACN_AVAILABLE = True
except ImportError:
    ACN_AVAILABLE = False
    # Define a mock SessionInfo for standalone testing that matches the real one
    @dataclass
    class SessionInfo:
        """Mock SessionInfo matching modified_interface.py structure."""
        station_id: str
        session_id: str
        requested_energy: float  # kWh
        energy_delivered: float  # kWh
        arrival: int  # time index (periods)
        departure: int  # time index (periods)
        estimated_departure: Optional[int] = None
        current_time: int = 0
        min_rates: Union[float, np.ndarray] = 0
        max_rates: Union[float, np.ndarray] = float('inf')
        
        def __post_init__(self):
            if self.estimated_departure is None:
                self.estimated_departure = self.departure
            # Convert min_rates/max_rates to arrays like real SessionInfo
            remaining = self.remaining_time
            if np.isscalar(self.min_rates):
                self.min_rates = np.array([self.min_rates] * remaining)
            if np.isscalar(self.max_rates):
                self.max_rates = np.array([self.max_rates] * remaining)
        
        @property
        def remaining_demand(self) -> float:
            """Return the Session's remaining demand in kWh."""
            return self.requested_energy - self.energy_delivered
        
        @property
        def arrival_offset(self) -> int:
            """Return the time (in periods) until arrival, or 0 if already arrived."""
            return max(self.arrival - self.current_time, 0)
        
        @property
        def remaining_time(self) -> int:
            """Return time remaining until departure in periods."""
            remaining = min(
                self.departure - self.arrival, 
                self.departure - self.current_time
            )
            return max(remaining, 0)


class PriorityReason(Enum):
    """Reasons for priority assignment - useful for analysis/debugging."""
    MODERATE_ENERGY_SHORT_WINDOW = "moderate_energy_short_window"
    LOW_ENERGY_URGENT = "low_energy_urgent"
    FLEET_CRITICAL = "fleet_critical"
    MANUAL_OVERRIDE = "manual_override"
    CANDIDATE_SELECTED = "candidate_selected"


@dataclass
class PriorityCandidate:
    """Represents a session that could be marked as priority."""
    session_id: str
    energy_demand: float  # kWh
    duration_hours: float  # total available charging time
    arrival_period: int
    departure_period: int
    score: float = 0.0  # priority score (higher = better candidate)
    reason: PriorityReason = PriorityReason.CANDIDATE_SELECTED
    is_high_demand: bool = False  # True if energy > 25 kWh
    
    def __lt__(self, other):
        """For sorting by score (descending)."""
        return self.score > other.score


@dataclass 
class PriorityConfig:
    """Configuration for priority selection rules."""
    # Maximum percentage of sessions that can be priority
    max_priority_pct: float = 0.30
    
    # Step size for priority percentage (not used currently but for future)
    priority_step: float = 0.05
    
    # Energy demand thresholds (kWh)
    preferred_energy_min: float = 10.0
    preferred_energy_max: float = 30.0
    high_demand_threshold: float = 25.0
    
    # Maximum percentage of priority vehicles that can be high-demand
    max_high_demand_pct: float = 0.06  # 5-6% of total priority
    
    # Minimum session duration (hours) for high-demand vehicles
    min_duration_high_demand: float = 3.0
    
    # Minimum session duration (hours) for any priority vehicle
    min_duration_any: float = 1.5
    
    # Period length in minutes (for time calculations)
    period_minutes: float = 5.0
    
    # Minimum charging rate for priority vehicles (Amps)
    min_charging_rate_priority: float = 11.0
    
    # Default max charging rate (Amps) - 21kW at 220V ≈ 95A
    default_max_rate: float = 95.0


class PrioritySelector:
    """
    Selects priority sessions based on configurable rules.
    
    This class encapsulates the priority selection logic that was previously
    done manually by inspecting session characteristics.
    
    Usage:
        selector = PrioritySelector(config=PriorityConfig(max_priority_pct=0.27))
        priority_ids = selector.select_from_sessions(active_sessions)
        
        # Or from raw session data before SessionInfo creation
        priority_ids = selector.select_from_raw_data(sessions_df)
    """
    
    def __init__(self, config: Optional[PriorityConfig] = None):
        self.config = config or PriorityConfig()
        self._selection_history: List[Dict] = []
    
    def _calculate_duration_hours(self, arrival: int, departure: int) -> float:
        """Calculate session duration in hours from period indices.
        
        Args:
            arrival: Arrival time in periods
            departure: Departure time in periods
            
        Returns:
            Duration in hours
        """
        periods = departure - arrival
        return periods * self.config.period_minutes / 60.0
    
    def _periods_to_hours(self, periods: int) -> float:
        """Convert periods to hours."""
        return periods * self.config.period_minutes / 60.0
    
    def _calculate_priority_score(
        self, 
        energy_demand: float, 
        duration_hours: float,
        arrival_period: int,
        total_periods: int
    ) -> Tuple[float, bool]:
        """
        Calculate a priority score for a session.
        
        Higher scores indicate better priority candidates.
        Returns (score, is_high_demand).
        
        Scoring factors:
        - Energy in preferred range (10-30 kWh): +2.0
        - Duration > 3h: +1.0
        - Duration > 4h: +0.5 additional
        - Earlier arrival: +0.5 * (1 - arrival/total)
        - High demand (>25 kWh) with short duration (<3h): -5.0 (penalty)
        """
        score = 0.0
        is_high_demand = energy_demand > self.config.high_demand_threshold
        
        # Energy range scoring
        if self.config.preferred_energy_min <= energy_demand <= self.config.preferred_energy_max:
            score += 2.0
        elif energy_demand < self.config.preferred_energy_min:
            score += 1.0  # Low demand is easier to fulfill
        else:
            score += 0.5  # High demand is harder
        
        # Duration scoring
        if duration_hours >= self.config.min_duration_high_demand:
            score += 1.0
        if duration_hours >= 4.0:
            score += 0.5
        
        # Arrival time scoring (prefer earlier arrivals for better scheduling flexibility)
        if total_periods > 0:
            arrival_ratio = arrival_period / total_periods
            score += 0.5 * (1 - arrival_ratio)
        
        # Penalty for high-demand with short duration (causes infeasibility)
        if is_high_demand and duration_hours < self.config.min_duration_high_demand:
            score -= 5.0
        
        # Penalty for very short durations
        if duration_hours < self.config.min_duration_any:
            score -= 2.0
        
        return score, is_high_demand
    
    def _create_candidates(
        self, 
        sessions: List[SessionInfo],
        total_periods: int = 288  # Default: 1 day with 5-min periods
    ) -> List[PriorityCandidate]:
        """Create priority candidate objects from sessions."""
        candidates = []
        
        for session in sessions:
            duration_hours = self._calculate_duration_hours(
                session.arrival, session.departure
            )
            
            # Skip sessions with already very short durations
            if duration_hours < 0.5:  # Less than 30 minutes
                continue
            
            score, is_high_demand = self._calculate_priority_score(
                session.remaining_demand,
                duration_hours,
                session.arrival,
                total_periods
            )
            
            # Determine reason
            if self.config.preferred_energy_min <= session.remaining_demand <= self.config.preferred_energy_max:
                if duration_hours < 4:
                    reason = PriorityReason.MODERATE_ENERGY_SHORT_WINDOW
                else:
                    reason = PriorityReason.CANDIDATE_SELECTED
            elif session.remaining_demand < self.config.preferred_energy_min:
                reason = PriorityReason.LOW_ENERGY_URGENT
            else:
                reason = PriorityReason.CANDIDATE_SELECTED
            
            candidates.append(PriorityCandidate(
                session_id=session.session_id,
                energy_demand=session.remaining_demand,
                duration_hours=duration_hours,
                arrival_period=session.arrival,
                departure_period=session.departure,
                score=score,
                reason=reason,
                is_high_demand=is_high_demand
            ))
        
        return candidates
    
    def select_from_sessions(
        self, 
        sessions: List[SessionInfo],
        total_periods: int = 288,
        manual_includes: Optional[Set[str]] = None,
        manual_excludes: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Select priority session IDs from a list of SessionInfo objects.
        
        Args:
            sessions: List of SessionInfo objects (from Interface.active_sessions())
            total_periods: Total periods in simulation (for arrival scoring)
            manual_includes: Session IDs to force-include as priority
            manual_excludes: Session IDs to force-exclude from priority
        
        Returns:
            Set of session_ids that should be treated as priority
        """
        if not sessions:
            return set()
        
        manual_includes = manual_includes or set()
        manual_excludes = manual_excludes or set()
        
        # Create candidates
        candidates = self._create_candidates(sessions, total_periods)
        
        # Sort by score (descending)
        candidates.sort()
        
        # Calculate limits
        max_priority_count = int(len(sessions) * self.config.max_priority_pct)
        max_high_demand_count = max(1, int(max_priority_count * self.config.max_high_demand_pct))
        
        # Select priority sessions
        selected: Set[str] = set()
        high_demand_count = 0
        
        # First, add manual includes (if they pass basic checks)
        for session_id in manual_includes:
            if session_id not in manual_excludes:
                selected.add(session_id)
                # Check if it's high demand
                for c in candidates:
                    if c.session_id == session_id and c.is_high_demand:
                        high_demand_count += 1
                        break
        
        # Then select from candidates based on score
        for candidate in candidates:
            if len(selected) >= max_priority_count:
                break
            
            if candidate.session_id in manual_excludes:
                continue
            
            if candidate.session_id in selected:
                continue
            
            # Check high-demand limit
            if candidate.is_high_demand:
                # Additional check: high-demand must have sufficient duration
                if candidate.duration_hours < self.config.min_duration_high_demand:
                    continue
                if high_demand_count >= max_high_demand_count:
                    continue
                high_demand_count += 1
            
            # Check minimum duration for any priority
            if candidate.duration_hours < self.config.min_duration_any:
                continue
            
            # Skip negative scores (likely to cause infeasibility)
            if candidate.score < 0:
                continue
            
            selected.add(candidate.session_id)
        
        # Record selection for debugging
        self._selection_history.append({
            'total_sessions': len(sessions),
            'candidates_evaluated': len(candidates),
            'selected_count': len(selected),
            'high_demand_count': high_demand_count,
            'max_allowed': max_priority_count,
            'selected_ids': list(selected)
        })
        
        return selected
    
    def select_from_raw_data(
        self,
        session_data: List[Dict],
        period_minutes: float = 5.0
    ) -> Set[str]:
        """
        Select priority sessions from raw session data (before SessionInfo creation).
        
        Args:
            session_data: List of dicts with keys:
                - session_id: str
                - energy_demand: float (kWh)
                - arrival: int (period index)
                - departure: int (period index)
            period_minutes: Length of each period in minutes
        
        Returns:
            Set of session_ids that should be treated as priority
        """
        # Convert to mock SessionInfo objects
        sessions = []
        for data in session_data:
            session = SessionInfo(
                station_id=data.get('station_id', 'unknown'),
                session_id=data['session_id'],
                requested_energy=data['energy_demand'],
                energy_delivered=0,
                arrival=data['arrival'],
                departure=data['departure'],
                current_time=0
            )
            sessions.append(session)
        
        # Update config period if different
        old_period = self.config.period_minutes
        self.config.period_minutes = period_minutes
        
        result = self.select_from_sessions(sessions)
        
        # Restore config
        self.config.period_minutes = old_period
        
        return result
    
    def get_selection_summary(self) -> Dict:
        """Get summary of the last selection."""
        if not self._selection_history:
            return {}
        return self._selection_history[-1]
    
    def validate_feasibility(
        self,
        priority_ids: Set[str],
        sessions: List[SessionInfo],
        network_capacity_kw: float,
        voltage: float = 220.0
    ) -> Tuple[bool, Set[str], str]:
        """
        Validate if selected priority sessions are likely feasible.
        
        This is a heuristic check - actual feasibility is determined by the solver.
        
        Returns:
            (is_likely_feasible, adjusted_priority_ids, message)
        """
        priority_sessions = [s for s in sessions if s.session_id in priority_ids]
        
        if not priority_sessions:
            return True, priority_ids, "No priority sessions selected"
        
        # Calculate total priority energy demand
        total_priority_energy = sum(s.remaining_demand for s in priority_sessions)
        
        # Calculate total available charging capacity (rough estimate)
        # Assume priority sessions can charge at min_rate (11A) continuously
        min_rate_kw = self.config.min_charging_rate_priority * voltage / 1000
        
        # Check if all priority sessions can be served
        avg_duration_hours = np.mean([
            self._calculate_duration_hours(s.arrival, s.departure) 
            for s in priority_sessions
        ])
        
        # Rough capacity check
        estimated_deliverable = min_rate_kw * avg_duration_hours * len(priority_sessions)
        
        if estimated_deliverable < total_priority_energy * 0.9:
            # May be infeasible - try reducing high-demand vehicles
            adjusted = set(priority_ids)
            
            # Sort priority sessions by energy demand (descending)
            sorted_priority = sorted(
                priority_sessions, 
                key=lambda s: s.remaining_demand, 
                reverse=True
            )
            
            # Remove highest demand sessions until likely feasible
            for session in sorted_priority:
                if session.remaining_demand > self.config.high_demand_threshold:
                    adjusted.remove(session.session_id)
                    
                    # Recalculate
                    remaining_sessions = [s for s in priority_sessions if s.session_id in adjusted]
                    new_total = sum(s.remaining_demand for s in remaining_sessions)
                    new_deliverable = min_rate_kw * avg_duration_hours * len(remaining_sessions)
                    
                    if new_deliverable >= new_total * 0.9:
                        return True, adjusted, f"Reduced priority set from {len(priority_ids)} to {len(adjusted)} for feasibility"
            
            return False, adjusted, "Priority demands may exceed capacity - solver may be infeasible"
        
        return True, priority_ids, "Priority selection appears feasible"


class PrioritySessionManager:
    """
    Manager class that integrates with AdaptiveChargingOptimization.
    
    This class provides a clean interface for the optimizer to query
    priority status without modifying the core algorithm structure.
    
    Usage in modified_adaptive_charging_optimization.py:
        
        # In __init__:
        self.priority_manager = PrioritySessionManager()
        
        # Or pass in constructor:
        def __init__(self, ..., priority_sessions: Optional[Set[str]] = None):
            self.priority_manager = PrioritySessionManager(priority_sessions)
        
        # In charging_rate_bounds:
        if self.priority_manager.is_priority(session.session_id):
            lb[i, session_slice] = np.maximum(session.min_rates, 11)
    """
    
    def __init__(
        self, 
        priority_sessions: Optional[Set[str]] = None,
        config: Optional[PriorityConfig] = None
    ):
        self._priority_sessions: Set[str] = priority_sessions or set()
        self._config = config or PriorityConfig()
        self._selector = PrioritySelector(self._config)
    
    @property
    def priority_sessions(self) -> Set[str]:
        """Get current priority session IDs."""
        return self._priority_sessions.copy()
    
    @priority_sessions.setter
    def priority_sessions(self, sessions: Set[str]):
        """Set priority session IDs."""
        self._priority_sessions = set(sessions)
    
    def is_priority(self, session_id: str) -> bool:
        """Check if a session is priority."""
        return session_id in self._priority_sessions
    
    def update_from_active_sessions(
        self, 
        active_sessions: List[SessionInfo],
        total_periods: int = 288
    ) -> Set[str]:
        """
        Update priority sessions based on currently active sessions.
        
        This can be called periodically during simulation to adjust
        priorities based on changing conditions.
        """
        self._priority_sessions = self._selector.select_from_sessions(
            active_sessions, total_periods
        )
        return self._priority_sessions
    
    def add_priority(self, session_id: str):
        """Manually add a session to priority."""
        self._priority_sessions.add(session_id)
    
    def remove_priority(self, session_id: str):
        """Manually remove a session from priority."""
        self._priority_sessions.discard(session_id)
    
    def get_min_rate_for_session(self, session: SessionInfo) -> Union[float, np.ndarray]:
        """Get the minimum charging rate for a session.
        
        Note: In the actual ACN-Sim, min_rates is a numpy array of length remaining_time.
        This method returns either:
        - A scalar (priority min rate) if session is priority
        - The original min_rates array if session is not priority
        """
        if self.is_priority(session.session_id):
            return self._config.min_charging_rate_priority
        # min_rates is a numpy array in actual SessionInfo
        if hasattr(session, 'min_rates'):
            if isinstance(session.min_rates, np.ndarray):
                return session.min_rates
            elif isinstance(session.min_rates, (int, float)):
                return session.min_rates
        return 0.0
    
    def should_enforce_energy_equality(self, session: SessionInfo) -> bool:
        """
        Determine if energy equality constraint should be enforced.
        
        For priority sessions, we want >= constraint (must deliver at least requested).
        For non-priority, we use <= constraint (can deliver less).
        """
        return self.is_priority(session.session_id)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_modified_charging_rate_bounds(priority_manager: PrioritySessionManager):
    """
    Factory function to create a modified charging_rate_bounds method.
    
    This can be used to patch the AdaptiveChargingOptimization class
    without modifying its source code.
    
    Usage:
        from priority_ev_automation import create_modified_charging_rate_bounds, PrioritySessionManager
        
        priority_mgr = PrioritySessionManager(priority_sessions={'session_1', 'session_2'})
        AdaptiveChargingOptimization.charging_rate_bounds = create_modified_charging_rate_bounds(priority_mgr)
    """
    import cvxpy as cp
    
    @staticmethod
    def charging_rate_bounds(
        rates: cp.Variable, 
        active_sessions: List[SessionInfo], 
        evse_index: List[str]
    ):
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        
        for session in active_sessions:
            i = evse_index.index(session.station_id)
            session_slice = slice(
                session.arrival_offset, 
                session.arrival_offset + session.remaining_time
            )
            
            if priority_manager.is_priority(session.session_id):
                # Higher minimum rate for priority sessions
                # session.min_rates is a numpy array of length remaining_time
                lb[i, session_slice] = np.maximum(
                    session.min_rates, 
                    priority_manager._config.min_charging_rate_priority
                )
            else:
                lb[i, session_slice] = session.min_rates
            
            ub[i, session_slice] = session.max_rates
        
        # Ensure feasibility
        ub[ub < lb] = lb[ub < lb]
        
        return {
            "charging_rate_bounds.lb": rates >= lb,
            "charging_rate_bounds.ub": rates <= ub,
        }
    
    return charging_rate_bounds


def create_modified_energy_constraints(priority_manager: PrioritySessionManager):
    """
    Factory function to create a modified energy_constraints method.
    """
    import cvxpy as cp
    
    @staticmethod
    def energy_constraints(
        rates: cp.Variable,
        active_sessions: List[SessionInfo],
        infrastructure,  # InfrastructureInfo
        period: int,
        enforce_energy_equality: bool = False,
    ):
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
            elif priority_manager.is_priority(session.session_id):
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
    
    return energy_constraints


# =============================================================================
# SCENARIO GENERATION INTEGRATION
# =============================================================================

def mark_priority_candidates_in_events(
    events: List,  # List of PluginEvent or (PluginEvent, priority) tuples
    selector: PrioritySelector,
    period_minutes: float = 5.0
) -> List:
    """
    Mark priority candidates in a list of events.
    
    This function processes generated events and marks those that
    should be considered for priority based on the selector rules.
    
    Args:
        events: List of PluginEvent objects or EventQueue
        selector: PrioritySelector instance
        period_minutes: Length of each period in minutes
    
    Returns:
        List of (event, is_priority) tuples for use with modified EventQueue
    """
    # Extract session data from events
    session_data = []
    event_list = []
    
    for event in events:
        # Handle both raw events and (event, priority) tuples
        if isinstance(event, tuple):
            ev_event = event[0]
        else:
            ev_event = event
        
        event_list.append(ev_event)
        
        # Extract EV data from PluginEvent
        ev = ev_event.ev
        session_data.append({
            'session_id': ev.session_id,
            'energy_demand': ev.requested_energy,
            'arrival': ev_event.timestamp,
            'departure': ev.departure,
            'station_id': ev.station_id
        })
    
    # Select priority candidates
    priority_ids = selector.select_from_raw_data(session_data, period_minutes)
    
    # Create output with priority flags
    result = []
    for i, event in enumerate(event_list):
        session_id = event.ev.session_id
        is_priority = session_id in priority_ids
        result.append((event, is_priority))
    
    return result


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    print("Priority EV Automation Module")
    print("=" * 50)
    
    # Create sample sessions for testing
    sample_sessions = [
        SessionInfo("EVSE-001", "session_0", 15.0, 0, 0, 72, current_time=0),    # 6h, 15kWh - good candidate
        SessionInfo("EVSE-002", "session_1", 25.0, 0, 10, 82, current_time=0),   # 6h, 25kWh - borderline
        SessionInfo("EVSE-003", "session_2", 35.0, 0, 20, 56, current_time=0),   # 3h, 35kWh - high demand, short
        SessionInfo("EVSE-004", "session_3", 12.0, 0, 30, 78, current_time=0),   # 4h, 12kWh - good candidate
        SessionInfo("EVSE-005", "session_4", 40.0, 0, 40, 112, current_time=0),  # 6h, 40kWh - high demand, ok duration
        SessionInfo("EVSE-006", "session_5", 8.0, 0, 50, 74, current_time=0),    # 2h, 8kWh - low energy
        SessionInfo("EVSE-007", "session_6", 22.0, 0, 60, 132, current_time=0),  # 6h, 22kWh - good candidate
        SessionInfo("EVSE-008", "session_7", 18.0, 0, 70, 106, current_time=0),  # 3h, 18kWh - good candidate
        SessionInfo("EVSE-009", "session_8", 30.0, 0, 80, 104, current_time=0),  # 2h, 30kWh - high demand, very short
        SessionInfo("EVSE-010", "session_9", 20.0, 0, 90, 162, current_time=0),  # 6h, 20kWh - good candidate
    ]
    
    # Create selector with default config
    config = PriorityConfig(max_priority_pct=0.30)
    selector = PrioritySelector(config)
    
    # Select priority sessions
    priority_ids = selector.select_from_sessions(sample_sessions, total_periods=288)
    
    print(f"\nTotal sessions: {len(sample_sessions)}")
    print(f"Max priority allowed: {int(len(sample_sessions) * config.max_priority_pct)}")
    print(f"Selected priority sessions: {len(priority_ids)}")
    print(f"Priority IDs: {priority_ids}")
    
    # Show details
    print("\nSession Details:")
    print("-" * 80)
    for session in sample_sessions:
        duration = (session.departure - session.arrival) * 5 / 60
        status = "PRIORITY" if session.session_id in priority_ids else "normal"
        print(f"  {session.session_id}: {session.remaining_demand:.1f}kWh, "
              f"{duration:.1f}h duration, [{status}]")
    
    # Show selection summary
    summary = selector.get_selection_summary()
    print(f"\nSelection Summary:")
    print(f"  High-demand priority vehicles: {summary.get('high_demand_count', 0)}")
    
    # Test PrioritySessionManager
    print("\n" + "=" * 50)
    print("Testing PrioritySessionManager")
    
    manager = PrioritySessionManager(priority_ids, config)
    
    for session in sample_sessions[:3]:
        min_rate = manager.get_min_rate_for_session(session)
        # Handle both scalar and array returns
        if isinstance(min_rate, np.ndarray):
            min_rate_display = f"{min_rate[0]:.1f}A (array)"
        else:
            min_rate_display = f"{min_rate:.1f}A"
        print(f"  {session.session_id}: is_priority={manager.is_priority(session.session_id)}, "
              f"min_rate={min_rate_display}")

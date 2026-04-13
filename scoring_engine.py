from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

# --- Data Structures (Mocking what comes from Correlation Engine) ---

@dataclass
class DetectionEvent:
    camera_id: str
    timestamp: datetime
    is_restricted_zone: bool = False

@dataclass
class Trail:
    trail_id: str
    events: List[DetectionEvent]
    
@dataclass
class ScoreResult:
    total_score: float
    factors: Dict[str, float]
    is_alert: bool
    
# --- Functional Scoring Rules ---

def calculate_time_penalty(current_time: datetime) -> float:
    """Assigns base penalty based on time of day."""
    hour = current_time.hour
    is_weekend = current_time.weekday() >= 5
    
    score = 0.0
    if is_weekend:
        score += 10.0
        
    if 23 <= hour or hour < 5:
        score += 30.0  # Deep night
    elif 18 <= hour < 23:
        score += 15.0  # Evening
    # Business hours (5-18) add 0
    return score

def calculate_zone_penalty(trail: Trail) -> float:
    """Checks if the latest event is in a restricted zone."""
    if not trail.events:
        return 0.0
    latest_event = trail.events[-1]
    return 30.0 if latest_event.is_restricted_zone else 0.0

def calculate_dwell_penalty(trail: Trail, threshold_seconds: int = 120) -> float:
    """Penalizes if a person stays on the SAME camera for too long."""
    if len(trail.events) < 2:
        return 0.0
    
    latest = trail.events[-1]
    previous = trail.events[-2]
    
    # If they are still on the same camera
    if latest.camera_id == previous.camera_id:
        dwell_time = (latest.timestamp - previous.timestamp).total_seconds()
        if dwell_time > threshold_seconds:
            return 20.0
    return 0.0

def calculate_speed_penalty(actual_transit_time: float, expected_transit_time: float) -> float:
    """
    Checks if someone moved unusually fast between cameras.
    Requires input from the Welford Transit Learner.
    """
    if expected_transit_time <= 0:
        return 0.0 # No data yet
    
    # If they moved in less than 50% of the expected time, it's a dead sprint
    if actual_transit_time < (expected_transit_time * 0.5):
        return 20.0
    return 0.0

def apply_routine_suppression(base_score: float, route_seen_count: int) -> float:
    """
    CRITICAL FIX: Suppresses the score if this is a known, routine action.
    A cleaner's routine shouldn't just 'not add points', it should suppress base points.
    """
    if route_seen_count == 0:
        return base_score + 25.0  # Never seen before: Add penalty
    elif route_seen_count < 10:
        return base_score + 10.0  # Rare route
    elif route_seen_count > 50:
        # Highly common route (e.g., standard patrol or cleaner). Suppress by 60%
        return base_score * 0.4 
    else:
        # Standard route, no suppression
        return base_score

# --- Main Engine ---

class IncidentScorer:
    def __init__(self, alert_threshold: float = 70.0):
        self.alert_threshold = alert_threshold

    def evaluate_trail(self, 
                       trail: Trail, 
                       route_seen_count: int, 
                       expected_transit_time: float = 0.0,
                       actual_transit_time: float = 0.0) -> ScoreResult:
        """
        Takes a trail and contextual data, outputs a final incident score.
        """
        if not trail.events:
            return ScoreResult(0.0, {}, False)
            
        latest_event = trail.events[-1]
        factors = {}
        
        # 1. Base Additive Scoring
        factors['time_penalty'] = calculate_time_penalty(latest_event.timestamp)
        factors['zone_penalty'] = calculate_zone_penalty(trail)
        factors['dwell_penalty'] = calculate_dwell_penalty(trail)
        factors['speed_penalty'] = calculate_speed_penalty(actual_transit_time, expected_transit_time)
        
        base_score = sum(factors.values())
        factors['base_raw_score'] = base_score
        
        # 2. Multiplicative/Suppressive Scoring based on History
        final_score = apply_routine_suppression(base_score, route_seen_count)
        factors['route_suppression_applied'] = final_score - base_score
        
        # Cap at 100
        final_score = min(final_score, 100.0)
        
        return ScoreResult(
            total_score=final_score,
            factors=factors,
            is_alert=(final_score >= self.alert_threshold)
        )

# --- Quick Test Block (To prove it works before we integrate) ---
if __name__ == "__main__":
    scorer = IncidentScorer()
    
    # Scenario: Cleaner at 3 AM on a Saturday, entering restricted Server Room
    # Additive alone would trigger this. Let's see suppression work.
    dt_3am_sat = datetime(2026, 4, 4, 3, 0, 0) # Saturday 3AM
    cleaner_trail = Trail(
        trail_id="T001",
        events=[
            DetectionEvent("cam_hallway", dt_3am_sat),
            DetectionEvent("cam_server_room", dt_3am_sat, is_restricted_zone=True)
        ]
    )
    
    # We've seen this route 80 times (it's their normal job)
    result = scorer.evaluate_trail(cleaner_trail, route_seen_count=80)
    
    print(f"Cleaner Score: {result.total_score:.1f} | Alert: {result.is_alert}")
    print(f"Factors: {result.factors}")
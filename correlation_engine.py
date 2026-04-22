"""
correlation_engine.py — Cross-Camera Correlation Engine (v3)
=============================================================
Added non-blocking UI push thread.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime
from typing import Optional

import requests
import database
from models import DetectionEvent, Trail
from transit_learner import TransitLearner
from scoring_engine import IncidentScorer, Trail as ScoringTrail, DetectionEvent as ScoringEvent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_COLOR_DISTANCE = 60.0
MAX_ASPECT_RATIO_DIFF = 0.20
TRAIL_STALE_TIMEOUT_SECONDS = 300
CLEANUP_INTERVAL_SECONDS = 60
MIN_SECURED_ZONES_FOR_INCIDENT = 2

AFTER_HOURS_START = 20  # 8pm
AFTER_HOURS_END = 6     # 6am


# ── Alert Dispatcher ──────────────────────────────────────────────────────────

class AlertDispatcher:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token.strip()
        self._chat_id = chat_id.strip()

    def send_incident(self, trail: Trail) -> None:
        path_str = trail.get_path_summary()
        start_time = trail.events[0].timestamp
        end_time = trail.events[-1].timestamp
        total_seconds = int(end_time - start_time)
        time_label = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

        logger.warning(
            "INCIDENT | trail=%s | path=%s | duration=%ds | started=%s",
            trail.trail_id, path_str, total_seconds, time_label,
        )

        if not self._bot_token or not self._chat_id:
            return

        self._push_telegram(trail.trail_id, path_str, total_seconds, time_label)

    def _push_telegram(
        self, trail_id: str, path_str: str, total_seconds: int, time_label: str
    ) -> None:
        message = (
            f"🚨 *Smart CCTV — Incident Detected*\n\n"
            f"*Trail ID:* `{trail_id}`\n"
            f"*Path:* {path_str}\n"
            f"*Duration:* {total_seconds}s\n"
            f"*Started:* {time_label}\n\n"
            f"_Subject traversed multiple secured zones within anomaly threshold._"
        )
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            requests.post(
                url,
                data={"chat_id": self._chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=8,
            )
        except Exception as exc:
            logger.error("Telegram push failed for trail %s: %s", trail_id, exc)


# ── Correlation Engine ────────────────────────────────────────────────────────

class CorrelationEngine:
    def __init__(
        self,
        camera_map: dict,
        bot_token: str = "",
        chat_id: str = "",
        transit_db_path: str = "transit_memory.db",
    ) -> None:
        self._camera_map = camera_map
        self._active_trails: list[Trail] = []
        self._trails_lock = threading.Lock()
        self._learner = TransitLearner(camera_map, db_path=transit_db_path)
        self._dispatcher = AlertDispatcher(bot_token, chat_id)
        self._scorer = IncidentScorer(alert_threshold=70.0)
        self._start_cleanup_thread()
        database.init_db()
        logger.info("CorrelationEngine ready.")

    def _push_to_ui(self, trail: Trail, event_type: str, score_result=None) -> None:
        """Sends real-time updates to the local web server asynchronously."""
        if not trail.events:
            return

        last_event = trail.events[-1]
        cam_info = self._camera_map.get("cameras", {}).get(last_event.camera_id, {})
        coords = cam_info.get("map_coords", {"x": 0, "y": 0})

        payload = {
            "type": event_type,
            "data": {
                "id": trail.trail_id,
                "x": coords["x"],
                "y": coords["y"],
                "history": [], 
                "score": score_result.total_score if score_result else 0,
                "timestamp": datetime.fromtimestamp(last_event.timestamp).isoformat(),
                "message": f"Activity at {cam_info.get('name', last_event.camera_id)}"
            }
        }
        
        for e in trail.events:
            e_coords = self._camera_map.get("cameras", {}).get(e.camera_id, {}).get("map_coords", {"x": 0, "y": 0})
            payload["data"]["history"].append({"x": e_coords["x"], "y": e_coords["y"]})

        # FIX: Fire-and-forget thread so AI never blocks waiting for the web server
        def background_push():
            try:
                requests.post("http://127.0.0.1:8000/api/internal/push", json=payload, timeout=0.5)
            except Exception:
                pass

        threading.Thread(target=background_push, daemon=True).start()

    def process_event(self, new_event: DetectionEvent) -> None:
        matched_trail = self._find_matching_trail(new_event)
        with self._trails_lock:
            if matched_trail:
                self._extend_trail(matched_trail, new_event)
            else:
                self._start_new_trail(new_event)

    def _find_matching_trail(self, new_event: DetectionEvent) -> Optional[Trail]:
        with self._trails_lock:
            active = [t for t in self._active_trails if t.status == "active"]

        for trail in active:
            last_event = trail.events[-1]
            if not self._is_adjacent_or_same(last_event.camera_id, new_event.camera_id):
                continue
            if not self._is_valid_transit(last_event, new_event):
                continue
            if self._is_appearance_match(last_event.appearance_vector, new_event.appearance_vector):
                return trail
        return None

    def _is_adjacent_or_same(self, from_cam: str, to_cam: str) -> bool:
        if from_cam == to_cam:
            return True
        cam_info = self._camera_map.get("cameras", {}).get(from_cam, {})
        return to_cam in cam_info.get("adjacent_to", [])

    def _is_valid_transit(self, last_event: DetectionEvent, new_event: DetectionEvent) -> bool:
        if last_event.camera_id == new_event.camera_id:
            return True

        elapsed = new_event.timestamp - last_event.timestamp
        hour = datetime.fromtimestamp(last_event.timestamp).hour
        window = self._learner.get_transit_window(
            last_event.camera_id, new_event.camera_id, hour
        )
        return window.contains(elapsed)

    def _is_appearance_match(self, vec1: dict, vec2: dict) -> bool:
        if not vec1 or not vec2:
            return False
        if vec1.get("type") != "heuristic" or vec2.get("type") != "heuristic":
            return False

        ar1, ar2 = vec1.get("aspect_ratio", 0), vec2.get("aspect_ratio", 0)
        if ar1 == 0 or ar2 == 0:
            return False
        if abs(ar1 - ar2) / max(ar1, ar2) > MAX_ASPECT_RATIO_DIFF:
            return False

        torso_dist = _euclidean_rgb(vec1.get("torso_rgb", []), vec2.get("torso_rgb", []))
        legs_dist = _euclidean_rgb(vec1.get("legs_rgb", []), vec2.get("legs_rgb", []))
        if torso_dist is None or legs_dist is None:
            return False

        return torso_dist < MAX_COLOR_DISTANCE and legs_dist < MAX_COLOR_DISTANCE

    def _start_new_trail(self, event: DetectionEvent) -> None:
        new_trail = Trail()
        new_trail.add_event(event)
        self._active_trails.append(new_trail)
        self._push_to_ui(new_trail, "trail_update")
        database.save_trail_and_incident(new_trail)

    def _extend_trail(self, trail: Trail, new_event: DetectionEvent) -> None:
        previous_camera = trail.events[-1].camera_id
        trail.add_event(new_event)

        if new_event.camera_id != previous_camera:
            self._record_confirmed_transit(trail, previous_camera, new_event)

        self._evaluate_incident_rules(trail)
        self._push_to_ui(trail, "trail_update")
        database.save_trail_and_incident(trail)

    def _record_confirmed_transit(
        self, trail: Trail, from_camera: str, arrival_event: DetectionEvent
    ) -> None:
        departure_events = [e for e in trail.events if e.camera_id == from_camera]
        if not departure_events:
            return

        elapsed = arrival_event.timestamp - departure_events[-1].timestamp
        hour = datetime.fromtimestamp(departure_events[-1].timestamp).hour

        self._learner.record_transit(
            from_camera=from_camera,
            to_camera=arrival_event.camera_id,
            elapsed_seconds=elapsed,
            hour=hour,
        )

    def _evaluate_incident_rules(self, trail: Trail) -> None:
        scoring_events = []
        for e in trail.events:
            is_restricted = self._camera_map.get("cameras", {}).get(e.camera_id, {}).get("is_restricted", False)
            scoring_events.append(ScoringEvent(e.camera_id, datetime.fromtimestamp(e.timestamp), is_restricted))
        
        scoring_trail = ScoringTrail(trail.trail_id, scoring_events)
        
        route_seen_count = 0
        if len(trail.events) >= 2:
            route_seen_count = database.get_route_seen_count([e.camera_id for e in trail.events[-2:]])
            
        score_result = self._scorer.evaluate_trail(scoring_trail, route_seen_count)
        
        if score_result.is_alert and trail.status != "incident":
            trail.status = "incident"
            if _is_in_alert_window():
                self._dispatcher.send_incident(trail)
                self._push_to_ui(trail, "new_incident", score_result)
            database.save_trail_and_incident(trail, score_result)

    def _start_cleanup_thread(self) -> None:
        threading.Thread(target=self._cleanup_loop, daemon=True, name="trail-cleanup").start()

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            self._close_stale_trails()

    def _close_stale_trails(self) -> None:
        now = time.time()
        with self._trails_lock:
            for trail in self._active_trails:
                if trail.status == "active" and (now - trail.last_updated) > TRAIL_STALE_TIMEOUT_SECONDS:
                    trail.status = "closed"

def _load_alert_schedule() -> dict | None:
    """Load alert schedule from file. Returns None if not configured (always alert)."""
    try:
        if os.path.exists("alert_schedule.json"):
            with open("alert_schedule.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _is_in_alert_window() -> bool:
    """Returns True if the current time falls within the configured alert window."""
    schedule = _load_alert_schedule()
    if schedule is None:
        return True  # No schedule set — maintain backward-compatible behaviour
    if not schedule.get("enabled", True):
        return False
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    start_minutes = schedule.get("start_hour", 22) * 60 + schedule.get("start_minute", 0)
    end_minutes = schedule.get("end_hour", 7) * 60 + schedule.get("end_minute", 0)
    # Overnight window (e.g. 22:00 → 07:00 crosses midnight)
    if start_minutes > end_minutes:
        return current_minutes >= start_minutes or current_minutes < end_minutes
    return start_minutes <= current_minutes < end_minutes


def _euclidean_rgb(rgb1: list, rgb2: list) -> Optional[float]:
    if len(rgb1) != 3 or len(rgb2) != 3:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
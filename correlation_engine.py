"""
correlation_engine.py — Cross-Camera Correlation Engine (v2)
=============================================================
Changes from v1:
  - Replaced hardcoded transit time checks with TransitLearner.
    Every confirmed transit is recorded so the model self-updates
    from real behaviour — no more hardcoded expected_seconds.
  - Replaced all print() with structured logging.
  - Incident rules now consider time-of-day and restricted zones,
    not just camera count alone.
  - AlertDispatcher extracted so the engine has one responsibility.
  - Stale trail cleanup runs in a background thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime
from typing import Optional

import requests

from models import DetectionEvent, Trail
from transit_learner import TransitLearner

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
    """Sends incident alerts. Decoupled from engine so channels can be extended."""

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
            logger.info("Telegram not configured — skipping push notification.")
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
            response = requests.post(
                url,
                data={"chat_id": self._chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=8,
            )
            response.raise_for_status()
            logger.info("Telegram alert sent for trail %s.", trail_id)
        except requests.exceptions.Timeout:
            logger.error("Telegram push timed out for trail %s.", trail_id)
        except requests.exceptions.RequestException as exc:
            logger.error("Telegram push failed for trail %s: %s", trail_id, exc)


# ── Correlation Engine ────────────────────────────────────────────────────────

class CorrelationEngine:
    """
    Reads DetectionEvents and stitches them into cross-camera Trails.

    Transit validation now uses TransitLearner — every confirmed transit
    is recorded as an observation so the model self-improves over time.
    """

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
        self._start_cleanup_thread()
        logger.info("CorrelationEngine ready.")

    def process_event(self, new_event: DetectionEvent) -> None:
        matched_trail = self._find_matching_trail(new_event)
        with self._trails_lock:
            if matched_trail:
                self._extend_trail(matched_trail, new_event)
            else:
                self._start_new_trail(new_event)

    # ── Trail matching ────────────────────────────────────────────────────────

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

        if window.contains(elapsed):
            return True

        logger.debug(
            "Transit rejected %s→%s: %.1fs not in %s",
            last_event.camera_id, new_event.camera_id, elapsed, window,
        )
        return False

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

    # ── Trail lifecycle ───────────────────────────────────────────────────────

    def _start_new_trail(self, event: DetectionEvent) -> None:
        new_trail = Trail()
        new_trail.add_event(event)
        self._active_trails.append(new_trail)
        logger.info("New trail %s started at %s.", new_trail.trail_id, event.camera_id)

    def _extend_trail(self, trail: Trail, new_event: DetectionEvent) -> None:
        previous_camera = trail.events[-1].camera_id
        trail.add_event(new_event)

        if new_event.camera_id != previous_camera:
            self._record_confirmed_transit(trail, previous_camera, new_event)

        self._evaluate_incident_rules(trail)

    def _record_confirmed_transit(
        self, trail: Trail, from_camera: str, arrival_event: DetectionEvent
    ) -> None:
        """A confirmed cross-camera transit — feed it to the learner."""
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
        logger.info(
            "Trail %s: transit %s→%s %.1fs recorded (h=%d).",
            trail.trail_id, from_camera, arrival_event.camera_id, elapsed, hour,
        )

    # ── Incident rules ────────────────────────────────────────────────────────

    def _evaluate_incident_rules(self, trail: Trail) -> None:
        if trail.status == "incident":
            return

        unique_cameras = {e.camera_id for e in trail.events}
        if len(unique_cameras) < MIN_SECURED_ZONES_FOR_INCIDENT:
            return

        hour = datetime.fromtimestamp(trail.events[0].timestamp).hour
        is_after_hours = hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END
        crossed_restricted = self._trail_crossed_restricted_zone(trail)

        if is_after_hours or crossed_restricted:
            trail.status = "incident"
            self._dispatcher.send_incident(trail)

    def _trail_crossed_restricted_zone(self, trail: Trail) -> bool:
        cameras_config = self._camera_map.get("cameras", {})
        return any(
            cameras_config.get(e.camera_id, {}).get("is_restricted", False)
            for e in trail.events
        )

    # ── Background cleanup ────────────────────────────────────────────────────

    def _start_cleanup_thread(self) -> None:
        threading.Thread(
            target=self._cleanup_loop, daemon=True, name="trail-cleanup"
        ).start()

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            self._close_stale_trails()

    def _close_stale_trails(self) -> None:
        now = time.time()
        closed = 0
        with self._trails_lock:
            for trail in self._active_trails:
                if trail.status == "active" and (now - trail.last_updated) > TRAIL_STALE_TIMEOUT_SECONDS:
                    trail.status = "closed"
                    closed += 1
        if closed:
            logger.info("Closed %d stale trail(s).", closed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euclidean_rgb(rgb1: list, rgb2: list) -> Optional[float]:
    if len(rgb1) != 3 or len(rgb2) != 3:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
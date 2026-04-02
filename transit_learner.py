"""
transit_learner.py — Adaptive Transit Learning Engine
======================================================
Replaces the hardcoded `expected_seconds` / `tolerance_seconds` values
in camera_map.json with a self-updating statistical model.

How it works:
  - Every time the correlation engine confirms a person moved from
    Camera A to Camera B, it calls `record_transit()` with the
    time it actually took.
  - This module stores each observation in SQLite (persists across
    restarts) and recalculates the running mean + std deviation.
  - `get_transit_window()` returns the current (min, max) acceptable
    range for a given camera pair and hour-of-day bucket.
  - While sample count is below MIN_SAMPLES_FOR_CONFIDENCE the system
    falls back to the hardcoded values in camera_map.json so it never
    fires blind.

Design decisions:
  - SQLite: zero external dependencies, single file, works on a Pi.
  - Hour-of-day bucketing: people move faster at 8am than 3pm.
    One model per (camera_pair, hour) prevents day/night averaging.
  - Welford's online algorithm for mean/variance: no need to store
    every raw observation — just running stats per bucket.
  - Thread-safe: a single threading.Lock guards all DB writes.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_SAMPLES_FOR_CONFIDENCE = 30   # Don't use learned model until we have this many
FALLBACK_TOLERANCE_MULTIPLIER = 2.0  # How wide the window is before we have confidence
LEARNED_STD_MULTIPLIER = 2.5      # Window = mean ± (N * std_dev) once confident
MIN_WINDOW_SECONDS = 3.0          # Never allow a window narrower than this
DB_FILENAME = "transit_memory.db"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransitWindow:
    """The acceptable time range for a camera-pair transit."""
    from_camera: str
    to_camera: str
    hour_bucket: int          # 0–23
    min_seconds: float
    max_seconds: float
    sample_count: int
    is_learned: bool          # True = from observations, False = from fallback config

    def contains(self, elapsed_seconds: float) -> bool:
        return self.min_seconds <= elapsed_seconds <= self.max_seconds

    def __str__(self) -> str:
        source = "learned" if self.is_learned else "fallback"
        return (
            f"{self.from_camera}→{self.to_camera} "
            f"[{self.min_seconds:.1f}s – {self.max_seconds:.1f}s] "
            f"({source}, n={self.sample_count})"
        )


# ── Database layer ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transit_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_camera     TEXT    NOT NULL,
    to_camera       TEXT    NOT NULL,
    hour_bucket     INTEGER NOT NULL,   -- 0-23 hour of day
    elapsed_seconds REAL    NOT NULL,
    recorded_at     REAL    NOT NULL    -- unix timestamp
);

CREATE TABLE IF NOT EXISTS transit_stats (
    from_camera  TEXT    NOT NULL,
    to_camera    TEXT    NOT NULL,
    hour_bucket  INTEGER NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    mean_seconds REAL    NOT NULL DEFAULT 0.0,
    -- Welford M2 accumulator (sum of squared deviations from mean)
    m2           REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (from_camera, to_camera, hour_bucket)
);
"""


class TransitDatabase:
    """
    Thin SQLite wrapper.  All public methods are thread-safe via a single lock.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._initialise_schema()

    @contextmanager
    def _connection(self):
        """Yield a connection that auto-commits or rolls back."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise exc
        finally:
            conn.close()

    def _initialise_schema(self) -> None:
        try:
            with self._connection() as conn:
                conn.executescript(_SCHEMA)
            logger.info("TransitDatabase ready at '%s'.", self._db_path)
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Failed to initialise transit database at '{self._db_path}': {exc}"
            ) from exc

    def upsert_welford(
        self,
        from_camera: str,
        to_camera: str,
        hour_bucket: int,
        new_value: float,
    ) -> None:
        """
        Update running mean and M2 using Welford's online algorithm.
        This lets us track variance without storing every raw observation.
        """
        with self._lock:
            try:
                with self._connection() as conn:
                    # Fetch existing stats
                    row = conn.execute(
                        """
                        SELECT sample_count, mean_seconds, m2
                        FROM transit_stats
                        WHERE from_camera=? AND to_camera=? AND hour_bucket=?
                        """,
                        (from_camera, to_camera, hour_bucket),
                    ).fetchone()

                    if row is None:
                        count, mean, m2 = 0, 0.0, 0.0
                    else:
                        count, mean, m2 = row["sample_count"], row["mean_seconds"], row["m2"]

                    # Welford update step
                    count += 1
                    delta = new_value - mean
                    mean += delta / count
                    delta2 = new_value - mean
                    m2 += delta * delta2

                    conn.execute(
                        """
                        INSERT INTO transit_stats
                            (from_camera, to_camera, hour_bucket, sample_count, mean_seconds, m2)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(from_camera, to_camera, hour_bucket)
                        DO UPDATE SET
                            sample_count = excluded.sample_count,
                            mean_seconds = excluded.mean_seconds,
                            m2           = excluded.m2
                        """,
                        (from_camera, to_camera, hour_bucket, count, mean, m2),
                    )
            except sqlite3.Error as exc:
                # CAUTION: A write failure here means one observation is lost.
                # This is acceptable — we log it and continue rather than
                # crashing the whole pipeline over a single data point.
                logger.error(
                    "Failed to record transit observation (%s→%s h=%d val=%.1f): %s",
                    from_camera, to_camera, hour_bucket, new_value, exc,
                )

    def get_stats(
        self,
        from_camera: str,
        to_camera: str,
        hour_bucket: int,
    ) -> Optional[tuple[int, float, float]]:
        """
        Returns (sample_count, mean_seconds, std_dev) or None if no data.
        """
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT sample_count, mean_seconds, m2
                    FROM transit_stats
                    WHERE from_camera=? AND to_camera=? AND hour_bucket=?
                    """,
                    (from_camera, to_camera, hour_bucket),
                ).fetchone()

            if row is None or row["sample_count"] < 2:
                return None

            count = row["sample_count"]
            mean = row["mean_seconds"]
            # Population variance = M2 / n  (we use population, not sample, for
            # conservative anomaly detection — tighter is safer here)
            variance = row["m2"] / count
            std_dev = math.sqrt(variance) if variance > 0 else 0.0

            return count, mean, std_dev

        except sqlite3.Error as exc:
            logger.error(
                "Failed to read transit stats (%s→%s h=%d): %s",
                from_camera, to_camera, hour_bucket, exc,
            )
            return None


# ── Public interface ──────────────────────────────────────────────────────────

class TransitLearner:
    """
    The self-learning transit time model.

    Usage (inside CorrelationEngine):

        learner = TransitLearner(camera_map, db_path="transit_memory.db")

        # When a trail confirms a real transit:
        learner.record_transit("cam_01_gate", "cam_02_corridor", elapsed_seconds=18.4)

        # When validating a candidate transit:
        window = learner.get_transit_window("cam_01_gate", "cam_02_corridor", hour=14)
        if not window.contains(elapsed_seconds):
            reject()
    """

    def __init__(
        self,
        camera_map: dict,
        db_path: str = DB_FILENAME,
        min_samples: int = MIN_SAMPLES_FOR_CONFIDENCE,
    ) -> None:
        self._camera_map = camera_map
        self._min_samples = min_samples
        self._db = TransitDatabase(Path(db_path))

        logger.info(
            "TransitLearner ready — confidence threshold: %d observations.",
            self._min_samples,
        )

    def record_transit(
        self,
        from_camera: str,
        to_camera: str,
        elapsed_seconds: float,
        hour: int,
    ) -> None:
        """
        Record a confirmed real-world transit observation.
        Call this ONLY for verified trails — not candidates.

        Args:
            from_camera:     Camera ID the subject left.
            to_camera:       Camera ID the subject arrived at.
            elapsed_seconds: Actual time between the two detections.
            hour:            Hour of day (0–23) when the transit occurred.
        """
        if elapsed_seconds <= 0:
            logger.warning(
                "Ignoring non-positive transit time (%.2fs) for %s→%s.",
                elapsed_seconds, from_camera, to_camera,
            )
            return

        if not (0 <= hour <= 23):
            raise ValueError(f"hour must be 0–23, got {hour}.")

        self._db.upsert_welford(from_camera, to_camera, hour, elapsed_seconds)

        stats = self._db.get_stats(from_camera, to_camera, hour)
        if stats:
            count, mean, std_dev = stats
            logger.debug(
                "Transit recorded %s→%s h=%d: %.1fs | "
                "n=%d mean=%.1fs σ=%.1fs",
                from_camera, to_camera, hour, elapsed_seconds,
                count, mean, std_dev,
            )

    def get_transit_window(
        self,
        from_camera: str,
        to_camera: str,
        hour: int,
    ) -> TransitWindow:
        """
        Return the acceptable transit window for this camera pair and hour.

        If we have enough observations → learned window (mean ± N*std).
        Otherwise → fallback to camera_map.json hardcoded values.
        """
        if not (0 <= hour <= 23):
            raise ValueError(f"hour must be 0–23, got {hour}.")

        stats = self._db.get_stats(from_camera, to_camera, hour)

        if stats and stats[0] >= self._min_samples:
            return self._build_learned_window(from_camera, to_camera, hour, stats)

        return self._build_fallback_window(from_camera, to_camera, hour, stats)

    def learning_status(self) -> dict:
        """
        Returns a summary of how confident the model is for each camera pair.
        Useful for the dashboard / admin UI.
        """
        status = {}
        transit_model = self._camera_map.get("transit_model", {})

        for key in transit_model:
            parts = key.split("_to_")
            if len(parts) != 2:
                continue
            from_cam, to_cam = parts[0], parts[1]
            pair_status = {}
            for hour in range(24):
                stats = self._db.get_stats(from_cam, to_cam, hour)
                if stats:
                    count, mean, std_dev = stats
                    pair_status[f"h{hour:02d}"] = {
                        "samples": count,
                        "mean_s": round(mean, 1),
                        "std_dev_s": round(std_dev, 1),
                        "confident": count >= self._min_samples,
                    }
            status[key] = pair_status

        return status

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_learned_window(
        self,
        from_camera: str,
        to_camera: str,
        hour: int,
        stats: tuple[int, float, float],
    ) -> TransitWindow:
        count, mean, std_dev = stats
        half_width = max(
            LEARNED_STD_MULTIPLIER * std_dev,
            MIN_WINDOW_SECONDS,
        )
        min_s = max(0.0, mean - half_width)
        max_s = mean + half_width

        logger.debug(
            "Learned window %s→%s h=%d: [%.1f–%.1f]s (n=%d)",
            from_camera, to_camera, hour, min_s, max_s, count,
        )
        return TransitWindow(
            from_camera=from_camera,
            to_camera=to_camera,
            hour_bucket=hour,
            min_seconds=min_s,
            max_seconds=max_s,
            sample_count=count,
            is_learned=True,
        )

    def _build_fallback_window(
        self,
        from_camera: str,
        to_camera: str,
        hour: int,
        stats: Optional[tuple],
    ) -> TransitWindow:
        """
        Use camera_map.json values, widened by FALLBACK_TOLERANCE_MULTIPLIER
        to be forgiving while still in learning mode.
        """
        transit_key = f"{from_camera}_to_{to_camera}"
        config = self._camera_map.get("transit_model", {}).get(transit_key)
        sample_count = stats[0] if stats else 0

        if config is None:
            # No config at all — be very permissive (up to 5 minutes)
            logger.warning(
                "No transit config found for %s→%s. Allowing up to 300s.",
                from_camera, to_camera,
            )
            return TransitWindow(
                from_camera=from_camera,
                to_camera=to_camera,
                hour_bucket=hour,
                min_seconds=0.0,
                max_seconds=300.0,
                sample_count=sample_count,
                is_learned=False,
            )

        expected = float(config["expected_seconds"])
        tolerance = float(config["tolerance_seconds"]) * FALLBACK_TOLERANCE_MULTIPLIER

        logger.debug(
            "Fallback window %s→%s h=%d: [%.1f–%.1f]s (n=%d/%d)",
            from_camera, to_camera, hour,
            max(0.0, expected - tolerance),
            expected + tolerance,
            sample_count, self._min_samples,
        )
        return TransitWindow(
            from_camera=from_camera,
            to_camera=to_camera,
            hour_bucket=hour,
            min_seconds=max(0.0, expected - tolerance),
            max_seconds=expected + tolerance,
            sample_count=sample_count,
            is_learned=False,
        )
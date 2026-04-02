"""
test_transit_learner.py — Unit tests for the Adaptive Transit Learner
=====================================================================
Run with: python -m pytest test_transit_learner.py -v

Tests cover:
  - Fallback behaviour while below confidence threshold
  - Window tightening as observations accumulate
  - Welford mean/std accuracy vs naive calculation
  - Hour-of-day bucketing keeps separate stats
  - Edge cases: zero elapsed, bad hour, missing config
"""

from __future__ import annotations

import math
import statistics
import tempfile
from pathlib import Path

import pytest

from transit_learner import TransitLearner, MIN_SAMPLES_FOR_CONFIDENCE

SAMPLE_CAMERA_MAP = {
    "cameras": {
        "gate": {"name": "Gate", "adjacent_to": ["corridor"]},
        "corridor": {"name": "Corridor", "adjacent_to": ["gate", "backdoor"]},
        "backdoor": {"name": "Back Door", "adjacent_to": ["corridor"]},
    },
    "transit_model": {
        "gate_to_corridor": {
            "expected_seconds": 20,
            "tolerance_seconds": 10,
        }
    },
}


@pytest.fixture
def learner(tmp_path):
    """Fresh learner backed by a temp DB for each test."""
    return TransitLearner(
        camera_map=SAMPLE_CAMERA_MAP,
        db_path=str(tmp_path / "test_transit.db"),
        min_samples=5,  # Low threshold so tests don't need 30 observations
    )


class TestFallbackBehaviour:
    def test_uses_fallback_when_no_observations(self, learner):
        window = learner.get_transit_window("gate", "corridor", hour=9)
        assert window.is_learned is False
        assert window.sample_count == 0

    def test_fallback_window_is_wider_than_config(self, learner):
        """Fallback applies FALLBACK_TOLERANCE_MULTIPLIER so it's forgiving."""
        window = learner.get_transit_window("gate", "corridor", hour=9)
        # Config: expected=20 tolerance=10 → raw range 10–30
        # Fallback multiplier=2.0 → tolerance becomes 20 → range 0–40
        assert window.min_seconds <= 10
        assert window.max_seconds >= 30

    def test_fallback_allows_up_to_300s_when_no_config(self, learner):
        window = learner.get_transit_window("corridor", "backdoor", hour=9)
        assert window.max_seconds == 300.0
        assert window.is_learned is False

    def test_stays_in_fallback_below_min_samples(self, learner):
        # Record fewer than min_samples observations
        for _ in range(4):
            learner.record_transit("gate", "corridor", elapsed_seconds=20.0, hour=9)
        window = learner.get_transit_window("gate", "corridor", hour=9)
        assert window.is_learned is False
        assert window.sample_count == 4


class TestLearningBehaviour:
    def _seed_observations(self, learner, values, hour=9):
        for v in values:
            learner.record_transit("gate", "corridor", elapsed_seconds=v, hour=hour)

    def test_switches_to_learned_at_min_samples(self, learner):
        self._seed_observations(learner, [20.0] * 5)
        window = learner.get_transit_window("gate", "corridor", hour=9)
        assert window.is_learned is True

    def test_learned_mean_matches_actual_mean(self, learner):
        values = [14.0, 18.0, 20.0, 22.0, 26.0]
        self._seed_observations(learner, values)
        window = learner.get_transit_window("gate", "corridor", hour=9)
        expected_mean = statistics.mean(values)
        # Window midpoint should be close to the true mean
        midpoint = (window.min_seconds + window.max_seconds) / 2
        assert abs(midpoint - expected_mean) < 1.0

    def test_learned_window_tightens_with_consistent_data(self, learner):
        # Tight cluster
        tight_values = [19.5, 20.0, 20.5, 19.8, 20.2]
        self._seed_observations(learner, tight_values)
        tight_window = learner.get_transit_window("gate", "corridor", hour=9)

        # Reset with a new learner backed by fresh DB
        import tempfile
        learner2 = TransitLearner(
            camera_map=SAMPLE_CAMERA_MAP,
            db_path=str(tempfile.mkdtemp() + "/t.db"),
            min_samples=5,
        )
        # Wide spread
        wide_values = [5.0, 12.0, 20.0, 28.0, 35.0]
        for v in wide_values:
            learner2.record_transit("gate", "corridor", elapsed_seconds=v, hour=9)

        wide_window = learner2.get_transit_window("gate", "corridor", hour=9)

        tight_width = tight_window.max_seconds - tight_window.min_seconds
        wide_width = wide_window.max_seconds - wide_window.min_seconds
        assert tight_width < wide_width

    def test_hour_buckets_are_independent(self, learner):
        """Morning rush vs late evening should learn different windows."""
        morning_values = [15.0, 16.0, 17.0, 15.5, 16.5]  # Fast
        evening_values = [35.0, 38.0, 36.0, 37.0, 39.0]  # Slow

        for v in morning_values:
            learner.record_transit("gate", "corridor", elapsed_seconds=v, hour=8)
        for v in evening_values:
            learner.record_transit("gate", "corridor", elapsed_seconds=v, hour=21)

        morning_window = learner.get_transit_window("gate", "corridor", hour=8)
        evening_window = learner.get_transit_window("gate", "corridor", hour=21)

        assert morning_window.max_seconds < evening_window.min_seconds

    def test_contains_rejects_outlier(self, learner):
        values = [18.0, 19.0, 20.0, 21.0, 22.0]
        self._seed_observations(learner, values)
        window = learner.get_transit_window("gate", "corridor", hour=9)
        assert window.contains(20.0) is True
        assert window.contains(0.5) is False   # Too fast — impossible
        assert window.contains(200.0) is False  # Too slow


class TestWelfordAccuracy:
    """Verify Welford running stats match the naive calculation."""

    def test_mean_accuracy(self, learner):
        values = [10.0, 15.0, 20.0, 25.0, 30.0, 12.0, 18.0]
        for v in values:
            learner.record_transit("gate", "corridor", elapsed_seconds=v, hour=9)

        stats = learner._db.get_stats("gate", "corridor", 9)
        assert stats is not None
        count, mean, std_dev = stats

        assert count == len(values)
        assert abs(mean - statistics.mean(values)) < 0.01

    def test_std_dev_accuracy(self, learner):
        values = [10.0, 15.0, 20.0, 25.0, 30.0, 12.0, 18.0]
        for v in values:
            learner.record_transit("gate", "corridor", elapsed_seconds=v, hour=9)

        stats = learner._db.get_stats("gate", "corridor", 9)
        assert stats is not None
        _, _, std_dev = stats

        # Population std dev
        expected_std = statistics.pstdev(values)
        assert abs(std_dev - expected_std) < 0.1


class TestEdgeCases:
    def test_rejects_zero_elapsed(self, learner):
        """Zero transit time is physically impossible — must be ignored."""
        learner.record_transit("gate", "corridor", elapsed_seconds=0.0, hour=9)
        stats = learner._db.get_stats("gate", "corridor", 9)
        assert stats is None  # Nothing recorded

    def test_rejects_negative_elapsed(self, learner):
        learner.record_transit("gate", "corridor", elapsed_seconds=-5.0, hour=9)
        stats = learner._db.get_stats("gate", "corridor", 9)
        assert stats is None

    def test_invalid_hour_raises(self, learner):
        with pytest.raises(ValueError, match="hour must be 0–23"):
            learner.record_transit("gate", "corridor", elapsed_seconds=20.0, hour=25)

    def test_get_window_invalid_hour_raises(self, learner):
        with pytest.raises(ValueError, match="hour must be 0–23"):
            learner.get_transit_window("gate", "corridor", hour=-1)

    def test_learning_status_returns_dict(self, learner):
        status = learner.learning_status()
        assert isinstance(status, dict)
        assert "gate_to_corridor" in status

    def test_persists_across_restarts(self, tmp_path):
        """Data written by one instance must survive a restart."""
        db_path = str(tmp_path / "persist_test.db")

        learner_a = TransitLearner(SAMPLE_CAMERA_MAP, db_path=db_path, min_samples=5)
        for _ in range(5):
            learner_a.record_transit("gate", "corridor", elapsed_seconds=20.0, hour=10)

        # Simulate restart — new instance, same DB file
        learner_b = TransitLearner(SAMPLE_CAMERA_MAP, db_path=db_path, min_samples=5)
        window = learner_b.get_transit_window("gate", "corridor", hour=10)

        assert window.is_learned is True
        assert window.sample_count == 5
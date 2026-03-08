"""Tests for app/monitoring/__init__.py — module-level singletons.

Follows the same conventions as the other test modules: plain pytest
functions, from __future__ import annotations, direct assert statements,
no test classes, no external fixtures.
"""

from __future__ import annotations

from app.monitoring import default_analyzer, default_store
from app.monitoring.analyzer import SessionAnalyzer
from app.monitoring.session_store import SessionStore


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


def test_default_store_is_session_store():
    """default_store must be a SessionStore instance."""
    assert isinstance(default_store, SessionStore)


def test_default_analyzer_is_session_analyzer():
    """default_analyzer must be a SessionAnalyzer instance."""
    assert isinstance(default_analyzer, SessionAnalyzer)


# ---------------------------------------------------------------------------
# Singleton identity
# ---------------------------------------------------------------------------


def test_singleton_store():
    """Two imports of default_store must return the same object."""
    from app.monitoring import default_store as store_a
    from app.monitoring import default_store as store_b

    assert store_a is store_b


def test_singleton_analyzer():
    """Two imports of default_analyzer must return the same object."""
    from app.monitoring import default_analyzer as analyzer_a
    from app.monitoring import default_analyzer as analyzer_b

    assert analyzer_a is analyzer_b


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------


def test_store_has_expected_defaults():
    """default_store must be configured with window_size=50, ttl_seconds=3600."""
    assert default_store._window_size == 50
    assert default_store._ttl_seconds == 3600


def test_analyzer_has_expected_defaults():
    """default_analyzer must carry all four default thresholds."""
    assert default_analyzer._convergence_threshold == 0.75
    assert default_analyzer._min_window == 5
    assert default_analyzer._perturbation_sim_threshold == 0.95
    assert default_analyzer._perturbation_time_window == 300

"""Tests for app/monitoring/schemas.py.

Follows the same conventions as tests/test_pipeline.py: plain pytest
functions, direct assert statements, no test classes, no external fixtures.
"""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.monitoring.schemas import (
    EMBEDDING_DIM,
    AnomalyAlert,
    ConvergenceResult,
    PerturbationResult,
    SessionEntry,
    SessionState,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_HASH = "a" * 64  # 64 lowercase hex chars → valid SHA-256
VALID_EMBEDDING = [0.1] * EMBEDDING_DIM
NOW = datetime.now(timezone.utc)


def _make_entry(**overrides) -> SessionEntry:
    """Return a valid SessionEntry, applying any field overrides."""
    kwargs = dict(
        sequence_hash=VALID_HASH,
        embedding=VALID_EMBEDDING,
        timestamp=NOW,
        risk_score=0.5,
        sequence_length=100,
    )
    kwargs.update(overrides)
    return SessionEntry(**kwargs)


def _make_convergence(**overrides) -> ConvergenceResult:
    kwargs = dict(
        mean_similarity=0.7,
        similarity_trend=0.05,
        window_size=5,
        is_flagged=False,
    )
    kwargs.update(overrides)
    return ConvergenceResult(**kwargs)


def _make_perturbation(**overrides) -> PerturbationResult:
    kwargs = dict(
        cluster_count=0,
        max_cluster_size=0,
        high_sim_pairs=[],
        is_flagged=False,
    )
    kwargs.update(overrides)
    return PerturbationResult(**kwargs)


def _make_alert(**overrides) -> AnomalyAlert:
    kwargs = dict(
        anomaly_score=0.3,
        convergence=_make_convergence(),
        perturbation=_make_perturbation(),
        flagged_indices=[],
        explanation="No significant anomaly detected.",
    )
    kwargs.update(overrides)
    return AnomalyAlert(**kwargs)


# ---------------------------------------------------------------------------
# SessionEntry — valid construction
# ---------------------------------------------------------------------------


def test_session_entry_valid_construction():
    entry = _make_entry()
    assert entry.sequence_hash == VALID_HASH
    assert len(entry.embedding) == EMBEDDING_DIM
    assert entry.risk_score == 0.5
    assert entry.sequence_length == 100


def test_session_entry_risk_score_boundary_zero():
    entry = _make_entry(risk_score=0.0)
    assert entry.risk_score == 0.0


def test_session_entry_risk_score_boundary_one():
    entry = _make_entry(risk_score=1.0)
    assert entry.risk_score == 1.0


# ---------------------------------------------------------------------------
# SessionEntry — validator rejection
# ---------------------------------------------------------------------------


def test_session_entry_rejects_wrong_embedding_length_short():
    with pytest.raises(ValidationError):
        _make_entry(embedding=[0.1] * (EMBEDDING_DIM - 1))


def test_session_entry_rejects_wrong_embedding_length_long():
    with pytest.raises(ValidationError):
        _make_entry(embedding=[0.1] * (EMBEDDING_DIM + 1))


def test_session_entry_rejects_embedding_length_zero():
    with pytest.raises(ValidationError):
        _make_entry(embedding=[])


def test_session_entry_rejects_risk_score_above_one():
    with pytest.raises(ValidationError):
        _make_entry(risk_score=1.01)


def test_session_entry_rejects_risk_score_below_zero():
    with pytest.raises(ValidationError):
        _make_entry(risk_score=-0.01)


def test_session_entry_rejects_hash_too_short():
    with pytest.raises(ValidationError):
        _make_entry(sequence_hash="abc123")


def test_session_entry_rejects_hash_too_long():
    with pytest.raises(ValidationError):
        _make_entry(sequence_hash="a" * 65)


def test_session_entry_rejects_hash_with_uppercase():
    # SHA-256 must be lowercase hex; uppercase letters are not valid
    with pytest.raises(ValidationError):
        _make_entry(sequence_hash="A" * 64)


def test_session_entry_rejects_hash_with_non_hex_chars():
    with pytest.raises(ValidationError):
        _make_entry(sequence_hash="g" * 64)  # 'g' is not a hex char


# ---------------------------------------------------------------------------
# SessionState — valid construction
# ---------------------------------------------------------------------------


def test_session_state_valid_with_entries():
    entry = _make_entry()
    state = SessionState(
        session_id="sess-001",
        entries=[entry],
        created_at=NOW,
        last_active=NOW,
    )
    assert state.session_id == "sess-001"
    assert len(state.entries) == 1
    assert state.anomaly_score == 0.0


def test_session_state_accepts_empty_entries():
    state = SessionState(
        session_id="sess-empty",
        entries=[],
        created_at=NOW,
        last_active=NOW,
    )
    assert state.entries == []


def test_session_state_anomaly_score_default():
    state = SessionState(
        session_id="sess-002",
        created_at=NOW,
        last_active=NOW,
    )
    assert state.anomaly_score == 0.0


def test_session_state_anomaly_score_explicit():
    state = SessionState(
        session_id="sess-003",
        created_at=NOW,
        last_active=NOW,
        anomaly_score=0.75,
    )
    assert state.anomaly_score == 0.75


# ---------------------------------------------------------------------------
# ConvergenceResult — valid construction
# ---------------------------------------------------------------------------


def test_convergence_result_valid():
    result = _make_convergence(mean_similarity=0.85, similarity_trend=0.1, window_size=8, is_flagged=True)
    assert result.mean_similarity == 0.85
    assert result.similarity_trend == 0.1
    assert result.window_size == 8
    assert result.is_flagged is True


def test_convergence_result_negative_trend():
    result = _make_convergence(similarity_trend=-0.05)
    assert result.similarity_trend < 0


# ---------------------------------------------------------------------------
# PerturbationResult — valid construction
# ---------------------------------------------------------------------------


def test_perturbation_result_valid_no_clusters():
    result = _make_perturbation()
    assert result.cluster_count == 0
    assert result.high_sim_pairs == []
    assert result.is_flagged is False


def test_perturbation_result_with_pairs():
    result = _make_perturbation(
        cluster_count=1,
        max_cluster_size=2,
        high_sim_pairs=[(0, 1), (1, 2)],
        is_flagged=True,
    )
    assert result.cluster_count == 1
    assert result.high_sim_pairs == [(0, 1), (1, 2)]
    assert result.is_flagged is True


# ---------------------------------------------------------------------------
# AnomalyAlert — valid construction
# ---------------------------------------------------------------------------


def test_anomaly_alert_valid():
    alert = _make_alert()
    assert alert.anomaly_score == 0.3
    assert isinstance(alert.convergence, ConvergenceResult)
    assert isinstance(alert.perturbation, PerturbationResult)
    assert alert.flagged_indices == []
    assert len(alert.explanation) > 0


def test_anomaly_alert_with_flagged_indices():
    alert = _make_alert(
        anomaly_score=0.9,
        flagged_indices=[0, 2, 4],
        explanation="Convergent optimization detected across 3 queries.",
    )
    assert alert.flagged_indices == [0, 2, 4]
    assert "Convergent" in alert.explanation


def test_anomaly_alert_rejects_score_above_one():
    with pytest.raises(ValidationError):
        _make_alert(anomaly_score=1.1)


def test_anomaly_alert_rejects_score_below_zero():
    with pytest.raises(ValidationError):
        _make_alert(anomaly_score=-0.1)


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_session_entry_round_trip():
    original = _make_entry()
    json_str = original.model_dump_json()
    restored = SessionEntry.model_validate_json(json_str)
    assert restored.sequence_hash == original.sequence_hash
    assert restored.risk_score == original.risk_score
    assert restored.sequence_length == original.sequence_length
    assert len(restored.embedding) == EMBEDDING_DIM


def test_session_state_round_trip():
    state = SessionState(
        session_id="rt-test",
        entries=[_make_entry()],
        created_at=NOW,
        last_active=NOW,
        anomaly_score=0.2,
    )
    json_str = state.model_dump_json()
    restored = SessionState.model_validate_json(json_str)
    assert restored.session_id == state.session_id
    assert len(restored.entries) == 1
    assert restored.anomaly_score == 0.2


def test_convergence_result_round_trip():
    original = _make_convergence(mean_similarity=0.6, similarity_trend=-0.02, window_size=3, is_flagged=False)
    json_str = original.model_dump_json()
    restored = ConvergenceResult.model_validate_json(json_str)
    assert restored.mean_similarity == original.mean_similarity
    assert restored.similarity_trend == original.similarity_trend
    assert restored.window_size == original.window_size


def test_perturbation_result_round_trip():
    original = _make_perturbation(
        cluster_count=2,
        max_cluster_size=3,
        high_sim_pairs=[(0, 1), (2, 3)],
        is_flagged=True,
    )
    json_str = original.model_dump_json()
    restored = PerturbationResult.model_validate_json(json_str)
    assert restored.cluster_count == 2
    assert restored.is_flagged is True
    # JSON round-trip: tuples serialise to arrays and are coerced back to tuples
    assert list(restored.high_sim_pairs[0]) == [0, 1]


def test_anomaly_alert_round_trip():
    original = _make_alert(
        anomaly_score=0.65,
        flagged_indices=[1, 3],
        explanation="High convergence signal.",
    )
    json_str = original.model_dump_json()
    restored = AnomalyAlert.model_validate_json(json_str)
    assert restored.anomaly_score == original.anomaly_score
    assert restored.flagged_indices == [1, 3]
    assert restored.explanation == original.explanation


def test_anomaly_alert_model_dump_produces_valid_json():
    alert = _make_alert()
    data = alert.model_dump()
    # model_dump() returns a dict; json.dumps must not raise
    json_str = json.dumps(data, default=str)
    assert isinstance(json_str, str)

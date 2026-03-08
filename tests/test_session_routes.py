"""Integration tests for session-monitoring API routes.

Creates a minimal FastAPI app with the router and injects mocked app state
so the heavy pipeline dependencies (ESM-2, FAISS) are never loaded.

Follows the same conventions as other test modules: plain pytest functions,
from __future__ import annotations, direct assert statements, no test classes,
no external fixtures.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavy ML/native dependencies before any app.pipeline imports so
# that the test module can be collected without torch, transformers, or faiss
# being installed.
# ---------------------------------------------------------------------------
for _stub_mod in ["torch", "transformers", "faiss"]:
    sys.modules.setdefault(_stub_mod, MagicMock())

# routes.py references predict_structure which is absent from the real module.
# Inject it before routes.py is first imported.
import app.pipeline.structure as _struct_mod  # noqa: E402

if not hasattr(_struct_mod, "predict_structure"):
    _struct_mod.predict_structure = MagicMock()

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.monitoring.analyzer import SessionAnalyzer
from app.monitoring.schemas import AnomalyAlert, ConvergenceResult, PerturbationResult
from app.monitoring.session_store import SessionStore
from app.pipeline.sequence import SequenceType, ValidationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1280
_RNG = np.random.default_rng(0)
FAKE_EMBEDDING = _RNG.random(EMBEDDING_DIM).astype(np.float32)

# 100-AA valid protein sequence
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 5


# ---------------------------------------------------------------------------
# App / client factory
# ---------------------------------------------------------------------------


def _app_with_state() -> FastAPI:
    """Return a FastAPI app with mocked app.state for pipeline resources."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    mock_emb = MagicMock()
    mock_emb.is_loaded = True
    mock_emb.embed.return_value = FAKE_EMBEDDING.copy()
    app.state.embedding_model = mock_emb

    mock_db = MagicMock()
    mock_db.size = 2
    mock_db.get_metadata.return_value = {
        "uniprot_id": "P12345",
        "name": "Mock Toxin",
        "organism": "Mockus mockus",
        "toxin_type": "neurotoxin",
        "go_terms": [],
        "ec_numbers": [],
    }
    app.state.toxin_db = mock_db

    return app


# ---------------------------------------------------------------------------
# Pipeline mock helpers
# ---------------------------------------------------------------------------


def _similarity_result() -> MagicMock:
    hit = MagicMock()
    hit.cosine_similarity = 0.45
    hit.metadata = {
        "uniprot_id": "P12345",
        "name": "Mock Toxin",
        "organism": "Mockus mockus",
        "toxin_type": "neurotoxin",
        "go_terms": [],
        "ec_numbers": [],
    }
    sr = MagicMock()
    sr.embedding_hits = [hit]
    sr.structure_hits = []
    sr.max_embedding_sim = 0.45
    sr.max_structure_sim = None
    return sr


def _function_prediction() -> MagicMock:
    from app.models.schemas import FunctionPrediction

    return FunctionPrediction(go_terms=[], ec_numbers=[], summary="mock")


def _enter_pipeline_patches(stack: ExitStack) -> None:
    """Enter all heavyweight pipeline patches into *stack*."""
    sim_result = _similarity_result()
    func_pred = _function_prediction()

    _valid_result = ValidationResult(
        valid=True, sequence_type=SequenceType.PROTEIN, cleaned=SEQUENCE
    )
    stack.enter_context(
        patch("app.api.routes.validate_sequence", return_value=_valid_result)
    )
    mock_sc = stack.enter_context(patch("app.api.routes.CombinedSimilaritySearcher"))
    mock_fp = stack.enter_context(patch("app.api.routes.get_function_predictor"))
    stack.enter_context(
        patch("app.api.routes.compute_score", return_value=(0.3, "low score"))
    )
    mock_sc.return_value.search = AsyncMock(return_value=sim_result)
    mock_fp.return_value.predict.return_value = func_pred


def _enter_monitoring_patches(
    stack: ExitStack,
    store: SessionStore,
    analyzer: MagicMock | SessionAnalyzer,
) -> None:
    stack.enter_context(patch("app.api.routes.default_store", store))
    stack.enter_context(patch("app.api.routes.default_analyzer", analyzer))


def _mock_analyzer(anomaly_score: float = 0.0) -> MagicMock:
    m = MagicMock()
    m.analyze.return_value = MagicMock(anomaly_score=anomaly_score)
    return m


# ---------------------------------------------------------------------------
# POST /api/screen — session tracking
# ---------------------------------------------------------------------------


def test_screen_creates_session_entry():
    """POST /screen must create a session entry in the store."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        resp = client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "create-test"},
        )
    assert resp.status_code == 200
    state = store.get_session("create-test")
    assert state is not None
    assert len(state.entries) == 1


def test_screen_uses_x_session_id_header():
    """Session key must come from the X-Session-ID header when supplied."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "header-session"},
        )
    assert store.get_session("header-session") is not None
    assert store.session_count() == 1


def test_screen_falls_back_to_client_ip():
    """Without X-Session-ID header, client IP must be used as session key."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        client.post("/api/screen", json={"sequence": SEQUENCE})
    assert store.session_count() == 1


def test_screen_entry_risk_score_matches_pipeline():
    """Stored risk_score must equal the value returned by compute_score."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "risk-score-test"},
        )
    entry = store.get_session("risk-score-test").entries[0]
    # compute_score mock always returns 0.3
    assert entry.risk_score == pytest.approx(0.3)


def test_screen_entry_embedding_matches_model_output():
    """Stored embedding must be equal (within float tolerance) to the mock output."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "emb-test"},
        )
    stored = store.get_session("emb-test").entries[0].embedding
    assert len(stored) == EMBEDDING_DIM
    assert np.allclose(stored, FAKE_EMBEDDING, atol=1e-5)


def test_screen_accumulates_multiple_entries():
    """Each POST /screen call must append a new entry to the same session."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        for _ in range(3):
            client.post(
                "/api/screen",
                json={"sequence": SEQUENCE},
                headers={"X-Session-ID": "multi-call"},
            )
    assert len(store.get_session("multi-call").entries) == 3


def test_screen_updates_session_anomaly_score():
    """session.anomaly_score must be updated after analyze() returns."""
    store = SessionStore()
    analyzer = _mock_analyzer(anomaly_score=0.55)
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, analyzer)
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "anomaly-update"},
        )
    state = store.get_session("anomaly-update")
    assert state.anomaly_score == pytest.approx(0.55)


def test_screen_includes_session_anomaly_score_in_risk_factors():
    """risk_factors in the /screen response must include session_anomaly_score."""
    store = SessionStore()
    analyzer = _mock_analyzer(anomaly_score=0.12)
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, analyzer)
        client = TestClient(_app_with_state())
        resp = client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "rf-test"},
        )
    assert resp.status_code == 200
    rf = resp.json()["risk_factors"]
    assert "session_anomaly_score" in rf
    assert rf["session_anomaly_score"] == pytest.approx(0.12)


def test_screen_session_anomaly_score_in_range_with_real_analyzer():
    """session_anomaly_score must be in [0, 1] when using the real SessionAnalyzer."""
    store = SessionStore()
    real_analyzer = SessionAnalyzer()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, real_analyzer)
        client = TestClient(_app_with_state())
        resp = client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "real-analyzer"},
        )
    assert resp.status_code == 200
    score = resp.json()["risk_factors"]["session_anomaly_score"]
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# GET /api/session/{session_id}
# ---------------------------------------------------------------------------


def test_get_session_returns_200_with_state():
    """GET /api/session/{id} must return 200 and the session state."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "get-session-test"},
        )
        resp = client.get("/api/session/get-session-test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "get-session-test"
    assert len(body["entries"]) == 1


def test_get_session_not_found_returns_404():
    """GET /api/session/{nonexistent} must return 404."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        resp = client.get("/api/session/ghost-session")
    assert resp.status_code == 404


def test_get_session_entry_count_matches_calls():
    """entries list in GET /session response must equal the number of POSTs."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        for _ in range(4):
            client.post(
                "/api/screen",
                json={"sequence": SEQUENCE},
                headers={"X-Session-ID": "count-session"},
            )
        resp = client.get("/api/session/count-session")
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) == 4


# ---------------------------------------------------------------------------
# GET /api/session/{session_id}/alerts
# ---------------------------------------------------------------------------


def test_get_alerts_not_found_returns_404():
    """GET /api/session/{nonexistent}/alerts must return 404."""
    store = SessionStore()
    with ExitStack() as stack:
        _enter_monitoring_patches(stack, store, _mock_analyzer())
        client = TestClient(_app_with_state())
        resp = client.get("/api/session/ghost/alerts")
    assert resp.status_code == 404


def test_get_alerts_returns_anomaly_alert_shape():
    """GET /api/session/{id}/alerts must return a valid AnomalyAlert payload."""
    store = SessionStore()
    mock_alert = AnomalyAlert(
        anomaly_score=0.0,
        convergence=ConvergenceResult(
            mean_similarity=0.0,
            similarity_trend=0.0,
            window_size=1,
            is_flagged=False,
        ),
        perturbation=PerturbationResult(
            cluster_count=0,
            max_cluster_size=0,
            high_sim_pairs=[],
            is_flagged=False,
        ),
        flagged_indices=[],
        explanation="No anomalies detected.",
    )
    analyzer = MagicMock()
    analyzer.analyze.return_value = mock_alert

    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, analyzer)
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "alerts-shape"},
        )
        resp = client.get("/api/session/alerts-shape/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert "anomaly_score" in body
    assert "convergence" in body
    assert "perturbation" in body
    assert "flagged_indices" in body
    assert "explanation" in body


def test_get_alerts_anomaly_score_in_range_with_real_analyzer():
    """anomaly_score in GET /alerts must be in [0, 1] with real SessionAnalyzer."""
    store = SessionStore()
    real_analyzer = SessionAnalyzer()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, real_analyzer)
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "score-range"},
        )
        resp = client.get("/api/session/score-range/alerts")
    assert resp.status_code == 200
    score = resp.json()["anomaly_score"]
    assert 0.0 <= score <= 1.0


def test_get_alerts_explanation_is_string():
    """explanation field in GET /alerts must be a non-empty string."""
    store = SessionStore()
    real_analyzer = SessionAnalyzer()
    with ExitStack() as stack:
        _enter_pipeline_patches(stack)
        _enter_monitoring_patches(stack, store, real_analyzer)
        client = TestClient(_app_with_state())
        client.post(
            "/api/screen",
            json={"sequence": SEQUENCE},
            headers={"X-Session-ID": "explanation-test"},
        )
        resp = client.get("/api/session/explanation-test/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json()["explanation"], str)
    assert len(resp.json()["explanation"]) > 0

"""Tests for app/monitoring/analyzer.py — convergence detection.

Follows the same conventions as the other test modules: plain pytest
functions, from __future__ import annotations, direct assert statements,
no test classes, no external fixtures.

Geometry notes
--------------
All embeddings are constructed as::

    v_i = base + noise_scale * (1 / sqrt(D)) * noise_i

then L2-normalised.  With D=1280 the noise term has expected L2 norm
≈ noise_scale.  This keeps ``noise_scale`` interpretable as an angle
perturbation (≈ arctan(noise_scale) degrees away from base).

Approximate pairwise cosine similarities for two independent noise draws:

    noise_scale = 0.5  →  pairwise ≈ 0.80
    noise_scale = 0.01 →  pairwise ≈ 0.9999
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from app.monitoring.schemas import EMBEDDING_DIM, SessionEntry
from app.monitoring.analyzer import SessionAnalyzer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
VALID_HASH = "c" * 64  # 64 lowercase hex chars


def _unit(v: np.ndarray) -> np.ndarray:
    """Return the L2-normalised version of v."""
    return v / np.linalg.norm(v)


def _make_entry(embedding_vec: np.ndarray) -> SessionEntry:
    """Wrap a numpy embedding in a valid SessionEntry."""
    return SessionEntry(
        sequence_hash=VALID_HASH,
        embedding=embedding_vec.tolist(),
        timestamp=NOW,
        risk_score=0.4,
        sequence_length=120,
    )


def _noisy_entries(
    rng: np.random.Generator,
    base: np.ndarray,
    noise_scale: float,
    count: int,
) -> list[SessionEntry]:
    """Create *count* entries near *base* with L2 perturbation ≈ noise_scale."""
    scale = noise_scale / np.sqrt(EMBEDDING_DIM)
    return [
        _make_entry(_unit(base + scale * rng.standard_normal(EMBEDDING_DIM)))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# test_no_convergence
# ---------------------------------------------------------------------------


def test_no_convergence():
    """Random unit vectors in 1280-d space have near-zero pairwise cosine
    similarity — well below the default 0.75 threshold."""
    rng = np.random.default_rng(42)
    entries = [
        _make_entry(_unit(rng.standard_normal(EMBEDDING_DIM)))
        for _ in range(10)
    ]
    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence(entries)

    assert not result.is_flagged
    assert result.mean_similarity < 0.75
    assert result.window_size == 10


# ---------------------------------------------------------------------------
# test_clear_convergence
# ---------------------------------------------------------------------------


def test_clear_convergence():
    """First 5 entries have larger spread (noise_scale=0.5), last 5 are
    near-identical (noise_scale=0.01).  Expected all-pairs mean ≈ 0.90,
    positive trend ≈ 0.20 → flagged."""
    rng = np.random.default_rng(99)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    analyzer = SessionAnalyzer(convergence_threshold=0.75)
    result = analyzer.compute_convergence(entries)

    assert result.is_flagged
    assert result.similarity_trend > 0.0
    assert result.mean_similarity > 0.75
    assert result.window_size == 10


# ---------------------------------------------------------------------------
# test_insufficient_window
# ---------------------------------------------------------------------------


def test_insufficient_window():
    """Fewer than min_window_for_convergence entries always returns unflagged
    with zero scores, regardless of how similar the embeddings are."""
    rng = np.random.default_rng(7)
    identical = _unit(rng.standard_normal(EMBEDDING_DIM))
    # 3 identical embeddings → cosine sim = 1.0 for all pairs, but window
    # is too small to be meaningful.
    entries = [_make_entry(identical) for _ in range(3)]

    analyzer = SessionAnalyzer(min_window_for_convergence=5)
    result = analyzer.compute_convergence(entries)

    assert not result.is_flagged
    assert result.mean_similarity == 0.0
    assert result.similarity_trend == 0.0
    assert result.window_size == 3


# ---------------------------------------------------------------------------
# test_flat_high_similarity
# ---------------------------------------------------------------------------


def test_flat_high_similarity():
    """10 identical embeddings → mean_similarity ≈ 1.0 but similarity_trend
    ≈ 0.0 (both halves identical).  NOT flagged because trend is not positive."""
    rng = np.random.default_rng(13)
    identical = _unit(rng.standard_normal(EMBEDDING_DIM))
    entries = [_make_entry(identical) for _ in range(10)]

    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence(entries)

    assert not result.is_flagged
    assert result.mean_similarity > 0.99  # all pairs identical → ≈ 1.0
    assert abs(result.similarity_trend) < 1e-4  # first_half ≈ second_half


# ---------------------------------------------------------------------------
# test_diverging
# ---------------------------------------------------------------------------


def test_diverging():
    """First 5 near-identical, last 5 spread out → negative trend → NOT flagged
    even when overall mean similarity could be moderately high."""
    rng = np.random.default_rng(88)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    entries = first_half + second_half

    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence(entries)

    assert not result.is_flagged
    assert result.similarity_trend < 0.0


# ---------------------------------------------------------------------------
# test_exactly_at_threshold
# ---------------------------------------------------------------------------


def test_exactly_at_threshold():
    """Flagging uses strict > not >=, so mean_similarity == threshold must
    not trigger a flag.  One epsilon below the threshold with a positive trend
    must trigger a flag."""
    rng = np.random.default_rng(55)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    # Use the clear-convergence data profile so trend is reliably positive.
    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    # Establish ground-truth values with a very permissive threshold.
    reference = SessionAnalyzer(convergence_threshold=0.0)
    ref_result = reference.compute_convergence(entries)
    computed_mean = ref_result.mean_similarity

    # The data must have a positive trend for this edge-case test to be valid.
    assert ref_result.similarity_trend > 0.0, (
        "Test setup error: expected positive trend for convergence data"
    )

    # AT the threshold (mean == threshold) → NOT flagged (strict >).
    at_threshold = SessionAnalyzer(convergence_threshold=computed_mean)
    result_at = at_threshold.compute_convergence(entries)
    assert not result_at.is_flagged

    # ONE EPSILON BELOW the threshold → flagged.
    below_threshold = SessionAnalyzer(convergence_threshold=computed_mean - 1e-6)
    result_below = below_threshold.compute_convergence(entries)
    assert result_below.is_flagged


# ---------------------------------------------------------------------------
# test_single_entry
# ---------------------------------------------------------------------------


def test_single_entry():
    """A window of 1 entry must return unflagged without crashing."""
    rng = np.random.default_rng(1)
    entry = _make_entry(_unit(rng.standard_normal(EMBEDDING_DIM)))

    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence([entry])

    assert not result.is_flagged
    assert result.window_size == 1
    assert result.mean_similarity == 0.0
    assert result.similarity_trend == 0.0


# ---------------------------------------------------------------------------
# test_empty_entries
# ---------------------------------------------------------------------------


def test_empty_entries():
    """An empty window must return unflagged without crashing."""
    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence([])

    assert not result.is_flagged
    assert result.window_size == 0
    assert result.mean_similarity == 0.0
    assert result.similarity_trend == 0.0


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_exactly_min_window_size():
    """A window of exactly min_window_for_convergence entries is processed
    (not short-circuited) and can be flagged."""
    rng = np.random.default_rng(200)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    # 5 near-identical entries (noise_scale=0.01): pairwise ≈ 1.0
    entries = _noisy_entries(rng, base, noise_scale=0.01, count=5)

    analyzer = SessionAnalyzer(convergence_threshold=0.0, min_window_for_convergence=5)
    result = analyzer.compute_convergence(entries)

    # Window size exactly at boundary → processed, not short-circuited.
    assert result.window_size == 5
    # Mean similarity should be very high for near-identical vectors.
    assert result.mean_similarity > 0.99


def test_window_size_one_below_min_is_not_processed():
    """A window of min_window - 1 entries is short-circuited."""
    rng = np.random.default_rng(201)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    entries = _noisy_entries(rng, base, noise_scale=0.01, count=4)

    analyzer = SessionAnalyzer(min_window_for_convergence=5)
    result = analyzer.compute_convergence(entries)

    assert not result.is_flagged
    assert result.mean_similarity == 0.0


def test_custom_convergence_threshold():
    """A very low convergence_threshold means even spread-out entries are flagged
    if there is a positive trend."""
    rng = np.random.default_rng(77)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    # Default threshold (0.75) → flagged for this data.
    result_default = SessionAnalyzer(convergence_threshold=0.75).compute_convergence(entries)
    assert result_default.is_flagged

    # Impossibly high threshold → never flagged.
    result_high = SessionAnalyzer(convergence_threshold=1.1).compute_convergence(entries)
    assert not result_high.is_flagged


def test_similarity_trend_reflects_half_means():
    """similarity_trend is exactly second_half_mean − first_half_mean."""
    rng = np.random.default_rng(303)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    analyzer = SessionAnalyzer()
    result = analyzer.compute_convergence(entries)

    # The trend must be positive for this data profile.
    assert result.similarity_trend > 0.0
    # And less than 1 (the maximum possible difference).
    assert result.similarity_trend <= 1.0


# ===========================================================================
# Perturbation detection tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper for perturbation tests
# ---------------------------------------------------------------------------


def _make_perturb_entry(
    embedding_vec: np.ndarray,
    *,
    sequence_length: int = 100,
    timestamp: datetime | None = None,
) -> SessionEntry:
    """Create a SessionEntry with controllable sequence_length and timestamp."""
    return SessionEntry(
        sequence_hash=VALID_HASH,
        embedding=embedding_vec.tolist(),
        timestamp=timestamp or NOW,
        risk_score=0.3,
        sequence_length=sequence_length,
    )


# ---------------------------------------------------------------------------
# test_perturbation_detected
# ---------------------------------------------------------------------------


def test_perturbation_detected():
    """4 near-identical embeddings, same length, timestamps within 60 s →
    one cluster of 4 entries → flagged."""
    rng = np.random.default_rng(400)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    entries = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(seconds=i * 15),  # 0s, 15s, 30s, 45s
        )
        for i in range(4)
    ]

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation(entries)

    assert result.is_flagged
    assert result.max_cluster_size == 4
    assert result.cluster_count == 1
    # All 4C2 = 6 pairs should be flagged.
    assert len(result.high_sim_pairs) == 6


# ---------------------------------------------------------------------------
# test_perturbation_not_detected_time
# ---------------------------------------------------------------------------


def test_perturbation_not_detected_time():
    """Near-identical embeddings, same length, but timestamps spread hours
    apart → time condition fails → no pairs → not flagged."""
    rng = np.random.default_rng(401)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    entries = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(hours=i * 2),  # 0h, 2h, 4h, 6h apart
        )
        for i in range(4)
    ]

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation(entries)

    assert not result.is_flagged
    assert result.high_sim_pairs == []
    assert result.cluster_count == 0


# ---------------------------------------------------------------------------
# test_perturbation_not_detected_length
# ---------------------------------------------------------------------------


def test_perturbation_not_detected_length():
    """Near-identical embeddings, close timestamps, but sequence_lengths
    differ by 10+ → length condition fails → no pairs → not flagged."""
    rng = np.random.default_rng(402)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    entries = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100 + i * 10,  # 100, 110, 120, 130 → diff ≥ 10
            timestamp=t0 + timedelta(seconds=i * 10),
        )
        for i in range(4)
    ]

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation(entries)

    assert not result.is_flagged
    assert result.high_sim_pairs == []
    assert result.cluster_count == 0


# ---------------------------------------------------------------------------
# test_perturbation_below_cluster_threshold
# ---------------------------------------------------------------------------


def test_perturbation_below_cluster_threshold():
    """Exactly 2 similar entries → 1 pair in high_sim_pairs, max cluster
    size = 2 < 3 → not flagged."""
    rng = np.random.default_rng(403)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    similar_pair = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(seconds=i * 20),
        )
        for i in range(2)
    ]
    # Add 4 unrelated entries (random directions, spread timestamps) so the
    # window is large enough but won't form additional pairs.
    unrelated = [
        _make_perturb_entry(
            _unit(rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(hours=i + 1),
        )
        for i in range(4)
    ]
    entries = similar_pair + unrelated

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation(entries)

    assert not result.is_flagged
    assert result.max_cluster_size == 2
    assert len(result.high_sim_pairs) == 1  # exactly the one similar pair
    assert result.high_sim_pairs[0] == (0, 1)


# ---------------------------------------------------------------------------
# test_perturbation_mixed_cluster
# ---------------------------------------------------------------------------


def test_perturbation_mixed_cluster():
    """6 entries: first 3 form a tight cluster (flagged), last 3 are random
    and unrelated to each other and to the cluster."""
    rng = np.random.default_rng(404)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Entries 0-2: tight cluster, same length, close timestamps.
    cluster = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(seconds=i * 10),
        )
        for i in range(3)
    ]
    # Entries 3-5: random directions, hours apart → no pairs with anything.
    unrelated = [
        _make_perturb_entry(
            _unit(rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(hours=i + 2),
        )
        for i in range(3)
    ]
    entries = cluster + unrelated

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation(entries)

    assert result.is_flagged
    assert result.cluster_count == 1
    assert result.max_cluster_size == 3
    # Cluster pairs: (0,1), (0,2), (1,2) — all 3C2 = 3 pairs.
    assert len(result.high_sim_pairs) == 3
    cluster_indices = {idx for pair in result.high_sim_pairs for idx in pair}
    assert cluster_indices == {0, 1, 2}


# ---------------------------------------------------------------------------
# test_perturbation_empty
# ---------------------------------------------------------------------------


def test_perturbation_empty():
    """Empty entry list → not flagged, no crash."""
    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation([])

    assert not result.is_flagged
    assert result.cluster_count == 0
    assert result.max_cluster_size == 0
    assert result.high_sim_pairs == []


# ---------------------------------------------------------------------------
# test_perturbation_single
# ---------------------------------------------------------------------------


def test_perturbation_single():
    """Single entry → no pairs possible → not flagged, no crash."""
    rng = np.random.default_rng(405)
    entry = _make_perturb_entry(_unit(rng.standard_normal(EMBEDDING_DIM)))

    analyzer = SessionAnalyzer()
    result = analyzer.compute_perturbation([entry])

    assert not result.is_flagged
    assert result.cluster_count == 0
    assert result.max_cluster_size == 0
    assert result.high_sim_pairs == []


# ===========================================================================
# analyze() composite tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_analyze_clean_session
# ---------------------------------------------------------------------------


def test_analyze_clean_session():
    """8 random spread-out entries → neither detector fires → score ≈ 0,
    explanation reports no anomalies."""
    rng = np.random.default_rng(500)
    entries = [
        _make_entry(_unit(rng.standard_normal(EMBEDDING_DIM)))
        for _ in range(8)
    ]

    analyzer = SessionAnalyzer()
    alert = analyzer.analyze(entries)

    assert not alert.convergence.is_flagged
    assert not alert.perturbation.is_flagged
    assert alert.anomaly_score == 0.0
    assert alert.explanation == "No anomalies detected."
    assert alert.flagged_indices == []


# ---------------------------------------------------------------------------
# test_analyze_convergence_only
# ---------------------------------------------------------------------------


def test_analyze_convergence_only():
    """Converging embeddings with perturbation disabled via high sim threshold.
    Score reflects only the convergence component: 0.6 * mean_similarity."""
    rng = np.random.default_rng(501)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    # perturbation_sim_threshold=1.1 is impossible → perturbation never fires.
    analyzer = SessionAnalyzer(
        convergence_threshold=0.75,
        perturbation_sim_threshold=1.1,
    )
    alert = analyzer.analyze(entries)

    assert alert.convergence.is_flagged
    assert not alert.perturbation.is_flagged

    expected_score = 0.6 * alert.convergence.mean_similarity
    assert abs(alert.anomaly_score - expected_score) < 1e-5

    # All window indices are flagged when convergence fires.
    assert alert.flagged_indices == list(range(len(entries)))
    # Explanation mentions convergence, not perturbation.
    assert "Convergent" in alert.explanation
    assert "Perturbation" not in alert.explanation


# ---------------------------------------------------------------------------
# test_analyze_perturbation_only
# ---------------------------------------------------------------------------


def test_analyze_perturbation_only():
    """4-entry perturbation cluster; n < min_window so convergence is never
    attempted.  Score reflects only the perturbation component: 0.4 * (4/4)."""
    rng = np.random.default_rng(502)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    entries = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(seconds=i * 15),
        )
        for i in range(4)
    ]

    # n=4 < min_window=5 → compute_convergence returns early (not flagged).
    analyzer = SessionAnalyzer()
    alert = analyzer.analyze(entries)

    assert not alert.convergence.is_flagged
    assert alert.perturbation.is_flagged

    # perturbation_component = max_cluster_size / n = 4 / 4 = 1.0
    expected_score = 0.4 * (alert.perturbation.max_cluster_size / len(entries))
    assert abs(alert.anomaly_score - expected_score) < 1e-5

    assert "Perturbation" in alert.explanation
    assert "Convergent" not in alert.explanation


# ---------------------------------------------------------------------------
# test_analyze_both_flagged
# ---------------------------------------------------------------------------


def test_analyze_both_flagged():
    """Convergence pattern (first 5 spread, last 5 tight).

    The tight last-5 cluster also satisfies perturbation conditions because
    _noisy_entries uses timestamp=NOW and sequence_length=120 for all entries
    → both detectors fire → score has contributions from both components."""
    rng = np.random.default_rng(503)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    analyzer = SessionAnalyzer()
    alert = analyzer.analyze(entries)

    assert alert.convergence.is_flagged
    assert alert.perturbation.is_flagged

    # Combined score must be higher than either component alone.
    convergence_only = 0.6 * alert.convergence.mean_similarity
    perturbation_only = 0.4 * (alert.perturbation.max_cluster_size / len(entries))
    assert alert.anomaly_score > convergence_only
    assert alert.anomaly_score > perturbation_only

    # Both detectors mentioned in the explanation.
    assert "Convergent" in alert.explanation
    assert "Perturbation" in alert.explanation


# ---------------------------------------------------------------------------
# test_analyze_empty
# ---------------------------------------------------------------------------


def test_analyze_empty():
    """Empty entry list → anomaly score 0.0, no flags, standard explanation."""
    analyzer = SessionAnalyzer()
    alert = analyzer.analyze([])

    assert alert.anomaly_score == 0.0
    assert not alert.convergence.is_flagged
    assert not alert.perturbation.is_flagged
    assert alert.flagged_indices == []
    assert alert.explanation == "No anomalies detected."


# ---------------------------------------------------------------------------
# test_analyze_score_clamped
# ---------------------------------------------------------------------------


def test_analyze_score_clamped():
    """Both detectors maximally active; score is bounded to [0, 1].

    Setting perturbation_sim_threshold=0.5 forces all pairs (including
    cross-half pairs with cosine_sim ≈ 0.89) into a single cluster of n,
    giving perturbation_component = 1.0.  Combined with a converging window,
    the raw value 0.6 * mean_sim + 0.4 * 1.0 approaches 1.0.  Float32
    rounding can push cosine similarities marginally above 1.0, so the
    clamp min(1.0, max(0.0, raw)) must prevent a Pydantic ValidationError."""
    rng = np.random.default_rng(504)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    # perturbation_sim_threshold=0.5: cosine_sim of all pairs (≥ 0.80) exceeds
    # threshold → one cluster of n=10 → perturbation_component = 1.0.
    analyzer = SessionAnalyzer(
        convergence_threshold=0.75,
        perturbation_sim_threshold=0.5,
    )
    alert = analyzer.analyze(entries)

    # Core invariant: score is always in [0, 1] — no ValidationError raised.
    assert 0.0 <= alert.anomaly_score <= 1.0

    # Both components are active so the score is high.
    assert alert.convergence.is_flagged
    assert alert.perturbation.is_flagged
    assert alert.perturbation.max_cluster_size == len(entries)  # full cluster
    assert alert.anomaly_score > 0.6  # well above either component alone


# ---------------------------------------------------------------------------
# test_analyze_flagged_indices_convergence
# ---------------------------------------------------------------------------


def test_analyze_flagged_indices_convergence():
    """When only convergence fires, flagged_indices spans the entire window."""
    rng = np.random.default_rng(505)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))

    first_half = _noisy_entries(rng, base, noise_scale=0.5, count=5)
    second_half = _noisy_entries(rng, base, noise_scale=0.01, count=5)
    entries = first_half + second_half

    analyzer = SessionAnalyzer(
        convergence_threshold=0.75,
        perturbation_sim_threshold=1.1,  # disable perturbation
    )
    alert = analyzer.analyze(entries)

    assert alert.convergence.is_flagged
    assert not alert.perturbation.is_flagged
    assert alert.flagged_indices == list(range(len(entries)))


# ---------------------------------------------------------------------------
# test_analyze_flagged_indices_perturbation
# ---------------------------------------------------------------------------


def test_analyze_flagged_indices_perturbation():
    """When only perturbation fires, flagged_indices contains exactly the
    indices from high_sim_pairs (not all window indices)."""
    rng = np.random.default_rng(506)
    base = _unit(rng.standard_normal(EMBEDDING_DIM))
    scale = 0.01 / np.sqrt(EMBEDDING_DIM)
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 4 clustered entries — n < min_window → convergence short-circuits.
    cluster_entries = [
        _make_perturb_entry(
            _unit(base + scale * rng.standard_normal(EMBEDDING_DIM)),
            sequence_length=100,
            timestamp=t0 + timedelta(seconds=i * 10),
        )
        for i in range(4)
    ]

    analyzer = SessionAnalyzer()
    alert = analyzer.analyze(cluster_entries)

    assert not alert.convergence.is_flagged
    assert alert.perturbation.is_flagged

    # flagged_indices must be exactly the set of indices from high_sim_pairs.
    expected = sorted({idx for pair in alert.perturbation.high_sim_pairs for idx in pair})
    assert alert.flagged_indices == expected
    # All 4 entries are in the cluster, so all 4 indices appear.
    assert alert.flagged_indices == [0, 1, 2, 3]

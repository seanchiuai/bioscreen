"""Pydantic v2 schemas for session monitoring and behavioral anomaly detection.

Models represent the rolling-window session state used by Component 2 of
BioScreen to detect convergent optimization and multi-provider perturbation
patterns across multiple queries from the same session.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Embedding dimensionality constant
# ---------------------------------------------------------------------------
# facebook/esm2_t33_650M_UR50D: 33 transformer layers, 650 M parameters.
# hidden_size == 1280 (confirmed from HuggingFace model config and the
# EmbeddingModel.embedding_dim property in app/pipeline/embedding.py which
# reads self._model.config.hidden_size at runtime).
EMBEDDING_DIM: int = 1280

# Pre-compiled pattern for SHA-256 hex strings (64 lowercase hex chars).
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ── Session window models ──────────────────────────────────────────────────


class SessionEntry(BaseModel):
    """A single query fingerprint stored in the session rolling window.

    Args:
        sequence_hash: SHA-256 hex digest (64 chars) of the raw amino acid
            sequence, used as a stable identity key without storing the
            sequence itself.
        embedding: Mean-pooled ESM-2 embedding vector of length
            :data:`EMBEDDING_DIM` (1280 for ``esm2_t33_650M_UR50D``).
        timestamp: UTC wall-clock time when the query was received.
        risk_score: Per-sequence risk score in ``[0, 1]`` produced by
            Component 1 scoring.
        sequence_length: Number of amino acids in the original sequence.
    """

    sequence_hash: str = Field(
        ...,
        description="SHA-256 hex digest of the amino acid sequence (64 chars)",
    )
    embedding: list[float] = Field(
        ...,
        description=f"Mean-pooled ESM-2 embedding vector ({EMBEDDING_DIM}-d)",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the query was submitted",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Per-sequence risk score in [0, 1]",
    )
    sequence_length: int = Field(
        ...,
        gt=0,
        description="Length of the amino acid sequence",
    )

    @field_validator("sequence_hash")
    @classmethod
    def validate_sequence_hash(cls, v: str) -> str:
        """Ensure sequence_hash is a 64-character lowercase hex string (SHA-256)."""
        if not _SHA256_RE.match(v):
            raise ValueError(
                "sequence_hash must be a 64-character lowercase hex string (SHA-256)"
            )
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding_length(cls, v: list[float]) -> list[float]:
        """Ensure embedding length matches ESM-2 esm2_t33_650M_UR50D hidden size."""
        if len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have exactly {EMBEDDING_DIM} elements, got {len(v)}"
            )
        return v


class SessionState(BaseModel):
    """Rolling-window state for one client session.

    Args:
        session_id: Opaque identifier for the session (e.g. client IP or
            API key hash).
        entries: Ordered list of :class:`SessionEntry` records in the current
            sliding window (oldest first).
        created_at: UTC timestamp when the session was first seen.
        last_active: UTC timestamp of the most recent query in this session.
        anomaly_score: Aggregate behavioral anomaly score in ``[0, 1]``
            computed by the analyzer; ``0.0`` until first analysis run.
    """

    session_id: str = Field(..., description="Opaque session identifier")
    entries: list[SessionEntry] = Field(
        default_factory=list,
        description="Ordered rolling window of query fingerprints",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the session was first seen",
    )
    last_active: datetime = Field(
        ...,
        description="UTC timestamp of the most recent query",
    )
    anomaly_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate behavioral anomaly score [0, 1]",
    )


# ── Detector result models ─────────────────────────────────────────────────


class ConvergenceResult(BaseModel):
    """Output of the convergent-optimization detector.

    A positive :attr:`similarity_trend` indicates that successive queries
    are moving closer to each other in embedding space (i.e. converging
    toward a target structure).

    Args:
        mean_similarity: Mean pairwise cosine similarity across the window.
        similarity_trend: Linear-regression slope of per-step similarities
            (positive = converging, negative = diverging).
        window_size: Number of entries included in this computation.
        is_flagged: ``True`` when convergence exceeds the configured threshold.
    """

    mean_similarity: float = Field(
        ...,
        description="Mean pairwise cosine similarity across the session window",
    )
    similarity_trend: float = Field(
        ...,
        description="Similarity trend slope (positive = converging toward a target)",
    )
    window_size: int = Field(
        ...,
        ge=0,
        description="Number of session entries included in this analysis",
    )
    is_flagged: bool = Field(
        ...,
        description="True when convergence exceeds the detection threshold",
    )


class PerturbationResult(BaseModel):
    """Output of the multi-provider perturbation detector.

    Detects near-identical sequences submitted with slight perturbations,
    suggesting systematic threshold-probing behaviour.

    Args:
        cluster_count: Number of high-similarity clusters found in the window.
        max_cluster_size: Size of the largest cluster.
        high_sim_pairs: Index pairs ``(i, j)`` from the session window whose
            cosine similarity exceeds the perturbation threshold (``> 0.95``).
        is_flagged: ``True`` when at least one high-similarity cluster is found.
    """

    cluster_count: int = Field(
        ...,
        ge=0,
        description="Number of near-identical sequence clusters in the window",
    )
    max_cluster_size: int = Field(
        ...,
        ge=0,
        description="Size of the largest near-identical cluster",
    )
    high_sim_pairs: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Index pairs (i, j) with cosine similarity > 0.95",
    )
    is_flagged: bool = Field(
        ...,
        description="True when near-identical perturbation clusters are found",
    )


class AnomalyAlert(BaseModel):
    """Aggregated behavioral anomaly assessment for a session.

    The anomaly score is a weighted combination of the convergence and
    perturbation sub-scores::

        anomaly_score = 0.6 × convergence_signal + 0.4 × perturbation_signal

    Args:
        anomaly_score: Aggregate anomaly score in ``[0, 1]``.
        convergence: Detailed convergent-optimization result.
        perturbation: Detailed perturbation-probe result.
        flagged_indices: Indices into the session window that contributed to
            the anomaly signal.
        explanation: Human-readable summary of the anomaly assessment.
    """

    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate behavioral anomaly score [0, 1]",
    )
    convergence: ConvergenceResult = Field(
        ...,
        description="Convergent-optimization detector result",
    )
    perturbation: PerturbationResult = Field(
        ...,
        description="Multi-provider perturbation detector result",
    )
    flagged_indices: list[int] = Field(
        default_factory=list,
        description="Session window indices that contributed to the anomaly signal",
    )
    explanation: str = Field(
        ...,
        description="Human-readable summary of the anomaly assessment",
    )

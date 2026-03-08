"""Risk scoring algorithm for protein toxicity assessment.

Combines embedding similarity, structural similarity, and functional overlap
to compute a unified risk score with interpretable explanations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from loguru import logger


def compute_score(
    embedding_sim: float,
    structural_sim: Optional[float] = None,
    function_overlap: float = 0.0,
    active_site_overlap: Optional[float] = None,
    sequence_length: Optional[int] = None,
) -> Tuple[float, str]:
    """Compute a unified risk score from multiple similarity metrics.

    Combines sequence embedding similarity, structural similarity (if available),
    functional annotation overlap, and active site geometry comparison into a
    single risk score between 0 and 1.

    Args:
        embedding_sim: ESM-2 embedding cosine similarity (0-1).
        structural_sim: Foldseek TM-score or None if not available (0-1).
        function_overlap: Jaccard similarity of GO terms/EC numbers (0-1).
        active_site_overlap: Active site geometric similarity (0-1), or None.

    Returns:
        Tuple of (risk_score, explanation):
            - risk_score: Float between 0 (safe) and 1 (high risk)
            - explanation: Human-readable explanation of the score
    """
    logger.debug(
        "Computing score: embedding_sim={:.3f}, structural_sim={}, function_overlap={:.3f}, active_site={}",
        embedding_sim,
        structural_sim,
        function_overlap,
        active_site_overlap,
    )

    # Validate inputs
    embedding_sim = max(0.0, min(1.0, embedding_sim))
    if structural_sim is not None:
        structural_sim = max(0.0, min(1.0, structural_sim))
    function_overlap = max(0.0, min(1.0, function_overlap))
    if active_site_overlap is not None:
        active_site_overlap = max(0.0, min(1.0, active_site_overlap))

    # Short sequence penalty: ESM-2 embeddings are less discriminative for
    # small peptides (<50 aa) because they cluster in embedding space.
    # Reduce embedding weight and overall confidence for short sequences.
    length_confidence = 1.0
    if sequence_length is not None and sequence_length < 50:
        length_confidence = max(0.3, sequence_length / 50.0)
        embedding_sim = embedding_sim * length_confidence

    # Scoring weights - these can be tuned based on validation data
    # Full path with active site: embedding 0.35, structure 0.25, active_site 0.25, function 0.15
    # Full path without active site: embedding 0.5, structure 0.3, function 0.2
    # Fast path (no structure): embedding 0.65, function 0.35
    if structural_sim is not None and active_site_overlap is not None:
        weights = {
            "embedding": 0.35,
            "structure": 0.25,
            "active_site": 0.25,
            "function": 0.15,
        }
    elif structural_sim is not None:
        weights = {
            "embedding": 0.5,
            "structure": 0.3,
            "active_site": 0.0,
            "function": 0.2,
        }
    else:
        weights = {
            "embedding": 0.65,
            "structure": 0.0,
            "active_site": 0.0,
            "function": 0.35,
        }

    # Base score components
    embedding_score = embedding_sim
    structure_score = structural_sim if structural_sim is not None else 0.0
    function_score = function_overlap

    # Apply non-linear transformations to emphasize high similarities.
    # Calibrated against SCOPe 2.08: non-toxins cluster at 0.90-0.96 sim,
    # true toxin matches are 0.97+. The transform should be decisive in
    # the 0.95-1.0 range while keeping 0.90-0.96 at low scores.

    # Embedding similarity: calibrated against SCOPe 2.08 + evasion benchmark.
    # Non-toxins: 0.90-0.96, mutated toxins: 0.96-0.985, exact matches: 0.98+.
    # The 0.96 boundary is where signal starts; 0.98+ is strong signal.
    if embedding_sim > 0.99:
        embedding_score = 0.8 + 0.2 * ((embedding_sim - 0.99) / 0.01)
    elif embedding_sim > 0.97:
        embedding_score = 0.5 + 0.3 * ((embedding_sim - 0.97) / 0.02)
    elif embedding_sim > 0.96:
        embedding_score = 0.35 + 0.15 * ((embedding_sim - 0.96) / 0.01)
    elif embedding_sim > 0.93:
        embedding_score = 0.1 + 0.25 * ((embedding_sim - 0.93) / 0.03)
    elif embedding_sim > 0.85:
        embedding_score = 0.1 * ((embedding_sim - 0.85) / 0.08)
    else:
        embedding_score = 0.0

    # Structural similarity: TM-score is already well-calibrated
    if structural_sim is not None:
        if structural_sim > 0.7:
            structure_score = 0.4 + 0.6 * ((structural_sim - 0.7) / 0.3) ** 1.5
        else:
            structure_score = 0.4 * (structural_sim / 0.7)

    # Function overlap: high overlap is very concerning
    if function_overlap > 0.5:
        function_score = 0.3 + 0.7 * ((function_overlap - 0.5) / 0.5) ** 2
    else:
        function_score = 0.3 * (function_overlap / 0.5)

    # Active site overlap: very high overlap is the strongest danger signal
    active_site_score = 0.0
    if active_site_overlap is not None:
        if active_site_overlap > 0.6:
            active_site_score = 0.4 + 0.6 * ((active_site_overlap - 0.6) / 0.4) ** 1.5
        else:
            active_site_score = 0.4 * (active_site_overlap / 0.6)

    # Weighted combination
    raw_score = (
        weights["embedding"] * embedding_score +
        weights["structure"] * structure_score +
        weights["active_site"] * active_site_score +
        weights["function"] * function_score
    )

    # Apply bonus for multiple high-confidence signals
    bonus = 0.0
    high_confidence_signals = 0

    if embedding_sim > 0.8:
        high_confidence_signals += 1
    if structural_sim is not None and structural_sim > 0.7:
        high_confidence_signals += 1
    if active_site_overlap is not None and active_site_overlap > 0.6:
        high_confidence_signals += 1
    if function_overlap > 0.4:
        high_confidence_signals += 1

    # Synergy bonus: multiple signals are more concerning than individual ones
    if high_confidence_signals >= 2:
        bonus = 0.1 * high_confidence_signals
    elif high_confidence_signals == 1 and embedding_sim > 0.9:
        bonus = 0.05  # Single very strong signal gets small bonus

    final_score = min(1.0, raw_score + bonus)

    # Structural confirmation boost: when Foldseek finds a real structural
    # match (TM > 0.4) alongside high embedding, this is a strong signal
    # that embedding alone can't provide. Boost to ensure detection.
    if structural_sim is not None and structural_sim > 0.4 and embedding_sim > 0.95:
        final_score = max(final_score, 0.55)
        high_confidence_signals = max(high_confidence_signals, 2)

    # Generate explanation
    explanation = _generate_explanation(
        final_score=final_score,
        embedding_sim=embedding_sim,
        structural_sim=structural_sim,
        function_overlap=function_overlap,
        active_site_overlap=active_site_overlap,
        high_confidence_signals=high_confidence_signals,
        bonus=bonus,
        length_confidence=length_confidence,
    )

    logger.debug(
        "Final score: {:.3f} (raw: {:.3f}, bonus: {:.3f})",
        final_score,
        raw_score,
        bonus,
    )

    return final_score, explanation


def _generate_explanation(
    final_score: float,
    embedding_sim: float,
    structural_sim: Optional[float],
    function_overlap: float,
    active_site_overlap: Optional[float] = None,
    high_confidence_signals: int = 0,
    bonus: float = 0.0,
    length_confidence: float = 1.0,
) -> str:
    """Generate human-readable explanation of the risk score."""
    parts = []

    # Overall assessment
    if final_score >= 0.8:
        parts.append("HIGH RISK: Strong similarity to known toxins")
    elif final_score >= 0.6:
        parts.append("MODERATE RISK: Notable similarity to toxins")
    elif final_score >= 0.4:
        parts.append("LOW-MODERATE RISK: Some similarity detected")
    elif final_score >= 0.2:
        parts.append("LOW RISK: Minimal similarity")
    else:
        parts.append("MINIMAL RISK: No significant similarity")

    # Detailed breakdown
    details = []

    # Embedding similarity
    if embedding_sim >= 0.85:
        details.append(f"very high sequence similarity ({embedding_sim:.3f})")
    elif embedding_sim >= 0.7:
        details.append(f"high sequence similarity ({embedding_sim:.3f})")
    elif embedding_sim >= 0.5:
        details.append(f"moderate sequence similarity ({embedding_sim:.3f})")
    else:
        details.append(f"low sequence similarity ({embedding_sim:.3f})")

    # Structural similarity
    if structural_sim is not None:
        if structural_sim >= 0.8:
            details.append(f"very high structural similarity ({structural_sim:.3f})")
        elif structural_sim >= 0.6:
            details.append(f"high structural similarity ({structural_sim:.3f})")
        elif structural_sim >= 0.4:
            details.append(f"moderate structural similarity ({structural_sim:.3f})")
        else:
            details.append(f"low structural similarity ({structural_sim:.3f})")
    else:
        details.append("structural analysis not performed")

    # Active site overlap
    if active_site_overlap is not None:
        if active_site_overlap >= 0.7:
            details.append(f"high active site similarity ({active_site_overlap:.3f}) — catalytic geometry matches known toxin")
        elif active_site_overlap >= 0.4:
            details.append(f"moderate active site similarity ({active_site_overlap:.3f})")
        else:
            details.append(f"low active site similarity ({active_site_overlap:.3f})")

    # Function overlap
    if function_overlap >= 0.6:
        details.append(f"high functional overlap ({function_overlap:.3f})")
    elif function_overlap >= 0.3:
        details.append(f"moderate functional overlap ({function_overlap:.3f})")
    elif function_overlap > 0.0:
        details.append(f"low functional overlap ({function_overlap:.3f})")
    else:
        details.append("no functional overlap detected")

    # Short sequence note
    if length_confidence < 1.0:
        details.append(f"short sequence — reduced embedding confidence ({length_confidence:.0%})")

    # Synergy information
    if bonus > 0.05:
        details.append(f"multiple strong signals detected (bonus: +{bonus:.2f})")

    if details:
        parts.append("Factors: " + "; ".join(details))

    # Risk interpretation
    if final_score >= 0.75:
        parts.append(
            "Recommend immediate review by biosafety expert and consider "
            "additional wet-lab validation before any synthesis"
        )
    elif final_score >= 0.5:
        parts.append(
            "Recommend biosafety review and enhanced monitoring protocols"
        )
    elif final_score >= 0.3:
        parts.append("Standard biosafety protocols should be sufficient")

    return ". ".join(parts) + "."


def score_batch(
    embedding_similarities: list[float],
    structural_similarities: Optional[list[Optional[float]]] = None,
    function_overlaps: Optional[list[float]] = None,
) -> list[Tuple[float, str]]:
    """Compute risk scores for multiple sequences.

    Args:
        embedding_similarities: List of embedding cosine similarities.
        structural_similarities: Optional list of structural similarities.
        function_overlaps: Optional list of function overlaps.

    Returns:
        List of (score, explanation) tuples.
    """
    n_sequences = len(embedding_similarities)

    # Prepare optional lists
    if structural_similarities is None:
        structural_similarities = [None] * n_sequences
    elif len(structural_similarities) != n_sequences:
        logger.warning(
            "Structural similarities length mismatch, padding with None"
        )
        structural_similarities.extend([None] * (n_sequences - len(structural_similarities)))

    if function_overlaps is None:
        function_overlaps = [0.0] * n_sequences
    elif len(function_overlaps) != n_sequences:
        logger.warning(
            "Function overlaps length mismatch, padding with 0.0"
        )
        function_overlaps.extend([0.0] * (n_sequences - len(function_overlaps)))

    # Compute scores
    results = []
    for i in range(n_sequences):
        score, explanation = compute_score(
            embedding_sim=embedding_similarities[i],
            structural_sim=structural_similarities[i],
            function_overlap=function_overlaps[i],
        )
        results.append((score, explanation))

    return results


def calibrate_thresholds(
    validation_scores: list[float],
    validation_labels: list[bool],
    target_specificity: float = 0.95,
) -> dict[str, float]:
    """Calibrate risk thresholds based on validation data.

    Args:
        validation_scores: List of computed risk scores.
        validation_labels: List of true labels (True = toxin, False = benign).
        target_specificity: Target specificity for the high-risk threshold.

    Returns:
        Dictionary with calibrated thresholds.
    """
    if len(validation_scores) != len(validation_labels):
        raise ValueError("Scores and labels must have same length")

    # Convert to numpy arrays for easier manipulation
    scores = np.array(validation_scores)
    labels = np.array(validation_labels)

    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Find threshold for target specificity
    n_negatives = np.sum(~labels)
    target_false_positives = int(n_negatives * (1 - target_specificity))

    cumulative_fp = np.cumsum(~sorted_labels)
    high_risk_idx = np.where(cumulative_fp <= target_false_positives)[0]

    if len(high_risk_idx) > 0:
        high_risk_threshold = sorted_scores[high_risk_idx[-1]]
    else:
        high_risk_threshold = 1.0  # Very conservative

    # Medium risk threshold at ~85% specificity
    target_fp_medium = int(n_negatives * 0.15)
    medium_risk_idx = np.where(cumulative_fp <= target_fp_medium)[0]

    if len(medium_risk_idx) > 0:
        medium_risk_threshold = sorted_scores[medium_risk_idx[-1]]
    else:
        medium_risk_threshold = high_risk_threshold * 0.7

    return {
        "high_risk_threshold": float(high_risk_threshold),
        "medium_risk_threshold": float(medium_risk_threshold),
        "validation_stats": {
            "total_samples": len(validation_scores),
            "positive_samples": int(np.sum(labels)),
            "negative_samples": int(np.sum(~labels)),
            "score_range": (float(np.min(scores)), float(np.max(scores))),
        },
    }
"""Behavioral anomaly detection for BioScreen session monitoring.

Implements convergence detection by analysing pairwise cosine similarities of
ESM-2 embeddings across a rolling session window.  No external dependencies
beyond ``numpy`` and the standard library.
"""

from __future__ import annotations

import numpy as np

from app.monitoring.schemas import AnomalyAlert, ConvergenceResult, PerturbationResult, SessionEntry


class SessionAnalyzer:
    """Analyses session rolling windows for convergent-optimization signals.

    Convergence is detected when the mean pairwise cosine similarity across
    the full window exceeds *convergence_threshold* **and** the similarity
    trend is positive — i.e. successive queries are moving closer together
    in ESM-2 embedding space, suggesting iteration toward a target structure.

    Perturbation is detected when a cluster of three or more entries share
    all of: high embedding similarity, similar sequence length, and close
    submission timestamps — suggesting systematic threshold-probing.

    Args:
        convergence_threshold: Mean pairwise cosine similarity above which a
            converging session is considered suspicious.  Default ``0.75``.
        min_window_for_convergence: Minimum number of session entries required
            before the detector produces a meaningful result.  Windows shorter
            than this return an unflagged result with all-zero scores.
            Default ``5``.
        perturbation_sim_threshold: Cosine similarity above which two entries
            are considered near-identical for perturbation detection.
            Default ``0.95``.
        perturbation_time_window_seconds: Maximum absolute difference in
            submission timestamps (seconds) for a pair to be considered a
            perturbation candidate.  Default ``300``.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.75,
        min_window_for_convergence: int = 5,
        perturbation_sim_threshold: float = 0.95,
        perturbation_time_window_seconds: int = 300,
    ) -> None:
        self._convergence_threshold = convergence_threshold
        self._min_window = min_window_for_convergence
        self._perturbation_sim_threshold = perturbation_sim_threshold
        self._perturbation_time_window = perturbation_time_window_seconds

    # ------------------------------------------------------------------
    # Public detector
    # ------------------------------------------------------------------

    def compute_convergence(self, entries: list[SessionEntry]) -> ConvergenceResult:
        """Compute pairwise embedding similarity and a convergence trend.

        The **mean similarity** is the mean cosine similarity over all unique
        ``(i, j)`` pairs (``i < j``) in the full window.

        The **similarity trend** is computed as::

            trend = mean_pairwise_sim(second_half) - mean_pairwise_sim(first_half)

        A positive trend means the second half of the window is more tightly
        clustered than the first — convergence toward a target.

        A result is flagged when **all** of the following hold:

        * ``mean_similarity > convergence_threshold``
        * ``similarity_trend > 0``
        * ``len(entries) >= min_window_for_convergence``

        When ``len(entries) < min_window_for_convergence`` the method returns
        immediately with ``mean_similarity=0.0``, ``similarity_trend=0.0``, and
        ``is_flagged=False``.

        Args:
            entries: Ordered list of :class:`~app.monitoring.schemas.SessionEntry`
                objects (oldest first).

        Returns:
            A populated :class:`~app.monitoring.schemas.ConvergenceResult`.
        """
        n = len(entries)

        if n < self._min_window:
            return ConvergenceResult(
                mean_similarity=0.0,
                similarity_trend=0.0,
                window_size=n,
                is_flagged=False,
            )

        # Build (N, D) float32 matrix from the embedding lists.
        matrix = np.array([e.embedding for e in entries], dtype=np.float32)

        # L2-normalise each row so that dot products equal cosine similarity.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1e-9, norms)
        normed = matrix / norms  # (N, D)

        # All-pairs cosine similarity via a single matrix multiply → (N, N).
        sim = normed @ normed.T

        # Upper-triangle indices (i < j) yield all unique pairs.
        ii, jj = np.triu_indices(n, k=1)
        mean_similarity = float(np.mean(sim[ii, jj]))

        # Trend: within-half mean similarity for second half minus first half.
        mid = n // 2
        first_mean = _half_mean_sim(normed[:mid])
        second_mean = _half_mean_sim(normed[mid:])
        similarity_trend = float(second_mean - first_mean)

        is_flagged = mean_similarity > self._convergence_threshold and similarity_trend > 0.0

        return ConvergenceResult(
            mean_similarity=mean_similarity,
            similarity_trend=similarity_trend,
            window_size=n,
            is_flagged=is_flagged,
        )

    def compute_perturbation(self, entries: list[SessionEntry]) -> PerturbationResult:
        """Detect near-identical sequence clusters suggesting threshold-probing.

        A pair ``(i, j)`` is flagged when **all three** conditions hold:

        * ``cosine_similarity(embedding_i, embedding_j) > perturbation_sim_threshold``
        * ``abs(entry_i.sequence_length - entry_j.sequence_length) < 5``
        * ``abs((entry_i.timestamp - entry_j.timestamp).total_seconds())
          < perturbation_time_window_seconds``

        Flagged pairs are grouped into connected components (clusters) using
        union-find.  The result is flagged when the largest cluster contains
        **at least 3** entries.

        Args:
            entries: Ordered list of :class:`~app.monitoring.schemas.SessionEntry`
                objects (oldest first).

        Returns:
            A populated :class:`~app.monitoring.schemas.PerturbationResult`.
        """
        n = len(entries)

        if n < 2:
            return PerturbationResult(
                cluster_count=0,
                max_cluster_size=0,
                high_sim_pairs=[],
                is_flagged=False,
            )

        # Build (N, D) float32 matrix and L2-normalise rows.
        matrix = np.array([e.embedding for e in entries], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1e-9, norms)
        normed = matrix / norms

        # All-pairs cosine similarity via matrix multiply → (N, N).
        sim = normed @ normed.T

        # Collect pairs that satisfy all three conditions.
        high_sim_pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = float(sim[i, j])
                length_diff = abs(entries[i].sequence_length - entries[j].sequence_length)
                time_diff = abs(
                    (entries[i].timestamp - entries[j].timestamp).total_seconds()
                )
                if (
                    cos_sim > self._perturbation_sim_threshold
                    and length_diff < 5
                    and time_diff < self._perturbation_time_window
                ):
                    high_sim_pairs.append((i, j))

        if not high_sim_pairs:
            return PerturbationResult(
                cluster_count=0,
                max_cluster_size=0,
                high_sim_pairs=[],
                is_flagged=False,
            )

        # Union-find: build connected components from flagged pairs.
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            rx, ry = _find(x), _find(y)
            if rx != ry:
                parent[rx] = ry

        for i, j in high_sim_pairs:
            _union(i, j)

        # Tally cluster sizes, counting only nodes that appear in a flagged pair.
        nodes_in_pairs: set[int] = set()
        for i, j in high_sim_pairs:
            nodes_in_pairs.add(i)
            nodes_in_pairs.add(j)

        root_to_members: dict[int, list[int]] = {}
        for node in nodes_in_pairs:
            root = _find(node)
            if root not in root_to_members:
                root_to_members[root] = []
            root_to_members[root].append(node)

        cluster_sizes = [len(members) for members in root_to_members.values()]
        cluster_count = len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)

        return PerturbationResult(
            cluster_count=cluster_count,
            max_cluster_size=max_cluster_size,
            high_sim_pairs=high_sim_pairs,
            is_flagged=max_cluster_size >= 3,
        )

    def analyze(self, entries: list[SessionEntry]) -> AnomalyAlert:
        """Run both detectors and produce a composite anomaly assessment.

        Runs :meth:`compute_convergence` and :meth:`compute_perturbation` then
        combines their outputs into a single :class:`~app.monitoring.schemas.AnomalyAlert`.

        **Score formula**::

            convergence_component  = convergence.mean_similarity  if convergence.is_flagged  else 0.0
            perturbation_component = max_cluster_size / len(entries) if perturbation.is_flagged else 0.0
            raw_score = 0.6 * convergence_component + 0.4 * perturbation_component
            anomaly_score = clamp(raw_score, 0.0, 1.0)

        The clamp protects against floating-point edge cases where
        ``float32`` cosine similarity can marginally exceed ``1.0``, which
        would otherwise cause a Pydantic ``ValidationError`` on
        :attr:`~AnomalyAlert.anomaly_score`.

        **Flagged indices** are the union of:

        * All window indices when convergence is flagged.
        * Indices from :attr:`~PerturbationResult.high_sim_pairs` when
          perturbation is flagged.

        **Explanation** is a single-sentence-per-detector factual summary;
        ``"No anomalies detected."`` when neither fires.

        Args:
            entries: Ordered list of :class:`~app.monitoring.schemas.SessionEntry`
                objects (oldest first).  An empty list is handled gracefully.

        Returns:
            A fully populated :class:`~app.monitoring.schemas.AnomalyAlert`.
        """
        n = len(entries)

        convergence = self.compute_convergence(entries)
        perturbation = self.compute_perturbation(entries)

        # Weighted combination of the two detector signals.
        convergence_component = convergence.mean_similarity if convergence.is_flagged else 0.0
        perturbation_component = (
            perturbation.max_cluster_size / n if (perturbation.is_flagged and n > 0) else 0.0
        )
        raw_score = 0.6 * convergence_component + 0.4 * perturbation_component
        anomaly_score = min(1.0, max(0.0, raw_score))

        # Collect indices that contributed to the anomaly signal.
        flagged_set: set[int] = set()
        if convergence.is_flagged:
            flagged_set.update(range(n))
        if perturbation.is_flagged:
            for i, j in perturbation.high_sim_pairs:
                flagged_set.add(i)
                flagged_set.add(j)
        flagged_indices = sorted(flagged_set)

        # Build a factual explanation (one sentence per active detector).
        parts: list[str] = []
        if convergence.is_flagged:
            parts.append(
                f"Convergent optimization detected: mean similarity "
                f"{convergence.mean_similarity:.3f}, "
                f"trend {convergence.similarity_trend:+.3f}."
            )
        if perturbation.is_flagged:
            parts.append(
                f"Perturbation probing detected: "
                f"{perturbation.cluster_count} cluster(s), "
                f"largest size {perturbation.max_cluster_size}."
            )
        explanation = " ".join(parts) if parts else "No anomalies detected."

        return AnomalyAlert(
            anomaly_score=anomaly_score,
            convergence=convergence,
            perturbation=perturbation,
            flagged_indices=flagged_indices,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _half_mean_sim(normed_half: np.ndarray) -> float:
    """Return mean pairwise cosine similarity within a normalised sub-window.

    Returns ``0.0`` when the sub-window has fewer than 2 entries (no pairs
    exist to compare).

    Args:
        normed_half: L2-normalised embedding matrix of shape ``(M, D)``.

    Returns:
        Scalar mean pairwise cosine similarity in ``[-1, 1]``, or ``0.0``
        if ``M < 2``.
    """
    m = len(normed_half)
    if m < 2:
        return 0.0
    sim = normed_half @ normed_half.T
    ii, jj = np.triu_indices(m, k=1)
    return float(np.mean(sim[ii, jj]))

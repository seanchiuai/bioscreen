"""Structural and embedding-based similarity searches.

Fast path  : ESM-2 embedding → FAISS cosine search.
Full path  : ESMFold structure → Foldseek TM-score, combined with embeddings.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from app.config import get_settings
from app.pipeline.embedding import EmbeddingArray, cosine_similarity


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class EmbeddingHit:
    """A single FAISS nearest-neighbour result."""

    index: int
    """Position in the FAISS index / metadata list."""
    cosine_similarity: float
    metadata: dict = field(default_factory=dict)


@dataclass
class StructureHit:
    """A single Foldseek alignment result."""

    target_id: str
    tm_score: float          # TM-score (query-normalised)
    lddt: float = 0.0
    aligned_length: int = 0
    query_coverage: float = 0.0
    qstart: int = 0          # 0-indexed start of aligned region in query
    qend: int = 0            # 0-indexed end of aligned region in query
    raw_line: str = ""


@dataclass
class SimilarityResult:
    """Combined similarity results for one query sequence."""

    embedding_hits: list[EmbeddingHit] = field(default_factory=list)
    structure_hits: list[StructureHit] = field(default_factory=list)
    max_embedding_sim: float = 0.0
    max_structure_sim: float = 0.0


# ── Embedding similarity (FAISS) ──────────────────────────────────────────────


class EmbeddingSimilaritySearcher:
    """Cosine similarity search over the FAISS toxin index.

    The searcher is thin wrapper around :class:`~app.database.toxin_db.ToxinDatabase`
    and is kept separate to decouple the pipeline from the database layer.
    """

    def __init__(self, toxin_db) -> None:
        """
        Args:
            toxin_db: A loaded :class:`~app.database.toxin_db.ToxinDatabase` instance.
        """
        self._db = toxin_db

    def search(
        self,
        query_embedding: EmbeddingArray,
        top_k: int = 10,
    ) -> list[EmbeddingHit]:
        """Find the *top_k* most similar toxins by embedding cosine similarity.

        Args:
            query_embedding: 1-D float32 numpy array.
            top_k: Number of nearest neighbours to return.

        Returns:
            List of :class:`EmbeddingHit` sorted by descending similarity.
        """
        if self._db is None or self._db.size == 0:
            logger.warning("Toxin DB is empty – returning no embedding hits.")
            return []

        distances, indices = self._db.search(query_embedding, k=top_k)
        hits: list[EmbeddingHit] = []
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue  # FAISS padding index
            meta = self._db.get_metadata(idx)
            hits.append(
                EmbeddingHit(
                    index=int(idx),
                    cosine_similarity=float(dist),
                    metadata=meta,
                )
            )
        return hits


# ── Structural similarity (Foldseek) ─────────────────────────────────────────


class FoldseekSearcher:
    """Run Foldseek structure alignments against the toxin PDB database.

    Requires:
        - ``foldseek`` binary on PATH (or configured in settings).
        - A pre-built Foldseek database at ``settings.foldseek_db_path``.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def available(self) -> bool:
        """Return True if the foldseek binary and database can be found."""
        import shutil

        binary_ok = shutil.which(self._settings.foldseek_bin) is not None
        db_ok = Path(self._settings.foldseek_db_path).exists()
        return binary_ok and db_ok

    def _parse_m8(self, m8_text: str) -> list[StructureHit]:
        """Parse Foldseek tab-separated m8 output into :class:`StructureHit` list.

        Foldseek default columns (``--format-output``):
        query, target, fident, alnlen, mismatch, gapopen,
        qstart, qend, tstart, tend, evalue, bits

        When ``--format-output`` includes ``lddt`` and ``qtmscore``, columns shift.
        We request: query,target,qtmscore,lddt,alnlen,qcov,qstart,qend
        """
        hits: list[StructureHit] = []
        for line in m8_text.strip().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                target_id = parts[1].split(".")[0]  # strip PDB chain suffix
                tm_score = float(parts[2])
                lddt = float(parts[3]) if len(parts) > 3 else 0.0
                aln_len = int(parts[4]) if len(parts) > 4 else 0
                qcov = float(parts[5]) if len(parts) > 5 else 0.0
                qstart = int(parts[6]) if len(parts) > 6 else 0
                qend = int(parts[7]) if len(parts) > 7 else 0
                hits.append(
                    StructureHit(
                        target_id=target_id,
                        tm_score=tm_score,
                        lddt=lddt,
                        aligned_length=aln_len,
                        query_coverage=qcov,
                        qstart=qstart,
                        qend=qend,
                        raw_line=line,
                    )
                )
            except (ValueError, IndexError):
                logger.debug("Skipping malformed Foldseek line: {!r}", line)

        return sorted(hits, key=lambda h: h.tm_score, reverse=True)

    async def search(
        self,
        pdb_string: str,
        top_k: int = 10,
        sensitivity: float = 9.5,
    ) -> list[StructureHit]:
        """Run an async Foldseek search for one query structure.

        Args:
            pdb_string: Query structure in PDB format.
            top_k: Maximum number of hits to return.
            sensitivity: Foldseek sensitivity (1–9.5; higher = more sensitive).

        Returns:
            List of :class:`StructureHit` sorted by descending TM-score.
        """
        if not self.available:
            logger.warning(
                "Foldseek is not available (binary or database missing)."
            )
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            query_pdb = tmp / "query.pdb"
            query_pdb.write_text(pdb_string)
            result_file = tmp / "result.m8"
            tmp_foldseek = tmp / "foldseek_tmp"
            tmp_foldseek.mkdir()

            cmd = [
                self._settings.foldseek_bin,
                "easy-search",
                str(query_pdb),
                str(self._settings.foldseek_db_path),
                str(result_file),
                str(tmp_foldseek),
                "--format-output", "query,target,qtmscore,lddt,alnlen,qcov,qstart,qend",
                "-s", str(sensitivity),
                "--max-seqs", str(top_k * 2),
                "--threads", "4",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                if proc.returncode != 0:
                    logger.error("Foldseek exited with code {}: {}", proc.returncode, stderr.decode())
                    return []
            except asyncio.TimeoutError:
                logger.error("Foldseek search timed out.")
                return []
            except FileNotFoundError:
                logger.error("Foldseek binary not found: {}", self._settings.foldseek_bin)
                return []

            if not result_file.exists():
                return []

            m8_text = result_file.read_text()
            hits = self._parse_m8(m8_text)
            return hits[:top_k]


# ── Combined searcher ─────────────────────────────────────────────────────────


class CombinedSimilaritySearcher:
    """Orchestrates both embedding and structure similarity searches.

    Args:
        toxin_db: Loaded :class:`~app.database.toxin_db.ToxinDatabase`.
    """

    def __init__(self, toxin_db) -> None:
        self._embedding_searcher = EmbeddingSimilaritySearcher(toxin_db)
        self._foldseek_searcher = FoldseekSearcher()

    async def search(
        self,
        query_embedding: EmbeddingArray,
        pdb_string: str | None = None,
        top_k: int = 10,
        run_structure: bool = False,
    ) -> SimilarityResult:
        """Run similarity search on the query sequence representation.

        Always runs embedding search (fast path).  Optionally runs Foldseek
        when *run_structure* is True and a *pdb_string* is provided.

        Args:
            query_embedding: Mean-pooled ESM-2 embedding of the query.
            pdb_string: ESMFold-predicted PDB string (required for structure search).
            top_k: Number of hits per modality.
            run_structure: Whether to run Foldseek structure comparison.

        Returns:
            :class:`SimilarityResult` with combined hits.
        """
        result = SimilarityResult()

        # Fast path: embedding search
        emb_hits = self._embedding_searcher.search(query_embedding, top_k=top_k)
        result.embedding_hits = emb_hits
        result.max_embedding_sim = max(
            (h.cosine_similarity for h in emb_hits), default=0.0
        )

        # Full path: structure search
        if run_structure and pdb_string:
            struct_hits = await self._foldseek_searcher.search(pdb_string, top_k=top_k)
            result.structure_hits = struct_hits
            result.max_structure_sim = max(
                (h.tm_score for h in struct_hits), default=0.0
            )

        return result

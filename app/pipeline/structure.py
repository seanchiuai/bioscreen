"""ESMFold structure prediction via the NVIDIA NIM API.

Wraps the NVIDIA-hosted ESMFold endpoint to obtain PDB-format 3-D structures
from amino acid sequences.  Falls back gracefully when the API key is absent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class StructureResult:
    """Result of an ESMFold prediction call."""

    pdb_string: str
    """Full PDB-format structure string."""

    mean_plddt: float
    """Mean per-residue pLDDT confidence (0–100)."""

    success: bool = True
    error: str = ""


# ── Client ────────────────────────────────────────────────────────────────────


class ESMFoldClient:
    """Async client for the NVIDIA NIM ESMFold endpoint.

    Usage::

        client = ESMFoldClient()
        result = await client.predict("MKTAYIAKQRQISFVKSH...")
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _post(self, sequence: str) -> dict:
        """POST to the NIM ESMFold endpoint and return parsed JSON."""
        payload = {"sequence": sequence}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self._settings.esmfold_api_url,
                headers=self._settings.nim_headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_plddt(pdb: str) -> float:
        """Extract mean pLDDT from B-factor column of ATOM records."""
        scores: list[float] = []
        for line in pdb.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    scores.append(float(line[60:66].strip()))
                except (ValueError, IndexError):
                    pass
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(self, sequence: str) -> StructureResult:
        """Predict the 3-D structure of an amino acid sequence.

        Args:
            sequence: Single-letter amino acid sequence (max ~400 aa for NIM).

        Returns:
            :class:`StructureResult` with PDB string and pLDDT score.
        """
        if not self._settings.nvidia_api_key:
            logger.warning(
                "NVIDIA_API_KEY not set – structure prediction unavailable."
            )
            return StructureResult(
                pdb_string="",
                mean_plddt=0.0,
                success=False,
                error="NVIDIA_API_KEY not configured.",
            )

        try:
            data = await self._post(sequence)
        except httpx.HTTPStatusError as exc:
            logger.error("ESMFold API returned {}: {}", exc.response.status_code, exc)
            return StructureResult(
                pdb_string="",
                mean_plddt=0.0,
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ESMFold prediction failed: {}", exc)
            return StructureResult(
                pdb_string="",
                mean_plddt=0.0,
                success=False,
                error=str(exc),
            )

        # The NIM response schema returns the PDB under "pdbs" (list) or "pdb"
        pdb_string: str = ""
        if "pdbs" in data and data["pdbs"]:
            pdb_string = data["pdbs"][0]
        elif "pdb" in data:
            pdb_string = data["pdb"]

        mean_plddt = self._parse_plddt(pdb_string)
        logger.debug(
            "ESMFold succeeded (len={}, pLDDT={:.1f})", len(sequence), mean_plddt
        )
        return StructureResult(pdb_string=pdb_string, mean_plddt=mean_plddt)

    async def predict_batch(
        self, sequences: list[str], max_concurrency: int = 3
    ) -> list[StructureResult]:
        """Predict structures for multiple sequences with rate-limiting.

        Args:
            sequences: List of amino acid sequences.
            max_concurrency: Maximum simultaneous requests to the NIM API.

        Returns:
            List of :class:`StructureResult` in the same order as *sequences*.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(seq: str) -> StructureResult:
            async with semaphore:
                return await self.predict(seq)

        return await asyncio.gather(*(_guarded(s) for s in sequences))


# ── Module-level singleton ────────────────────────────────────────────────────

_client: ESMFoldClient | None = None


def get_esmfold_client() -> ESMFoldClient:
    """Return a module-level ESMFoldClient singleton."""
    global _client
    if _client is None:
        _client = ESMFoldClient()
    return _client


async def predict_structure(sequence: str) -> str:
    """Convenience wrapper: predict structure and return PDB string.

    Raises on failure so callers can catch and handle gracefully.
    """
    client = get_esmfold_client()
    result = await client.predict(sequence)
    if not result.success:
        raise RuntimeError(result.error)
    return result.pdb_string

"""Protein function prediction via InterProScan API with mock fallback.

Submits sequences to the EBI InterProScan REST API for real GO term and
EC number predictions. Falls back to a lightweight mock predictor when
the API is unavailable or for testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import Dict, List, Optional

import httpx
from loguru import logger

from app.models.schemas import FunctionPrediction


# ── InterPro API predictor ───────────────────────────────────────────────────


class InterProPredictor:
    """Predicts protein function by submitting to the EBI InterProScan API.

    API docs: https://www.ebi.ac.uk/Tools/services/rest/iprscan5
    Free, no API key required.
    """

    BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
    POLL_INTERVAL = 5  # seconds between status checks
    MAX_WAIT = 300  # max seconds to wait for a job

    async def predict(self, sequence: str) -> Optional[FunctionPrediction]:
        """Submit sequence to InterProScan and return parsed results."""
        try:
            job_id = await self._submit(sequence)
            if not job_id:
                return None

            result_json = await self._poll_until_done(job_id)
            if not result_json:
                return None

            return self._parse_results(result_json)

        except Exception as e:
            logger.warning("InterPro API error: {}", e)
            return None

    async def _submit(self, sequence: str) -> Optional[str]:
        """Submit a sequence for analysis. Returns job ID."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.BASE_URL}/run",
                data={
                    "email": "bioscreen@example.com",
                    "sequence": sequence,
                    "goterms": "true",
                    "pathways": "true",
                },
            )
            if resp.status_code == 200:
                job_id = resp.text.strip()
                logger.debug("InterProScan job submitted: {}", job_id)
                return job_id
            else:
                logger.warning("InterPro submit failed: {} {}", resp.status_code, resp.text[:200])
                return None

    async def _poll_until_done(self, job_id: str) -> Optional[dict]:
        """Poll job status until FINISHED, then fetch JSON results."""
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=30) as client:
            while time.monotonic() - start < self.MAX_WAIT:
                resp = await client.get(f"{self.BASE_URL}/status/{job_id}")
                status = resp.text.strip()

                if status == "FINISHED":
                    result_resp = await client.get(
                        f"{self.BASE_URL}/result/{job_id}/json",
                    )
                    if result_resp.status_code == 200:
                        return result_resp.json()
                    return None

                if status in ("FAILURE", "ERROR", "NOT_FOUND"):
                    logger.warning("InterProScan job {} failed: {}", job_id, status)
                    return None

                await asyncio.sleep(self.POLL_INTERVAL)

        logger.warning("InterProScan job {} timed out after {}s", job_id, self.MAX_WAIT)
        return None

    def _parse_results(self, data: dict) -> FunctionPrediction:
        """Parse InterProScan JSON into FunctionPrediction."""
        go_terms: list[dict[str, str]] = []
        ec_numbers: list[dict[str, str]] = []
        seen_go = set()
        seen_ec = set()
        family_names: list[str] = []

        for result in data.get("results", []):
            for match in result.get("matches", []):
                signature = match.get("signature", {})
                entry = signature.get("entry") or {}

                # Collect family/domain names
                sig_name = signature.get("name", "")
                if sig_name:
                    family_names.append(sig_name)

                # GO terms from the InterPro entry
                for go in entry.get("goXRefs", []):
                    go_id = go.get("id", "")
                    go_name = go.get("name", "")
                    if go_id and go_id not in seen_go:
                        seen_go.add(go_id)
                        go_terms.append({
                            "term": go_id,
                            "name": go_name,
                            "confidence": "0.9",  # InterPro matches are high-confidence
                        })

                # Pathways (for EC numbers, look in pathways or entry description)
                for pathway in entry.get("pathways", {}):
                    db = pathway.get("databaseName", "")
                    if db == "EC":
                        ec_id = pathway.get("id", "")
                        ec_name = pathway.get("name", "")
                        if ec_id and ec_id not in seen_ec:
                            seen_ec.add(ec_id)
                            ec_numbers.append({
                                "number": ec_id,
                                "name": ec_name,
                                "confidence": "0.85",
                            })

        # Build summary
        if go_terms or ec_numbers:
            parts = []
            if family_names:
                unique = list(dict.fromkeys(family_names))[:3]
                parts.append(f"Matched protein families: {', '.join(unique)}")
            if go_terms:
                top_go = [g["name"] for g in go_terms[:3]]
                parts.append(f"functions: {', '.join(top_go)}")
            if ec_numbers:
                top_ec = [e["name"] for e in ec_numbers[:2]]
                parts.append(f"enzymatic activity: {', '.join(top_ec)}")
            summary = "; ".join(parts).capitalize() + "."
        else:
            summary = "No known protein family or function matches found."

        return FunctionPrediction(
            go_terms=go_terms,
            ec_numbers=ec_numbers,
            summary=summary,
        )


# ── Mock predictor (fallback) ────────────────────────────────────────────────


class MockFunctionPredictor:
    """Deterministic mock predictor based on sequence features. Used as fallback."""

    TOXIN_MOTIFS = ["RGD", "NGR", "KGD", "LDV", "REDV"]

    def predict(self, sequence: str) -> FunctionPrediction:
        features = self._analyze(sequence)
        toxin_score = self._toxin_likelihood(sequence, features)

        seq_hash = int(hashlib.md5(sequence.encode()).hexdigest(), 16) % (2**32)
        random.seed(seq_hash)

        go_terms = self._pick_go_terms(toxin_score)
        ec_numbers = self._pick_ec_numbers(toxin_score)

        if toxin_score > 0.7:
            summary = "Potentially harmful protein with toxin-like characteristics."
        elif toxin_score > 0.4:
            summary = "Protein with some toxin-like features."
        else:
            summary = "Likely benign protein."

        return FunctionPrediction(go_terms=go_terms, ec_numbers=ec_numbers, summary=summary)

    def _analyze(self, seq: str) -> Dict[str, float]:
        n = len(seq) or 1
        return {
            "charged": sum(seq.count(a) for a in "RHKDE") / n,
            "hydrophobic": sum(seq.count(a) for a in "AILMFWYV") / n,
            "cysteine": seq.count("C") / n,
            "signal": sum(1 for a in seq[:30] if a in "AILMFWYV") / min(30, n),
            "motifs": sum(seq.count(m) for m in self.TOXIN_MOTIFS) / (n / 100),
        }

    def _toxin_likelihood(self, seq: str, f: Dict[str, float]) -> float:
        s = 0.0
        if f["signal"] > 0.6: s += 0.3
        if f["charged"] > 0.3: s += 0.2
        if f["motifs"] > 1.0: s += 0.4
        if f["cysteine"] > 0.05: s += 0.2
        if 0.3 < f["hydrophobic"] < 0.6: s += 0.15
        h = int(hashlib.md5(seq.encode()).hexdigest(), 16) % (2**32)
        random.seed(h)
        s += random.uniform(-0.1, 0.1)
        return min(1.0, max(0.0, s))

    def _pick_go_terms(self, toxin_score: float) -> list[dict]:
        toxin_pool = [
            {"term": "GO:0090729", "name": "toxin activity", "confidence": "0.9"},
            {"term": "GO:0005576", "name": "extracellular region", "confidence": "0.8"},
            {"term": "GO:0019835", "name": "cytolysis", "confidence": "0.85"},
        ]
        benign_pool = [
            {"term": "GO:0003824", "name": "catalytic activity", "confidence": "0.9"},
            {"term": "GO:0005515", "name": "protein binding", "confidence": "0.85"},
            {"term": "GO:0008152", "name": "metabolic process", "confidence": "0.8"},
        ]
        if toxin_score > 0.5:
            return random.sample(toxin_pool, min(2, len(toxin_pool)))
        return random.sample(benign_pool, min(2, len(benign_pool)))

    def _pick_ec_numbers(self, toxin_score: float) -> list[dict]:
        if random.random() < 0.4:
            return []
        if toxin_score > 0.5:
            return [{"number": "3.4.21.-", "name": "serine endopeptidase", "confidence": "0.8"}]
        return [{"number": "1.1.1.-", "name": "oxidoreductase", "confidence": "0.85"}]


# ── Unified predictor ────────────────────────────────────────────────────────


class FunctionPredictor:
    """Unified function predictor: tries InterPro API, falls back to mock."""

    def __init__(self, use_api: bool = True) -> None:
        self._use_api = use_api
        self._interpro = InterProPredictor()
        self._mock = MockFunctionPredictor()

    async def predict_async(self, sequence: str) -> FunctionPrediction:
        """Predict function, trying InterPro API first."""
        if self._use_api:
            result = await self._interpro.predict(sequence)
            if result is not None:
                return result
            logger.info("InterPro unavailable, falling back to mock predictor")
        return self._mock.predict(sequence)

    def predict(self, sequence: str) -> FunctionPrediction:
        """Synchronous predict (mock only, for backwards compatibility)."""
        return self._mock.predict(sequence)

    def batch_predict(self, sequences: List[str]) -> List[FunctionPrediction]:
        """Synchronous batch predict (mock only)."""
        return [self._mock.predict(seq) for seq in sequences]

    async def batch_predict_async(self, sequences: List[str]) -> List[FunctionPrediction]:
        """Async batch predict with InterPro API (sequential to respect rate limits)."""
        results = []
        for seq in sequences:
            results.append(await self.predict_async(seq))
        return results

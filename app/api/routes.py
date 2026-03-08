"""API routes for the bioscreen application."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from loguru import logger

from app import __version__
from app.config import get_settings
from app.models.schemas import (
    BatchScreeningRequest,
    BatchScreeningResult,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    RiskLevel,
    ScreeningRequest,
    ScreeningResult,
    ToxinListResponse,
    ToxinSummary,
    FunctionPrediction,
    ToxinMatch,
)
from app.monitoring import default_analyzer, default_store
from app.monitoring.schemas import AnomalyAlert, SessionEntry, SessionState
from app.pipeline.embedding import get_embedding_model
from app.pipeline.function import FunctionPredictor
from app.pipeline.scoring import compute_score
from app.pipeline.sequence import validate_sequence, SequenceType
from app.pipeline.similarity import CombinedSimilaritySearcher
from app.pipeline.structure import predict_structure

router = APIRouter()

# Global function predictor instance
_function_predictor = None

def get_function_predictor() -> FunctionPredictor:
    """Get the global FunctionPredictor instance."""
    global _function_predictor
    if _function_predictor is None:
        _function_predictor = FunctionPredictor()
    return _function_predictor


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()

    # Check if toxin DB is loaded
    toxin_db = getattr(request.app.state, "toxin_db", None)
    toxin_db_loaded = toxin_db is not None and toxin_db.size > 0

    # Check if ESM-2 model is loaded
    embedding_model = getattr(request.app.state, "embedding_model", None)
    esm2_loaded = embedding_model is not None and embedding_model.is_loaded

    # Check if Foldseek is available
    from app.pipeline.similarity import FoldseekSearcher
    foldseek_searcher = FoldseekSearcher()
    foldseek_available = foldseek_searcher.available

    return HealthResponse(
        status="ok",
        version=__version__,
        toxin_db_loaded=toxin_db_loaded,
        esm2_loaded=esm2_loaded,
        foldseek_available=foldseek_available,
    )


@router.get("/toxins", response_model=ToxinListResponse)
async def list_toxins(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> ToxinListResponse:
    """List available toxin proteins in the database."""
    toxin_db = getattr(request.app.state, "toxin_db", None)

    if not toxin_db or toxin_db.size == 0:
        raise HTTPException(
            status_code=503,
            detail="Toxin database not loaded. Run scripts/build_db.py first.",
        )

    # Get toxin summaries from the database
    toxins = []
    total = toxin_db.size
    end_idx = min(offset + limit, total)

    for i in range(offset, end_idx):
        try:
            meta = toxin_db.get_metadata(i)
            toxins.append(
                ToxinSummary(
                    uniprot_id=meta.get("uniprot_id", f"UNK_{i}"),
                    name=meta.get("name", "Unknown"),
                    organism=meta.get("organism", ""),
                    toxin_type=meta.get("toxin_type", ""),
                    sequence_length=meta.get("sequence_length", 0),
                    danger_description=meta.get("danger_description", ""),
                    mechanism=meta.get("mechanism", ""),
                )
            )
        except Exception as e:
            logger.warning(f"Error getting metadata for index {i}: {e}")
            continue

    return ToxinListResponse(total=total, toxins=toxins)


@router.post("/screen", response_model=ScreeningResult)
async def screen_sequence(
    request_data: ScreeningRequest,
    app_request: Request,
    x_session_id: str | None = Header(None),
) -> ScreeningResult:
    """Screen a single sequence for toxin similarity."""
    settings = get_settings()

    # Get required dependencies from app state
    toxin_db = getattr(app_request.app.state, "toxin_db", None)
    embedding_model = getattr(app_request.app.state, "embedding_model", None)

    if not toxin_db or toxin_db.size == 0:
        raise HTTPException(
            status_code=503,
            detail="Toxin database not loaded. Run scripts/build_db.py first.",
        )

    if not embedding_model or not embedding_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ESM-2 embedding model not loaded.",
        )

    try:
        # Validate sequence
        validation = validate_sequence(request_data.sequence)
        if not validation.valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence: {validation.message}",
            )
        if validation.sequence_type != SequenceType.PROTEIN:
            raise HTTPException(
                status_code=400,
                detail=f"Only protein sequences are supported, got {validation.sequence_type.value}",
            )

        # Reject extremely long sequences (>5000 aa) to prevent resource exhaustion
        if len(request_data.sequence) > 5000:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence too long ({len(request_data.sequence)} aa). Maximum is 5000.",
            )

        # Generate sequence ID if not provided
        sequence_id = request_data.sequence_id or f"query_{hash(request_data.sequence) % 100000}"

        # Generate embedding
        logger.info(f"Generating embedding for sequence {sequence_id}")
        query_embedding = embedding_model.embed(request_data.sequence)

        # Always predict structure
        pdb_string = None
        structure_predicted = False
        logger.info(f"Predicting structure for sequence {sequence_id}")
        try:
            pdb_string = await predict_structure(request_data.sequence)
            structure_predicted = True
        except Exception as e:
            logger.warning(f"Structure prediction failed: {e}")

        # Similarity search
        logger.info(f"Running similarity search for sequence {sequence_id}")
        similarity_searcher = CombinedSimilaritySearcher(toxin_db)
        similarity_result = await similarity_searcher.search(
            query_embedding=query_embedding,
            pdb_string=pdb_string,
            top_k=request_data.top_k,
        )

        # Function prediction
        function_predictor = get_function_predictor()
        function_prediction = function_predictor.predict(request_data.sequence)

        # Convert similarity hits to ToxinMatch objects
        top_matches = []
        for hit in similarity_result.embedding_hits:
            meta = hit.metadata

            # Get structure similarity and sequence identity for this hit if available
            struct_sim = None
            seq_identity = None
            uniprot_id = meta.get("uniprot_id", "")
            for struct_hit in similarity_result.structure_hits:
                if struct_hit.target_id == uniprot_id:
                    struct_sim = struct_hit.tm_score
                    seq_identity = struct_hit.fident if struct_hit.fident > 0 else None
                    break

            top_matches.append(
                ToxinMatch(
                    uniprot_id=uniprot_id,
                    name=meta.get("name", "Unknown"),
                    organism=meta.get("organism", ""),
                    toxin_type=meta.get("toxin_type", ""),
                    embedding_similarity=hit.cosine_similarity,
                    structure_similarity=struct_sim,
                    sequence_identity=seq_identity,
                    go_terms=meta.get("go_terms", []),
                    ec_numbers=meta.get("ec_numbers", []),
                    danger_description=meta.get("danger_description", ""),
                    biological_target=meta.get("biological_target", ""),
                    mechanism=meta.get("mechanism", ""),
                )
            )

        # Compute risk score
        max_embedding_sim = similarity_result.max_embedding_sim
        max_structure_sim = similarity_result.max_structure_sim

        # Active site score: combine Foldseek lDDT (local geometry) with
        # pocket-based RMSD comparison when structure is available
        active_site_score = None
        pocket_residues: list[int] = []
        danger_residues: list[int] = []
        aligned_regions: list[list[int]] = []

        if similarity_result.structure_hits:
            # Foldseek lDDT captures local structural conservation
            max_lddt = max(h.lddt for h in similarity_result.structure_hits)
            active_site_score = max_lddt  # lDDT is already 0-1

            # Extract aligned regions from top Foldseek hits
            # Foldseek qstart/qend are 0-indexed; convert to 1-indexed for PDB residue numbering
            for hit in similarity_result.structure_hits[:3]:
                if hit.qend >= hit.qstart:
                    aligned_regions.append([hit.qstart + 1, hit.qend + 1])

            # Refine with pocket RMSD if query structure available
            if pdb_string:
                try:
                    from app.pipeline.active_site import detect_pockets, compute_active_site_score
                    from pathlib import Path

                    # Detect pockets in query structure
                    query_pockets = detect_pockets(pdb_string)
                    for pocket in query_pockets:
                        pocket_residues.extend(pocket.residue_indices)

                    # Compare against top Foldseek hits
                    target_pdbs = {}
                    for hit in similarity_result.structure_hits[:5]:
                        pdb_path = Path(f"data/toxin_structures/{hit.target_id}.pdb")
                        if pdb_path.exists():
                            target_pdbs[hit.target_id] = pdb_path.read_text()
                    if target_pdbs:
                        site_matches = compute_active_site_score(pdb_string, target_pdbs, top_k=3)
                        if site_matches:
                            pocket_score = site_matches[0].overlap_score
                            active_site_score = 0.6 * max_lddt + 0.4 * pocket_score
                            # Danger residues: pocket residues from the best-matching site
                            if pocket_score > 0.3:
                                danger_residues = list(site_matches[0].query_pocket.residue_indices)
                except Exception as e:
                    logger.warning(f"Active site comparison failed: {e}")

        # Calculate function overlap (simplified)
        function_overlap = 0.0
        if top_matches and function_prediction.go_terms:
            query_go_set = {term["term"] for term in function_prediction.go_terms}
            for match in top_matches[:3]:  # Check top 3 matches
                match_go_set = set(match.go_terms)
                if query_go_set and match_go_set:
                    overlap = len(query_go_set & match_go_set) / len(query_go_set | match_go_set)
                    function_overlap = max(function_overlap, overlap)

        risk_score, score_explanation = compute_score(
            embedding_sim=max_embedding_sim,
            structural_sim=max_structure_sim,
            function_overlap=function_overlap,
            active_site_overlap=active_site_score,
            sequence_length=len(request_data.sequence),
        )

        # Determine risk level
        if risk_score >= settings.risk_high_threshold:
            risk_level = RiskLevel.HIGH
        elif risk_score >= settings.risk_medium_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Session monitoring — track this query and run anomaly analysis.
        _session_id = x_session_id or (
            app_request.client.host if app_request.client else "unknown"
        )
        _seq_hash = hashlib.sha256(request_data.sequence.encode()).hexdigest()
        _entry = SessionEntry(
            sequence_hash=_seq_hash,
            embedding=query_embedding.tolist(),
            timestamp=datetime.now(timezone.utc),
            risk_score=risk_score,
            sequence_length=len(request_data.sequence),
        )
        _state = default_store.add_entry(_session_id, _entry)
        _alert = default_analyzer.analyze(list(_state.entries))
        _state.anomaly_score = _alert.anomaly_score

        # Build risk factors
        risk_factors = {
            "max_embedding_similarity": max_embedding_sim,
            "max_structure_similarity": max_structure_sim,
            "function_overlap": function_overlap,
            "score_explanation": score_explanation,
            "top_match_count": len(top_matches),
            "session_anomaly_score": _alert.anomaly_score,
        }

        # Warnings
        warnings = []
        if len(request_data.sequence) > 1000:
            warnings.append("Long sequence may have truncated embeddings.")

        return ScreeningResult(
            sequence_id=sequence_id,
            sequence_length=len(request_data.sequence),
            risk_score=risk_score,
            risk_level=risk_level,
            top_matches=top_matches,
            function_prediction=function_prediction,
            structure_predicted=structure_predicted,
            pdb_string=pdb_string,
            pocket_residues=pocket_residues,
            danger_residues=danger_residues,
            aligned_regions=aligned_regions,
            risk_factors=risk_factors,
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error screening sequence: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch", response_model=BatchScreeningResult)
async def batch_screen_sequences(
    request_data: BatchScreeningRequest,
    app_request: Request,
) -> BatchScreeningResult:
    """Screen multiple sequences in batch."""
    results = []
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0

    for seq_request in request_data.sequences:
        try:
            result = await screen_sequence(seq_request, app_request)
            results.append(result)

            # Count risk levels
            if result.risk_level == RiskLevel.HIGH:
                high_risk_count += 1
            elif result.risk_level == RiskLevel.MEDIUM:
                medium_risk_count += 1
            else:
                low_risk_count += 1

        except Exception as e:
            logger.error(f"Error screening sequence {seq_request.sequence_id}: {e}")
            # Create a failed result
            failed_result = ScreeningResult(
                sequence_id=seq_request.sequence_id or "unknown",
                sequence_length=len(seq_request.sequence),
                risk_score=0.0,
                risk_level=RiskLevel.UNKNOWN,
                top_matches=[],
                function_prediction=None,
                structure_predicted=False,
                risk_factors={},
                warnings=[f"Screening failed: {str(e)}"],
            )
            results.append(failed_result)

    return BatchScreeningResult(
        results=results,
        total=len(results),
        high_risk_count=high_risk_count,
        medium_risk_count=medium_risk_count,
        low_risk_count=low_risk_count,
    )


# ── AlphaFold PDB cache ───────────────────────────────────────────────────────
_alphafold_pdb_cache: dict[str, str] = {}

ALPHAFOLD_DB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


@router.post("/compare", response_model=CompareResponse)
async def compare_structures(
    request_data: CompareRequest,
    app_request: Request,
) -> CompareResponse:
    """Compare a query protein structure with a toxin reference via superposition.

    Fetches the toxin's PDB from AlphaFold DB, aligns it to the query structure
    using Kabsch alignment, and returns both PDB strings for overlay rendering.
    """
    import httpx
    from app.pipeline.active_site import superimpose_structures

    uniprot_id = request_data.target_uniprot_id.strip().upper()

    # Look up toxin metadata from the database
    toxin_db = getattr(app_request.app.state, "toxin_db", None)
    target_name = ""
    target_organism = ""
    if toxin_db:
        for i in range(toxin_db.size):
            meta = toxin_db.get_metadata(i)
            if meta.get("uniprot_id") == uniprot_id:
                target_name = meta.get("name", "")
                target_organism = meta.get("organism", "")
                break

    # Fetch target PDB from AlphaFold DB (with in-memory cache)
    if uniprot_id in _alphafold_pdb_cache:
        target_pdb = _alphafold_pdb_cache[uniprot_id]
    else:
        url = ALPHAFOLD_DB_URL.format(uniprot_id=uniprot_id)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
            if resp.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"No AlphaFold structure found for {uniprot_id}",
                )
            resp.raise_for_status()
            target_pdb = resp.text
            _alphafold_pdb_cache[uniprot_id] = target_pdb
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"AlphaFold DB returned error: {e.response.status_code}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach AlphaFold DB: {e}",
            )

    # Superimpose target onto query
    result = superimpose_structures(request_data.query_pdb, target_pdb)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Failed to align structures. One or both PDB inputs may be invalid.",
        )

    return CompareResponse(
        query_pdb=result.query_pdb,
        target_pdb=result.aligned_target_pdb,
        target_name=target_name,
        target_organism=target_organism,
        rmsd=result.rmsd,
        aligned_residues=result.aligned_residues,
    )


@router.get("/session/{session_id}", response_model=SessionState)
async def get_session(session_id: str) -> SessionState:
    """Return the stored session state for *session_id*."""
    state = default_store.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@router.get("/session/{session_id}/alerts", response_model=AnomalyAlert)
async def get_session_alerts(session_id: str) -> AnomalyAlert:
    """Run anomaly analysis on the current session window and return the result."""
    state = default_store.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return default_analyzer.analyze(list(state.entries))
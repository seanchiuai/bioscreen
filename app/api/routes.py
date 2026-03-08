"""API routes for the bioscreen application."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from app import __version__
from app.config import get_settings
from app.models.schemas import (
    BatchScreeningRequest,
    BatchScreeningResult,
    HealthResponse,
    RiskLevel,
    ScreeningRequest,
    ScreeningResult,
    ToxinListResponse,
    ToxinSummary,
    FunctionPrediction,
    ToxinMatch,
)
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

        # Predict structure if requested
        pdb_string = None
        structure_predicted = False
        if request_data.run_structure:
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
            run_structure=request_data.run_structure,
        )

        # Function prediction
        function_predictor = get_function_predictor()
        function_prediction = function_predictor.predict(request_data.sequence)

        # Convert similarity hits to ToxinMatch objects
        top_matches = []
        for hit in similarity_result.embedding_hits:
            meta = hit.metadata

            # Get structure similarity for this hit if available
            struct_sim = None
            uniprot_id = meta.get("uniprot_id", "")
            for struct_hit in similarity_result.structure_hits:
                if struct_hit.target_id == uniprot_id:
                    struct_sim = struct_hit.tm_score
                    break

            top_matches.append(
                ToxinMatch(
                    uniprot_id=uniprot_id,
                    name=meta.get("name", "Unknown"),
                    organism=meta.get("organism", ""),
                    toxin_type=meta.get("toxin_type", ""),
                    embedding_similarity=hit.cosine_similarity,
                    structure_similarity=struct_sim,
                    go_terms=meta.get("go_terms", []),
                    ec_numbers=meta.get("ec_numbers", []),
                )
            )

        # Compute risk score
        max_embedding_sim = similarity_result.max_embedding_sim
        max_structure_sim = similarity_result.max_structure_sim if request_data.run_structure else None

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
        )

        # Determine risk level
        if risk_score >= settings.risk_high_threshold:
            risk_level = RiskLevel.HIGH
        elif risk_score >= settings.risk_medium_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Build risk factors
        risk_factors = {
            "max_embedding_similarity": max_embedding_sim,
            "max_structure_similarity": max_structure_sim,
            "function_overlap": function_overlap,
            "score_explanation": score_explanation,
            "top_match_count": len(top_matches),
        }

        # Warnings
        warnings = []
        if not request_data.run_structure and max_embedding_sim > 0.8:
            warnings.append("High sequence similarity detected. Consider running structure analysis.")
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
        # Apply batch-level structure setting if not specified per sequence
        if not hasattr(seq_request, 'run_structure'):
            seq_request.run_structure = request_data.run_structure

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
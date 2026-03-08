"""Pydantic v2 schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class SequenceType(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"


# ── Request models ────────────────────────────────────────────────────────────


class ScreeningRequest(BaseModel):
    """Single-sequence screening request."""

    sequence: str = Field(
        ...,
        min_length=10,
        description="Amino acid (FASTA alphabet) or nucleotide sequence",
    )
    sequence_id: str | None = Field(
        None, description="Optional caller-supplied identifier"
    )
    top_k: int = Field(5, ge=1, le=50, description="Number of top toxin matches to return")

    @field_validator("sequence")
    @classmethod
    def clean_sequence(cls, v: str) -> str:
        """Strip whitespace / FASTA header lines."""
        lines = v.strip().splitlines()
        if lines and lines[0].startswith(">"):
            lines = lines[1:]
        return "".join(lines).strip().upper()


class BatchScreeningRequest(BaseModel):
    """Batch screening for multiple sequences."""

    sequences: list[ScreeningRequest] = Field(
        ..., min_length=1, max_length=100, description="List of sequences to screen"
    )


# ── Result / response models ──────────────────────────────────────────────────


class ToxinMatch(BaseModel):
    """A single matched toxin entry."""

    uniprot_id: str = Field(..., description="UniProt accession")
    name: str = Field(..., description="Protein name")
    organism: str = Field("", description="Source organism")
    toxin_type: str = Field("", description="Toxin classification")
    embedding_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity of ESM-2 embeddings"
    )
    structure_similarity: float | None = Field(
        None, ge=0.0, le=1.0, description="Foldseek TM-score (if structure run)"
    )
    sequence_identity: float | None = Field(
        None, ge=0.0, le=1.0, description="Foldseek sequence identity fraction (if structure run)"
    )
    go_terms: list[str] = Field(default_factory=list, description="GO term annotations")
    ec_numbers: list[str] = Field(
        default_factory=list, description="EC number annotations"
    )
    danger_description: str = Field("", description="Human-readable danger description")
    biological_target: str = Field("", description="What this toxin targets")
    mechanism: str = Field("", description="Mechanism of action")


class FunctionPrediction(BaseModel):
    """Predicted molecular function."""

    go_terms: list[dict[str, str]] = Field(
        default_factory=list,
        description="Predicted GO terms with confidence scores",
    )
    ec_numbers: list[dict[str, str]] = Field(
        default_factory=list,
        description="Predicted EC numbers with confidence scores",
    )
    summary: str = Field("", description="Human-readable function summary")


class ScreeningResult(BaseModel):
    """Full screening result for one sequence."""

    sequence_id: str = Field(..., description="Sequence identifier")
    sequence_length: int = Field(..., description="Length of query sequence (aa)")
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated risk score [0, 1]"
    )
    risk_level: RiskLevel = Field(..., description="Categorical risk level")
    top_matches: list[ToxinMatch] = Field(
        default_factory=list, description="Highest-similarity toxin matches"
    )
    function_prediction: FunctionPrediction | None = Field(
        None, description="Predicted molecular function"
    )
    structure_predicted: bool = Field(
        True, description="Whether ESMFold was used in this run"
    )
    pdb_string: str | None = Field(
        None, description="ESMFold PDB output for 3D viewer"
    )
    pocket_residues: list[int] = Field(
        default_factory=list, description="Active site pocket residue indices"
    )
    danger_residues: list[int] = Field(
        default_factory=list, description="Residue indices matching toxin active sites"
    )
    aligned_regions: list[list[int]] = Field(
        default_factory=list,
        description="Regions of query structurally aligned to toxin, as [start, end] pairs (1-indexed, inclusive)",
    )
    risk_factors: dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate signals contributing to the risk score",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings from the pipeline"
    )


class BatchScreeningResult(BaseModel):
    """Results for a batch screening run."""

    results: list[ScreeningResult]
    total: int = Field(..., description="Total sequences processed")
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# ── Health / info models ──────────────────────────────────────────────────────


class CompareRequest(BaseModel):
    """Request to compare a query structure with a toxin reference."""

    query_pdb: str = Field(..., description="PDB string of the query protein")
    target_uniprot_id: str = Field(..., description="UniProt accession of the toxin to compare")


class CompareResponse(BaseModel):
    """Structural superposition result for 3D overlay."""

    query_pdb: str = Field(..., description="Original query PDB string (unchanged)")
    target_pdb: str = Field(..., description="Target toxin PDB aligned to query coordinate frame")
    target_name: str = Field("", description="Toxin name")
    target_organism: str = Field("", description="Toxin source organism")
    rmsd: float = Field(..., description="C-alpha RMSD after alignment (angstroms)")
    aligned_residues: int = Field(..., description="Number of C-alpha pairs used for alignment")
    sequence_identity: float | None = Field(None, description="Sequence identity if available")
    tm_score: float | None = Field(None, description="TM-score if available from prior screening")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    toxin_db_loaded: bool
    esm2_loaded: bool
    foldseek_available: bool


class ToxinSummary(BaseModel):
    """Lightweight toxin entry for the /api/toxins listing."""

    uniprot_id: str
    name: str
    organism: str
    toxin_type: str
    sequence_length: int
    danger_description: str = ""
    mechanism: str = ""


class ToxinListResponse(BaseModel):
    total: int
    toxins: list[ToxinSummary]

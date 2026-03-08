# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioScreen is a structure-based biosecurity screening tool for AI-designed proteins. It detects dangerous proteins by evaluating **function and structure**, not just sequence homology — addressing the gap where AI-designed proteins (RFdiffusion, ProteinMPNN) can fold into toxin structures while sharing near-zero sequence similarity.

## Commands

```bash
# Run API server (dev mode with hot reload)
uvicorn app.main:app --reload

# Run Streamlit frontend (separate terminal)
streamlit run frontend/streamlit_app.py

# Run all tests
pytest

# Run a single test
pytest tests/test_pipeline.py::test_score_returns_value_in_range

# Build toxin reference database (requires network access to UniProt)
python scripts/build_db.py
```

## Architecture

### Screening Pipeline (`app/pipeline/`)

The core screening flow for a protein sequence:

1. **Sequence validation** (`sequence.py`) — validates input, detects type (protein/DNA/RNA), translates nucleotides to protein if needed
2. **Embedding** (`embedding.py`) — generates ESM-2 embeddings (`facebook/esm2_t33_650M_UR50D`)
3. **Similarity search** (`similarity.py`) — cosine similarity via FAISS (fast path) + optional Foldseek structural alignment (full path)
4. **Structure prediction** (`structure.py`) — ESMFold via NVIDIA NIM API (only when `run_structure=True`)
5. **Function prediction** (`function.py`) — GO term / EC number classification
6. **Risk scoring** (`scoring.py`) — weighted combination of embedding similarity (0.5), structural similarity (0.3), function overlap (0.2), with non-linear transforms and synergy bonuses for multiple high-confidence signals

### Two Screening Paths

- **Fast path** (~seconds, CPU): Steps 1→2→3 (FAISS only)→6. Controlled by `run_structure=False` (default).
- **Full path** (~15s, GPU): All steps including ESMFold + Foldseek. Controlled by `run_structure=True`.

### Key Design Patterns

- **App state via lifespan** (`main.py`): ESM-2 model and FAISS toxin DB are loaded once at startup via FastAPI's `asynccontextmanager` lifespan, then attached to `app.state` for route access.
- **Configuration** (`config.py`): All settings via `pydantic-settings` `BaseSettings`, read from `.env` file. Cached singleton via `@lru_cache`. Includes screening thresholds, model paths, and API keys.
- **Pydantic v2 schemas** (`models/schemas.py`): Request/response models with validators (e.g., FASTA header stripping on `ScreeningRequest.sequence`).
- **Toxin database** (`database/`): FAISS index + JSON metadata sidecar. Built from UniProt Tox-Prot via `scripts/build_db.py`.

### API Endpoints (all under `/api` prefix)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/screen` | POST | Screen single sequence |
| `/api/batch` | POST | Screen multiple sequences |
| `/api/health` | GET | Health/readiness check |
| `/api/proteins` | GET | List toxin DB entries |

## Environment Setup

Requires `.env` file (copy from `.env.example`). Key variables:
- `NVIDIA_API_KEY` — for ESMFold NIM API (structure prediction)
- `DEVICE` — `cpu` or `cuda`
- `APP_ENV` — `development` or `production`

## Testing

Tests use `pytest` + `pytest-asyncio`. Test file at `tests/test_pipeline.py` covers sequence validation and risk scoring. Tests do not require GPU or external APIs.

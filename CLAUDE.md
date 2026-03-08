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
4. **Structure prediction** (`structure.py`) — ESMFold via NVIDIA NIM API (always runs)
5. **Active site detection** (`active_site.py`) — identifies binding pockets in PDB structures and compares active site geometry between query and known toxins (BioPython + numpy)
6. **Function prediction** (`function.py`) — GO term / EC number classification
7. **Risk scoring** (`scoring.py`) — weighted combination of embedding similarity (0.5), structural similarity (0.3), function overlap (0.2), with non-linear transforms and synergy bonuses for multiple high-confidence signals

### Screening Path

All screening runs the full pipeline: Steps 1→2→3→4→5→6→7. Structure prediction (ESMFold + Foldseek) is always enabled.

### Key Design Patterns

- **App state via lifespan** (`main.py`): ESM-2 model and FAISS toxin DB are loaded once at startup via FastAPI's `asynccontextmanager` lifespan, then attached to `app.state` for route access.
- **Configuration** (`config.py`): All settings via `pydantic-settings` `BaseSettings`, read from `.env` file. Cached singleton via `@lru_cache`. Includes screening thresholds, model paths, and API keys.
- **Pydantic v2 schemas** (`models/schemas.py`): Request/response models with validators (e.g., FASTA header stripping on `ScreeningRequest.sequence`).
- **Toxin database** (`database/`): FAISS index + JSON metadata sidecar. Built from UniProt Tox-Prot via `scripts/build_db.py`.
- **Routes** (`api/routes.py`): All API endpoints defined in a single `APIRouter`, mounted under `/api` prefix in `main.py`.

### Session Monitoring (`app/monitoring/`)

Behavioral monitoring layer that detects convergent optimization patterns (e.g., a user iteratively modifying sequences toward a toxin). Key components:

- `session_store.py` — rolling-window session store (50-entry, 1-hour TTL) tracking per-session screening history
- `analyzer.py` — `SessionAnalyzer` that detects anomalous patterns across a session's entries
- `schemas.py` — Pydantic models for `SessionEntry`, `SessionState`, `AnomalyAlert`
- Module-level singletons (`default_store`, `default_analyzer`) used by the API layer

### API Endpoints (all under `/api` prefix)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/screen` | POST | Screen single sequence |
| `/api/batch` | POST | Screen multiple sequences |
| `/api/health` | GET | Health/readiness check |
| `/api/toxins` | GET | List toxin DB entries |
| `/api/compare` | POST | Compare query structure with toxin via superposition |
| `/api/session/{id}` | GET | Get session state/history |
| `/api/session/{id}/alerts` | GET | Get anomaly alerts for session |

### Frontend (`frontend/`)

Streamlit-based UI with multi-page layout:

- `streamlit_app.py` — main app entry point
- `pages/single_screen.py` — single sequence screening page
- `pages/session_analysis.py` — session history and anomaly analysis
- `components/api_client.py` — HTTP client for the backend API
- `components/protein_3d.py` — py3Dmol 3D protein structure viewer
- `components/result_viewer.py` — screening result display
- `components/summary_cards.py` — summary card widgets
- `components/styles.py` — shared CSS/styling
- `video_generator.py` — captures py3Dmol via headless Playwright + composites stats overlays with PIL, outputs MP4 via ffmpeg

## Environment Setup

Requires `.env` file (copy from `.env.example`). Key variables:
- `NVIDIA_API_KEY` — for ESMFold NIM API (structure prediction)
- `ESMFOLD_API_URL` — ESMFold NIM endpoint URL
- `DEVICE` — `cpu` or `cuda`
- `APP_ENV` — `development` or `production`
- `LOG_LEVEL` — logging level (default `INFO`)
- `API_HOST` / `API_PORT` — server bind address (default `0.0.0.0:8000`)
- `ESM2_MODEL_NAME` — HuggingFace model ID for embeddings
- `TOXIN_DB_PATH` / `TOXIN_META_PATH` — paths to FAISS index and metadata JSON
- `FOLDSEEK_BIN` / `FOLDSEEK_DB_PATH` — Foldseek binary and database paths
- `UNIPROT_BATCH_SIZE` / `MAX_TOXIN_RECORDS` — UniProt build settings for `scripts/build_db.py`
- `EMBEDDING_SIM_THRESHOLD` / `STRUCTURE_SIM_THRESHOLD` / `RISK_HIGH_THRESHOLD` / `RISK_MEDIUM_THRESHOLD` — screening thresholds

## Testing

Tests use `pytest` + `pytest-asyncio`. Test files in `tests/`:
- `test_pipeline.py` — sequence validation and risk scoring
- `test_schemas.py` — Pydantic model validation
- `test_analyzer.py` — session anomaly detection logic
- `test_monitoring_init.py` — monitoring module singleton setup
- `test_session_store.py` — session store CRUD and TTL behavior
- `test_session_routes.py` — session API endpoint integration tests

Tests do not require GPU or external APIs.

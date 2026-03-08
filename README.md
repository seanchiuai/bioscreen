# BioScreen

**Structure-based biosecurity screening for AI-designed proteins.**

Current DNA synthesis screening tools compare sequences against databases of known pathogens using sequence homology. AI protein design tools (RFdiffusion, ProteinMPNN) can now generate novel sequences that fold into the same 3D structure as known toxins while sharing near-zero sequence similarity — **over 75% of these bypass conventional screening** ([Wittmann et al., *Science* 2025](https://doi.org/10.1126/science.adu8578)).

BioScreen evaluates **what a protein does**, not what it looks like. It predicts 3D structure, active-site geometry, and biological function to flag dangerous sequences even when sequence similarity is effectively zero.

## How It Works

```
Amino Acid Sequence
       │
       ├──→ ESM-2 Embedding ──→ FAISS cosine sim vs toxin DB ──→ Embedding Risk Score
       │
       ├──→ ESMFold (3D structure) ──→ Foldseek structural alignment ──→ Structure Risk Score
       │
       └──→ Function Prediction (GO terms / EC numbers) ──→ Function Risk Score
       │
       └──→ Combined Risk Assessment (0–1)
```

**Fast path** (~seconds, CPU): ESM-2 embeddings → cosine similarity against pre-computed toxin database.

**Full path** (~15s, GPU): Adds ESMFold structure prediction → Foldseek structural search → function prediction.

## Tech Stack

- **Structure Prediction**: ESMFold via NVIDIA NIM API (or local HuggingFace)
- **Embeddings**: ESM-2 (`facebook/esm2_t33_650M_UR50D`)
- **Structural Search**: Foldseek (3Di alphabet, 4000× faster than TM-align)
- **Function Prediction**: GO term / EC number classification
- **Vector DB**: FAISS for embedding similarity search
- **Reference Data**: UniProt Tox-Prot + PDB toxin-annotated structures
- **Backend**: FastAPI (async)
- **Frontend**: Streamlit

## Quick Start

```bash
# Clone
git clone https://github.com/seanchiuai/bioscreen.git
cd bioscreen

# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your NVIDIA NIM API key

# Build toxin reference database
python scripts/build_db.py

# Run API
uvicorn app.main:app --reload

# Run frontend (separate terminal)
streamlit run frontend/streamlit_app.py
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screen` | POST | Screen a single protein sequence |
| `/api/batch` | POST | Screen multiple sequences |
| `/api/health` | GET | Health check |
| `/api/toxins` | GET | Reference database stats |

### Screen a sequence

```bash
curl -X POST http://localhost:8000/api/screen \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH..."}'
```

### Response

```json
{
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "matched_toxins": [
    {"name": "Botulinum neurotoxin type A", "similarity": 0.91, "uniprot_id": "P0DPI1"}
  ],
  "predicted_functions": ["metalloprotease activity", "neurotoxin"],
  "explanation": "High structural similarity to known neurotoxin active site despite <15% sequence identity."
}
```

## Project Structure

```
bioscreen/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py             # Settings from .env
│   ├── api/routes.py         # API endpoints
│   ├── pipeline/
│   │   ├── sequence.py       # Input validation, DNA→protein
│   │   ├── structure.py      # ESMFold prediction
│   │   ├── embedding.py      # ESM-2 embeddings
│   │   ├── similarity.py     # Foldseek + cosine similarity
│   │   ├── function.py       # GO/EC prediction
│   │   └── scoring.py        # Combined risk scoring
│   ├── database/
│   │   ├── toxin_db.py       # FAISS index management
│   │   └── build_db.py       # Build reference DB from UniProt
│   └── models/schemas.py     # Pydantic models
├── frontend/streamlit_app.py # Web UI
├── scripts/
│   ├── build_db.py           # DB builder CLI
│   └── demo.py               # Demo script
├── data/                     # Toxin embeddings/structures
└── tests/test_pipeline.py    # Tests
```

## Key References

- Wittmann et al. (2025) "Strengthening nucleic acid biosecurity screening against generative protein design tools" — *Science*. [DOI: 10.1126/science.adu8578](https://doi.org/10.1126/science.adu8578)
- Wheeler et al. (2026) "The Limits of Sequence-Based Biosecurity Screening Tools in the Age of AI-Assisted Protein Design" — *bioRxiv*. [DOI: 10.64898/2026.03.04.709671](https://doi.org/10.64898/2026.03.04.709671)
- Baker & Church (2024) "Protein design meets biosecurity" — *Science*.

## License

MIT

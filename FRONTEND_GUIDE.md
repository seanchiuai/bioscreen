# Frontend Integration Guide

This guide describes how the backend works, what API endpoints are available, what data they return, and the ideal demo flow. Use this alongside `DESIGN.md` for the visual design.

## Starting the Backend

```bash
# Use Python 3.12 (Python 3.14 has FAISS/torch segfaults)
source venv312/bin/activate

# Required env vars (already set if .env exists)
# KMP_DUPLICATE_LIB_OK=TRUE — prevents OpenMP crash between torch and FAISS
# TOKENIZERS_PARALLELISM=false — prevents tokenizer fork crash

# Start server
KMP_DUPLICATE_LIB_OK=TRUE uvicorn app.main:app --reload --port 8000
```

The server loads ESM-2 model + FAISS toxin DB at startup (~10s). After that, endpoints respond in 0.5-3s.

---

## API Endpoints

Base URL: `http://localhost:8000/api`

### 1. `POST /api/screen` — Screen a single sequence

This is the main endpoint. Returns risk score, top toxin matches, function prediction, and session monitoring data.

**Request:**
```json
{
  "sequence": "MKTLLLAVAVVAFVCLGSADQLGLGRQQIDWGQGQAVGPPYTLCFECNRM...",
  "run_structure": true,
  "top_k": 5,
  "sequence_id": "demo-query-1"
}
```

- `sequence`: amino acid string (required, min 10 chars). FASTA headers are auto-stripped.
- `run_structure`: `false` = fast path (~0.5s, embedding only). `true` = full path (~3s, includes ESMFold + Foldseek + active site). **Use `true` for the demo.**
- `top_k`: number of top toxin matches to return (default 5)
- `sequence_id`: optional label

**Headers (optional):**
- `X-Session-Id: my-session-123` — ties this query to a session for behavioral monitoring. If omitted, uses client IP.

**Response:**
```json
{
  "sequence_id": "demo-query-1",
  "sequence_length": 109,
  "risk_score": 0.876,
  "risk_level": "HIGH",
  "top_matches": [
    {
      "uniprot_id": "A0S864",
      "name": "Irditoxin subunit A",
      "organism": "Boiga irregularis",
      "toxin_type": "ion_channel_toxin",
      "embedding_similarity": 1.0,
      "structure_similarity": 1.0,
      "go_terms": [],
      "ec_numbers": []
    }
  ],
  "function_prediction": {
    "go_terms": [{"term": "GO:0090729", "name": "toxin activity", "confidence": "0.9"}],
    "ec_numbers": [],
    "summary": "Likely benign protein."
  },
  "structure_predicted": true,
  "risk_factors": {
    "max_embedding_similarity": 1.0,
    "max_structure_similarity": 1.0,
    "function_overlap": 0.0,
    "score_explanation": "HIGH RISK: Strong similarity to known toxins...",
    "top_match_count": 5,
    "session_anomaly_score": 0.0
  },
  "warnings": []
}
```

**Key fields for the frontend:**
- `risk_score` (0-1) and `risk_level` ("LOW" / "MEDIUM" / "HIGH") → risk gauge
- `top_matches` → toxin match table
- `risk_factors.max_embedding_similarity` → embedding signal bar
- `risk_factors.max_structure_similarity` → structure signal bar (null if `run_structure=false`)
- `risk_factors.score_explanation` → human-readable explanation text
- `risk_factors.session_anomaly_score` → session monitoring indicator
- `function_prediction.summary` → function annotation section

### 2. `GET /api/health` — Health check

```json
{
  "status": "ok",
  "version": "0.1.0",
  "toxin_db_loaded": true,
  "esm2_loaded": true,
  "foldseek_available": true
}
```

Use this on page load to show system status. If `esm2_loaded` is false, the model is still loading.

### 3. `GET /api/toxins?limit=50&offset=0` — List toxin database

Returns paginated list of toxins in the reference database. Good for a "Database" tab or stats display.

```json
{
  "total": 2000,
  "toxins": [
    {
      "uniprot_id": "A0S864",
      "name": "Irditoxin subunit A",
      "organism": "Boiga irregularis",
      "toxin_type": "ion_channel_toxin",
      "sequence_length": 109
    }
  ]
}
```

### 4. `GET /api/session/{session_id}` — Session state

Returns the current session state including all queries and anomaly score.

### 5. `GET /api/session/{session_id}/alerts` — Session anomaly analysis

Returns detailed anomaly assessment for a session.

```json
{
  "anomaly_score": 0.798,
  "convergence": {
    "mean_similarity": 0.913,
    "similarity_trend": 0.045,
    "window_size": 8,
    "is_flagged": true
  },
  "perturbation": {
    "is_flagged": false,
    "cluster_count": 0,
    "max_cluster_size": 0
  },
  "explanation": "Convergent optimization detected: mean similarity 0.913, trend +0.045."
}
```

---

## Risk Score Thresholds

| Score Range | Level | Color |
|---|---|---|
| 0.00 - 0.44 | LOW | Green |
| 0.45 - 0.74 | MEDIUM | Orange/Yellow |
| 0.75 - 1.00 | HIGH | Red |

---

## Backend Changes Needed for DESIGN.md

The current `/api/screen` response does NOT include `pdb_string`, `pocket_residues`, or `danger_residues` (as described in DESIGN.md). These need to be added to `app/models/schemas.py` and `app/api/routes.py`:

### Schema additions (`app/models/schemas.py`):
```python
# Add to ScreeningResult class:
pdb_string: str | None = Field(None, description="ESMFold PDB output for 3D viewer")
pocket_residues: list[int] = Field(default_factory=list, description="Active site pocket residue indices")
danger_residues: list[int] = Field(default_factory=list, description="Residue indices matching toxin active sites")
```

### Route changes (`app/api/routes.py`):
The PDB string is already computed in `screen_sequence()` (stored in `pdb_string` variable) but not returned. The pocket/danger residues come from `detect_pockets()` and `compute_active_site_score()` which are already called. Just pass them through to the response.

---

## Ideal Demo Flow (4 sequences, ~60 seconds)

### Sequence 1: Known Safe — Human Insulin B Chain
```
Sequence: FVNQHLCGSHLVEALYLVCGERGFFYTPKT
Expected: LOW risk (0.126), green gauge, no structure viewer needed
Story: "This is insulin — the system correctly clears it."
```

### Sequence 2: Known Toxin — Scorpion Toxin (Aah4)
```
Sequence: (use the first ~80 chars from the /api/toxins endpoint, uniprot_id P45658)
Expected: HIGH risk (1.0), red gauge, structure shows red active site
Story: "Known toxin — any screening tool catches this."
```

### Sequence 3: AI-Designed Evasion — The Key Demo
```
Sequence: AKGLWREVWWSAFRCGNQPDQQMLPEKQIDWSPPLAEHKQWNLCFDCNEYMTSKDCSTALRCYRGSCYTLYRPDENCELKWCDEGFYCSCPHCSSNPAQCHRPCSNKD
Expected: MEDIUM-HIGH risk, Foldseek finds structural match
Story: "This sequence has <40% identity to any known toxin. BLAST says it's safe.
       But BioScreen's structure analysis finds it folds into a snake venom shape.
       This is what AI protein design tools can create today."
```
This is the Irditoxin sequence scrambled at 70% mutation rate with cysteines preserved. It's pre-computed in `scripts/demo_scenarios.py` (Scenario 1).

### Sequence 4: Show Session Monitoring
Submit 3-4 more variants of Sequence 3 (change a few residues each time). After each submission:
- Per-sequence risk stays MEDIUM
- Session anomaly score climbs: 0.0 → 0.3 → 0.5 → 0.8
- Story: "Each query alone looks borderline. But the session monitor detects
         someone iterating toward a dangerous design."

Use `X-Session-Id: demo-session` header to group them.

---

## Pre-built Demo Sequences

These are tested and produce good results. Available in `scripts/demo_scenarios.py` and `scripts/demo_10_scenarios.py`.

| Label | Sequence (first 50 chars) | Length | Expected |
|---|---|---|---|
| Insulin (safe) | `FVNQHLCGSHLVEALYLVCGERGFFYTPKT` | 30 | LOW (0.13) |
| Lysozyme (safe) | `KVFERCELARTLKRLGMDGYRGISLANWMCL...` | 130 | LOW (0.38) |
| GFP (safe) | `MSKGEELFTGVVPILVELDGDVNGHKFSVSGE...` | 238 | LOW (0.41) |
| Scorpion toxin | Get from `/api/toxins` P45658 | 84 | HIGH (1.0) |
| AI-evasion snake | `AKGLWREVWWSAFRCGNQPDQQMLPEKQIDWS...` | 109 | MEDIUM+ |

The full sequences are in the demo scripts — import them or hardcode in the frontend.

---

## Session Monitoring Integration

To show session monitoring in the UI:

1. Generate a session ID on page load: `session_id = f"demo-{uuid4().hex[:8]}"`
2. Pass it as `X-Session-Id` header with every `/api/screen` call
3. After each screen call, read `risk_factors.session_anomaly_score` from the response
4. Display as a gauge or timeline: "Session Anomaly: 0.12 (normal)" → "0.78 ⚠️ (convergent optimization detected)"
5. Optionally call `GET /api/session/{session_id}/alerts` for detailed breakdown

The anomaly score updates automatically with each query — no extra calls needed if you just use `session_anomaly_score` from the screen response.

---

## Error States to Handle

| HTTP Code | Meaning | UI Action |
|---|---|---|
| 200 | Success | Show results |
| 400 | Invalid sequence / too long (>5000 aa) | Show validation error |
| 503 | Model still loading or DB not found | Show "Loading..." spinner |
| 500 | Internal error | Show generic error |

---

## Performance Expectations

| Mode | Latency | When to use |
|---|---|---|
| Fast path (`run_structure: false`) | ~0.5s | Quick triage, session monitoring demo |
| Full path (`run_structure: true`) | ~2-4s | Main demo, shows structure + active site |

First request after server start takes ~10s (model loading). Subsequent requests are fast.

# BioScreen Demo — Frontend Visualization Design

> **Note:** `FRONTEND_GUIDE.md` is the source of truth for API details, request/response formats, and demo sequences. This document covers the visual design and 3D viewer implementation only.

## Goal
Show a screened protein's 3D structure with dangerous regions highlighted. No folding animation, no side-by-side superposition. Single structure, clear visual signal.

## Visualization: Single 3D Protein Viewer

**Library:** py3Dmol (Python wrapper for 3Dmol.js) inside Streamlit via `st.components.v1.html()`

### What the viewer shows:
1. **Full protein** — cartoon representation, light gray/blue
2. **Active site pocket residues** — highlighted in **orange**, stick representation (these are the detected binding pockets from `active_site.py`)
3. **Flagged danger residues** — highlighted in **red**, thicker stick + transparent surface (residues that geometrically match a known toxin's active site)
4. **Confidence coloring** — optional toggle: color by pLDDT (ESMFold confidence) using a blue→red gradient, so the user can see where the model is confident about the fold

### Interactions:
- Rotate (click + drag)
- Zoom (scroll)
- Hover → residue name + index tooltip (built into 3Dmol.js)

### No:
- No folding animation
- No side-by-side toxin comparison
- No sequence alignment view
- No BioRender

---

## Page Layout (Streamlit)

```
┌─────────────────────────────────────────────────────┐
│  🧬 BioScreen — Protein Toxin Screening             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Sequence Input]          [Risk Score: 0.87 HIGH]   │
│  [textarea]                [risk gauge/indicator]     │
│                                                      │
│  [🔍 Screen Sequence]                                │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────┐  ┌───────────────────────┐  │
│  │                     │  │ Top Toxin Matches      │  │
│  │   3D Protein View   │  │ • Botulinum (0.92)     │  │
│  │   (py3Dmol)         │  │ • Ricin A (0.78)       │  │
│  │                     │  │ • Abrin (0.65)          │  │
│  │  🔵 = normal        │  │                        │  │
│  │  🟠 = active site   │  ├───────────────────────┤  │
│  │  🔴 = danger match  │  │ Risk Factors           │  │
│  │                     │  │ Embedding sim: 0.91     │  │
│  │                     │  │ Structure sim: 0.87     │  │
│  │                     │  │ Active site:   0.82     │  │
│  │                     │  │ Function:      0.65     │  │
│  └─────────────────────┘  └───────────────────────┘  │
│                                                      │
│  [View Toggle: Cartoon | Surface | Stick]            │
│  [Color Toggle: Default | pLDDT Confidence]          │
│                                                      │
├─────────────────────────────────────────────────────┤
│  Function Prediction                                 │
│  GO Terms: toxin activity, metal ion binding          │
│  EC Numbers: 3.4.24.69 (metalloprotease)             │
│  Summary: Predicted zinc metalloprotease with ...     │
├─────────────────────────────────────────────────────┤
│  Session Monitoring                                  │
│  Queries this session: 4                             │
│  Anomaly score: 0.12 (normal)                        │
└─────────────────────────────────────────────────────┘
```

---

## Backend Changes Required

### 1. Extend `ScreeningResult` schema (`app/models/schemas.py`)

Add these fields:
```python
pdb_string: str | None = Field(None, description="PDB format structure (when structure analysis runs)")
pocket_residues: list[int] = Field(default_factory=list, description="Residue indices of detected active site pockets")
danger_residues: list[int] = Field(default_factory=list, description="Residue indices matching toxin active sites")
```

### 2. Update `/api/screen` route (`app/api/routes.py`)

- Pass `pdb_string` through to the response (currently computed and discarded)
- Extract `pocket.residue_indices` from `detect_pockets()` result
- Extract danger residues from `ActiveSiteMatch.query_pocket.residue_indices` when overlap_score > threshold
- Include all three in the `ScreeningResult`

### 3. No new endpoints needed

Everything goes through the existing `/api/screen` response.

---

## Frontend Implementation (`frontend/streamlit_app.py`)

### py3Dmol Viewer Function

```python
def render_protein_3d(pdb_string, pocket_residues, danger_residues, view_style="cartoon", color_mode="default"):
    """Render protein structure with highlighted regions using py3Dmol."""
    # 1. Create viewer
    # 2. Load PDB
    # 3. Apply base style (cartoon, light blue)
    # 4. Highlight pocket_residues (orange, stick)
    # 5. Highlight danger_residues (red, stick + transparent surface)
    # 6. Optional: pLDDT coloring from B-factor column
    # 7. Render via st.components.v1.html()
```

### View/Color Toggles
- `st.radio()` for view style: Cartoon | Surface | Stick
- `st.radio()` for color mode: Default (blue/orange/red) | pLDDT Confidence

### Risk Gauge
- Simple colored metric: green (0–0.44 LOW) | orange (0.45–0.74 MEDIUM) | red (0.75–1.0 HIGH)
- Use `st.metric()` with custom HTML for the color bar

---

## Dependencies to Add

```
py3Dmol>=2.0.0
```

That's it. py3Dmol has no heavy deps — it just generates HTML/JS for 3Dmol.js.

---

## Demo Script (4 sequences, ~60 seconds)

See `FRONTEND_GUIDE.md` for full sequences and expected results. See also `scripts/demo_scenarios.py` and `scripts/demo_10_scenarios.py` for pre-built sequences.

1. **Insulin B Chain** (known safe) → LOW risk (~0.13), structure is blue, no red highlights. _"The system correctly clears it."_
2. **Scorpion Toxin Aah4** (known toxin, UniProt P45658) → HIGH risk (1.0), red active site highlights. _"Any screening tool catches this."_
3. **AI-Designed Evasion** (Irditoxin scrambled at 70% mutation, cysteines preserved) → MEDIUM-HIGH risk, Foldseek finds structural match despite <40% sequence identity. _"BLAST says it's safe. BioScreen's structure analysis finds it folds into a snake venom shape."_
4. **Session Monitoring** — Submit 3-4 variants of sequence 3 with `X-Session-Id: demo-session`. Per-sequence risk stays MEDIUM, but session anomaly score climbs 0.0 → 0.3 → 0.5 → 0.8. _"Each query alone looks borderline. But the session monitor detects someone iterating toward a dangerous design."_

---

## Implementation Order

1. Add `py3Dmol` to requirements.txt
2. Extend schema + route to return PDB + residue data (see `FRONTEND_GUIDE.md` § "Backend Changes Needed")
3. Build the 3D viewer function
4. Redesign Streamlit layout per the wireframe above
5. Add view/color toggles
6. Test with real ESMFold output

---

## Related Docs

- **`FRONTEND_GUIDE.md`** — API endpoints, request/response formats, error states, performance expectations, and pre-built demo sequences

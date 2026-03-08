# Session Analysis Multipage Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the monolithic `frontend/streamlit_app.py` into a multipage Streamlit app with shared components, and add a new "Session Analysis" page that screens multiple sequences sequentially with live monitoring dashboard.

**Architecture:** Extract shared UI components (API client, CSS, 3D viewer, summary cards, result tabs) into `frontend/components/`. Create two pages under `frontend/pages/`: the existing single-screen flow and a new session analysis page. The entry point (`streamlit_app.py`) uses `st.navigation` to switch between pages. The session analysis page calls `GET /api/session/{id}` and `GET /api/session/{id}/alerts` to display convergence/perturbation details.

**Tech Stack:** Streamlit (multipage via `st.navigation`), pandas, py3Dmol, requests, existing FastAPI backend.

**Note:** The current `streamlit_app.py` has unresolved git merge conflicts (lines 119-282 and 725-801) in the `render_protein_3d` function and the Structure tab. These must be resolved first (keeping the newer overlay-capable version) before extraction.

---

### Task 0: Resolve merge conflicts in streamlit_app.py

**Files:**
- Modify: `frontend/streamlit_app.py:119-282` and `frontend/streamlit_app.py:725-801`

**Step 1: Resolve merge conflicts**

Keep the `>>>>>>> 3d2df1b` (overlay-capable) version for both conflict regions. Remove all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).

For the `render_protein_3d` function (lines 119-282): keep the version that uses `{"model": 0}` selectors and supports `overlay_pdb`.

For the Structure tab legend (lines 725-801): keep the version that shows overlay stats and comparison panel.

**Step 2: Verify the file has no remaining conflict markers**

Run: `grep -n '<<<<<<\|======\|>>>>>>' frontend/streamlit_app.py`
Expected: no output

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "fix: resolve merge conflicts in streamlit_app.py"
```

---

### Task 1: Create directory structure and `__init__.py` files

**Files:**
- Create: `frontend/components/__init__.py`
- Create: `frontend/pages/__init__.py`

**Step 1: Create directories and init files**

```bash
mkdir -p frontend/components frontend/pages
touch frontend/components/__init__.py frontend/pages/__init__.py
```

**Step 2: Commit**

```bash
git add frontend/components/__init__.py frontend/pages/__init__.py
git commit -m "chore: create components and pages directories"
```

---

### Task 2: Extract API client helpers

**Files:**
- Create: `frontend/components/api_client.py`

**Step 1: Create `frontend/components/api_client.py`**

Extract from `streamlit_app.py`:
- `API_BASE_URL` constant
- `DEMO_SEQUENCES` dict
- `check_api_health()` function (lines 37-45)
- `screen_sequence()` function (lines 48-78)

Add two new helpers:

```python
def get_session_state(session_id: str) -> dict | None:
    """Fetch session state from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/session/{session_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def get_session_alerts(session_id: str) -> dict | None:
    """Fetch anomaly alerts for a session."""
    try:
        resp = requests.get(f"{API_BASE_URL}/session/{session_id}/alerts", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None
```

**Step 2: Verify imports work**

Run: `cd frontend && python -c "from components.api_client import check_api_health, screen_sequence, get_session_alerts, DEMO_SEQUENCES; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/components/api_client.py
git commit -m "refactor: extract API client helpers into components/api_client.py"
```

---

### Task 3: Extract CSS styles

**Files:**
- Create: `frontend/components/styles.py`

**Step 1: Create `frontend/components/styles.py`**

Move the `inject_custom_css()` function (lines 292-426 of streamlit_app.py) verbatim. It uses only `streamlit` — no other local imports needed.

```python
import streamlit as st

def inject_custom_css() -> None:
    # ... exact CSS from original ...
```

**Step 2: Verify import**

Run: `cd frontend && python -c "from components.styles import inject_custom_css; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/components/styles.py
git commit -m "refactor: extract CSS styles into components/styles.py"
```

---

### Task 4: Extract 3D protein viewer

**Files:**
- Create: `frontend/components/protein_3d.py`

**Step 1: Create `frontend/components/protein_3d.py`**

Move from `streamlit_app.py`:
- `_aligned_residue_set()` helper (lines 81-87)
- `render_protein_3d()` function (lines 90-289, post-conflict-resolution)

Imports needed: `py3Dmol`, `streamlit.components.v1 as components`.

**Step 2: Verify import**

Run: `cd frontend && python -c "from components.protein_3d import render_protein_3d; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/components/protein_3d.py
git commit -m "refactor: extract 3D protein viewer into components/protein_3d.py"
```

---

### Task 5: Extract summary cards

**Files:**
- Create: `frontend/components/summary_cards.py`

**Step 1: Create `frontend/components/summary_cards.py`**

Move `render_summary_cards()` (lines 429-486) verbatim. Only needs `streamlit`.

**Step 2: Verify import**

Run: `cd frontend && python -c "from components.summary_cards import render_summary_cards; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/components/summary_cards.py
git commit -m "refactor: extract summary cards into components/summary_cards.py"
```

---

### Task 6: Extract result viewer (6-tab display)

**Files:**
- Create: `frontend/components/result_viewer.py`

**Step 1: Create `frontend/components/result_viewer.py`**

Create a `render_results(data: dict) -> None` function that contains the entire tabbed results section (lines 601-1013 of streamlit_app.py): Overview, Matches, Structure, Score Breakdown, Function, Explain tabs, plus the Copy JSON button and video generation section.

Imports needed:
- `json`, `streamlit as st`, `streamlit.components.v1 as components`, `pandas as pd`, `requests`
- `from components.summary_cards import render_summary_cards`
- `from components.protein_3d import render_protein_3d, _aligned_residue_set`
- `from components.api_client import API_BASE_URL`
- `from video_generator import ProteinVideoData, generate_video`

The function takes the result `data` dict and renders everything. It also needs access to `st.session_state` for compare caching and query count (already available via st.session_state).

**Step 2: Verify import**

Run: `cd frontend && python -c "from components.result_viewer import render_results; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/components/result_viewer.py
git commit -m "refactor: extract result viewer tabs into components/result_viewer.py"
```

---

### Task 7: Create single screen page

**Files:**
- Create: `frontend/pages/single_screen.py`

**Step 1: Create the single screen page**

This page contains the current main flow from `streamlit_app.py` lines 489-1023, but using imported components:

```python
import uuid
import streamlit as st

from components.api_client import (
    check_api_health, screen_sequence, DEMO_SEQUENCES,
)
from components.styles import inject_custom_css
from components.result_viewer import render_results


def page():
    inject_custom_css()

    # Session state init
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Header with API status
    # ... (same as current lines 508-526)

    # Input strip
    # ... (same as current lines 529-594)

    # Results
    data = st.session_state.last_result
    if data:
        st.divider()
        render_results(data)

    # Footer
    # ... (same as current lines 1016-1023)
```

**Step 2: Verify import**

Run: `cd frontend && python -c "from pages.single_screen import page; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/pages/single_screen.py
git commit -m "refactor: create single screen page using shared components"
```

---

### Task 8: Build session analysis page

**Files:**
- Create: `frontend/pages/session_analysis.py`

**Step 1: Create `frontend/pages/session_analysis.py`**

This is the new page. Key sections:

```python
import uuid
import streamlit as st
import pandas as pd

from components.api_client import (
    check_api_health, screen_sequence, get_session_alerts,
    DEMO_SEQUENCES,
)
from components.styles import inject_custom_css
from components.result_viewer import render_results


def page():
    inject_custom_css()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_results" not in st.session_state:
        st.session_state.session_results = []  # list of (label, result_data) tuples

    # Header + API health check (same pattern as single screen)

    st.markdown("# Session Analysis")
    st.markdown("Screen multiple sequences sequentially and monitor for behavioral patterns.")

    # --- Input section ---
    st.markdown("### Input Sequences")
    input_mode = st.radio(
        "Input mode",
        ["Paste sequences", "Use demo sequences"],
        horizontal=True,
        label_visibility="collapsed",
    )

    sequences_to_run = []  # list of (label, sequence) tuples

    if input_mode == "Paste sequences":
        raw = st.text_area(
            "Sequences (one per block, separate with blank line or FASTA headers)",
            height=200,
            placeholder=">Seq1\nMVLS...\n\n>Seq2\nDVSF...",
        )
        if raw.strip():
            # Parse FASTA-like blocks
            sequences_to_run = _parse_multi_sequences(raw)
            st.caption(f"{len(sequences_to_run)} sequences detected")
    else:
        selected = st.multiselect(
            "Select demo sequences",
            [k for k in DEMO_SEQUENCES if not k.startswith("--")],
        )
        sequences_to_run = [(name, DEMO_SEQUENCES[name]) for name in selected]

    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, key="sa_topk")

    # --- Run button ---
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        run_all = st.button(
            f"Screen {len(sequences_to_run)} Sequences",
            type="primary",
            disabled=len(sequences_to_run) == 0,
            use_container_width=True,
        )

    # --- Execute sequentially ---
    if run_all and sequences_to_run:
        progress = st.progress(0, text="Starting...")
        results = []
        for i, (label, seq) in enumerate(sequences_to_run):
            progress.progress(
                (i) / len(sequences_to_run),
                text=f"Screening {i+1}/{len(sequences_to_run)}: {label[:40]}...",
            )
            result = screen_sequence(
                sequence=seq,
                session_id=st.session_state.session_id,
                sequence_id=label,
                top_k=top_k,
            )
            if result["success"]:
                results.append((label, result["data"]))
            else:
                results.append((label, {"error": result["error"]}))
        progress.progress(1.0, text="Done!")
        st.session_state.session_results = results
        st.rerun()

    # --- Results display ---
    results = st.session_state.session_results
    if results:
        st.divider()

        # Anomaly alert banner
        alerts = get_session_alerts(st.session_state.session_id)
        if alerts and alerts["anomaly_score"] > 0.3:
            if alerts["anomaly_score"] > 0.5:
                st.error(f"⚠ Session Alert (score: {alerts['anomaly_score']:.2f}): {alerts['explanation']}")
            else:
                st.warning(f"Elevated activity (score: {alerts['anomaly_score']:.2f}): {alerts['explanation']}")

        # Summary table + risk trend
        col_table, col_chart = st.columns([3, 2])

        with col_table:
            st.markdown("### Results Summary")
            table_data = []
            for i, (label, data) in enumerate(results, 1):
                if "error" in data:
                    table_data.append({"#": i, "Sequence": label[:50], "Risk Score": None, "Risk Level": "ERROR", "Top Match": data["error"]})
                else:
                    top = data.get("top_matches", [{}])[0] if data.get("top_matches") else {}
                    table_data.append({
                        "#": i,
                        "Sequence": label[:50],
                        "Risk Score": data["risk_score"],
                        "Risk Level": data["risk_level"],
                        "Top Match": top.get("name", "None"),
                    })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col_chart:
            st.markdown("### Risk Trend")
            scores = [d["risk_score"] for _, d in results if "risk_score" in d]
            if scores:
                chart_df = pd.DataFrame({"Query #": range(1, len(scores)+1), "Risk Score": scores})
                st.line_chart(chart_df.set_index("Query #"))

        # Convergence / Perturbation detail cards
        if alerts:
            st.markdown("### Session Monitoring")
            col_c, col_p = st.columns(2)

            conv = alerts["convergence"]
            with col_c:
                st.markdown("**Convergence Detector**")
                flagged_label = "FLAGGED" if conv["is_flagged"] else "Normal"
                st.metric("Status", flagged_label)
                st.metric("Mean Similarity", f"{conv['mean_similarity']:.3f}")
                st.metric("Trend", f"{conv['similarity_trend']:+.3f}")
                st.caption(f"Window: {conv['window_size']} entries | Threshold: 0.75")

            pert = alerts["perturbation"]
            with col_p:
                st.markdown("**Perturbation Detector**")
                flagged_label = "FLAGGED" if pert["is_flagged"] else "Normal"
                st.metric("Status", flagged_label)
                st.metric("Clusters", str(pert["cluster_count"]))
                st.metric("Max Cluster", str(pert["max_cluster_size"]))
                st.caption(f"High-sim pairs: {len(pert['high_sim_pairs'])}")

        # Expandable per-sequence detail
        st.markdown("### Detailed Results")
        for i, (label, data) in enumerate(results):
            if "error" in data:
                st.error(f"**{label}**: {data['error']}")
                continue
            risk = data["risk_score"]
            level = data["risk_level"]
            with st.expander(f"**{label}** — Risk: {risk:.3f} ({level})"):
                render_results(data)


def _parse_multi_sequences(raw: str) -> list[tuple[str, str]]:
    """Parse multi-FASTA or blank-line-separated sequences."""
    sequences = []
    lines = raw.strip().split("\n")
    current_label = ""
    current_seq = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                seq = "".join(current_seq)
                label = current_label or f"Sequence {len(sequences)+1}"
                sequences.append((label, seq))
            current_label = line[1:].strip() or f"Sequence {len(sequences)+1}"
            current_seq = []
        elif line == "":
            if current_seq:
                seq = "".join(current_seq)
                label = current_label or f"Sequence {len(sequences)+1}"
                sequences.append((label, seq))
                current_label = ""
                current_seq = []
        else:
            current_seq.append(line)

    if current_seq:
        seq = "".join(current_seq)
        label = current_label or f"Sequence {len(sequences)+1}"
        sequences.append((label, seq))

    return sequences
```

**Step 2: Verify import**

Run: `cd frontend && python -c "from pages.session_analysis import page; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add frontend/pages/session_analysis.py
git commit -m "feat: add session analysis page with sequential screening and monitoring dashboard"
```

---

### Task 9: Rewrite entry point with st.navigation

**Files:**
- Modify: `frontend/streamlit_app.py`

**Step 1: Replace `streamlit_app.py` contents**

The entry point becomes minimal:

```python
"""BioScreen — Structure-based biosecurity screening for AI-designed proteins."""

import streamlit as st

single_screen = st.Page("pages/single_screen.py", title="Single Screen", icon="🔬", default=True)
session_analysis = st.Page("pages/session_analysis.py", title="Session Analysis", icon="📊")

nav = st.navigation([single_screen, session_analysis])
st.set_page_config(
    page_title="BioScreen",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
nav.run()
```

**Step 2: Run the app and verify both pages load**

Run: `cd frontend && streamlit run streamlit_app.py`
Expected: App opens, sidebar shows "Single Screen" and "Session Analysis" navigation. Single Screen page behaves identically to the original.

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "refactor: rewrite entry point as multipage navigation"
```

---

### Task 10: End-to-end smoke test

**Step 1: Start the API server**

Run: `uvicorn app.main:app --reload`

**Step 2: Start the frontend**

Run: `cd frontend && streamlit run streamlit_app.py`

**Step 3: Test single screen page**

1. Select a demo sequence (e.g., Trichosanthin)
2. Click "Screen Sequence"
3. Verify all 6 tabs render correctly
4. Verify video generation button appears

**Step 4: Test session analysis page**

1. Switch to "Session Analysis" in nav
2. Select 3+ demo sequences from multiselect
3. Click "Screen N Sequences"
4. Verify progress bar advances
5. Verify results summary table shows all sequences
6. Verify risk trend chart renders
7. Verify convergence/perturbation cards appear
8. Expand a result and verify full detail renders
9. If 5+ similar sequences screened, verify anomaly alert banner appears

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify multipage frontend end-to-end"
```

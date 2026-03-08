# Dashboard Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Streamlit frontend for visual clarity — light mode, no sidebar, summary cards, tabbed detail area with 5 tabs (Matches, Structure, Score Breakdown, Function, Explain).

**Architecture:** Single-file rewrite of `frontend/streamlit_app.py`. No backend changes. All existing helper functions (`check_api_health`, `screen_sequence`, `render_protein_3d`, `render_risk_gauge`, `render_session_monitor`) are preserved or adapted. Layout shifts from sidebar+columns to top-down flow with inline controls and `st.tabs()`.

**Tech Stack:** Streamlit, py3Dmol, pandas, custom HTML/CSS via `unsafe_allow_html`

---

### Task 1: Inject Global CSS and Rewrite Header

**Files:**
- Modify: `frontend/streamlit_app.py:210-282` (the `main()` function start)

**Step 1: Add CSS injection function**

Add this function before `main()`:

```python
def inject_custom_css() -> None:
    """Inject custom CSS for clean light-mode dashboard styling."""
    st.markdown("""
    <style>
    /* Card styling */
    .summary-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }
    .summary-card h3 {
        margin: 0 0 0.25rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    /* Risk colors */
    .risk-low { color: #22c55e; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    /* Risk gauge bar */
    .risk-bar-bg {
        background: #e2e8f0;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
        margin-top: 6px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s;
    }
    /* API status pill */
    .api-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 2px 10px;
    }
    .api-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .api-dot-ok { background: #22c55e; }
    .api-dot-err { background: #ef4444; }
    /* Score breakdown bars */
    .score-bar-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .score-bar-label {
        width: 160px;
        font-size: 0.85rem;
        color: #475569;
        text-align: right;
        flex-shrink: 0;
    }
    .score-bar-track {
        flex: 1;
        background: #e2e8f0;
        border-radius: 4px;
        height: 20px;
        overflow: hidden;
        position: relative;
    }
    .score-bar-value {
        height: 100%;
        border-radius: 4px;
        background: #6366f1;
    }
    .score-bar-num {
        width: 50px;
        font-size: 0.85rem;
        color: #334155;
        font-weight: 600;
        flex-shrink: 0;
    }
    /* Explain tab */
    .verdict-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }
    .verdict-low { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .verdict-medium { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .verdict-high { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .recommend-box {
        background: #f8fafc;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 1rem;
    }
    /* Function cards */
    .func-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
    }
    .func-card-id {
        font-family: monospace;
        font-size: 0.8rem;
        color: #6366f1;
    }
    .func-card-name {
        font-size: 0.9rem;
        color: #1e293b;
        font-weight: 500;
    }
    .conf-bar-bg {
        background: #e2e8f0;
        border-radius: 3px;
        height: 6px;
        margin-top: 4px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: #6366f1;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Reduce top padding */
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)
```

**Step 2: Rewrite the header and API status**

Replace the `main()` function opening (title, sidebar API check) with:

```python
def main():
    st.set_page_config(
        page_title="BioScreen",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    inject_custom_css()

    # Persistent state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Header with API status pill
    health = check_api_health()
    api_ok = health.get("status") == "ok"
    dot_class = "api-dot-ok" if api_ok else "api-dot-err"
    api_text = "API Connected" if api_ok else "API Unavailable"

    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.markdown("# BioScreen")
        st.markdown("Structure-based biosecurity screening for AI-designed proteins")
    with col_status:
        st.markdown(
            f'<div style="text-align:right; margin-top:1.5rem;">'
            f'<span class="api-pill"><span class="api-dot {dot_class}"></span>{api_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if not api_ok:
        st.error(f"Cannot reach API: {health.get('message', 'Unknown error')}")
        st.stop()
```

**Step 3: Verify it renders**

Run: `streamlit run frontend/streamlit_app.py` and confirm:
- Title "BioScreen" renders without emoji
- API status pill appears top-right (green dot if server running, red if not)
- No sidebar visible

**Step 4: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add custom CSS and rewrite header with API status pill"
```

---

### Task 2: Rewrite Input Strip (Inline Controls, No Sidebar)

**Files:**
- Modify: `frontend/streamlit_app.py` — replace sidebar + input column with inline input strip

**Step 1: Write inline input section**

Replace the sidebar block and `col_input, col_gauge` section with:

```python
    # ── Input Strip ──────────────────────────────────────────
    col_seq, col_controls = st.columns([3, 2])

    with col_seq:
        sequence_input = st.text_area(
            "Protein Sequence",
            height=100,
            placeholder="Paste protein sequence here (FASTA headers auto-stripped)...",
            value=getattr(st.session_state, "example_sequence", ""),
            label_visibility="collapsed",
        )
        sequence_id = st.text_input(
            "Sequence ID",
            placeholder="Optional sequence ID",
            label_visibility="collapsed",
        )

    with col_controls:
        demo_choice = st.selectbox(
            "Demo sequence",
            options=list(DEMO_SEQUENCES.keys()),
            label_visibility="collapsed",
        )
        if DEMO_SEQUENCES.get(demo_choice):
            st.session_state.example_sequence = DEMO_SEQUENCES[demo_choice]

        c1, c2 = st.columns(2)
        with c1:
            run_structure = st.toggle(
                "Structure analysis",
                value=False,
                help="Include ESMFold + Foldseek (slower, more accurate)",
            )
        with c2:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, label_visibility="collapsed")

        screen_button = st.button(
            "Screen Sequence",
            type="primary",
            disabled=not sequence_input.strip(),
            use_container_width=True,
        )

    # Validation feedback
    if sequence_input.strip():
        cleaned = sequence_input.strip().replace("\n", "").replace("\r", "")
        if cleaned.startswith(">"):
            cleaned = "".join(cleaned.split("\n")[1:])
        seq_len = len(cleaned)
        msg = f"{seq_len} amino acids"
        if seq_len < 10:
            st.warning(f"Sequence very short ({seq_len} aa) — results may be unreliable.")
        elif seq_len > 1000:
            st.warning(f"Long sequence ({seq_len} aa) — embeddings may be truncated.")
        else:
            st.caption(msg)
```

**Step 2: Keep the screening execution block unchanged**

The `if screen_button and sequence_input.strip():` block stays as-is.

**Step 3: Verify it renders**

Run: `streamlit run frontend/streamlit_app.py` and confirm:
- Sequence textarea on left, controls on right
- Demo dropdown, structure toggle, top-k, screen button all inline
- No sidebar

**Step 4: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): inline input strip replaces sidebar"
```

---

### Task 3: Summary Cards Section

**Files:**
- Modify: `frontend/streamlit_app.py` — add summary cards between input and tabs

**Step 1: Add summary card rendering functions**

```python
def render_summary_cards(data: dict) -> None:
    """Render the three summary cards: Risk Score, Top Match, Screening Mode."""
    risk_score = data["risk_score"]
    risk_level = data["risk_level"]

    # Determine colors
    if risk_score < 0.45:
        bar_color, level_class = "#22c55e", "risk-low"
    elif risk_score < 0.75:
        bar_color, level_class = "#f59e0b", "risk-medium"
    else:
        bar_color, level_class = "#ef4444", "risk-high"

    # Top match info
    top_match = data.get("top_matches", [{}])[0] if data.get("top_matches") else {}
    match_name = top_match.get("name", "No match")
    match_org = top_match.get("organism", "")
    emb_sim = top_match.get("embedding_similarity", 0)
    str_sim = top_match.get("structure_similarity")
    best_sim = max(emb_sim, str_sim or 0)
    sim_label = "structure" if (str_sim and str_sim >= emb_sim) else "embedding"

    # Screening mode
    structure_ran = data.get("structure_predicted", False)
    mode_label = "Full" if structure_ran else "Fast"
    mode_detail = "Embedding + Structure + Function" if structure_ran else "Embedding + Function"

    pct = int(risk_score * 100)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="summary-card">
            <h3>Risk Score</h3>
            <div style="font-size:2.2rem; font-weight:700; color:{bar_color};">{risk_score:.3f}</div>
            <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{pct}%; background:{bar_color};"></div></div>
            <div style="margin-top:6px;">
                <span class="{level_class}" style="font-weight:700; font-size:0.95rem;">{risk_level}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="summary-card">
            <h3>Top Match</h3>
            <div style="font-size:1.1rem; font-weight:600; color:#1e293b; margin-bottom:2px;">{match_name}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-bottom:6px;">{match_org}</div>
            <div style="font-size:1.3rem; font-weight:700; color:#334155;">{best_sim:.3f}
                <span style="font-size:0.75rem; font-weight:400; color:#94a3b8;">({sim_label})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="summary-card">
            <h3>Screening Mode</h3>
            <div style="font-size:1.3rem; font-weight:700; color:#334155;">{mode_label}</div>
            <div style="font-size:0.85rem; color:#64748b; margin-top:4px;">{mode_detail}</div>
        </div>
        """, unsafe_allow_html=True)
```

**Step 2: Wire it into main() after screening results are available**

```python
    data = st.session_state.last_result
    if data:
        st.divider()
        render_summary_cards(data)
```

**Step 3: Verify it renders**

Run the app, screen a sequence, confirm three cards appear in a row with risk score (colored bar), top match name + similarity, and screening mode.

**Step 4: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add summary cards — risk score, top match, screening mode"
```

---

### Task 4: Tabbed Results — Matches Tab

**Files:**
- Modify: `frontend/streamlit_app.py` — add tabs section, implement Matches tab

**Step 1: Create tab container and Matches tab**

After summary cards, replace the current 3D viewer + matches + function sections with:

```python
        # ── Tabbed Detail Area ────────────────────────────────
        has_structure = data.get("pdb_string") is not None
        tab_labels = ["Matches", "Structure", "Score Breakdown", "Function", "Explain"]
        tabs = st.tabs(tab_labels)

        # Tab: Matches
        with tabs[0]:
            if data.get("top_matches"):
                matches_data = []
                for i, match in enumerate(data["top_matches"], 1):
                    matches_data.append({
                        "Rank": i,
                        "Name": match["name"],
                        "Organism": match["organism"],
                        "Toxin Type": match["toxin_type"],
                        "Embedding Sim": match["embedding_similarity"],
                        "Structure Sim": match.get("structure_similarity") if match.get("structure_similarity") is not None else None,
                    })
                df = pd.DataFrame(matches_data)

                # Column config for colored similarity bars
                col_config = {
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Name": st.column_config.TextColumn(width="medium"),
                    "Organism": st.column_config.TextColumn(width="medium"),
                    "Toxin Type": st.column_config.TextColumn(width="small"),
                    "Embedding Sim": st.column_config.ProgressColumn(
                        "Embedding Sim", min_value=0, max_value=1, format="%.3f",
                    ),
                    "Structure Sim": st.column_config.ProgressColumn(
                        "Structure Sim", min_value=0, max_value=1, format="%.3f",
                    ) if has_structure else st.column_config.TextColumn("Structure Sim"),
                }

                # Replace None with "—" for display when no structure
                if not has_structure:
                    df["Structure Sim"] = "—"

                st.dataframe(df, column_config=col_config, use_container_width=True, hide_index=True)
            else:
                st.info("No significant matches found.")
```

**Step 2: Verify**

Run app, screen a sequence, confirm Matches tab shows full-width table with progress-bar columns for similarity scores. Names are not truncated.

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add tabbed layout with Matches tab"
```

---

### Task 5: Structure Tab (Move Existing 3D Viewer)

**Files:**
- Modify: `frontend/streamlit_app.py` — populate Structure tab

**Step 1: Add Structure tab content**

```python
        # Tab: Structure
        with tabs[1]:
            pdb_string = data.get("pdb_string")
            if pdb_string:
                col_view, col_color = st.columns(2)
                with col_view:
                    view_style = st.radio(
                        "View",
                        ["Cartoon", "Surface", "Stick"],
                        horizontal=True,
                        key="view_style",
                    )
                with col_color:
                    color_mode = st.radio(
                        "Color",
                        ["Default", "pLDDT"],
                        horizontal=True,
                        key="color_mode",
                        help="pLDDT: ESMFold confidence (blue=high, red=low)",
                    )

                pocket_res = data.get("pocket_residues", [])
                danger_res = data.get("danger_residues", [])

                render_protein_3d(
                    pdb_string=pdb_string,
                    pocket_residues=pocket_res,
                    danger_residues=danger_res,
                    view_style=view_style,
                    color_mode=color_mode,
                    width=800,
                    height=500,
                )

                # Legend
                legend_parts = ["Blue: normal structure"]
                if pocket_res:
                    legend_parts.append(f"Orange: active site pocket ({len(pocket_res)} residues)")
                if danger_res:
                    legend_parts.append(f"Red: danger residues ({len(danger_res)} residues)")
                st.caption(" | ".join(legend_parts))
            else:
                st.info("Structure analysis was not run. Enable 'Structure analysis' and re-screen to see the 3D viewer.")
```

**Step 2: Verify**

Run app with structure analysis enabled. Confirm 3D viewer appears inside Structure tab with toggles and legend. Confirm the tab shows the info message when structure analysis is off.

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): move 3D structure viewer into Structure tab"
```

---

### Task 6: Score Breakdown Tab

**Files:**
- Modify: `frontend/streamlit_app.py` — populate Score Breakdown tab

**Step 1: Add Score Breakdown tab content**

```python
        # Tab: Score Breakdown
        with tabs[2]:
            factors = data.get("risk_factors", {})

            emb_sim = factors.get("max_embedding_similarity", 0)
            struct_sim = factors.get("max_structure_similarity")
            func_overlap = factors.get("function_overlap", 0)
            structure_ran = data.get("structure_predicted", False)

            # Determine which weight set was used
            if structure_ran and struct_sim is not None:
                weight_set = {"Embedding": 0.50, "Structure": 0.30, "Function": 0.20}
                weight_note = "Full path weights (embedding 0.50, structure 0.30, function 0.20)"
            else:
                weight_set = {"Embedding": 0.65, "Function": 0.35}
                weight_note = "Fast path weights (embedding 0.65, function 0.35)"

            st.markdown(f"**Weight set:** {weight_note}")
            st.markdown("")

            # Build bars
            components_list = [
                ("Embedding Similarity", emb_sim, weight_set.get("Embedding", 0)),
            ]
            if structure_ran:
                components_list.append(
                    ("Structure Similarity", struct_sim if struct_sim is not None else 0, weight_set.get("Structure", 0)),
                )
            components_list.append(
                ("Function Overlap", func_overlap, weight_set.get("Function", 0)),
            )

            for label, raw_val, weight in components_list:
                pct = int(raw_val * 100)
                weighted = raw_val * weight
                st.markdown(f"""
                <div class="score-bar-row">
                    <span class="score-bar-label">{label}</span>
                    <div class="score-bar-track">
                        <div class="score-bar-value" style="width:{pct}%;"></div>
                    </div>
                    <span class="score-bar-num">{raw_val:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"  Weight: {weight:.2f}  |  Contribution: {weighted:.3f}")

            # Synergy bonus
            top_match_count = factors.get("top_match_count", 0)
            risk_score = data["risk_score"]
            total_weighted = sum(raw * w for _, raw, w in components_list)
            bonus = max(0, risk_score - min(1.0, total_weighted))
            if bonus > 0.01:
                st.markdown(f"""
                <div class="score-bar-row">
                    <span class="score-bar-label">Synergy Bonus</span>
                    <div class="score-bar-track">
                        <div class="score-bar-value" style="width:{int(bonus*100)}%; background:#a78bfa;"></div>
                    </div>
                    <span class="score-bar-num">+{bonus:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption("  Bonus for multiple high-confidence signals")

            st.markdown(f"**Final Score: {risk_score:.3f}**")
```

**Step 2: Verify**

Run app, screen a sequence, open Score Breakdown tab. Confirm horizontal bars render with correct values and weights. Check fast vs full path shows different weight sets.

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add Score Breakdown tab with component bars"
```

---

### Task 7: Function Tab

**Files:**
- Modify: `frontend/streamlit_app.py` — populate Function tab

**Step 1: Add Function tab content**

```python
        # Tab: Function
        with tabs[3]:
            function_pred = data.get("function_prediction")
            if function_pred:
                # Summary callout
                summary = function_pred.get("summary", "")
                if summary:
                    st.markdown(f"""
                    <div class="recommend-box">{summary}</div>
                    """, unsafe_allow_html=True)

                # GO Terms
                go_terms = function_pred.get("go_terms", [])
                if go_terms:
                    st.markdown("**GO Terms**")
                    for term in go_terms:
                        term_id = term.get("term", "Unknown")
                        name = term.get("name", "")
                        conf = term.get("confidence", "0")
                        conf_float = float(conf) if conf else 0
                        conf_pct = int(conf_float * 100)
                        st.markdown(f"""
                        <div class="func-card">
                            <span class="func-card-id">{term_id}</span>
                            <span class="func-card-name" style="margin-left:8px;">{name}</span>
                            <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                                <div class="conf-bar-bg" style="flex:1;">
                                    <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
                                </div>
                                <span style="font-size:0.75rem; color:#64748b;">{conf_float:.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # EC Numbers
                ec_numbers = function_pred.get("ec_numbers", [])
                if ec_numbers:
                    st.markdown("**EC Numbers**")
                    for ec in ec_numbers:
                        number = ec.get("number", "Unknown")
                        conf = ec.get("confidence", "0")
                        conf_float = float(conf) if conf else 0
                        conf_pct = int(conf_float * 100)
                        st.markdown(f"""
                        <div class="func-card">
                            <span class="func-card-id">{number}</span>
                            <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                                <div class="conf-bar-bg" style="flex:1;">
                                    <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
                                </div>
                                <span style="font-size:0.75rem; color:#64748b;">{conf_float:.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                if not go_terms and not ec_numbers:
                    st.info("No GO terms or EC numbers predicted.")
            else:
                st.info("No function prediction available.")
```

**Step 2: Verify**

Run app, screen a sequence, open Function tab. Confirm GO terms and EC numbers render as styled cards with confidence bars.

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add Function tab with GO/EC confidence cards"
```

---

### Task 8: Explain Tab

**Files:**
- Modify: `frontend/streamlit_app.py` — populate Explain tab

**Step 1: Add Explain tab content**

```python
        # Tab: Explain
        with tabs[4]:
            risk_score = data["risk_score"]
            risk_level = data["risk_level"]
            factors = data.get("risk_factors", {})
            explanation = factors.get("score_explanation", "")

            # Verdict box
            if risk_score >= 0.75:
                verdict_class = "verdict-high"
            elif risk_score >= 0.45:
                verdict_class = "verdict-medium"
            else:
                verdict_class = "verdict-low"

            # Parse explanation into parts (split on ". ")
            parts = [p.strip() for p in explanation.split(". ") if p.strip()]
            verdict_text = parts[0] if parts else f"{risk_level} RISK"

            st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)

            # Factor details
            if len(parts) > 1:
                # The second part usually starts with "Factors: "
                for part in parts[1:]:
                    if part.startswith("Factors:"):
                        factor_text = part[len("Factors:"):].strip()
                        factor_items = [f.strip() for f in factor_text.split(";") if f.strip()]
                        for item in factor_items:
                            st.markdown(f"- {item}")
                    elif part.startswith("Recommend"):
                        st.markdown(f'<div class="recommend-box">{part}.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {part}")

            # Session anomaly
            anomaly_score = factors.get("session_anomaly_score", 0.0)
            query_count = st.session_state.get("query_count", 0)
            if anomaly_score > 0 or query_count > 1:
                st.markdown("---")
                st.markdown("**Session Monitoring**")
                if anomaly_score > 0.5:
                    st.warning(f"Session anomaly score: {anomaly_score:.2f} — convergent optimization pattern detected across {query_count} queries.")
                elif anomaly_score > 0.3:
                    st.info(f"Session anomaly score: {anomaly_score:.2f} — elevated activity across {query_count} queries.")
                else:
                    st.caption(f"Session anomaly score: {anomaly_score:.2f} (normal) | {query_count} queries this session")

            # Warnings
            warnings = data.get("warnings", [])
            if warnings:
                st.markdown("---")
                st.markdown("**Warnings**")
                for w in warnings:
                    st.warning(w)
```

**Step 2: Verify**

Run app, screen sequences at different risk levels. Confirm:
- Verdict box is green for LOW, amber for MEDIUM, red for HIGH
- Factors render as bullet list
- Recommendation renders in styled callout
- Session anomaly shows after multiple queries

**Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): add Explain tab with verdict, factors, recommendations"
```

---

### Task 9: Cleanup and Footer

**Files:**
- Modify: `frontend/streamlit_app.py` — remove old sections, add footer and copy-JSON button

**Step 1: Remove old code**

Delete:
- The old `render_session_monitor` function call and section
- The old `render_risk_gauge` call in the col_gauge section
- The old `📋 Risk Factor Breakdown` expander
- The old `🧬 Function Prediction` section
- The old `📡 Session Monitoring` section
- The old `format_function_prediction` function (no longer used)
- The old sidebar block entirely

Keep:
- `render_protein_3d` function (used in Structure tab)
- `render_risk_gauge` function (can remove if not called anywhere)
- `render_session_monitor` function (can remove if not called anywhere)
- `check_api_health`, `screen_sequence` functions
- `get_risk_level_color`, `get_risk_score_color` helper functions

**Step 2: Add Copy JSON button and footer after tabs**

```python
        # Copy JSON button
        col_spacer, col_json = st.columns([5, 1])
        with col_json:
            if st.button("Copy JSON", key="copy_json"):
                st.code(json.dumps(data, indent=2), language="json")

    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align:center; color:#94a3b8; font-size:0.8rem;">'
        'BioScreen — Structure-based biosecurity screening for AI-designed proteins. '
        'For research purposes only. Always validate results with experimental methods.'
        '</div>',
        unsafe_allow_html=True,
    )
```

**Step 3: Verify full flow**

Run the app end-to-end:
1. Page loads with header + API pill
2. Input strip with inline controls
3. Screen a sequence → summary cards appear
4. All 5 tabs work: Matches, Structure, Score Breakdown, Function, Explain
5. Copy JSON button works
6. Footer renders clean

**Step 4: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): cleanup old layout, add footer and copy-JSON"
```

---

### Task 10: Final Visual Polish

**Files:**
- Modify: `frontend/streamlit_app.py` — minor spacing/alignment fixes

**Step 1: Visual review and fix**

Review the full page and fix:
- Card heights should be consistent across the 3 summary cards
- Tab content should have consistent top padding
- Table column widths should feel balanced
- The 3D viewer width should fill the tab container (use `width=-1` or measure container)

**Step 2: Run full manual test**

Test with all demo sequences:
- Insulin (LOW) → green card, no structure tab content
- Scorpion toxin (HIGH) → red card
- AI-evasion (MEDIUM-HIGH) → amber card
- Multiple submissions → session anomaly shows in Explain tab

**Step 3: Final commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat(frontend): visual polish — consistent spacing, card heights, tab padding"
```

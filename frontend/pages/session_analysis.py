"""Session Analysis page — sequential multi-sequence screening with behavioral monitoring."""

import uuid

import streamlit as st
import pandas as pd

from components.api_client import (
    check_api_health,
    screen_sequence,
    get_session_alerts,
    DEMO_CONVERGENCE_SERIES,
)
from components.styles import inject_custom_css
from components.result_viewer import render_results


def _parse_multi_sequences(raw: str) -> list[tuple[str, str]]:
    """Parse FASTA-like text into a list of (label, sequence) tuples.

    Lines starting with ``>`` are treated as headers/labels.  Blank lines
    separate sequences.  Consecutive non-header, non-blank lines are
    concatenated into a single sequence string.
    """
    sequences: list[tuple[str, str]] = []
    current_label: str | None = None
    current_seq_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            # Blank line — flush current sequence if any
            if current_seq_lines:
                label = current_label or f"Sequence {len(sequences) + 1}"
                sequences.append((label, "".join(current_seq_lines)))
                current_label = None
                current_seq_lines = []
            continue
        if stripped.startswith(">"):
            # New header — flush previous sequence first
            if current_seq_lines:
                label = current_label or f"Sequence {len(sequences) + 1}"
                sequences.append((label, "".join(current_seq_lines)))
                current_seq_lines = []
            current_label = stripped[1:].strip() or f"Sequence {len(sequences) + 1}"
        else:
            current_seq_lines.append(stripped)

    # Flush any remaining sequence
    if current_seq_lines:
        label = current_label or f"Sequence {len(sequences) + 1}"
        sequences.append((label, "".join(current_seq_lines)))

    return sequences


def page() -> None:
    """Render the Session Analysis page."""
    inject_custom_css()

    # Goto navigation
    nav_cols = st.columns([1, 1, 6])
    with nav_cols[0]:
        st.page_link("pages/single_screen.py", label="Single Screen", icon="\U0001f52c")
    with nav_cols[1]:
        st.page_link("pages/session_analysis.py", label="Session Analysis", icon="\U0001f4ca")

    # ------------------------------------------------------------------
    # 1. Header + health check
    # ------------------------------------------------------------------
    st.title("Orthogon — Session Analysis")
    st.caption("Screen multiple sequences sequentially and monitor for behavioral patterns.")

    health = check_api_health()
    if health.get("status") == "ok":
        st.success("API status: **online**", icon=":material/check_circle:")
    else:
        msg = health.get("message", "unknown error")
        st.error(f"API status: **offline** — {msg}", icon=":material/error:")

    st.divider()

    # ------------------------------------------------------------------
    # 2. Input section
    # ------------------------------------------------------------------
    input_mode = st.radio(
        "Input mode",
        ["Paste sequences", "Use demo sequences"],
        horizontal=True,
    )

    sequences: list[tuple[str, str]] = []

    if input_mode == "Paste sequences":
        raw_text = st.text_area(
            "Multi-FASTA input",
            height=200,
            placeholder=(
                ">Sequence 1\nMKTAYIAKQRQISFVKSH...\n\n"
                ">Sequence 2\nMVLSPADKTNVKAAWGKV..."
            ),
        )
        if raw_text.strip():
            sequences = _parse_multi_sequences(raw_text)
    else:
        st.info(
            "**Convergence attack demo** — 8 sequences that individually look benign "
            "but progressively converge toward irditoxin snake venom. "
            "Early queries score LOW; later queries score MEDIUM/HIGH. "
            "The session monitor flags the pattern even before the final query."
        )
        sequences = list(DEMO_CONVERGENCE_SERIES)

    top_k = st.number_input("Top K matches", min_value=1, max_value=20, value=5)

    # ------------------------------------------------------------------
    # 3. Run button
    # ------------------------------------------------------------------
    n_seq = len(sequences)
    run_disabled = n_seq == 0
    run_label = f"Screen {n_seq} Sequence{'s' if n_seq != 1 else ''}" if n_seq else "Screen Sequences"

    if st.button(run_label, disabled=run_disabled, type="primary"):
        # Generate a fresh session id for this batch
        session_id = str(uuid.uuid4())
        st.session_state["sa_session_id"] = session_id

        # ---------------------------------------------------------------
        # 4. Sequential execution
        # ---------------------------------------------------------------
        results: list[tuple[str, dict]] = []
        progress_bar = st.progress(0, text="Screening sequences...")

        for idx, (label, seq) in enumerate(sequences):
            progress_bar.progress(
                (idx) / n_seq,
                text=f"Screening {idx + 1}/{n_seq}: {label[:60]}...",
            )
            result = screen_sequence(
                sequence=seq,
                session_id=session_id,
                sequence_id=label,
                top_k=int(top_k),
            )
            results.append((label, result))

        progress_bar.progress(1.0, text="Done!")
        st.session_state["session_results"] = results
        st.rerun()

    # ------------------------------------------------------------------
    # 5. Results display
    # ------------------------------------------------------------------
    if "session_results" not in st.session_state:
        return

    results: list[tuple[str, dict]] = st.session_state["session_results"]
    session_id: str = st.session_state.get("sa_session_id", "")

    # 5a — Alert banner
    alerts = get_session_alerts(session_id) if session_id else None
    if alerts:
        anomaly_score = alerts.get("anomaly_score", 0)
        explanation = alerts.get("explanation", "")
        if anomaly_score > 0.5:
            st.error(
                f"**Anomaly detected** (score {anomaly_score:.2f}): {explanation}",
                icon=":material/warning:",
            )
        elif anomaly_score > 0.3:
            st.warning(
                f"**Elevated anomaly score** ({anomaly_score:.2f}): {explanation}",
                icon=":material/info:",
            )

    # 5b — Summary table + risk trend chart
    rows = []
    for idx, (label, res) in enumerate(results, start=1):
        if res.get("success") and res.get("data"):
            data = res["data"]
            risk_score = data.get("risk_score", 0)
            risk_level = data.get("risk_level", "unknown")
            matches = data.get("top_matches", [])
            top_match = matches[0]["name"] if matches else "—"
            rows.append(
                {
                    "#": idx,
                    "Sequence": label[:50],
                    "Risk Score": round(risk_score, 3),
                    "Risk Level": risk_level,
                    "Top Match": top_match,
                }
            )
        else:
            error_msg = res.get("error", "unknown error")
            rows.append(
                {
                    "#": idx,
                    "Sequence": label[:50],
                    "Risk Score": None,
                    "Risk Level": "error",
                    "Top Match": error_msg[:40],
                }
            )

    if rows:
        df = pd.DataFrame(rows)

        col_table, col_chart = st.columns([3, 2])

        with col_table:
            st.subheader("Results Summary")
            st.dataframe(df, width="stretch", hide_index=True)

        with col_chart:
            st.subheader("Risk Trend")
            scores = [r["Risk Score"] for r in rows if r["Risk Score"] is not None]
            if scores:
                chart_df = pd.DataFrame({"Sequence #": range(1, len(scores) + 1), "Risk Score": scores})
                st.line_chart(chart_df, x="Sequence #", y="Risk Score")

    # 5c — Session Monitoring section
    if alerts:
        st.subheader("Session Monitoring")
        convergence = alerts.get("convergence", {})
        perturbation = alerts.get("perturbation", {})

        col_conv, col_pert = st.columns(2)

        with col_conv:
            st.markdown("**Convergence Detector**")
            is_flagged = convergence.get("is_flagged", False)
            status_text = "FLAGGED" if is_flagged else "Normal"
            st.metric("Status", status_text)
            st.metric("Mean Similarity", f"{convergence.get('mean_similarity', 0):.4f}")
            st.metric("Similarity Trend", f"{convergence.get('similarity_trend', 0):.4f}")
            st.caption(f"Window size: {convergence.get('window_size', 0)}")

        with col_pert:
            st.markdown("**Perturbation Detector**")
            is_flagged = perturbation.get("is_flagged", False)
            status_text = "FLAGGED" if is_flagged else "Normal"
            st.metric("Status", status_text)
            st.metric("Cluster Count", perturbation.get("cluster_count", 0))
            st.metric("Max Cluster Size", perturbation.get("max_cluster_size", 0))
            high_sim_pairs = perturbation.get("high_sim_pairs", [])
            st.caption(f"High-similarity pairs: {len(high_sim_pairs)}")

    # 5d — Expandable per-sequence detail
    st.subheader("Per-Sequence Detail")
    for idx, (label, res) in enumerate(results):
        with st.expander(label):
            if res.get("success") and res.get("data"):
                render_results(res["data"], key_prefix=f"sa_{idx}_")
            else:
                st.error(res.get("error", "Screening failed for this sequence."))


page()

"""Single-sequence screening page for BioScreen."""

import uuid
import streamlit as st

from components.api_client import (
    check_api_health,
    screen_sequence,
    DEMO_SEQUENCES,
)
from components.styles import inject_custom_css
from components.result_viewer import render_results


def page():
    """Render the single-sequence screening page."""
    inject_custom_css()

    # Session state initialization
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

    # Input strip
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
        if seq_len < 10:
            st.warning(f"Sequence very short ({seq_len} aa) — results may be unreliable.")
        elif seq_len > 1000:
            st.warning(f"Long sequence ({seq_len} aa) — embeddings may be truncated.")
        else:
            st.caption(f"{seq_len} amino acids")

    # Screening execution
    if screen_button and sequence_input.strip():
        with st.spinner("Analyzing sequence..."):
            result = screen_sequence(
                sequence=sequence_input,
                session_id=st.session_state.session_id,
                sequence_id=sequence_id if sequence_id.strip() else None,
                top_k=top_k,
            )

        if result["success"]:
            st.session_state.last_result = result["data"]
            st.session_state.query_count += 1
            st.rerun()
        else:
            st.error(f"Error: {result['error']}")
            if "details" in result:
                with st.expander("Error Details"):
                    st.code(result["details"])

    # Results
    data = st.session_state.last_result
    if data:
        render_results(data)

    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align:center; color:#94a3b8; font-size:0.8rem;">'
        'BioScreen — Structure-based biosecurity screening for AI-designed proteins. '
        'For research purposes only. Always validate results with experimental methods.'
        '</div>',
        unsafe_allow_html=True,
    )

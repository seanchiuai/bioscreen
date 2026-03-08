"""Streamlit web interface for protein toxin screening.

Provides an easy-to-use frontend for submitting protein sequences
and visualizing similarity results and risk assessments.
"""

import json
import uuid
from typing import Optional

import pandas as pd
import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components


# Configuration
API_BASE_URL = "http://localhost:8000/api"

# Demo sequences for the sidebar selector
DEMO_SEQUENCES = {
    "-- Select a demo sequence --": "",
    "Insulin (full, 110aa)": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "Insulin B chain (short, 30aa)": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "Human Lysozyme (benign, 130aa)": "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
    "GFP (benign, 238aa)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Hemoglobin alpha (benign, 141aa)": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLAS",
    "Short test (13aa)": "MKAIFVLKGWWRT",
}


def check_api_health() -> dict:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Cannot connect to API: {str(e)}"}


def screen_sequence(
    sequence: str,
    session_id: str,
    sequence_id: Optional[str] = None,
    run_structure: bool = False,
    top_k: int = 5
) -> dict:
    """Submit a sequence for screening via the API."""
    payload = {
        "sequence": sequence,
        "sequence_id": sequence_id,
        "run_structure": run_structure,
        "top_k": top_k
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/screen",
            json=payload,
            headers={"X-Session-Id": session_id},
            timeout=120,  # Allow time for structure prediction
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


def get_risk_level_color(risk_level: str) -> str:
    """Get color for risk level display."""
    color_map = {
        "LOW": "green",
        "MEDIUM": "orange",
        "HIGH": "red",
        "UNKNOWN": "gray"
    }
    return color_map.get(risk_level, "gray")


def get_risk_score_color(risk_score: float) -> str:
    """Get color for risk score display based on thresholds."""
    if risk_score < 0.5:
        return "green"
    elif risk_score < 0.7:
        return "orange"
    else:
        return "red"


def format_function_prediction(function_prediction: Optional[dict]) -> str:
    """Format function prediction for display."""
    if not function_prediction:
        return "No function prediction available"

    parts = []

    if function_prediction.get("summary"):
        parts.append(f"**Summary:** {function_prediction['summary']}")

    if function_prediction.get("go_terms"):
        go_terms = function_prediction["go_terms"][:3]  # Show top 3
        go_text = ", ".join([f"{term.get('term', 'Unknown')} ({term.get('confidence', 'N/A')})" for term in go_terms])
        parts.append(f"**GO Terms:** {go_text}")

    if function_prediction.get("ec_numbers"):
        ec_numbers = function_prediction["ec_numbers"][:3]  # Show top 3
        ec_text = ", ".join([f"{ec.get('number', 'Unknown')} ({ec.get('confidence', 'N/A')})" for ec in ec_numbers])
        parts.append(f"**EC Numbers:** {ec_text}")

    return "\n\n".join(parts) if parts else "No detailed function information available"


def render_protein_3d(
    pdb_string: str,
    pocket_residues: list[int],
    danger_residues: list[int],
    view_style: str = "Cartoon",
    color_mode: str = "Default",
    width: int = 600,
    height: int = 480,
) -> None:
    """Render an interactive 3D protein viewer using py3Dmol.

    Args:
        pdb_string: PDB format string from ESMFold.
        pocket_residues: Residue indices for active site pockets (orange).
        danger_residues: Residue indices matching toxin active sites (red).
        view_style: One of "Cartoon", "Surface", "Stick".
        color_mode: "Default" for light blue or "pLDDT" for confidence coloring.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")

    # Base style depends on view_style and color_mode
    if color_mode == "pLDDT":
        # ESMFold stores pLDDT in the B-factor column (0-100)
        color_spec = {
            "prop": "b",
            "gradient": "roygb",
            "min": 50,
            "max": 100,
        }
    else:
        color_spec = "lightblue"

    if view_style == "Cartoon":
        if color_mode == "pLDDT":
            view.setStyle({"cartoon": {"colorscheme": color_spec}})
        else:
            view.setStyle({"cartoon": {"color": color_spec}})
    elif view_style == "Surface":
        # Show cartoon underneath, then add surface
        if color_mode == "pLDDT":
            view.setStyle({"cartoon": {"colorscheme": color_spec, "opacity": 0.5}})
            view.addSurface(
                py3Dmol.VDW,
                {"opacity": 0.7, "colorscheme": color_spec},
            )
        else:
            view.setStyle({"cartoon": {"color": color_spec, "opacity": 0.5}})
            view.addSurface(
                py3Dmol.VDW,
                {"opacity": 0.7, "color": color_spec},
            )
    elif view_style == "Stick":
        if color_mode == "pLDDT":
            view.setStyle({"stick": {"colorscheme": color_spec}})
        else:
            view.setStyle({"stick": {"color": color_spec}})

    # Highlight pocket residues in orange (stick representation)
    if pocket_residues:
        view.addStyle(
            {"resi": pocket_residues},
            {"stick": {"color": "orange", "radius": 0.2}},
        )

    # Highlight danger residues in red (thick stick + transparent surface)
    if danger_residues:
        view.addStyle(
            {"resi": danger_residues},
            {"stick": {"color": "red", "radius": 0.3}},
        )
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.3, "color": "red"},
            {"resi": danger_residues},
        )

    view.zoomTo()
    view.spin(False)

    # Render into Streamlit via HTML iframe
    html = view._make_html()
    components.html(html, width=width, height=height, scrolling=False)


def render_risk_gauge(risk_score: float, risk_level: str) -> None:
    """Render a colored risk gauge bar with score and level label."""
    color = get_risk_score_color(risk_score)
    pct = int(risk_score * 100)
    level_color = get_risk_level_color(risk_level)
    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem;">
          <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-size: 2rem; font-weight: bold;">{risk_score:.3f}</span>
            <span style="font-size: 1.4rem; font-weight: bold; color: {level_color};">{risk_level}</span>
          </div>
          <div style="background: #e0e0e0; border-radius: 6px; height: 18px; margin-top: 4px; overflow: hidden;">
            <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 6px;
                        transition: width 0.3s;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #888; margin-top: 2px;">
            <span>0 — Safe</span><span>0.5 — Medium</span><span>1.0 — High</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_session_monitor(data: dict) -> None:
    """Render the session monitoring section from screening response data."""
    factors = data.get("risk_factors", {})
    anomaly_score = factors.get("session_anomaly_score", 0.0)
    query_count = st.session_state.get("query_count", 0)

    if anomaly_score > 0.5:
        anomaly_color = "red"
        anomaly_label = "convergent pattern detected"
    elif anomaly_score > 0.3:
        anomaly_color = "orange"
        anomaly_label = "elevated"
    else:
        anomaly_color = "green"
        anomaly_label = "normal"

    col_q, col_a = st.columns(2)
    with col_q:
        st.metric("Queries this session", query_count)
    with col_a:
        st.markdown(
            f"**Anomaly Score**<br>"
            f"<span style='font-size:1.6rem; font-weight:bold; color:{anomaly_color};'>"
            f"{anomaly_score:.2f}</span> "
            f"<span style='color:{anomaly_color};'>({anomaly_label})</span>",
            unsafe_allow_html=True,
        )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="BioScreen - Protein Toxin Screening",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Persistent state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    st.title("🧬 BioScreen - Protein Toxin Screening")
    st.markdown("**Structure-based biosecurity screening for AI-designed proteins**")

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("API Status")
        health = check_api_health()

        if health["status"] == "ok":
            st.success("✅ API is healthy")
            st.write(f"**Version:** {health.get('version', 'Unknown')}")
            st.write(f"**Database Loaded:** {'✅' if health.get('toxin_db_loaded') else '❌'}")
            st.write(f"**ESM-2 Loaded:** {'✅' if health.get('esm2_loaded') else '❌'}")
            st.write(f"**Foldseek Available:** {'✅' if health.get('foldseek_available') else '❌'}")
        else:
            st.error(f"❌ API unavailable: {health.get('message', 'Unknown error')}")
            st.stop()

        st.divider()

        st.subheader("Screening Options")
        run_structure = st.checkbox(
            "Run Structure Analysis",
            value=False,
            help="Include ESMFold structure prediction and Foldseek comparison (slower but more accurate)"
        )

        top_k = st.slider(
            "Number of Top Matches",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top similar toxins to display"
        )

        st.divider()
        st.subheader("Demo Sequences")
        demo_choice = st.selectbox(
            "Load a demo sequence",
            options=list(DEMO_SEQUENCES.keys()),
            help="Select a pre-built sequence to load into the input area"
        )
        if DEMO_SEQUENCES.get(demo_choice):
            st.session_state.example_sequence = DEMO_SEQUENCES[demo_choice]

    # ── Top section: Input (left) + Risk gauge (right) ───────
    col_input, col_gauge = st.columns([3, 2])

    with col_input:
        st.subheader("📝 Sequence Input")
        sequence_input = st.text_area(
            "Protein Sequence",
            height=120,
            placeholder="Paste your protein sequence here (FASTA format with header is accepted)...",
            value=getattr(st.session_state, 'example_sequence', ''),
            help="Enter a protein sequence in single-letter amino acid code. FASTA headers will be automatically removed."
        )

        col_id, col_btn = st.columns([2, 1])
        with col_id:
            sequence_id = st.text_input(
                "Sequence ID (Optional)",
                placeholder="e.g., my_protein_001",
                label_visibility="collapsed",
            )
        with col_btn:
            screen_button = st.button(
                "🔍 Screen Sequence",
                type="primary",
                disabled=not sequence_input.strip(),
                use_container_width=True,
            )

        # Validation feedback
        if sequence_input.strip():
            cleaned = sequence_input.strip().replace('\n', '').replace('\r', '')
            if cleaned.startswith('>'):
                cleaned = ''.join(cleaned.split('\n')[1:])
            seq_len = len(cleaned)
            st.caption(f"📊 {seq_len} amino acids")
            if seq_len < 10:
                st.warning("⚠️ Sequence is very short (< 10 aa). Results may be unreliable.")
            elif seq_len > 1000:
                st.warning("⚠️ Long sequence (> 1000 aa). Embeddings may be truncated.")

    with col_gauge:
        st.subheader("🎯 Risk Assessment")
        if st.session_state.last_result:
            data = st.session_state.last_result
            render_risk_gauge(data["risk_score"], data["risk_level"])
        else:
            st.caption("Submit a sequence to see risk assessment.")

    # ── Run screening ────────────────────────────────────────
    if screen_button and sequence_input.strip():
        with st.spinner("🔬 Analyzing sequence..."):
            result = screen_sequence(
                sequence=sequence_input,
                session_id=st.session_state.session_id,
                sequence_id=sequence_id if sequence_id.strip() else None,
                run_structure=run_structure,
                top_k=top_k
            )

        if result["success"]:
            st.session_state.last_result = result["data"]
            st.session_state.query_count += 1
            st.rerun()
        else:
            st.error(f"❌ **Error:** {result['error']}")
            if "details" in result:
                with st.expander("Error Details"):
                    st.code(result["details"])

    # ── Middle section: 3D viewer (left) + Matches & factors (right) ──
    data = st.session_state.last_result
    if data:
        st.divider()

        pdb_string = data.get("pdb_string")
        has_structure = pdb_string is not None

        if has_structure:
            col_viewer, col_matches = st.columns([3, 2])
        else:
            # No 3D viewer — use full width for matches
            col_matches = st.container()

        # 3D Viewer column
        if has_structure:
            with col_viewer:
                st.subheader("🔬 3D Structure")
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
                )

                # Legend
                legend_parts = []
                if pocket_res:
                    legend_parts.append(f"🟠 Active site pocket ({len(pocket_res)} residues)")
                if danger_res:
                    legend_parts.append(f"🔴 Danger residues ({len(danger_res)} residues)")
                if legend_parts:
                    st.caption(" | ".join(legend_parts))

        # Matches + Risk factors column
        with col_matches:
            st.subheader("🎯 Top Toxin Matches")
            if data.get("top_matches"):
                matches_data = []
                for i, match in enumerate(data["top_matches"], 1):
                    matches_data.append({
                        "Rank": i,
                        "UniProt ID": match["uniprot_id"],
                        "Name": match["name"][:50] + ("..." if len(match["name"]) > 50 else ""),
                        "Organism": match["organism"][:30] + ("..." if len(match["organism"]) > 30 else ""),
                        "Type": match["toxin_type"],
                        "Emb Sim": f"{match['embedding_similarity']:.3f}",
                        "Str Sim": f"{match['structure_similarity']:.3f}" if match.get("structure_similarity") else "N/A"
                    })
                matches_df = pd.DataFrame(matches_data)
                st.dataframe(matches_df, use_container_width=True, hide_index=True)
            else:
                st.info("No significant matches found.")

            # Risk factors breakdown
            factors = data.get("risk_factors", {})
            if factors:
                with st.expander("📋 Risk Factor Breakdown", expanded=True):
                    col_e, col_s = st.columns(2)
                    with col_e:
                        emb_sim = factors.get("max_embedding_similarity", 0)
                        st.metric("Embedding Similarity", f"{emb_sim:.3f}")
                    with col_s:
                        struct_sim = factors.get("max_structure_similarity")
                        if struct_sim is not None:
                            st.metric("Structure Similarity", f"{struct_sim:.3f}")
                        else:
                            st.metric("Structure Similarity", "N/A")

                    col_f, col_x = st.columns(2)
                    with col_f:
                        func_overlap = factors.get("function_overlap", 0)
                        st.metric("Function Overlap", f"{func_overlap:.3f}")
                    with col_x:
                        if factors.get("score_explanation"):
                            st.caption(factors["score_explanation"][:200])

        # ── Function prediction (full width) ─────────────────
        st.divider()
        st.subheader("🧬 Function Prediction")
        function_pred = data.get("function_prediction")
        if function_pred:
            st.markdown(format_function_prediction(function_pred))
        else:
            st.info("No function prediction available.")

        # ── Warnings ─────────────────────────────────────────
        if data.get("warnings"):
            for warning in data["warnings"]:
                st.warning(warning)

        # ── Session Monitoring (full width) ──────────────────
        st.divider()
        st.subheader("📡 Session Monitoring")
        render_session_monitor(data)

        # ── Raw data (collapsed) ─────────────────────────────
        with st.expander("🔍 Raw Response Data", expanded=False):
            st.json(data)

    # ── Footer ───────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "🧬 BioScreen - Structure-based biosecurity screening for AI-designed proteins<br>"
        "For research purposes only. Always validate results with experimental methods."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
"""Streamlit web interface for protein toxin screening.

Provides an easy-to-use frontend for submitting protein sequences
and visualizing similarity results and risk assessments.
"""

import json
from typing import Optional

import pandas as pd
import requests
import streamlit as st


# Configuration
API_BASE_URL = "http://localhost:8000/api"


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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="BioScreen - Protein Toxin Screening",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧬 BioScreen - Protein Toxin Screening")
    st.markdown("**Identify potential toxin similarity in protein sequences using ESM-2 embeddings and structure analysis**")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # API health check
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

        # Screening options
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

        # Example sequences
        st.divider()
        st.subheader("Example Sequences")

        if st.button("Load Insulin Example"):
            st.session_state.example_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

        if st.button("Load Short Test Sequence"):
            st.session_state.example_sequence = "MKAIFVLKGWWRT"

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Sequence Input")

        # Sequence input
        sequence_input = st.text_area(
            "Protein Sequence",
            height=150,
            placeholder="Paste your protein sequence here (FASTA format with header is accepted)...",
            value=getattr(st.session_state, 'example_sequence', ''),
            help="Enter a protein sequence in single-letter amino acid code. FASTA headers will be automatically removed."
        )

        # Sequence ID (optional)
        sequence_id = st.text_input(
            "Sequence ID (Optional)",
            placeholder="e.g., my_protein_001",
            help="Optional identifier for your sequence"
        )

        # Screen button
        screen_button = st.button(
            "🔍 **Screen Sequence**",
            type="primary",
            disabled=not sequence_input.strip(),
            help="Submit sequence for toxin similarity screening"
        )

        # Input validation feedback
        if sequence_input.strip():
            cleaned_sequence = sequence_input.strip().replace('\n', '').replace('\r', '')
            # Remove FASTA header if present
            if cleaned_sequence.startswith('>'):
                lines = cleaned_sequence.split('\n')
                cleaned_sequence = ''.join(lines[1:])

            seq_length = len(cleaned_sequence)
            st.info(f"📊 Sequence length: {seq_length} amino acids")

            if seq_length < 10:
                st.warning("⚠️ Sequence is very short (< 10 aa). Results may be unreliable.")
            elif seq_length > 1000:
                st.warning("⚠️ Long sequence (> 1000 aa). Embeddings may be truncated.")

    with col2:
        st.header("📊 Results")

        if screen_button and sequence_input.strip():
            # Show progress
            with st.spinner("🔬 Analyzing sequence..."):
                result = screen_sequence(
                    sequence=sequence_input,
                    sequence_id=sequence_id if sequence_id.strip() else None,
                    run_structure=run_structure,
                    top_k=top_k
                )

            if result["success"]:
                data = result["data"]

                # Risk score and level
                st.subheader("🎯 Risk Assessment")

                col_score, col_level = st.columns(2)

                with col_score:
                    risk_score = data["risk_score"]
                    st.metric(
                        "Risk Score",
                        f"{risk_score:.3f}",
                        delta=f"{risk_score - 0.5:.3f}" if risk_score != 0.5 else None,
                        delta_color="inverse"
                    )

                with col_level:
                    risk_level = data["risk_level"]
                    color = get_risk_level_color(risk_level)
                    st.markdown(f"**Risk Level**")
                    st.markdown(f"<h2 style='color: {color}; margin-top: 0;'>{risk_level}</h2>", unsafe_allow_html=True)

                # Risk factors
                if data.get("risk_factors"):
                    with st.expander("📋 Risk Factors", expanded=False):
                        factors = data["risk_factors"]

                        col_emb, col_struct = st.columns(2)
                        with col_emb:
                            emb_sim = factors.get("max_embedding_similarity", 0)
                            st.metric("Max Embedding Similarity", f"{emb_sim:.3f}")

                        with col_struct:
                            struct_sim = factors.get("max_structure_similarity")
                            if struct_sim is not None:
                                st.metric("Max Structure Similarity", f"{struct_sim:.3f}")
                            else:
                                st.metric("Structure Similarity", "Not analyzed")

                        if factors.get("function_overlap"):
                            st.metric("Function Overlap", f"{factors['function_overlap']:.3f}")

                        if factors.get("score_explanation"):
                            st.text_area(
                                "Score Explanation",
                                factors["score_explanation"],
                                height=100,
                                disabled=True
                            )

                # Top matches
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
                            "Embedding Sim": f"{match['embedding_similarity']:.3f}",
                            "Structure Sim": f"{match['structure_similarity']:.3f}" if match.get("structure_similarity") else "N/A"
                        })

                    matches_df = pd.DataFrame(matches_data)
                    st.dataframe(
                        matches_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No significant matches found.")

                # Function prediction
                st.subheader("🧬 Function Prediction")

                function_pred = data.get("function_prediction")
                if function_pred:
                    formatted_function = format_function_prediction(function_pred)
                    st.markdown(formatted_function)
                else:
                    st.info("No function prediction available.")

                # Warnings
                if data.get("warnings"):
                    st.subheader("⚠️ Warnings")
                    for warning in data["warnings"]:
                        st.warning(warning)

                # Raw data (collapsible)
                with st.expander("🔍 Raw Response Data", expanded=False):
                    st.json(data)

            else:
                st.error(f"❌ **Error:** {result['error']}")
                if "details" in result:
                    with st.expander("Error Details"):
                        st.code(result["details"])

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        🧬 BioScreen - Protein toxin similarity screening using ESM-2 embeddings and structure analysis<br>
        For research purposes only. Always validate results with experimental methods.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
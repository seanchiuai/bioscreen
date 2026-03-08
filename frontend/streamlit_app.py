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

DEMO_SEQUENCES = {
    "-- Select a demo sequence --": "",
    # Dangerous proteins with low BLAST identity to known toxins (demonstrates BioScreen advantage)
    "Trichosanthin (RIP toxin, ~28% identity to ricin, 289aa)": "DVSFRLSGATSSSYGVFISNLRKALPNERKLYDIPLLRSSLPGSQRYALIHLTNYADETISVAIDVTNVYIMGYRAGDTSYFFNEASATEAAKYVFKDAMRKVTLPYSGNYERLQTAAGKIRENIPLGLPALDSAITTLFYYNANSAASALMVLIQSTSEAARYKFIEQQIGKRVDKTFLPSLAIISLENSWSALSKQIQIASTNNGQFESPVVLINAQNQRVTITNVDAGVVTSNIALLLNRNNMAAMDDDVPMTQSFGCGSYAI",
    "Saporin (RIP toxin, ~30% identity to ricin, 253aa)": "DAVTSITLDLVNPTAGQYSSFVDKIRNNVKDPNLKYGGTDIAVIGPPSKEKFLRINFQSSRGTVSLGLKRDNLYVVAYLAMDNTNVNRAYYFRSEITSAESTALFPEATTANQKALEYTEDYQSIEKNAQITQGDQSRKELGLGIDLLSTSMEAVNKKARVVKDEARFLLIAIQMTAEAARFRYIQNLVIKNFPNKFNSENKVIQFEVNWKKISTAIYGDAKNGVFNKDYDFGFGKVRQVKDLQMGLLMYLGKPKSSNEANSTVRHYGPLKPTLLIT",
    "MAP30 (RIP toxin, bitter melon, 263aa)": "DVNFDLSTATAKTYTKFIEDFRATLPFSHKVYDIPLLYSTISDSRRFILLNLTSYAYETISVAIDVTNVYVVAYRTRDVSYFFKESPPEAYNILFKGTRKITLPYTGNYENLQTAAHKIRENIDLGLPALSSAITTLFYYNAQSAPSALLVLIQTTAEAARFKYIERHVAKYVATNFKPNLAIISLENQWSALSKQIFLAQNQGGKFRNPVDLIKPTGERFQVTNVDSDVVKGNIKLLLNSRASTADENFITTMTLLGESVVN",
    "Bouganin (RIP toxin, ~32% identity to ricin, 70aa)": "EFQESVKSQHTERCIDFLTKELKVSNEKEAAERVFFVSARETLQARLEEAKGNPPHLGAIAEGFQIRYFE",
    "Cholix toxin (ADP-ribosyltransferase, <20% identity to DT, 95aa)": "YPTKGRGGKGIKTANITAKNGPLAGLVTVNDDEDIMIITDTGVIIRTSVADISQTGRSAMGVKVMRLDENAKIVTFALVKSEVIEGTSLNNNENE",
    # Benign controls
    "GFP (benign control, 238aa)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Hemoglobin alpha (benign control, 141aa)": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLAS",
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


def inject_custom_css() -> None:
    st.markdown("""
    <style>
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
    .risk-low { color: #22c55e; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)


def render_summary_cards(data: dict) -> None:
    risk_score = data["risk_score"]
    risk_level = data["risk_level"]

    if risk_score < 0.45:
        bar_color, level_class = "#22c55e", "risk-low"
    elif risk_score < 0.75:
        bar_color, level_class = "#f59e0b", "risk-medium"
    else:
        bar_color, level_class = "#ef4444", "risk-high"

    top_match = data.get("top_matches", [{}])[0] if data.get("top_matches") else {}
    match_name = top_match.get("name", "No match")
    match_org = top_match.get("organism", "")
    emb_sim = top_match.get("embedding_similarity", 0)
    str_sim = top_match.get("structure_similarity")
    best_sim = max(emb_sim, str_sim or 0)
    sim_label = "structure" if (str_sim and str_sim >= emb_sim) else "embedding"

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


def main():
    st.set_page_config(
        page_title="BioScreen",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    inject_custom_css()

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
            width="stretch",
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
                run_structure=run_structure,
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
        st.divider()
        render_summary_cards(data)

        # Tabbed detail area
        has_structure = data.get("pdb_string") is not None
        tab_labels = ["Matches", "Structure", "Score Breakdown", "Function", "Explain"]
        tabs = st.tabs(tab_labels)

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
                        "Structure Sim": match.get("structure_similarity") if has_structure else None,
                    })
                df = pd.DataFrame(matches_data)

                col_config = {
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Name": st.column_config.TextColumn(width="medium"),
                    "Organism": st.column_config.TextColumn(width="medium"),
                    "Toxin Type": st.column_config.TextColumn(width="small"),
                    "Embedding Sim": st.column_config.ProgressColumn(
                        "Embedding Sim", min_value=0, max_value=1, format="%.3f",
                    ),
                }
                if has_structure:
                    col_config["Structure Sim"] = st.column_config.ProgressColumn(
                        "Structure Sim", min_value=0, max_value=1, format="%.3f",
                    )
                else:
                    df["Structure Sim"] = "\u2014"
                    col_config["Structure Sim"] = st.column_config.TextColumn("Structure Sim")

                st.dataframe(df, column_config=col_config, width="stretch", hide_index=True)
            else:
                st.info("No significant matches found.")

        with tabs[1]:
            pdb_string = data.get("pdb_string")
            if pdb_string:
                col_view, col_color = st.columns(2)
                with col_view:
                    view_style = st.radio(
                        "View", ["Cartoon", "Surface", "Stick"],
                        horizontal=True, key="view_style",
                    )
                with col_color:
                    color_mode = st.radio(
                        "Color", ["Default", "pLDDT"],
                        horizontal=True, key="color_mode",
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

                legend_parts = ["Blue: normal structure"]
                if pocket_res:
                    legend_parts.append(f"Orange: active site pocket ({len(pocket_res)} residues)")
                if danger_res:
                    legend_parts.append(f"Red: danger residues ({len(danger_res)} residues)")
                st.caption(" | ".join(legend_parts))
            else:
                st.info("Structure analysis was not run. Enable 'Structure analysis' and re-screen to see the 3D viewer.")

        with tabs[2]:
            factors = data.get("risk_factors", {})
            emb_sim = factors.get("max_embedding_similarity", 0)
            struct_sim = factors.get("max_structure_similarity")
            func_overlap = factors.get("function_overlap", 0)
            structure_ran = data.get("structure_predicted", False)

            if structure_ran and struct_sim is not None:
                weight_set = {"Embedding": 0.50, "Structure": 0.30, "Function": 0.20}
                weight_note = "Full path weights (embedding 0.50, structure 0.30, function 0.20)"
            else:
                weight_set = {"Embedding": 0.65, "Function": 0.35}
                weight_note = "Fast path weights (embedding 0.65, function 0.35)"

            st.markdown(f"**Weight set:** {weight_note}")

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
                st.caption(f"Weight: {weight:.2f} | Contribution: {weighted:.3f}")

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
                st.caption("Bonus for multiple high-confidence signals")

            st.markdown(f"**Final Score: {risk_score:.3f}**")

        with tabs[3]:
            function_pred = data.get("function_prediction")
            if function_pred:
                summary = function_pred.get("summary", "")
                if summary:
                    st.markdown(f'<div class="recommend-box">{summary}</div>', unsafe_allow_html=True)

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

        with tabs[4]:
            risk_score = data["risk_score"]
            risk_level = data["risk_level"]
            factors = data.get("risk_factors", {})
            explanation = factors.get("score_explanation", "")

            if risk_score >= 0.75:
                verdict_class = "verdict-high"
            elif risk_score >= 0.45:
                verdict_class = "verdict-medium"
            else:
                verdict_class = "verdict-low"

            parts = [p.strip() for p in explanation.split(". ") if p.strip()]
            verdict_text = parts[0] if parts else f"{risk_level} RISK"

            st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)

            if len(parts) > 1:
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

            warnings = data.get("warnings", [])
            if warnings:
                st.markdown("---")
                st.markdown("**Warnings**")
                for w in warnings:
                    st.warning(w)

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


if __name__ == "__main__":
    main()
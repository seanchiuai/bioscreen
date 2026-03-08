"""Shared 6-tab result display for BioScreen screening results."""

import html as _html
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components.summary_cards import render_summary_cards
from components.protein_3d import render_protein_3d, _aligned_residue_set

try:
    from video_generator import ProteinVideoData, generate_video
    _VIDEO_AVAILABLE = True
except ImportError:
    _VIDEO_AVAILABLE = False


def render_results(data: dict, key_prefix: str = "") -> None:
    """Render the full tabbed results section for a screening result.

    Parameters
    ----------
    data : dict
        The screening result dict (from the API ``/api/screen`` response).
    key_prefix : str
        Optional prefix for widget keys to avoid collisions when rendering
        multiple results on the same page (e.g. in session analysis expanders).
    """

    # Tabbed detail area
    has_structure = data.get("pdb_string") is not None
    tab_labels = ["Overview", "Matches", "Structure", "Score Breakdown", "Function", "Explain"]
    tabs = st.tabs(tab_labels)

    # ── Tab 0: Overview ──────────────────────────────────────────────────
    with tabs[0]:
        render_summary_cards(data)

    # ── Tab 1: Matches ───────────────────────────────────────────────────
    with tabs[1]:
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

            st.dataframe(df, column_config=col_config, use_container_width=True, hide_index=True)
        else:
            st.info("No significant matches found.")

    # ── Tab 2: Structure ─────────────────────────────────────────────────
    with tabs[2]:
        pdb_string = data.get("pdb_string")
        if pdb_string:
            col_view, col_color = st.columns(2)
            with col_view:
                view_style = st.radio(
                    "View", ["Cartoon", "Surface", "Stick"],
                    horizontal=True, key=f"{key_prefix}view_style",
                )
            with col_color:
                color_options = ["Default", "pLDDT", "Risk Layers"]
                color_mode = st.radio(
                    "Color", color_options,
                    horizontal=True, key=f"{key_prefix}color_mode",
                    help="Risk Layers: gray=no match, yellow=structurally aligned to toxin, orange=pocket, red=active site match",
                )

            pocket_res = data.get("pocket_residues", [])
            danger_res = data.get("danger_residues", [])
            aligned_regions = data.get("aligned_regions", [])

            overlay_pdb = None
            overlay_name = ""
            compare_data = None

            render_protein_3d(
                pdb_string=pdb_string,
                pocket_residues=pocket_res,
                danger_residues=danger_res,
                aligned_regions=aligned_regions,
                view_style=view_style,
                color_mode=color_mode,
                overlay_pdb=overlay_pdb,
                overlay_name=overlay_name,
                width=800,
                height=500,
            )

            # Dynamic legend based on color mode
            if color_mode == "Risk Layers":
                aligned_res = _aligned_residue_set(aligned_regions)
                legend_parts = ["Gray: no structural match"]
                if aligned_res:
                    legend_parts.append(f"Yellow: structurally aligned to toxin ({len(aligned_res)} residues)")
                if pocket_res:
                    legend_parts.append(f"Orange: active site pocket ({len(pocket_res)} residues)")
                if danger_res:
                    legend_parts.append(f"Red: active site match ({len(danger_res)} residues)")
                st.caption(" | ".join(legend_parts))

                # Framing: explain fold-level vs residue-level risk
                if danger_res or aligned_res:
                    top_match = data.get("top_matches", [{}])[0] if data.get("top_matches") else {}
                    match_name = _html.escape(top_match.get("name", "a known toxin"))
                    st.markdown(f"""
                    <div class="recommend-box" style="margin-top: 0.5rem;">
                        <strong>Interpreting this view:</strong> The yellow regions show where this protein's
                        backbone folds like <em>{match_name}</em>. Red highlights mark residues whose
                        geometry matches the toxin's functional site. Danger is a property of the
                        <strong>overall fold</strong> positioning these residues — not the residues alone.
                        Mutating red residues does not necessarily make the protein safe, because the
                        surrounding scaffold exists to position them.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                legend_parts = ["Blue: query protein"]
                if overlay_pdb:
                    legend_parts.append(f"Red (transparent): {overlay_name}")
                if pocket_res:
                    legend_parts.append(f"Orange: active site pocket ({len(pocket_res)} residues)")
                if danger_res:
                    legend_parts.append(f"Red (solid): danger residues ({len(danger_res)} residues)")
                st.caption(" | ".join(legend_parts))

            # Comparison stats panel
            if overlay_pdb and compare_data:
                st.markdown("---")
                st.markdown("**Structural Comparison**")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                rmsd = compare_data.get("rmsd", 0)
                aligned_res = compare_data.get("aligned_residues", 0)

                # Get sequence identity and TM-score from the top match
                seq_identity = top_match.get("sequence_identity")
                tm_score = top_match.get("structure_similarity")

                col_s1.metric("RMSD", f"{rmsd:.1f} A")
                col_s2.metric("TM-score", f"{tm_score:.2f}" if tm_score is not None else "N/A")
                col_s3.metric("Seq. Identity", f"{seq_identity:.0%}" if seq_identity is not None else "N/A")
                col_s4.metric("Aligned Residues", str(aligned_res))

                # Interpretive callout
                if seq_identity is not None and tm_score is not None:
                    if seq_identity < 0.3 and tm_score > 0.5:
                        st.warning(
                            f"Low sequence identity ({seq_identity:.0%}) with high structural similarity "
                            f"(TM-score {tm_score:.2f}) — this pattern is characteristic of AI-designed "
                            f"structural mimicry that BLAST-based screening would miss."
                        )
                    elif tm_score > 0.7:
                        st.info(
                            f"High structural similarity (TM-score {tm_score:.2f}) to {overlay_name}. "
                            f"The overlay shows where the two structures align."
                        )
        else:
            st.info("Structure prediction did not return a result. Try again or check the ESMFold API.")

    # ── Tab 3: Score Breakdown ───────────────────────────────────────────
    with tabs[3]:
        factors = data.get("risk_factors", {})
        emb_sim = factors.get("max_embedding_similarity", 0)
        struct_sim = factors.get("max_structure_similarity")
        func_overlap = factors.get("function_overlap", 0)
        if struct_sim is not None:
            weight_set = {"Embedding": 0.50, "Structure": 0.30, "Function": 0.20}
            weight_note = "Weights: embedding 0.50, structure 0.30, function 0.20"
        else:
            weight_set = {"Embedding": 0.65, "Function": 0.35}
            weight_note = "Weights: embedding 0.65, function 0.35 (no structure data available)"

        st.markdown(f"**Weight set:** {weight_note}")

        components_list = [
            ("Embedding Similarity", emb_sim, weight_set.get("Embedding", 0)),
        ]
        if struct_sim is not None:
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

    # ── Tab 4: Function ──────────────────────────────────────────────────
    with tabs[4]:
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

    # ── Tab 5: Explain ───────────────────────────────────────────────────
    with tabs[5]:
        risk_score = data["risk_score"]
        risk_level = data["risk_level"]
        factors = data.get("risk_factors", {})
        top_matches = data.get("top_matches", [])
        top_match = top_matches[0] if top_matches else {}
        match_name = _html.escape(top_match.get("name", "")) if top_match else ""
        match_organism = _html.escape(top_match.get("organism", "")) if top_match else ""
        match_toxin_type = top_match.get("toxin_type", "") if top_match else ""

        emb_sim = factors.get("max_embedding_similarity", 0)
        struct_sim = factors.get("max_structure_similarity")
        func_overlap = factors.get("function_overlap", 0)
        active_site = factors.get("active_site_overlap")
        seq_length = data.get("sequence_length", 0)

        if risk_score >= 0.75:
            verdict_class = "verdict-high"
        elif risk_score >= 0.45:
            verdict_class = "verdict-medium"
        else:
            verdict_class = "verdict-low"

        # Verdict banner
        if risk_score >= 0.8:
            verdict_text = "HIGH RISK: Strong similarity to known toxins"
        elif risk_score >= 0.6:
            verdict_text = "MODERATE RISK: Notable similarity to toxins"
        elif risk_score >= 0.4:
            verdict_text = "LOW-MODERATE RISK: Some similarity detected"
        elif risk_score >= 0.2:
            verdict_text = "LOW RISK: Minimal similarity"
        else:
            verdict_text = "MINIMAL RISK: No significant similarity"

        st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)

        # ── Overall Summary ──
        st.markdown("**What does this mean?**")

        if risk_score >= 0.75 and match_name:
            summary = (
                f"This protein scored **{risk_score:.3f}** on the BioScreen risk scale, placing it in the "
                f"**high risk** category. The submitted sequence shows strong similarity to "
                f"**{match_name}**{f', a {match_toxin_type} toxin from *{match_organism}*' if match_toxin_type and match_organism else ''}. "
                f"Multiple independent lines of evidence — including sequence embedding analysis, "
                f"structural fold comparison, and functional annotation — converge to suggest that "
                f"this protein could have biological activity consistent with a known dangerous agent. "
                f"Immediate review by a biosafety expert is strongly recommended before any synthesis "
                f"or experimental work proceeds."
            )
        elif risk_score >= 0.45 and match_name:
            summary = (
                f"This protein scored **{risk_score:.3f}** on the BioScreen risk scale, placing it in the "
                f"**moderate risk** category. The screening pipeline detected notable similarity to "
                f"**{match_name}**{f', a {match_toxin_type} toxin from *{match_organism}*' if match_toxin_type and match_organism else ''}. "
                f"While the evidence is not conclusive enough to classify this as high risk, the signals "
                f"are sufficient to warrant further investigation. A biosafety review and enhanced "
                f"monitoring protocols are recommended."
            )
        elif risk_score >= 0.2:
            summary = (
                f"This protein scored **{risk_score:.3f}** on the BioScreen risk scale, placing it in the "
                f"**low risk** category. The screening pipeline detected only minimal similarity to "
                f"known toxins in the reference database. "
                f"{'The closest match was **' + match_name + '**, but the similarity was not strong enough to raise concern. ' if match_name else ''}"
                f"Standard biosafety protocols should be sufficient for this sequence."
            )
        else:
            summary = (
                f"This protein scored **{risk_score:.3f}** on the BioScreen risk scale, indicating "
                f"**minimal risk**. No significant similarity to any known toxin was detected across "
                f"sequence embeddings, structural folds, or functional annotations. This protein does "
                f"not appear to resemble any dangerous agent in the BioScreen reference database."
            )
        st.markdown(summary)

        # ── Evidence Breakdown ──
        st.markdown("---")
        st.markdown("**Evidence Breakdown**")

        # Embedding similarity explanation
        if emb_sim >= 0.97:
            emb_paragraph = (
                f"**Sequence Embedding Similarity ({emb_sim:.3f}):** The ESM-2 protein language model "
                f"found very high similarity between this protein's learned representation and those of "
                f"known toxins. A score this high (above 0.97) means that the model considers this sequence "
                f"to occupy nearly the same region of protein embedding space as established dangerous "
                f"proteins, suggesting conserved biochemical properties even if the raw sequence alignment "
                f"might differ."
            )
        elif emb_sim >= 0.93:
            emb_paragraph = (
                f"**Sequence Embedding Similarity ({emb_sim:.3f}):** The ESM-2 protein language model "
                f"detected moderate similarity between this protein and known toxins. This score falls "
                f"in a range where some structural or functional resemblance exists, but it is not high "
                f"enough on its own to be definitive. The embedding captures learned patterns of amino acid "
                f"interactions, so this similarity may reflect shared folds or binding properties rather "
                f"than direct sequence homology."
            )
        elif emb_sim >= 0.85:
            emb_paragraph = (
                f"**Sequence Embedding Similarity ({emb_sim:.3f}):** The ESM-2 protein language model "
                f"found only weak similarity between this protein and known toxins in the reference "
                f"database. Scores in this range are typical of proteins that share some general "
                f"structural features (such as common folds) but lack the specific characteristics "
                f"that define toxin function."
            )
        else:
            emb_paragraph = (
                f"**Sequence Embedding Similarity ({emb_sim:.3f}):** The ESM-2 protein language model "
                f"found no meaningful similarity between this protein and known toxins. This sequence "
                f"occupies a distinctly different region of embedding space, indicating that its learned "
                f"biochemical properties are unrelated to any dangerous protein in the reference database."
            )
        st.markdown(emb_paragraph)

        # Structural similarity explanation
        if struct_sim is not None:
            if struct_sim >= 0.8:
                struct_paragraph = (
                    f"**Structural Similarity ({struct_sim:.3f}):** Foldseek structural alignment found "
                    f"very high 3D fold similarity (TM-score {struct_sim:.3f}) between the predicted "
                    f"structure of this protein and a known toxin. A TM-score above 0.8 strongly suggests "
                    f"that these proteins share the same overall fold topology, which is particularly "
                    f"concerning because structure determines function — two proteins with the same fold "
                    f"can have similar biological activity regardless of how different their amino acid "
                    f"sequences appear. This is the key gap that BioScreen addresses: AI-designed proteins "
                    f"from tools like RFdiffusion can adopt toxin-like folds while evading traditional "
                    f"BLAST-based sequence screening."
                )
            elif struct_sim >= 0.5:
                struct_paragraph = (
                    f"**Structural Similarity ({struct_sim:.3f}):** Foldseek structural alignment detected "
                    f"moderate 3D fold similarity (TM-score {struct_sim:.3f}) between this protein and a "
                    f"known toxin. TM-scores in this range indicate partial structural overlap — the "
                    f"proteins may share some common structural motifs or sub-domains without having an "
                    f"identical overall fold. This warrants attention because functional domains (such as "
                    f"binding pockets or catalytic sites) can be conserved even when overall structures diverge."
                )
            else:
                struct_paragraph = (
                    f"**Structural Similarity ({struct_sim:.3f}):** Foldseek structural alignment found "
                    f"low 3D fold similarity (TM-score {struct_sim:.3f}) between this protein and the "
                    f"closest toxin match. A TM-score below 0.5 generally indicates that these proteins "
                    f"have distinct overall folds, making it unlikely that this protein could replicate "
                    f"the structural basis of the matched toxin's function."
                )
            st.markdown(struct_paragraph)

            # Sequence-structure divergence callout
            seq_identity = top_match.get("sequence_identity")
            if seq_identity is not None and seq_identity < 0.3 and struct_sim > 0.5:
                st.markdown(
                    f"> This protein shows **low sequence identity ({seq_identity:.0%})** but "
                    f"**moderate-to-high structural similarity (TM-score {struct_sim:.3f})**. "
                    f"This divergence between sequence and structure is a hallmark of AI-designed "
                    f"proteins that can mimic toxin folds while appearing completely different at the "
                    f"sequence level — exactly the scenario that traditional BLAST-based biosecurity "
                    f"screening fails to catch."
                )
        else:
            st.markdown(
                "**Structural Similarity:** Structure prediction was not available for this screening. "
                "Without 3D structural data, the risk assessment relies solely on sequence embeddings "
                "and functional annotations, which may miss cases where a protein adopts a toxin-like "
                "fold despite having a dissimilar sequence."
            )

        # Active site explanation
        if active_site is not None:
            if active_site >= 0.7:
                active_paragraph = (
                    f"**Active Site Similarity ({active_site:.3f}):** Analysis of binding pocket geometry "
                    f"revealed high similarity between this protein's active site and that of the matched "
                    f"toxin. This means that not only does the overall fold resemble a toxin, but the "
                    f"specific 3D arrangement of residues responsible for the toxin's biological "
                    f"activity — its catalytic or binding site — is also conserved. This is the strongest "
                    f"indicator that the protein could have functional toxicity."
                )
            elif active_site >= 0.4:
                active_paragraph = (
                    f"**Active Site Similarity ({active_site:.3f}):** Binding pocket analysis detected "
                    f"moderate geometric similarity between this protein's predicted active site and "
                    f"that of the matched toxin. While not an exact match, there is enough spatial "
                    f"resemblance to suggest partial conservation of the functional site architecture."
                )
            else:
                active_paragraph = (
                    f"**Active Site Similarity ({active_site:.3f}):** The predicted active site geometry "
                    f"of this protein shows little resemblance to known toxin catalytic sites. Even if "
                    f"the overall fold has some similarity, the lack of active site conservation reduces "
                    f"the likelihood that this protein could replicate toxin function."
                )
            st.markdown(active_paragraph)

        # Function overlap explanation
        if func_overlap > 0:
            if func_overlap >= 0.6:
                func_paragraph = (
                    f"**Functional Annotation Overlap ({func_overlap:.3f}):** The predicted Gene Ontology "
                    f"(GO) terms and Enzyme Commission (EC) numbers for this protein overlap significantly "
                    f"with those of the matched toxin. This means independent computational methods predict "
                    f"that this protein performs similar biological functions to a known dangerous agent, "
                    f"corroborating the structural and sequence-based evidence."
                )
            elif func_overlap >= 0.3:
                func_paragraph = (
                    f"**Functional Annotation Overlap ({func_overlap:.3f}):** There is moderate overlap "
                    f"between the predicted functional annotations (GO terms and EC numbers) of this "
                    f"protein and those of the closest toxin match. This suggests some shared biological "
                    f"function, though it could also reflect broadly conserved activities common to many "
                    f"protein families."
                )
            else:
                func_paragraph = (
                    f"**Functional Annotation Overlap ({func_overlap:.3f}):** There is minimal overlap "
                    f"between the predicted functional annotations of this protein and those of known "
                    f"toxins. The protein appears to have a different predicted biological role."
                )
            st.markdown(func_paragraph)
        else:
            st.markdown(
                "**Functional Annotation Overlap (0.000):** No overlap was detected between the predicted "
                "functional annotations (GO terms and EC numbers) of this protein and those of any "
                "known toxin in the reference database. This suggests the protein is predicted to "
                "serve a different biological function."
            )

        # Short sequence note
        if seq_length and seq_length < 50:
            st.markdown(
                f"> **Note on sequence length:** This protein is only {seq_length} amino acids long. "
                f"Short peptides (under 50 residues) tend to cluster in ESM-2 embedding space regardless "
                f"of their function, which can inflate embedding similarity scores. The screening pipeline "
                f"has automatically reduced the weight of embedding similarity for this assessment."
            )

        # ── Recommendation ──
        if risk_score >= 0.75:
            st.markdown("---")
            st.markdown(
                '<div class="recommend-box">'
                "<strong>Recommendation:</strong> Immediate review by a biosafety expert is strongly "
                "recommended. This protein should not proceed to synthesis or experimental work "
                "without thorough manual evaluation. Consider additional wet-lab validation "
                "(e.g., functional assays) to confirm or rule out toxin-like activity."
                "</div>",
                unsafe_allow_html=True,
            )
        elif risk_score >= 0.45:
            st.markdown("---")
            st.markdown(
                '<div class="recommend-box">'
                "<strong>Recommendation:</strong> A biosafety review is recommended before proceeding. "
                "Enhanced monitoring protocols should be applied if this protein moves forward in "
                "any experimental pipeline. The moderate risk score warrants closer scrutiny but "
                "does not necessarily indicate confirmed danger."
                "</div>",
                unsafe_allow_html=True,
            )
        elif risk_score >= 0.3:
            st.markdown("---")
            st.markdown(
                '<div class="recommend-box">'
                "<strong>Recommendation:</strong> Standard biosafety protocols should be sufficient "
                "for this protein. The low similarity signals detected do not indicate meaningful "
                "toxin resemblance, but routine screening documentation should be maintained."
                "</div>",
                unsafe_allow_html=True,
            )

        # Session monitoring
        anomaly_score = factors.get("session_anomaly_score", 0.0)
        query_count = st.session_state.get("query_count", 0)
        if anomaly_score > 0 or query_count > 1:
            st.markdown("---")
            st.markdown("**Session Monitoring**")
            if anomaly_score > 0.5:
                st.warning(
                    f"Session anomaly score: {anomaly_score:.2f} — A convergent optimization pattern "
                    f"has been detected across {query_count} queries in this session. This means that "
                    f"successive submissions appear to be iteratively modifying a sequence toward "
                    f"greater similarity with a known toxin, which is a behavioral pattern of concern."
                )
            elif anomaly_score > 0.3:
                st.info(
                    f"Session anomaly score: {anomaly_score:.2f} — Elevated activity has been detected "
                    f"across {query_count} queries. The screening system is monitoring for patterns "
                    f"that might indicate iterative convergence toward dangerous sequences."
                )
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
        if st.button("Copy JSON", key=f"{key_prefix}copy_json"):
            st.code(json.dumps(data, indent=2), language="json")

    # Video generation
    if _VIDEO_AVAILABLE and data.get("pdb_string"):
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; margin: 1rem 0 0.5rem 0;">'
            '<span style="color:#94a3b8; font-size:0.85rem;">'
            'Generate an MP4 video showing the 3D structure rotating with risk annotations and stats overlay'
            '</span></div>',
            unsafe_allow_html=True,
        )
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            generate_clicked = st.button(
                "Generate Video Analysis",
                key=f"{key_prefix}gen_video",
                use_container_width=True,
                type="primary",
            )
        if generate_clicked:
            video_input = ProteinVideoData(
                pdb_string=data["pdb_string"],
                risk_score=data["risk_score"],
                risk_level=data["risk_level"],
                sequence_length=data.get("sequence_length", 0),
                top_matches=data.get("top_matches", []),
                pocket_residues=data.get("pocket_residues", []),
                danger_residues=data.get("danger_residues", []),
                risk_factors=data.get("risk_factors", {}),
                structure_predicted=data.get("structure_predicted", False),
                function_prediction=data.get("function_prediction"),
            )
            with st.spinner("Rendering video... this may take 30-90 seconds"):
                try:
                    video_bytes = generate_video(video_input)
                    st.video(video_bytes, format="video/mp4")
                    st.download_button(
                        label="Download Video",
                        data=video_bytes,
                        file_name="bioscreen_analysis.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Video generation failed: {e}")

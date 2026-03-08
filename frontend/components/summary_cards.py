import streamlit as st

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

    mode_label = "Full"
    mode_detail = "Embedding + Structure + Function"

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

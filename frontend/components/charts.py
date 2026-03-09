"""Plotly chart builders for BioScreen screening results.

Each function returns a ``plotly.graph_objects.Figure`` or ``None`` when
the input data is missing or insufficient to render a meaningful chart.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ── Color palette (matches mockup) ────────────────────────────────────────────
COLOR_RED = "#dc3545"
COLOR_ORANGE = "#d48c0e"
COLOR_GREEN = "#2a9d5c"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#6b6560"

# ── Shared layout helper ─────────────────────────────────────────────────────

def _base_layout(**overrides) -> dict:
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=COLOR_TEXT),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    defaults.update(overrides)
    return defaults


# ── 1. Radar chart ────────────────────────────────────────────────────────────

def build_radar_chart(risk_factors: dict) -> go.Figure | None:
    if not risk_factors:
        return None

    categories = [
        "Embedding Sim",
        "Structure Sim",
        "Function Overlap",
        "Active Site",
        "Session Anomaly",
    ]
    values = [
        risk_factors.get("max_embedding_similarity", 0),
        risk_factors.get("max_structure_similarity", 0) or 0,
        risk_factors.get("function_overlap", 0),
        risk_factors.get("active_site_overlap", 0) or 0,
        risk_factors.get("session_anomaly_score", 0),
    ]
    # Close the polygon
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(220,53,69,0.15)",
            line=dict(color=COLOR_RED, width=2),
            marker=dict(size=5, color=COLOR_RED),
        )
    )
    fig.update_layout(
        **_base_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            ),
            showlegend=False,
            title=dict(text="Risk Factor Radar", font=dict(size=14)),
        )
    )
    return fig


# ── 2. Donut chart ────────────────────────────────────────────────────────────

def build_donut_chart(risk_factors: dict, risk_score: float) -> go.Figure | None:
    if not risk_factors:
        return None

    emb = risk_factors.get("max_embedding_similarity", 0)
    struct = risk_factors.get("max_structure_similarity", None)
    func = risk_factors.get("function_overlap", 0)

    if struct is not None:
        weights = (0.50, 0.30, 0.20)
        labels = ["Embedding", "Structure", "Function"]
        raw_vals = [emb, struct, func]
        colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
    else:
        weights = (0.65, 0.35)
        labels = ["Embedding", "Function"]
        raw_vals = [emb, func]
        colors = [COLOR_RED, COLOR_GREEN]

    contributions = [r * w for r, w in zip(raw_vals, weights)]
    # Avoid all-zero pie
    if sum(contributions) == 0:
        contributions = [0.001] * len(contributions)

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=contributions,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            showlegend=False,
            title=dict(text="Score Composition", font=dict(size=14)),
            annotations=[
                dict(
                    text=f"<b>{risk_score:.2f}</b>",
                    x=0.5, y=0.5,
                    font=dict(size=22, color=COLOR_TEXT),
                    showarrow=False,
                )
            ],
        )
    )
    return fig


# ── 3. Matches bar chart ─────────────────────────────────────────────────────

def build_matches_bar_chart(top_matches: list[dict]) -> go.Figure | None:
    if not top_matches:
        return None

    names = [m.get("name", "?")[:20] for m in top_matches]
    emb_sims = [m.get("embedding_similarity", 0) for m in top_matches]
    struct_sims = [m.get("structure_similarity", 0) or 0 for m in top_matches]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Embedding Sim",
        x=names, y=emb_sims,
        marker_color=COLOR_TEXT,
    ))
    fig.add_trace(go.Bar(
        name="Structure Sim",
        x=names, y=struct_sims,
        marker_color=COLOR_RED,
    ))
    fig.update_layout(
        **_base_layout(
            barmode="group",
            title=dict(text="Top Matches — Similarity Comparison", font=dict(size=14)),
            yaxis=dict(range=[0, 1], title="Similarity"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    )
    return fig


# ── 4. Similarity heatmap ────────────────────────────────────────────────────

def build_similarity_heatmap(top_matches: list[dict]) -> go.Figure | None:
    if not top_matches:
        return None

    names = [m.get("name", "?")[:20] for m in top_matches]
    emb_row = [m.get("embedding_similarity", 0) for m in top_matches]
    struct_row = [m.get("structure_similarity", 0) or 0 for m in top_matches]

    z = [emb_row, struct_row]
    y_labels = ["Embedding Sim", "Structure Sim"]

    # Build annotation text
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=names[j], y=y_labels[i],
                    text=f"{val:.2f}",
                    font=dict(color="white" if val > 0.5 else COLOR_TEXT, size=11),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        go.Heatmap(
            z=z, x=names, y=y_labels,
            colorscale=[[0, "#fee"], [0.5, "#f88"], [1, COLOR_RED]],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(title="Sim", len=0.8),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Similarity Heatmap", font=dict(size=14)),
            annotations=annotations,
            yaxis=dict(autorange="reversed"),
        )
    )
    return fig


# ── 5. Waterfall chart ───────────────────────────────────────────────────────

def build_waterfall_chart(risk_factors: dict, risk_score: float) -> go.Figure | None:
    if not risk_factors:
        return None

    emb = risk_factors.get("max_embedding_similarity", 0)
    struct = risk_factors.get("max_structure_similarity", None)
    func = risk_factors.get("function_overlap", 0)

    if struct is not None:
        weights = {"Embedding": 0.50, "Structure": 0.30, "Function": 0.20}
        items = [
            ("Embedding", emb * 0.50),
            ("Structure", struct * 0.30),
            ("Function", func * 0.20),
        ]
    else:
        weights = {"Embedding": 0.65, "Function": 0.35}
        items = [
            ("Embedding", emb * 0.65),
            ("Function", func * 0.35),
        ]

    total_weighted = sum(v for _, v in items)
    bonus = max(0, risk_score - min(1.0, total_weighted))

    labels = [name for name, _ in items]
    values = [val for _, val in items]
    measures = ["relative"] * len(items)
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN][:len(items)]

    if bonus > 0.01:
        labels.append("Synergy")
        values.append(bonus)
        measures.append("relative")
        colors.append("#a78bfa")

    labels.append("Total")
    values.append(risk_score)
    measures.append("total")
    colors.append(COLOR_TEXT)

    fig = go.Figure(
        go.Waterfall(
            x=labels, y=values,
            measure=measures,
            connector=dict(line=dict(color=COLOR_MUTED, width=1)),
            increasing=dict(marker=dict(color=COLOR_RED)),
            decreasing=dict(marker=dict(color=COLOR_GREEN)),
            totals=dict(marker=dict(color=COLOR_TEXT)),
            textposition="outside",
            text=[f"{v:.3f}" for v in values],
        )
    )
    # Override bar colors
    fig.update_traces(
        marker_color=colors,
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Score Waterfall", font=dict(size=14)),
            yaxis=dict(range=[0, max(risk_score * 1.3, 0.3)], title="Score"),
            showlegend=False,
        )
    )
    return fig


# ── 6. Threshold chart ───────────────────────────────────────────────────────

def build_threshold_chart(risk_score: float) -> go.Figure:
    fig = go.Figure()

    # Stacked background segments
    fig.add_trace(go.Bar(
        x=[0.3], y=["Risk"], orientation="h",
        marker_color=COLOR_GREEN, name="Low",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[0.25], y=["Risk"], orientation="h",
        marker_color=COLOR_ORANGE, name="Medium",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[0.45], y=["Risk"], orientation="h",
        marker_color=COLOR_RED, name="High",
        showlegend=False, hoverinfo="skip",
    ))

    # Marker for current score
    fig.add_trace(go.Scatter(
        x=[risk_score], y=["Risk"],
        mode="markers+text",
        marker=dict(size=16, color=COLOR_TEXT, symbol="diamond"),
        text=[f"{risk_score:.2f}"],
        textposition="top center",
        textfont=dict(size=12, color=COLOR_TEXT),
        showlegend=False,
        hovertemplate=f"Risk Score: {risk_score:.3f}<extra></extra>",
    ))

    # Threshold labels
    annotations = [
        dict(x=0.15, y=-0.4, text="Low", showarrow=False, font=dict(size=10, color=COLOR_GREEN), yref="paper"),
        dict(x=0.425, y=-0.4, text="Medium", showarrow=False, font=dict(size=10, color=COLOR_ORANGE), yref="paper"),
        dict(x=0.775, y=-0.4, text="High", showarrow=False, font=dict(size=10, color=COLOR_RED), yref="paper"),
    ]

    fig.update_layout(
        **_base_layout(
            barmode="stack",
            xaxis=dict(range=[0, 1], showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=100,
            margin=dict(l=10, r=10, t=30, b=30),
            title=dict(text="Risk Threshold", font=dict(size=12)),
            annotations=annotations,
        )
    )
    return fig


# ── 7. Function confidence bars ──────────────────────────────────────────────

def build_function_bars(function_prediction: dict) -> go.Figure | None:
    if not function_prediction:
        return None

    items: list[tuple[str, float]] = []
    for term in function_prediction.get("go_terms", []):
        label = term.get("term", "?")
        name = term.get("name", "")
        if name:
            label = f"{label} — {name}"
        conf = float(term.get("confidence", 0))
        items.append((label, conf))

    for ec in function_prediction.get("ec_numbers", []):
        label = f"EC {ec.get('number', '?')}"
        conf = float(ec.get("confidence", 0))
        items.append((label, conf))

    if not items:
        return None

    # Sort by confidence descending
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    confs = [i[1] for i in items]

    fig = go.Figure(
        go.Bar(
            y=labels, x=confs,
            orientation="h",
            marker_color=COLOR_RED,
            text=[f"{c:.2f}" for c in confs],
            textposition="auto",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Function Prediction Confidence", font=dict(size=14)),
            xaxis=dict(range=[0, 1], title="Confidence"),
            yaxis=dict(autorange="reversed"),
            height=max(200, 40 * len(items) + 80),
        )
    )
    return fig


# ── 8. Function overlap (Venn-like data) ─────────────────────────────────────

def build_function_overlap(
    function_prediction: dict,
    top_match: dict,
) -> tuple[list[str], list[str], list[str]] | None:
    """Return (query_only, shared, match_only) GO term lists.

    Caller renders with ``st.columns`` for a Venn-like display.
    """
    if not function_prediction or not top_match:
        return None

    query_terms = {
        t.get("term", "") for t in function_prediction.get("go_terms", [])
    }
    match_terms = {
        t.get("term", "") for t in top_match.get("go_terms", [])
    }

    if not query_terms and not match_terms:
        return None

    shared = sorted(query_terms & match_terms)
    query_only = sorted(query_terms - match_terms)
    match_only = sorted(match_terms - query_terms)

    if not shared and not query_only and not match_only:
        return None

    return query_only, shared, match_only

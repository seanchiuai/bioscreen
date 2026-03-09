import streamlit as st


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
    [data-testid="stPageLink"] > a {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.82rem;
        transition: all 0.2s;
    }
    [data-testid="stPageLink"] > a:hover {
        border-color: #1a1a1a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

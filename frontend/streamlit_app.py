"""BioScreen — Structure-based biosecurity screening for AI-designed proteins.

Multipage Streamlit entry point. Configures the app and sets up navigation
between the Single Screen and Session Analysis pages.
"""

import streamlit as st

st.set_page_config(
    page_title="BioScreen",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="collapsed",
)

single_screen = st.Page("pages/single_screen.py", title="Single Screen", icon="\U0001f52c", default=True)
session_analysis = st.Page("pages/session_analysis.py", title="Session Analysis", icon="\U0001f4ca")

nav = st.navigation([single_screen, session_analysis])
nav.run()

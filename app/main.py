import streamlit as st

from app.editor.ui import editor_tab
from app.viewer.ui import map_tab


def main():
    # Page configuration
    st.set_page_config(layout="wide")
    st.markdown(
        """
    <style>
        .stAppDeployButton {display:none;}
        .stAppHeader {display:none;}
        .block-container {
               padding-top: 0.5rem;
               padding-bottom: 0rem;
            }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize state
    if "model" not in st.session_state:
        st.session_state["model"] = None

    # Page
    st.markdown(
        "<h1 style='text-align: center;'>FIM 2D Model Development Tool</h1>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        editor_tab()
    with c2:
        map_tab()


main()

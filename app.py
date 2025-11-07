"""
TwoDimFIM - 2D Hydrodynamic Flood Modeling Streamlit Application
This application supports OWP flood inundation mapping (FIM) efforts.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="TwoDimFIM",
    page_icon="🌊",
    layout="wide",
)

# Main title
st.title("🌊 TwoDimFIM")
st.subheader("2D Hydrodynamic Flood Modeling")

# Introduction
st.markdown("""
This application supports the Office of Water Prediction (OWP) flood inundation mapping (FIM) efforts 
through 2D hydrodynamic flood modeling capabilities.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **TwoDimFIM** is a tool for 2D hydrodynamic flood modeling to support 
    OWP flood inundation mapping efforts.
    """)

# Main content area
st.header("Getting Started")
st.markdown("""
Welcome to the TwoDimFIM application. This tool is designed to help with flood inundation mapping 
through advanced 2D hydrodynamic modeling.

### Features
- 🌊 2D Hydrodynamic Modeling
- 📊 Flood Inundation Mapping
- 🗺️ Support for OWP FIM Efforts

### Status
Application is ready for internal deployment.
""")

# Footer
st.divider()
st.caption("TwoDimFIM - 2D Hydrodynamic Flood Modeling")

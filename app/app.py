"""BerryBox AI – Landing page."""
import streamlit as st

st.set_page_config(
    page_title="BerryBox AI",
    page_icon="🫐",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.style import inject_css
inject_css()

st.markdown("# BerryBox AI")
st.markdown("### Cranberry / Blueberry Fruit Analysis System")
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
**📷 Interactive Mode**

Connect to the Raspberry Pi and Nikon camera for real-time capture and analysis.
Scan barcodes and process samples on the fly.

→ Navigate to **Interactive** in the sidebar
""")
with col2:
    st.markdown("""
**📁 Batch Mode**

Point to a folder of existing images and run the full analysis pipeline.
Supports both `berry-seg` and `rot-det` modules.

→ Navigate to **Batch** in the sidebar
""")
with col3:
    st.markdown("""
**📊 Results Viewer**

Explore session outputs: annotated images, detection summaries,
per-berry feature distributions, and CSV exports.

→ Navigate to **Results** in the sidebar
""")

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#57606a;">
BerryBox AI v1.0 &nbsp;·&nbsp; USDA ARS &nbsp;·&nbsp; YOLOv8-based cranberry phenotyping
</div>
""", unsafe_allow_html=True)

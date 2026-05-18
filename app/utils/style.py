"""
Shared light-mode CSS injected on every page.
Import and call inject_css() at the top of each page (after set_page_config).
"""
import streamlit as st

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background & text ── */
.stApp { background-color: #f6f8fa; color: #1f2328; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #d0d7de;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #57606a;
    margin-bottom: 0.3rem;
}

/* ── Headers ── */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #1f2328 !important; }
h2 { font-family: 'IBM Plex Mono', monospace !important; color: #0969da !important;
     font-size: 0.9rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #1f2328 !important; font-weight: 400 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label {
    color: #57606a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0969da !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.7rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.stButton > button:hover {
    background-color: #f3f4f6;
    border-color: #0969da;
    color: #0969da;
}
.stButton > button[kind="primary"] {
    background-color: #0969da;
    border-color: #0969da;
    color: #ffffff;
}
.stButton > button[kind="primary"]:hover {
    background-color: #0860ca;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background-color: #ffffff !important;
    border: 1px solid #d0d7de !important;
    color: #1f2328 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Status badges ── */
.status-ok   { display:inline-block; padding:2px 10px; background:#dafbe1; color:#116329;
               border:1px solid #aef0b5; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:.7rem; }
.status-err  { display:inline-block; padding:2px 10px; background:#ffebe9; color:#a40e26;
               border:1px solid #ffc1ba; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:.7rem; }
.status-warn { display:inline-block; padding:2px 10px; background:#fff8c5; color:#7d4e00;
               border:1px solid #f5d742; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:.7rem; }

/* ── Divider ── */
.bb-divider { border: none; border-top: 1px solid #d0d7de; margin: 1.5rem 0; }

/* ── Log box ── */
.log-box {
    background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px;
    padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #57606a; max-height: 220px; overflow-y: auto;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #d0d7de; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #57606a; padding: 0.5rem 1.25rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #0969da !important; border-bottom: 2px solid #0969da !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #57606a !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #d0d7de; border-radius: 6px; }
</style>
"""

def inject_css():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

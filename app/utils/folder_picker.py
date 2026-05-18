"""
Folder picker for BerryBox Streamlit app.

Opens the OS-native folder browser by spawning a subprocess that runs
tkinter on its own main thread — avoiding the macOS NSWindow/background-thread
crash that occurs when tkinter is called directly inside Streamlit.
"""

import os
import sys
import subprocess
import streamlit as st
from pathlib import Path


def _open_native_dialog(title: str = "Select folder") -> str:
    """Spawn a subprocess to show the native OS folder dialog on its own main thread."""
    script = (
        "import tkinter as tk;"
        "from tkinter import filedialog;"
        "root = tk.Tk();"
        "root.withdraw();"
        "root.wm_attributes('-topmost', 1);"
        f"path = filedialog.askdirectory(title={repr(title)});"
        "print(path);"
        "root.destroy()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def folder_picker(label: str, key: str, default: str = "",
                  dialog_title: str = "Select folder") -> str:
    """
    Render a labelled folder path input + Browse button.
    The text input has NO widget key — Streamlit cannot block writes to
    session state that way. The single state key drives everything.
    """
    state_key = f"_fp_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default

    st.markdown(f"**{label}**")
    col_input, col_browse = st.columns([5, 1])

    with col_browse:
        st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)
        if st.button("📂 Browse", key=f"_fp_btn_{key}", use_container_width=True):
            with st.spinner("Opening folder browser…"):
                chosen = _open_native_dialog(title=dialog_title)
            if chosen:
                st.session_state[state_key] = chosen
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_input:
        # No key= on this widget — Streamlit won't track it internally,
        # so we're free to set value= from session state after Browse.
        typed = st.text_input(
            label,
            value=st.session_state[state_key],
            placeholder="Paste a path or click Browse…",
            label_visibility="collapsed",
        )
        # If the user typed manually, persist it
        if typed != st.session_state[state_key]:
            st.session_state[state_key] = typed

    # Live validation
    path = st.session_state[state_key]
    if path:
        resolved = str(Path(path).resolve())
        if os.path.isdir(path):
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;'
                f'color:#116329;margin-top:-0.5rem;">✓ {resolved}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;'
                f'color:#7d4e00;margin-top:-0.5rem;">'
                f'⚠ Not found — will be created: {resolved}</div>',
                unsafe_allow_html=True,
            )

    return path

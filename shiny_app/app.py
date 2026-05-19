import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Shiny Imports
from shiny import reactive, render, ui
from shiny.express import input, output, ui

# 1. FIX PATHING
# Current file is at: berryboxai/shiny_app/app.py
# Weights are at:    berryboxai/data/weights/
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = BASE_DIR / "berryboxai"
WEIGHTS_DIR = BASE_DIR / "data" / "weights"
APP_DIR = BASE_DIR / "shiny_app"

print(f"--- Path Debugging ---")
print(f"App File: {__file__}")
print(f"Base Directory detected as: {BASE_DIR}")
print(f"Looking for weights in: {WEIGHTS_DIR}")
print(f"Weights folder exists: {WEIGHTS_DIR.exists()}")
print(f"----------------------")

# Ensure utils are importable
sys.path.insert(0, str(APP_DIR))

from utils.style import LIGHT_CSS
from utils.helpers import annotated_image_rgb, load_model, build_model_params

# --- GLOBAL REACTIVE STATE (Replaces st.session_state) ---
ssh_client = reactive.Value(None)
camera_ok = reactive.Value(False)
log_lines = reactive.Value([])
batch_results = reactive.Value(None)
rv_df = reactive.Value(None)

def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    new_logs = log_lines().copy()
    new_logs.append(f"[{ts}] {tag} {msg}")
    log_lines.set(new_logs)

# --- UI DEFINITION ---
ui.page_opts(title="BerryBox AI", fillable=True)

# Inject your existing CSS
ui.head_content(ui.tags.style(LIGHT_CSS))

with ui.sidebar():
    ui.markdown("## BerryBox AI")
    ui.input_select("module", "Module", {"berry-seg": "Segmentation", "rot-det": "Rot Detection"})
    
    ui.input_slider("conf", "Confidence", 0.1, 1.0, 0.5)
    ui.input_slider("iou", "IoU", 0.1, 0.9, 0.25)
    
    ui.hr()
    ui.markdown("### Raspberry Pi")
    ui.input_text("rpi_ip", "IP", "169.254.111.10")
    ui.input_password("rpi_pwd", "Password", value="usdacran")

with ui.navset_bar(title="BerryBox AI"):
    
    # --- PAGE 1: INTERACTIVE ---
    with ui.nav_panel("📷 Interactive"):
        with ui.layout_columns(col_widths=[4, 8]):
            with ui.card():
                ui.card_header("System Control")
                ui.input_action_button("btn_connect", "🔌 Connect Pi", class_="btn-primary")
                ui.input_action_button("btn_check_cam", "📷 Check Camera")
                
                @render.text
                def connection_status():
                    status = "✅ Connected" if ssh_client() else "❌ Disconnected"
                    return f"Status: {status}"

            with ui.card():
                ui.card_header("Log Output")
                @render.text
                def display_logs():
                    return "\n".join(log_lines()[-10:])

    # --- PAGE 2: BATCH ---
    with ui.nav_panel("📁 Batch"):
        with ui.card():
            ui.input_text("batch_path", "Folder Path", placeholder="C:/path/to/images")
            ui.input_action_button("btn_run_batch", "▶ Run Analysis", class_="btn-success")
        
        with ui.card():
            @render.data_frame
            def batch_table():
                if batch_results() is not None:
                    return render.DataTable(batch_results())

    # --- PAGE 3: RESULTS ---
    with ui.nav_panel("📊 Results"):
        # FIX: The error happened here. Arguments MUST be named.
        with ui.layout_columns():
            ui.value_box(
                title="Total Images", 
                value="0", 
                theme="primary"
            )
            ui.value_box(
                title="Mean % Rot", 
                value="0.0%", 
                theme="info"
            )

# --- SERVER LOGIC ---

@reactive.effect
@reactive.event(input.btn_connect)
def _connect_pi():
    try:
        # In a real scenario, weights are loaded using WEIGHTS_DIR
        # e.g. model_path = WEIGHTS_DIR / f"berrybox_{input.module()}.pt"
        add_log(f"Attempting connection to {input.rpi_ip()}...", "info")
        # Logic from your 1_interactive.py goes here
        time.sleep(1) # Simulate
        add_log("Pi Connected Successfully", "ok")
    except Exception as e:
        add_log(f"Error: {e}", "err")

@reactive.effect
@reactive.event(input.btn_run_batch)
def _run_batch_logic():
    # Logic from 2_batch.py goes here
    add_log(f"Starting batch in {input.batch_path()}", "info")
    # Simulate data result
    df = pd.DataFrame({"Image": ["img1.jpg"], "Status": ["Processed"]})
    batch_results.set(df)
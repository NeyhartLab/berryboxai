import os
import sys
import cv2
import glob
import paramiko
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Shiny Imports
from shiny import reactive, render, ui
from shiny.express import input, output, ui

# --- 1. PATHING & IMPORTS ---
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
WEIGHTS_DIR = BASE_DIR / "data" / "weights"

sys.path.insert(0, str(APP_DIR))

from utils.style import LIGHT_CSS
from utils.helpers import (
    annotated_image_rgb, 
    load_model, 
    build_model_params, 
    setup_nikon_camera
)

# --- 2. GLOBAL REACTIVE STATE ---
ssh_client = reactive.Value(None)
camera_ok = reactive.Value(False)
log_lines = reactive.Value(["[System] Ready."])
last_processed_img = reactive.Value(None)
last_metrics = reactive.Value({})

def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    # Reactive update
    current_logs = log_lines.get()
    new_logs = current_logs + [f"[{ts}] {tag} {msg}"]
    log_lines.set(new_logs)

# --- 3. MODEL LOADER ---
@reactive.calc
def get_model():
    module_name = input.module()
    task = "segment" if module_name == "berry-seg" else "detect"
    add_log(f"Loading weights for {module_name}...", "info")
    model = load_model(module_name, task, str(WEIGHTS_DIR))
    add_log(f"Model {module_name} is active.", "ok")
    return model

# --- 4. UI ---
ui.page_opts(title="BerryBox AI", fillable=True)
ui.head_content(ui.tags.style(LIGHT_CSS))

with ui.sidebar():
    ui.markdown("## BerryBox AI")
    ui.input_select("module", "Module", {"berry-seg": "Segmentation", "rot-det": "Rot Detection"})
    
    ui.input_slider("conf", "Confidence", 0.1, 1.0, 0.5, step=0.05)
    ui.input_slider("iou", "IoU", 0.1, 0.9, 0.25, step=0.05)
    
    ui.hr()
    ui.markdown("### Raspberry Pi")
    ui.input_text("rpi_ip", "IP Address", "169.254.111.10")
    ui.input_text("rpi_user", "Username", "cranpi2") # Added this back!
    ui.input_password("rpi_pwd", "Password", value="usdacran")
    
    ui.hr()
    ui.input_text("session_name", "Session ID", value=datetime.now().strftime("%Y%m%d_%H%M"))

with ui.navset_bar(title="BerryBox AI"):
    
    with ui.nav_panel("📷 Interactive"):
        with ui.layout_columns(col_widths=[4, 8]):
            with ui.card():
                ui.card_header("System Control")
                ui.input_action_button("btn_connect", "🔌 Connect Pi", class_="btn-primary")
                ui.input_action_button("btn_check_cam", "📷 Setup Nikon")
                ui.hr()
                ui.input_action_button("btn_capture", "📸 CAPTURE & ANALYZE", class_="btn-success btn-lg")
                
                @render.text
                def display_logs():
                    # Reverse to show newest at top, or keep standard
                    return "\n".join(log_lines.get()[-12:])

            with ui.layout_columns(col_widths=[12, 12]):
                with ui.card():
                    ui.card_header("Last Capture Preview")
                    @render.image
                    def interactive_preview():
                        img = last_processed_img()
                        if img is None: return None
                        temp_path = APP_DIR / "temp_interactive.jpg"
                        cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        return {"src": str(temp_path), "width": "100%"}

                with ui.layout_columns():
                    @render.ui
                    def metric_berries():
                        m = last_metrics()
                        return ui.value_box(title="Total Berries", value=str(m.get("total", 0)), theme="primary")
                    
                    @render.ui
                    def metric_rot():
                        m = last_metrics()
                        val = f"{m.get('pct_rot', 0):.1f}%"
                        return ui.value_box(title="Rot Percentage", value=val, theme="danger")

    with ui.nav_panel("📁 Batch"):
        with ui.card():
            ui.input_text("batch_path", "Folder Path", placeholder="C:/path/to/images")
            ui.input_action_button("btn_run_batch", "▶ Run Analysis", class_="btn-success")
        
        with ui.card():
            @render.data_frame
            def batch_table():
                df = batch_results()
                if df is not None:
                    return render.DataTable(df)

# --- 5. SERVER LOGIC ---

@reactive.effect
@reactive.event(input.btn_connect)
def _connect_pi():
    # 1. Provide immediate feedback
    add_log(f"Connecting to {input.rpi_ip()}...", "info")
    
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Use the inputs from sidebar
        client.connect(
            input.rpi_ip(), 
            username=input.rpi_user(), 
            password=input.rpi_pwd(), 
            timeout=8
        )
        
        ssh_client.set(client)
        add_log("SSH connection established.", "ok")
    except Exception as e:
        add_log(f"Connection failed: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_check_cam)
def _check_cam():
    client = ssh_client.get()
    if not client:
        add_log("Cannot setup: SSH not connected.", "warn")
        return
    
    try:
        add_log("Initializing Nikon D7500 settings...", "info")
        # setup_nikon_camera is now in helpers.py
        msg = setup_nikon_camera(client)
        camera_ok.set(True)
        add_log(msg, "ok")
    except Exception as e:
        camera_ok.set(False)
        add_log(f"Nikon error: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_capture)
def _handle_capture():
    client = ssh_client.get()
    if not client:
        add_log("Connect SSH first!", "err")
        return
    if not camera_ok.get():
        add_log("Setup camera first!", "warn")
        return

    try:
        with ui.Progress(min=1, max=4) as p:
            p.set(1, message="Nikon: Triggering shutter...")
            remote_img = "/tmp/capture.jpg"
            # Command specifically for Nikon tethering
            cmd = f"gphoto2 --capture-image-and-download --filename {remote_img} --force-overwrite"
            stdin, stdout, stderr = client.exec_command(cmd)
            
            # Wait for command to finish
            if stdout.channel.recv_exit_status() != 0:
                err_msg = stderr.read().decode()
                add_log(f"Capture failed: {err_msg}", "err")
                return

            p.set(2, message="Downloading image...")
            local_raw = APP_DIR / "last_raw.jpg"
            sftp = client.open_sftp()
            sftp.get(remote_img, str(local_raw))
            sftp.close()

            p.set(3, message="AI Inference...")
            model = get_model()
            img_bgr = cv2.imread(str(local_raw))
            
            cfg = {"module": input.module(), "conf": input.conf(), "iou": input.iou(), "imgsz": (1600, 2400)}
            params = build_model_params(cfg)
            results = model.predict(img_bgr, **params)
            res = results[0]

            p.set(4, message="Finalizing UI...")
            class_names = ["berry", "rotten", "sound"]
            annotated = annotated_image_rgb(img_bgr, res, class_names)
            
            total = len(res.boxes)
            rotten = int(sum(res.boxes.cls == 1)) if total > 0 else 0
            pct = (rotten / total * 100) if total > 0 else 0

            last_processed_img.set(annotated)
            last_metrics.set({"total": total, "pct_rot": pct})
            add_log(f"Analyzed {total} objects.", "ok")

    except Exception as e:
        add_log(f"Analysis Error: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_run_batch)
def _handle_batch():
    # (Same batch logic as before, just use ssh_client.get() or get_model())
    pass
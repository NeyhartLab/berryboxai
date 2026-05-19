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
# File is at berryboxai/shiny_app/app.py
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent  # This is the 'berryboxai' root
WEIGHTS_DIR = BASE_DIR / "data" / "weights"

# Add shiny_app to path so we can find utils
sys.path.insert(0, str(APP_DIR))

from utils.style import LIGHT_CSS
from utils.helpers import annotated_image_rgb, load_model, build_model_params

# --- 2. GLOBAL REACTIVE STATE ---
ssh_client = reactive.Value(None)
camera_ok = reactive.Value(False)
log_lines = reactive.Value([])
batch_results = reactive.Value(None)
last_processed_img = reactive.Value(None)
last_metrics = reactive.Value({})

def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    new_logs = log_lines().copy()
    new_logs.append(f"[{ts}] {tag} {msg}")
    log_lines.set(new_logs)

# --- 3. SMART MODEL LOADER (Optimized) ---
@reactive.calc
def get_model():
    """ 
    Loads the YOLO model into memory. 
    Only re-runs if the 'Module' selection changes in the sidebar.
    """
    module_name = input.module()
    task = "segment" if module_name == "berry-seg" else "detect"
    
    add_log(f"Initializing {module_name} model weights...", "info")
    # We pass the absolute path to our weights folder
    model = load_model(module_name, task, str(WEIGHTS_DIR))
    add_log(f"Model {module_name} loaded and ready.", "ok")
    return model

# --- 4. UI DEFINITION ---
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
    ui.input_password("rpi_pwd", "Password", value="usdacran")
    
    ui.hr()
    ui.markdown("### Session")
    ui.input_text("session_name", "Session ID", value=datetime.now().strftime("%Y%m%d_%H%M"))

with ui.navset_bar(title="BerryBox AI"):
    
    # --- PAGE 1: INTERACTIVE ---
    with ui.nav_panel("📷 Interactive"):
        with ui.layout_columns(col_widths=[4, 8]):
            with ui.card():
                ui.card_header("System Control")
                ui.input_action_button("btn_connect", "🔌 Connect Pi", class_="btn-primary")
                ui.input_action_button("btn_check_cam", "📷 Check Camera")
                ui.hr()
                ui.input_action_button("btn_capture", "📸 CAPTURE & ANALYZE", class_="btn-success btn-lg")
                
                @render.text
                def display_logs():
                    return "\n".join(log_lines()[-10:])

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

    # --- PAGE 2: BATCH ---
    with ui.nav_panel("📁 Batch"):
        with ui.card():
            ui.card_header("Batch Input")
            ui.input_text("batch_path", "Folder Path", placeholder="C:/path/to/images")
            ui.input_action_button("btn_run_batch", "▶ Run Analysis", class_="btn-success")
        
        with ui.card():
            @render.data_frame
            def batch_table():
                if batch_results() is not None:
                    return render.DataTable(batch_results())

    # --- PAGE 3: RESULTS ---
    with ui.nav_panel("📊 Results"):
        with ui.layout_columns():
            @render.ui
            def total_images_box():
                df = batch_results()
                val = str(len(df)) if df is not None else "0"
                return ui.value_box(title="Total Images", value=val, theme="primary")

            @render.ui
            def mean_rot_box():
                df = batch_results()
                val = f"{df['FruitRotPer'].mean():.1f}%" if df is not None and 'FruitRotPer' in df.columns else "0.0%"
                return ui.value_box(title="Mean % Rot", value=val, theme="info")

# --- 5. SERVER LOGIC ---

# 5.1 SSH Operations
@reactive.effect
@reactive.event(input.btn_connect)
def _connect_pi():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(input.rpi_ip(), username="cranpi2", password=input.rpi_pwd(), timeout=5)
        ssh_client.set(client)
        add_log("SSH connection successful.", "ok")
    except Exception as e:
        add_log(f"Connection failed: {e}", "err")

@reactive.effect
@reactive.event(input.btn_check_cam)
def _check_cam():
    client = ssh_client()
    if not client:
        add_log("Must connect SSH first.", "warn")
        return
    _, stdout, _ = client.exec_command("libcamera-still --list-cameras")
    if "Available cameras" in stdout.read().decode():
        camera_ok.set(True)
        add_log("Camera ready.", "ok")
    else:
        camera_ok.set(False)
        add_log("No camera detected.", "err")

# 5.2 Interactive Capture & Inference
@reactive.effect
@reactive.event(input.btn_capture)
def _handle_capture():
    client = ssh_client()
    if not client:
        add_log("SSH not connected!", "err")
        return

    try:
        with ui.Progress(min=1, max=4) as p:
            # 1. Trigger Pi
            p.set(1, message="RPi: Taking photo...")
            remote_img = "/tmp/capture.jpg"
            client.exec_command(f"libcamera-still -o {remote_img} --immediate --nopreview --width 2400 --height 1600")
            
            # 2. SFTP Transfer
            p.set(2, message="Transferring image to local...")
            local_raw = APP_DIR / "last_raw.jpg"
            sftp = client.open_sftp()
            sftp.get(remote_img, str(local_raw))
            sftp.close()

            # 3. Inference (USING CACHED MODEL)
            p.set(3, message="Running AI Inference...")
            model = get_model() # This is fast after the first load
            img_bgr = cv2.imread(str(local_raw))
            
            cfg = {"module": input.module(), "conf": input.conf(), "iou": input.iou(), "imgsz": (1600, 2400)}
            params = build_model_params(cfg)
            results = model.predict(img_bgr, **params)
            res = results[0]

            # 4. Processing Results
            p.set(4, message="Rendering results...")
            class_names = ["berry", "rotten", "sound"]
            annotated = annotated_image_rgb(img_bgr, res, class_names)
            
            # Calculate metrics
            total = len(res.boxes)
            rotten = int(sum(res.boxes.cls == 1)) if total > 0 else 0
            pct = (rotten / total * 100) if total > 0 else 0

            # Update UI
            last_processed_img.set(annotated)
            last_metrics.set({"total": total, "pct_rot": pct})
            add_log(f"Analyzed {total} berries.", "ok")

    except Exception as e:
        add_log(f"Capture Error: {e}", "err")

# 5.3 Batch Operations
@reactive.effect
@reactive.event(input.btn_run_batch)
def _handle_batch():
    in_dir = Path(input.batch_path())
    if not in_dir.exists():
        add_log("Folder path not found.", "err")
        return

    images = glob.glob(str(in_dir / "*.jpg")) + glob.glob(str(in_dir / "*.png"))
    if not images:
        add_log("No images found in folder.", "warn")
        return

    add_log(f"Starting batch of {len(images)} images...", "info")
    model = get_model()
    cfg = {"module": input.module(), "conf": input.conf(), "iou": input.iou(), "imgsz": (1600, 2400)}
    params = build_model_params(cfg)
    
    rows = []
    with ui.Progress(min=1, max=len(images)) as p:
        for i, path in enumerate(images):
            p.set(i+1, message=f"Processing {Path(path).name}")
            img = cv2.imread(path)
            res = model.predict(img, **params)[0]
            
            total = len(res.boxes)
            rotten = int(sum(res.boxes.cls == 1)) if total > 0 else 0
            
            rows.append({
                "Image Name": Path(path).name,
                "Total Berries": total,
                "FruitRotPer": (rotten/total*100) if total > 0 else 0
            })

    batch_results.set(pd.DataFrame(rows))
    add_log("Batch analysis complete.", "ok")
"""Page 1 – Interactive Mode: RPi + Nikon live capture loop."""

import streamlit as st
import os, time, cv2, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="BerryBox · Interactive", page_icon="📷", layout="wide")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.style        import inject_css
from utils.sidebar      import render_sidebar
from utils.helpers      import annotated_image_rgb, load_model, build_model_params, ensure_session_dirs

inject_css()
cfg = render_sidebar()

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("ssh_client",None),("camera_ok",False),("session_active",False),
              ("session_name",""),("paths",{}),("log_lines",[]),
              ("images_captured",0),("last_result_img",None),("last_metrics",{}),
              ("cumulative_df",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def log(msg, level="info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    st.session_state.log_lines.append(f"[{ts}] {tag} {msg}")

def badge(ok, ok_lbl, err_lbl):
    cls = "status-ok" if ok else "status-err"
    return f'<span class="{cls}">{"" if ok else ""} {ok_lbl if ok else err_lbl}</span>'

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("# 📷 Interactive Mode")
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

# Connection status row
st.markdown("## Connection Status")
c1, c2, _ = st.columns([1, 1, 3])
with c1:
    st.markdown(f"**Raspberry Pi** &nbsp; {badge(st.session_state.ssh_client is not None, 'CONNECTED', 'DISCONNECTED')}",
                unsafe_allow_html=True)
with c2:
    st.markdown(f"**Nikon D7500** &nbsp; {badge(st.session_state.camera_ok, 'READY', 'NOT DETECTED')}",
                unsafe_allow_html=True)

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

# Connect / Check / Disconnect
b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    if st.button("🔌  Connect to RPi", use_container_width=True,
                 disabled=st.session_state.ssh_client is not None):
        with st.spinner("Connecting…"):
            try:
                import paramiko
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(cfg["rpi_ip"], username=cfg["rpi_user"],
                            password=cfg["rpi_pwd"], timeout=10)
                st.session_state.ssh_client = ssh
                log(f"SSH connected to {cfg['rpi_ip']}", "ok")
                st.rerun()
            except Exception as e:
                log(f"SSH failed: {e}", "err")
                st.error(str(e))

with b2:
    if st.button("📷  Check Camera", use_container_width=True,
                 disabled=st.session_state.ssh_client is None):
        try:
            _, stdout, _ = st.session_state.ssh_client.exec_command("gphoto2 --auto-detect")
            ok = "Nikon DSC D7500" in stdout.read().decode()
            st.session_state.camera_ok = ok
            log("Nikon D7500 ready" if ok else "Camera not found", "ok" if ok else "warn")
            st.rerun()
        except Exception as e:
            log(f"Camera check failed: {e}", "err")

with b3:
    if st.button("⏹  Disconnect", use_container_width=True,
                 disabled=st.session_state.ssh_client is None):
        try: st.session_state.ssh_client.close()
        except: pass
        st.session_state.ssh_client = None
        st.session_state.camera_ok  = False
        log("SSH disconnected", "info")
        st.rerun()

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

# Session start/stop
st.markdown("## Session")
s1, s2 = st.columns([1, 2])
with s1:
    if not st.session_state.session_active:
        if st.button("▶  Start Session", type="primary", use_container_width=True,
                     disabled=st.session_state.ssh_client is None):
            ts    = datetime.now().strftime("%Y-%m-%d")
            name  = f"berrybox_{cfg['module']}_{ts}"
            paths = ensure_session_dirs(cfg["output_dir"], cfg["module"], name)
            st.session_state.update(session_name=name, paths=paths,
                                    session_active=True, images_captured=0,
                                    cumulative_df=None)
            ssh = st.session_state.ssh_client
            try:
                for cmd in ["pkill -f gphoto2",
                            "gphoto2 --set-config iso=100",
                            "gphoto2 --set-config whitebalance=7",
                            "gphoto2 --set-config /main/capturesettings/f-number=7.1",
                            "gphoto2 --set-config /main/capturesettings/shutterspeed=25"]:
                    ssh.exec_command(cmd); time.sleep(1.5)
                log("Camera configured (ISO100, WB7, f/7.1, 1/25s)", "ok")
            except Exception as e:
                log(f"Camera config warning: {e}", "warn")
            log(f"Session started: {name}", "ok")
            st.rerun()
    else:
        if st.button("⏹  End Session", use_container_width=True):
            st.session_state.session_active = False
            log("Session ended", "info")
            st.rerun()

with s2:
    if st.session_state.session_active:
        st.markdown(
            f'<span class="status-ok">SESSION ACTIVE</span> &nbsp;'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.78rem;color:#57606a;">'
            f'{st.session_state.session_name}</span>',
            unsafe_allow_html=True)

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

# Capture interface
if st.session_state.session_active:
    st.markdown("## Capture")
    barcode = st.text_input("Sample ID / Barcode",
                            placeholder="Scan or type sample ID, then press Enter…")
    cap_btn = st.button("📸  Capture & Analyse", type="primary",
                        disabled=not barcode.strip())

    if cap_btn and barcode.strip():
        paths  = st.session_state.paths
        ssh    = st.session_state.ssh_client
        module = cfg["module"]
        bc     = barcode.strip()

        with st.spinner("Triggering camera…"):
            try:
                remote = "/home/cranpi2/berrybox/captured_image.jpg"
                ssh.exec_command(
                    f"gphoto2 --capture-image-and-download --filename {remote} --force-overwrite")
                time.sleep(3)
                from scp import SCPClient
                ts_str     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                local_path = os.path.join(paths["raw_image_dir"], f"{bc}_{ts_str}.jpg")
                scp = SCPClient(ssh.get_transport())
                scp.get(remote, local_path); scp.close()
                log(f"Image captured: {os.path.basename(local_path)}", "ok")
            except Exception as e:
                log(f"Capture failed: {e}", "err"); st.error(str(e)); st.stop()

        with st.spinner("Running inference…"):
            try:
                model        = load_model(module, cfg["model_path"])
                model_params = build_model_params(cfg)
                newH, newW   = cfg["imgsz"]
                image  = cv2.imread(local_path)
                image  = cv2.resize(image, (newW, newH))
                result = model.predict(source=image, **model_params)[0].to("cpu")
                log("Inference complete", "ok")
            except Exception as e:
                log(f"Inference failed: {e}", "err"); st.error(str(e)); st.stop()

        with st.spinner("Extracting features…"):
            try:
                from berryboxai import functions as fn
                image_name = os.path.basename(local_path)
                current_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                if module == "berry-seg":
                    if not cfg["no_cc"]:
                        result, cc_ps = fn.color_correction(result)
                        cc_ps = float(np.min(cc_ps))
                        if cfg["save_cc"]:
                            cv2.imwrite(os.path.join(paths["cc_dir"], "cc_" + image_name), result.orig_img)
                    else:
                        cc_ps = float(cfg["patch_size"])
                    df1 = fn.get_all_features_parallel(result, name="berry")
                    df2 = fn.get_all_features_parallel(result, name="rotten")
                    df  = pd.concat([pd.DataFrame({"name":(["berry"]*len(df1))+(["rotten"]*len(df2))}),
                                     pd.concat([df1, df2], ignore_index=True)], axis=1)
                    w = len(df)
                    df.insert(0,"Date",current_dt); df.insert(1,"Image Name",image_name)
                    df.insert(2,"QR_info",bc); df.insert(3,"Object_ID",range(w)); df.insert(4,"Patch_size",cc_ps)
                    df["Ellipsoid_model_volume"] = (4/3)*np.pi*(df["RP_Minor_axis_length"]/2)*((df["RP_Major_axis_length"]/2)**2)
                    df["Eccentricity"] = np.sqrt(1-((0.5*df["RP_Major_axis_length"])**2/(0.5*df["RP_Minor_axis_length"])**2))
                    df = df.sort_values(["RP_BB_y","RP_BB_x"]).reset_index(drop=True)
                    df["Object_ID"] = df.index
                    metrics = {"berries_detected": w, "rotten_count": int((df["name"] == "rotten").sum())}
                    class_names = ["ColorCard", "berry", "info", "rotten"]

                elif module == "rot-det":
                    from berryboxai import main as bb_main
                    n_total,n_rotten,n_sound,perc_rot,wperc = bb_main.summarize_rot_det_results(result)
                    df = pd.DataFrame([{"Date":current_dt,"Image Name":image_name,"QR_info":bc,
                                        "NumberSoundBerries":n_sound,"NumberRottenBerries":n_rotten,
                                        "FruitRotPer":perc_rot,"FruitRotPerWtd":wperc}])
                    metrics = {"total":n_total,"sound":n_sound,"rotten":n_rotten,"pct_rot":perc_rot}
                    class_names = ["rotten","sound"]

                ff = paths["feature_file"]
                if os.path.exists(ff):
                    df = pd.concat([pd.read_csv(ff), df], ignore_index=True)
                df.to_csv(ff, index=False)
                st.session_state.cumulative_df = pd.read_csv(ff)

                ann_rgb = annotated_image_rgb(result.orig_img, result, class_names,
                                             show_masks=(module == "berry-seg"),
                                             show_count=(module == "rot-det"))
                cv2.imwrite(os.path.join(paths["pred_dir"], image_name),
                            cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))
                st.session_state.last_result_img = ann_rgb
                st.session_state.last_metrics    = metrics
                st.session_state.images_captured += 1
                log(f"Sample '{bc}' complete", "ok")
            except Exception as e:
                log(f"Feature extraction failed: {e}", "err"); st.error(str(e))
        st.rerun()

    # Last result
    if st.session_state.last_result_img is not None:
        st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
        st.markdown("## Last Result")
        img_col, met_col = st.columns([3, 1])
        with img_col:
            st.image(st.session_state.last_result_img, use_container_width=True)
        with met_col:
            m = st.session_state.last_metrics
            if cfg["module"] == "rot-det":
                st.metric("Total Berries", m.get("total", "—"))
                st.metric("Sound",         m.get("sound", "—"))
                st.metric("Rotten",        m.get("rotten", "—"))
                st.metric("% Rot",         f"{m.get('pct_rot', 0):.1f}%")
            else:
                st.metric("Berries Detected", m.get("berries_detected", "—"))
                st.metric("Rotten (seg)",     m.get("rotten_count", "—"))
            st.metric("Images This Session", st.session_state.images_captured)

    # Cumulative table
    if st.session_state.cumulative_df is not None:
        st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
        st.markdown("## Session Data")
        st.dataframe(st.session_state.cumulative_df, use_container_width=True, height=260)
        st.download_button("⬇  Download CSV",
                           data=st.session_state.cumulative_df.to_csv(index=False).encode(),
                           file_name=f"{st.session_state.session_name}_features.csv",
                           mime="text/csv")

# Activity log
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
st.markdown("## Activity Log")
log_html = "<br>".join(st.session_state.log_lines[-60:]) or "No activity yet."
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

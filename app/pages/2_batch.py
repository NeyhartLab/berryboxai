"""Page 2 – Batch Mode."""

import streamlit as st
import os, cv2, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="BerryBox · Batch", page_icon="📁", layout="wide")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.style         import inject_css
from utils.sidebar       import render_sidebar
from utils.helpers       import annotated_image_rgb, load_model, build_model_params, ensure_session_dirs
from utils.folder_picker import folder_picker

inject_css()
cfg = render_sidebar()

for k, v in [("batch_running",False),("batch_log",[]),("batch_results",None),
              ("batch_thumbs",[]),("batch_session",""),("batch_paths",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

def log(msg, level="info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    st.session_state.batch_log.append(f"[{ts}] {tag} {msg}")

st.markdown("# 📁 Batch Mode")
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
st.markdown("## Input")

input_dir = folder_picker("Image folder", key="batch_input_dir", default="")

col_ext, _ = st.columns([1, 3])
with col_ext:
    ext = st.selectbox("File extension", [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"])

image_list = []
if input_dir and os.path.isdir(input_dir):
    image_list = [f for f in os.listdir(input_dir)
                  if f.upper().endswith(ext.lstrip(".").upper())]
    st.caption(f"**{len(image_list)}** images found in `{input_dir}`")
elif input_dir:
    st.warning("Directory not found.")

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

run_btn = st.button("▶  Run Batch", type="primary",
                    disabled=(len(image_list) == 0 or st.session_state.batch_running))

if run_btn and image_list:
    st.session_state.update(batch_running=True, batch_log=[], batch_thumbs=[], batch_results=None)
    module = cfg["module"]
    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sname  = f"berrybox_{module}_batch_{ts}"
    paths  = ensure_session_dirs(cfg["output_dir"], module, sname)
    st.session_state.batch_session = sname
    st.session_state.batch_paths   = paths
    feature_file = paths["feature_file"]
    log(f"Batch session: {sname}", "ok")
    log(f"Images to process: {len(image_list)}", "info")

    model        = load_model(module, cfg["model_path"])
    model_params = build_model_params(cfg)
    newH, newW   = cfg["imgsz"]
    patch_size_use = 0.0

    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    for p, img_name in enumerate(image_list):
        local_path = os.path.join(input_dir, img_name)
        status_text.caption(f"Processing {p+1}/{len(image_list)}: {img_name}")
        try:
            image  = cv2.imread(local_path)
            image  = cv2.resize(image, (newW, newH))
            result = model.predict(source=image, **model_params)[0].to("cpu")
            current_date = datetime.now().strftime("%Y-%m-%d")

            if module == "berry-seg":
                from berryboxai import functions as fn
                if not cfg["no_cc"]:
                    result, patch_size = fn.color_correction(result)
                    patch_size = float(np.min(patch_size))
                    if cfg["save_cc"]:
                        cv2.imwrite(os.path.join(paths["cc_dir"], "cc_" + img_name), result.orig_img)
                else:
                    patch_size = float(cfg["patch_size"])
                if patch_size > 0 and (patch_size < patch_size_use or patch_size_use == 0):
                    patch_size_use = patch_size
                if cfg["no_qr"]:
                    barcode = img_name
                else:
                    try:
                        info_ids = fn.get_ids(result, 'info')
                        barcode  = fn.read_QR_code(result) if len(info_ids) and any(result.boxes.cls == info_ids[0]) else img_name
                    except Exception:
                        barcode = img_name
                df1 = fn.get_all_features_parallel(result, name="berry")
                df2 = fn.get_all_features_parallel(result, name="rotten")
                df  = pd.concat([pd.DataFrame({"name":(["berry"]*len(df1))+(["rotten"]*len(df2))}),
                                 pd.concat([df1,df2],ignore_index=True)], axis=1)
                w = len(df)
                if w == 0:
                    log(f"{img_name}: no berries, skipping", "warn"); continue
                df.insert(0,"Date",current_date); df.insert(1,"Image Name",img_name)
                df.insert(2,"QR_info",barcode); df.insert(3,"Object_ID",range(w)); df.insert(4,"Patch_size",patch_size)
                df["Ellipsoid_model_volume"] = (4/3)*np.pi*(df["RP_Minor_axis_length"]/2)*((df["RP_Major_axis_length"]/2)**2)
                df["Eccentricity"] = np.sqrt(1-((0.5*df["RP_Major_axis_length"])**2/(0.5*df["RP_Minor_axis_length"])**2))
                df = df.sort_values(["RP_BB_y","RP_BB_x"]).reset_index(drop=True)
                df["Object_ID"] = df.index
                class_names = ["ColorCard","berry","info","rotten"]
                log(f"{img_name}: {w} berries", "ok")

            elif module == "rot-det":
                from berryboxai import functions as fn
                from berryboxai import main as bb_main
                n_total,n_rotten,n_sound,perc_rot,wperc = bb_main.summarize_rot_det_results(result)
                if n_total == 0:
                    log(f"{img_name}: no berries, skipping", "warn"); continue
                df = pd.DataFrame([{"Date":current_date,"Image Name":img_name,"QR_info":img_name,
                                    "NumberSoundBerries":n_sound,"NumberRottenBerries":n_rotten,
                                    "FruitRotPer":perc_rot,"FruitRotPerWtd":wperc}])
                class_names = ["rotten","sound"]
                log(f"{img_name}: {n_sound} sound, {n_rotten} rotten ({perc_rot:.1f}%)", "ok")

            if os.path.exists(feature_file):
                df = pd.concat([pd.read_csv(feature_file), df], ignore_index=True)
            df.to_csv(feature_file, index=False)

            ann_rgb = annotated_image_rgb(result.orig_img, result, class_names,
                                         show_masks=(module=="berry-seg"),
                                         show_count=(module=="rot-det"))
            cv2.imwrite(os.path.join(paths["pred_dir"], img_name), cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))
            st.session_state.batch_thumbs.append((img_name, ann_rgb))

        except Exception as e:
            log(f"{img_name}: error — {e}", "err")

        progress_bar.progress((p+1)/len(image_list), text=f"{p+1}/{len(image_list)} processed")

    # cm conversion for berry-seg
    if module == "berry-seg" and os.path.exists(feature_file) and patch_size_use > 0:
        try:
            df_f = pd.read_csv(feature_file)
            cpp  = float(cfg["patch_size"]) / patch_size_use
            df_f["Area"]   = df_f["RP_Area"]              * (cpp**2)
            df_f["Length"] = df_f["RP_Minor_axis_length"] * cpp
            df_f["Width"]  = df_f["RP_Major_axis_length"] * cpp
            df_f["Ellipsoid_model_volume"] = df_f["Ellipsoid_model_volume"] * (cpp**3)
            df_f["cm_per_pixel"] = cpp
            if cfg["reduce_features"]:
                keep = ["Date","Image Name","QR_info","Object_ID","Patch_size","name",
                        "Area","Length","Width","Ellipsoid_model_volume","Eccentricity",
                        "Red_Color_Mean","Red_Color_Median","Red_Color_Std",
                        "Green_Color_Mean","Green_Color_Median","Green_Color_Std",
                        "Blue_Color_Mean","Blue_Color_Median","Blue_Color_Std",
                        "L_Color_Mean","L_Color_Median","L_Color_Std",
                        "a_Color_Mean","a_Color_Median","a_Color_Std",
                        "b_Color_Mean","b_Color_Median","b_Color_Std"]
                df_f = df_f[[c for c in keep if c in df_f.columns]]
            else:
                df_f = df_f.drop(columns=[c for c in ["RP_Area","RP_Major_axis_length","RP_Minor_axis_length"] if c in df_f.columns])
            df_f.to_csv(feature_file, index=False)
            log("Units converted to cm", "ok")
        except Exception as e:
            log(f"cm conversion failed: {e}", "warn")

    if os.path.exists(feature_file):
        st.session_state.batch_results = pd.read_csv(feature_file)

    progress_bar.progress(1.0, text="✓ Complete")
    status_text.empty()
    st.session_state.batch_running = False
    log(f"Done. Results saved to: {feature_file}", "ok")
    st.rerun()

# Results
if st.session_state.batch_results is not None:
    df_out = st.session_state.batch_results
    st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
    st.markdown("## Results")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Images Processed", df_out["Image Name"].nunique() if "Image Name" in df_out else "—")
    with m2: st.metric("Total Records",    len(df_out))
    if "FruitRotPer" in df_out.columns:
        with m3: st.metric("Mean % Rot", f"{df_out['FruitRotPer'].mean():.1f}%")
    elif "name" in df_out.columns:
        with m3: st.metric("Rotten Detected", int((df_out["name"]=="rotten").sum()))
    st.dataframe(df_out, use_container_width=True, height=300)
    st.download_button("⬇  Download CSV", data=df_out.to_csv(index=False).encode(),
                       file_name=f"{st.session_state.batch_session}_features.csv", mime="text/csv")

# Thumbnail grid
if st.session_state.batch_thumbs:
    st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
    st.markdown("## Annotated Previews")
    cols = st.columns(4)
    for i, (name, img_rgb) in enumerate(st.session_state.batch_thumbs):
        with cols[i % 4]:
            st.image(img_rgb, caption=name, use_container_width=True)

# Log
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
st.markdown("## Activity Log")
log_html = "<br>".join(st.session_state.batch_log[-80:]) or "No activity yet."
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

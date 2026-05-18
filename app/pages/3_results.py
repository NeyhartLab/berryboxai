"""Page 3 – Results Viewer."""

import streamlit as st
import os, glob, numpy as np, pandas as pd
from pathlib import Path

st.set_page_config(page_title="BerryBox · Results", page_icon="📊", layout="wide")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.style         import inject_css
from utils.sidebar       import render_sidebar
from utils.folder_picker import folder_picker

inject_css()
cfg = render_sidebar()

for k, v in [("rv_df",None),("rv_session_name",""),("rv_pred_dir","")]:
    if k not in st.session_state:
        st.session_state[k] = v

def find_sessions(base_dir):
    sessions = []
    if not os.path.isdir(base_dir):
        return sessions
    for session_dir in sorted(Path(base_dir).iterdir(), reverse=True):
        if not session_dir.is_dir(): continue
        output_sub = session_dir / "output"
        csvs   = list(output_sub.glob("*_features.csv")) if output_sub.exists() else []
        pred_d = output_sub / "predictions"
        images = list(pred_d.glob("*.jpg")) + list(pred_d.glob("*.png")) if pred_d.exists() else []
        if csvs:
            sessions.append({"name":session_dir.name,"csv":str(csvs[0]),
                             "n_images":len(images),"pred_dir":str(pred_d)})
    return sessions

def is_rot_det(df): return "FruitRotPer" in df.columns

st.markdown("# 📊 Results Viewer")
st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

# Session browser
st.markdown("## Session History")
sessions = find_sessions(cfg["output_dir"])

if not sessions:
    st.info("No sessions found in the configured output directory.")
else:
    opts = {s["name"]: s for s in sessions}
    chosen_name = st.selectbox("Select session", list(opts.keys()),
                               format_func=lambda n: f"📂 {n}  ({opts[n]['n_images']} images)")
    chosen = opts[chosen_name]
    col_load, col_info = st.columns([1, 3])
    with col_load:
        if st.button("📂  Load Session", type="primary"):
            st.session_state.rv_df           = pd.read_csv(chosen["csv"])
            st.session_state.rv_session_name = chosen_name
            st.session_state.rv_pred_dir     = chosen["pred_dir"]
            st.rerun()
    with col_info:
        st.caption(f"CSV: `{chosen['csv']}`")

with st.expander("Or load a CSV file directly"):
    manual_csv  = st.text_input("CSV path", placeholder="/path/to/features.csv")
    manual_pred = folder_picker("Predictions folder (optional)", key="rv_pred_manual", default="")
    if st.button("Load CSV"):
        if os.path.isfile(manual_csv):
            st.session_state.rv_df           = pd.read_csv(manual_csv)
            st.session_state.rv_session_name = Path(manual_csv).stem
            st.session_state.rv_pred_dir     = manual_pred
            st.rerun()
        else:
            st.error("File not found.")

df = st.session_state.rv_df
if df is None:
    st.stop()

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)
st.markdown(f"## {st.session_state.rv_session_name}")
rot_det_mode = is_rot_det(df)

# Summary metrics
st.markdown("## Summary Statistics")
if rot_det_mode:
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("Total Images",  df["Image Name"].nunique() if "Image Name" in df else len(df))
    with m2: st.metric("Total Berries", int(df["NumberSoundBerries"].sum()+df["NumberRottenBerries"].sum()))
    with m3: st.metric("Total Sound",   int(df["NumberSoundBerries"].sum()))
    with m4: st.metric("Total Rotten",  int(df["NumberRottenBerries"].sum()))
    with m5: st.metric("Mean % Rot",    f"{df['FruitRotPer'].mean():.2f}%")
else:
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Total Images",    df["Image Name"].nunique() if "Image Name" in df else "—")
    with m2: st.metric("Total Detections",len(df))
    with m3: st.metric("Sound Berries",   int((df["name"]=="berry").sum()) if "name" in df.columns else "—")
    with m4: st.metric("Rotten (seg)",    int((df["name"]=="rotten").sum()) if "name" in df.columns else "—")
    if "Area" in df.columns:
        m5,m6,m7 = st.columns(3)
        with m5: st.metric("Mean Area (cm²)",  f"{df['Area'].mean():.3f}")
        with m6: st.metric("Mean Length (cm)", f"{df['Length'].mean():.3f}" if "Length" in df.columns else "—")
        with m7: st.metric("Mean Width (cm)",  f"{df['Width'].mean():.3f}"  if "Width"  in df.columns else "—")

st.markdown('<hr class="bb-divider">', unsafe_allow_html=True)

tab_data, tab_charts, tab_images, tab_export = st.tabs(
    ["📋 Data Table","📈 Distributions","🖼 Annotated Images","⬇ Export"])

# ── Tab 1: Data ───────────────────────────────────────────────────────────────
with tab_data:
    all_cols  = list(df.columns)
    show_cols = st.multiselect("Columns to display", all_cols,
                               default=all_cols[:min(12,len(all_cols))])
    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=420)
    if rot_det_mode and "Image Name" in df.columns:
        st.markdown("#### Per-Image Summary")
        summary = df.groupby("Image Name").agg(
            Sound  =("NumberSoundBerries","sum"), Rotten=("NumberRottenBerries","sum"),
            PctRot =("FruitRotPer","mean"),       WtdRot=("FruitRotPerWtd","mean")
        ).reset_index()
        st.dataframe(summary, use_container_width=True, height=300)

# ── Tab 2: Charts ─────────────────────────────────────────────────────────────
with tab_charts:
    try:
        import plotly.express as px

        PLOT_BG  = "#ffffff"
        GRID_COL = "#e5e7eb"
        TEXT_COL = "#57606a"
        ACCENT   = "#0969da"
        ROT_COL  = "#cf222e"

        def theme(fig):
            fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                              font_color=TEXT_COL,
                              xaxis=dict(gridcolor=GRID_COL,linecolor=GRID_COL),
                              yaxis=dict(gridcolor=GRID_COL,linecolor=GRID_COL),
                              legend=dict(bgcolor="#f6f8fa",bordercolor=GRID_COL),
                              margin=dict(l=40,r=20,t=40,b=40))
            return fig

        if rot_det_mode:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### % Rot per Image")
                if "Image Name" in df.columns:
                    fig = px.bar(df.sort_values("FruitRotPer"), x="Image Name", y="FruitRotPer",
                                 color="FruitRotPer",
                                 color_continuous_scale=[[0,"#dafbe1"],[0.5,"#fff8c5"],[1,"#ffebe9"]],
                                 labels={"FruitRotPer":"% Rot"})
                    fig.update_coloraxes(showscale=False)
                    fig.update_xaxes(tickangle=45, tickfont_size=8)
                    st.plotly_chart(theme(fig), use_container_width=True)
            with c2:
                st.markdown("#### Sound vs Rotten per Image")
                if "Image Name" in df.columns:
                    melt = df.melt(id_vars="Image Name",
                                   value_vars=["NumberSoundBerries","NumberRottenBerries"],
                                   var_name="Class",value_name="Count")
                    melt["Class"] = melt["Class"].map({"NumberSoundBerries":"Sound","NumberRottenBerries":"Rotten"})
                    fig2 = px.bar(melt, x="Image Name", y="Count", color="Class",
                                  color_discrete_map={"Sound":ACCENT,"Rotten":ROT_COL}, barmode="stack")
                    fig2.update_xaxes(tickangle=45, tickfont_size=8)
                    st.plotly_chart(theme(fig2), use_container_width=True)
            st.markdown("#### % Rot Distribution")
            fig3 = px.histogram(df, x="FruitRotPer", nbins=20,
                                color_discrete_sequence=[ACCENT], labels={"FruitRotPer":"% Rot"})
            st.plotly_chart(theme(fig3), use_container_width=True)
        else:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                            if c not in ["Object_ID","Patch_size"]]
            col_f, col_g = st.columns([2,1])
            with col_f:
                feature = st.selectbox("Feature to plot", numeric_cols,
                                       index=numeric_cols.index("Area") if "Area" in numeric_cols else 0)
            with col_g:
                split = st.checkbox("Split by class", value=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### Distribution: {feature}")
                if split and "name" in df.columns:
                    fig = px.histogram(df, x=feature, color="name", nbins=30,
                                       color_discrete_map={"berry":ACCENT,"rotten":ROT_COL},
                                       barmode="overlay", opacity=0.75)
                else:
                    fig = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=[ACCENT])
                st.plotly_chart(theme(fig), use_container_width=True)
            with c2:
                st.markdown(f"#### {feature} by Image")
                if "Image Name" in df.columns:
                    agg = df.groupby("Image Name")[feature].mean().reset_index()
                    fig2 = px.bar(agg, x="Image Name", y=feature, color_discrete_sequence=[ACCENT])
                    fig2.update_xaxes(tickangle=45, tickfont_size=8)
                    st.plotly_chart(theme(fig2), use_container_width=True)

            if "Length" in df.columns and "Width" in df.columns:
                st.markdown("#### Length vs Width")
                fig3 = px.scatter(df, x="Width", y="Length",
                                  color="name" if "name" in df.columns else None,
                                  color_discrete_map={"berry":ACCENT,"rotten":ROT_COL}, opacity=0.6)
                st.plotly_chart(theme(fig3), use_container_width=True)

            color_cols = [c for c in df.columns if any(t in c for t in
                          ["Red_Color","Green_Color","Blue_Color","L_Color","a_Color","b_Color"])
                          and "Mean" in c]
            if color_cols and "name" in df.columns:
                st.markdown("#### Mean Color Channel Values")
                cm = df.groupby("name")[color_cols].mean().T.reset_index()
                cm.columns = ["Channel"] + list(cm.columns[1:])
                fig4 = px.bar(cm.melt(id_vars="Channel"), x="Channel", y="value", color="variable",
                              barmode="group", color_discrete_map={"berry":ACCENT,"rotten":ROT_COL},
                              labels={"value":"Mean value","variable":"Class"})
                fig4.update_xaxes(tickangle=45, tickfont_size=9)
                st.plotly_chart(theme(fig4), use_container_width=True)

    except ImportError:
        st.warning("Install plotly to enable charts: `pip install plotly`")

# ── Tab 3: Images ─────────────────────────────────────────────────────────────
with tab_images:
    pred_dir = st.session_state.rv_pred_dir
    if not pred_dir or not os.path.isdir(pred_dir):
        st.info("No predictions folder linked to this session. "
                "Load one via the manual CSV loader above, or re-run with Save enabled.")
    else:
        image_files = sorted(glob.glob(os.path.join(pred_dir,"*.jpg")) +
                             glob.glob(os.path.join(pred_dir,"*.png")))
        if not image_files:
            st.info("No annotated images in predictions folder.")
        else:
            st.caption(f"{len(image_files)} annotated images")
            filter_q = st.text_input("Filter by filename / sample ID", "")
            filtered = [f for f in image_files if filter_q.lower() in Path(f).name.lower()] if filter_q else image_files
            n_cols   = st.slider("Columns", 2, 5, 3)
            cols     = st.columns(n_cols)
            for i, fpath in enumerate(filtered):
                import cv2 as _cv2
                img = _cv2.imread(fpath)
                if img is None: continue
                with cols[i % n_cols]:
                    st.image(_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB),
                             caption=Path(fpath).name, use_container_width=True)

# ── Tab 4: Export ─────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("#### Full Feature CSV")
    st.download_button("⬇  Download full CSV", data=df.to_csv(index=False).encode(),
                       file_name=f"{st.session_state.rv_session_name}_features.csv", mime="text/csv")
    if rot_det_mode and "Image Name" in df.columns:
        st.markdown("#### Per-Image Summary CSV")
        summary = df.groupby("Image Name").agg(
            Sound  =("NumberSoundBerries","sum"), Rotten=("NumberRottenBerries","sum"),
            PctRot =("FruitRotPer","mean"),       WtdRot=("FruitRotPerWtd","mean")
        ).reset_index()
        st.download_button("⬇  Download summary CSV", data=summary.to_csv(index=False).encode(),
                           file_name=f"{st.session_state.rv_session_name}_summary.csv", mime="text/csv")
    if not rot_det_mode and "name" in df.columns:
        st.markdown("#### Reduced Feature Set")
        keep = ["Date","Image Name","QR_info","Object_ID","name","Area","Length","Width",
                "Ellipsoid_model_volume","Eccentricity",
                "Red_Color_Mean","Green_Color_Mean","Blue_Color_Mean",
                "L_Color_Mean","a_Color_Mean","b_Color_Mean"]
        df_r = df[[c for c in keep if c in df.columns]]
        st.download_button("⬇  Download reduced CSV", data=df_r.to_csv(index=False).encode(),
                           file_name=f"{st.session_state.rv_session_name}_features_reduced.csv", mime="text/csv")

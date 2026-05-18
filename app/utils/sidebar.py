"""Shared sidebar configuration rendered on every page."""
import streamlit as st
from utils.folder_picker import folder_picker


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## BerryBox AI")
        st.markdown('<hr style="border-color:#d0d7de;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

        st.markdown("### Module")
        module = st.selectbox(
            "Analysis module",
            options=["berry-seg", "rot-det"],
            format_func=lambda x: "Berry Segmentation" if x == "berry-seg" else "Fruit Rot Detection",
            label_visibility="collapsed"
        )

        st.markdown('<hr style="border-color:#d0d7de;margin:0.75rem 0">', unsafe_allow_html=True)
        st.markdown("### Model Parameters")

        default_conf = 0.75 if module == "berry-seg" else 0.50
        conf = st.slider("Confidence threshold", 0.1, 1.0, default_conf, 0.05)
        iou  = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.25, 0.05)

        with st.expander("Image size & advanced"):
            default_h, default_w = (1856, 2784) if module == "berry-seg" else (1600, 2400)
            img_h           = st.number_input("Image height (px)", value=default_h, step=32)
            img_w           = st.number_input("Image width (px)",  value=default_w, step=32)
            reduce_features = st.checkbox("Reduce feature set", value=False)
            no_cc           = st.checkbox("Disable color correction", value=False)
            no_qr           = st.checkbox("Disable QR / OCR reader", value=False)
            save_cc         = st.checkbox("Save color-corrected images", value=False)
            patch_size      = st.number_input("Color card patch size (cm)", value=1.2, step=0.1)
            model_path      = st.text_input("Custom model path (.pt)", value="",
                                            placeholder="Leave blank for default weights")
            verbose         = st.checkbox("Verbose model output", value=False)

        st.markdown('<hr style="border-color:#d0d7de;margin:0.75rem 0">', unsafe_allow_html=True)
        st.markdown("### Raspberry Pi")
        rpi_ip   = st.text_input("IP address", value="169.254.111.10")
        rpi_user = st.text_input("Username",   value="cranpi2")
        rpi_pwd  = st.text_input("Password",   value="usdacran", type="password")

        st.markdown('<hr style="border-color:#d0d7de;margin:0.75rem 0">', unsafe_allow_html=True)
        st.markdown("### Output")
        output_dir = folder_picker("Output directory", key="sidebar_output_dir",
                                   default="./berrybox_output")

    return dict(
        module=module, conf=conf, iou=iou,
        imgsz=(int(img_h), int(img_w)),
        reduce_features=reduce_features, no_cc=no_cc, no_qr=no_qr,
        save_cc=save_cc, patch_size=patch_size,
        model_path=model_path.strip() if model_path.strip() else ".",
        verbose=verbose,
        rpi_ip=rpi_ip, rpi_user=rpi_user, rpi_pwd=rpi_pwd,
        output_dir=output_dir,
    )

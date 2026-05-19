"""Shared helpers: image annotation, model loading, directory setup."""
import cv2
import numpy as np
import os
import platform
from pathlib import Path

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def annotated_image_rgb(image, result, class_names, show_masks=True, show_count=False):
    img = image.copy()
    colors_bgr = {
        "berry": (255, 100, 0), "rotten": (0, 200, 50),
        "sound": (50, 100, 255), "ColorCard": (200, 200, 0), "info": (180, 0, 180),
    }
    fallback = [(0,255,255),(255,0,255),(0,165,255)]

    def get_color(name):
        return colors_bgr.get(name, fallback[hash(name) % len(fallback)])

    masks = None
    if show_masks and hasattr(result, "masks") and result.masks is not None:
        masks = result.masks.data.numpy()

    boxes       = result.boxes.xyxy.numpy()
    class_ids   = result.boxes.cls.numpy()
    confidences = result.boxes.conf.numpy()
    count_dict  = {n: 0 for n in class_names}
    sorted_idx  = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))

    for rank, i in enumerate(sorted_idx, 1):
        class_name = result.names[int(class_ids[i])]
        color = get_color(class_name)
        if class_name in count_dict:
            count_dict[class_name] += 1
        if masks is not None and i < len(masks):
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            for c in range(3):
                colored_mask[:,:,c] = masks[i] * color[c]
            img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)
        x1,y1,x2,y2 = map(int, boxes[i])
        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
        cv2.putText(img, f"{class_name} {confidences[i]:.2f}",
                    (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if show_count:
        lines = []
        if "sound"  in count_dict: lines.append(f"Sound: {count_dict['sound']}")
        if "rotten" in count_dict: lines.append(f"Rotten: {count_dict['rotten']}")
        text = "   ".join(lines)
        if text:
            cv2.putText(img, text, (20,45), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0,255,80), 3, lineType=cv2.LINE_AA)

    return bgr_to_rgb(img)


def load_model(module: str, task: str, weights_dir: str):
    """
    Loads the YOLO model based on the operating system and architecture.
    Accepts the absolute path to the weights directory from app.py.
    """
    from ultralytics import YOLO
    
    system, machine = platform.system(), platform.machine()
    
    # Use OpenVINO for Windows and Intel Macs for performance
    if system == "Windows" or (system == "Darwin" and machine == "x86_64"):
        model_name = f"berrybox_{module}_openvino_model"
    else:
        model_name = f"berrybox_{module}.pt"
        
    model_path = os.path.join(weights_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
        
    return YOLO(model_path, task=task)


def get_device() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def build_model_params(cfg: dict) -> dict:
    return dict(
        save=False, show_labels=True, show_conf=True, save_crop=False,
        line_width=3, conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
        exist_ok=False, half=True, cache=False, retina_masks=False,
        device=get_device(), verbose=cfg.get("verbose", False),
        agnostic_nms=(cfg["module"] == "rot-det"),
    )


def ensure_session_dirs(base_dir: str, module: str, session_name: str) -> dict:
    session_dir   = os.path.join(base_dir, session_name)
    raw_image_dir = os.path.join(session_dir, "images")
    output_dir    = os.path.join(session_dir, "output")
    pred_dir      = os.path.join(output_dir,  "predictions")
    cc_dir        = os.path.join(output_dir,  "color_corrected_images")
    for d in [base_dir, session_dir, raw_image_dir, output_dir, pred_dir, cc_dir]:
        os.makedirs(d, exist_ok=True)
    return dict(
        session_dir=session_dir, raw_image_dir=raw_image_dir,
        output_dir=output_dir, pred_dir=pred_dir, cc_dir=cc_dir,
        feature_file=os.path.join(output_dir, f"{session_name}_features.csv"),
    )
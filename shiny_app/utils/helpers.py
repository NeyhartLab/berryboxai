"""Shared helpers: image annotation, model loading, directory setup."""
import cv2
import numpy as np
import os
import platform
import time
from pathlib import Path

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def annotated_image_rgb(image, result, class_names, show_masks=True, show_count=False):
    """
    Annotates BGR image with YOLO results. 
    Handles mask resizing to match original Nikon high-res dimensions.
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    colors_bgr = {
        "berry": (255, 100, 0), "rotten": (0, 200, 50),
        "sound": (50, 100, 255), "ColorCard": (200, 200, 0), "info": (180, 0, 180),
    }
    fallback = [(0, 255, 255), (255, 0, 255), (0, 165, 255)]

    def get_color(name):
        return colors_bgr.get(name, fallback[hash(name) % len(fallback)])

    # Extract Masks
    masks = None
    if show_masks and hasattr(result, "masks") and result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    # Extract Boxes, Classes, and Confidences safely
    if result.boxes is None:
        return bgr_to_rgb(img)

    boxes       = result.boxes.xyxy.cpu().numpy()
    class_ids   = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    count_dict  = {n: 0 for n in class_names}
    # Sort detections by Y-coordinate (top to bottom)
    sorted_idx  = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))

    for i in sorted_idx:
        idx = int(class_ids[i])
        class_name = result.names[idx] if hasattr(result, "names") else class_names[idx]
        color = get_color(class_name)
        
        if class_name in count_dict:
            count_dict[class_name] += 1
        
        # --- MASK DRAWING (With Resize Fix) ---
        if masks is not None and i < len(masks):
            mask = masks[i] 
            # Resize mask from inference resolution to original image resolution
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Apply colored overlay
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * color[c]
                
            img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)
        
        # --- BOX & LABEL DRAWING ---
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {confidences[i]:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 10, 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return bgr_to_rgb(img)

def load_model(module: str, task: str, weights_dir: str):
    """
    Loads YOLO model. If on Windows/Intel Mac and OpenVINO version is missing,
    it automatically converts the .pt file to OpenVINO format.
    """
    from ultralytics import YOLO
    
    system, machine = platform.system(), platform.machine()
    is_openvino_eligible = (system == "Windows") or (system == "Darwin" and machine == "x86_64")
    
    pt_name = f"berrybox_{module}.pt"
    ov_name = f"berrybox_{module}_openvino_model"
    
    pt_path = os.path.join(weights_dir, pt_name)
    ov_path = os.path.join(weights_dir, ov_name)

    # 1. Handle OpenVINO Conversion for eligible platforms
    if is_openvino_eligible:
        if not os.path.exists(ov_path):
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Model file not found: {pt_path}")
            
            print(f"--- Exporting {module} to OpenVINO... ---")
            model_to_convert = YOLO(pt_path)
            # Use standard export sizes
            imgsz = (1856, 2784) if "seg" in module else (1600, 2400)
            model_to_convert.export(format='openvino', imgsz=imgsz, half=True)
            print(f"--- Export complete. ---")
        
        return YOLO(ov_path, task=task)
    
    # 2. Standard .pt loading
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model file not found: {pt_path}")
        
    return YOLO(pt_path, task=task)

def setup_nikon_camera(ssh, camera_name: str = "Nikon DSC D7500", sleeps = 2):
    """Checks Nikon connection via gphoto2 and sets exposure/WB configs."""
    stdin, stdout, stderr = ssh.exec_command("gphoto2 --auto-detect")
    det = stdout.read().decode("utf-8")
    
    if camera_name not in det:
        raise RuntimeError(f"{camera_name} not found in gphoto2 auto-detect!")
    
    # Free the camera lock
    ssh.exec_command("pkill -f gphoto2")
    time.sleep(0.5)
    
    configs = [
        "iso=100",
        "whitebalance=7",
        "/main/capturesettings/f-number=7.1",
        "/main/capturesettings/shutterspeed=25"
    ]
    
    for cfg in configs:
        ssh.exec_command(f"gphoto2 --set-config {cfg}")
        time.sleep(sleeps)

    return f"{camera_name} connected and configured."

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
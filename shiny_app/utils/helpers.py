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
    img = image.copy()
    h, w = img.shape[:2]
    
    colors_bgr = {
        "berry": (255, 100, 0), "rotten": (0, 200, 50),
        "sound": (50, 100, 255), "ColorCard": (200, 200, 0), "info": (180, 0, 180),
    }
    fallback = [(0,255,255),(255,0,255),(0,165,255)]

    def get_color(name):
        return colors_bgr.get(name, fallback[hash(name) % len(fallback)])

    masks = None
    if show_masks and hasattr(result, "masks") and result.masks is not None:
        # Move to CPU and get numpy array
        masks = result.masks.data.cpu().numpy()

    boxes       = result.boxes.xyxy.numpy()
    class_ids   = result.boxes.cls.numpy()
    confidences = result.boxes.conf.numpy()
    count_dict  = {n: 0 for n in class_names}
    sorted_idx  = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))

for rank, i in enumerate(sorted_idx, 1):
        class_name = result.names[int(class_ids[i])]
        color = get_color(class_name)
        
        # --- MASK DRAWING WITH RESIZING FIX ---
        if masks is not None and i < len(masks):
            # 1. Take the single mask
            mask = masks[i] 
            
            # 2. Resize mask to match the original image dimensions (w, h)
            # This prevents the "Could not broadcast" error
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 3. Apply color
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            for c in range(3):
                # Ensure mask is treated as a boolean multiplier
                colored_mask[:,:,c] = (mask_resized > 0.5) * color[c]
                
            img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)
        
        # --- BOX DRAWING ---
        x1,y1,x2,y2 = map(int, boxes[i])
        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
        cv2.putText(img, f"{class_name} {confidences[i]:.2f}",
                    (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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
                raise FileNotFoundError(f"Neither .pt nor OpenVINO model found for {module} in {weights_dir}")
            
            print(f"--- Converting {module} to OpenVINO for first-time use ---")
            model_to_convert = YOLO(pt_path)
            imgsz = (1856, 2784) if "seg" in module else (1600, 2400)
            
            # This creates the ov_path folder
            model_to_convert.export(format='openvino', imgsz=imgsz, half=True)
            print(f"--- Conversion complete: {ov_path} ---")
        
        return YOLO(ov_path, task=task)
    
    # 2. Handle standard .pt loading (Linux / ARM Mac)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model weights (.pt) not found at: {pt_path}")
        
    return YOLO(pt_path, task=task)

def setup_nikon_camera(ssh, camera_name: str = "Nikon DSC D7500", sleeps = 2):
    """Checks Nikon connection via gphoto2 and sets exposure/WB configs."""
    stdin, stdout, stderr = ssh.exec_command("gphoto2 --auto-detect")
    det = stdout.read().decode("utf-8")
    
    if camera_name not in det:
        raise RuntimeError(f"{camera_name} not found in gphoto2 auto-detect!")
    
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
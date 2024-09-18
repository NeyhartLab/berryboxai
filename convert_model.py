## BerryBox AI
## 
## Convert a model to open vino format
## 
# Import packages here
from ultralytics import YOLO
import os

image_size = (1600, 2400)

# Set the path to the model
# model_path = os.path.join(proj_dir, "models/berrybox_inst_seg_nano_best_20240324.pt")
model_path = "data/models/berrybox_fruit_rot_objdet_small_2400_best_20240819.pt"
# OpenVino model path
ov_model_path = model_path.replace(".pt", "_openvino_model")

# Attempt to find the openvino version of the model;
# If it does not exist, export the model
if not os.path.exists(ov_model_path):

    model = YOLO(model_path)

    # Export the model using openVINO
    model.export(format = "openvino", imgsz = image_size, half = True)

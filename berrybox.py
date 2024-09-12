# Open ultralytics
import ultralytics
ultralytics.checks()


# Set project directory and change directory
import os
import shutil
proj_dir = os.getcwd()
proj_dir
%ls


# Set the path to the model
# model_path = os.path.join(proj_dir, "models/berrybox_best_20240316.pt")
model_path = "C:/Users/jeffrey.neyhart/OneDrive - USDA/Documents/Repositories/berryboxai/models/berrybox_inst_seg_nano_best_20240318.pt"
# OpenVino model path
ov_model_path = "../models/" + os.path.basename(model_path).replace(".pt", "_openvino_model")

# Attempt to find the openvino version of the model;
# If it does not exist, export the model
if not os.path.exists(ov_model_path):

    # Copy the model locally
    model_path_local = os.path.join("models", os.path.basename(model_path))
    shutil.copyfile(model_path, model_path_local)

    # Load the model with YOLO
    from ultralytics import YOLO
    model = YOLO(model_path_local)

    # Export the model using openVINO
    # model.export(format = "openvino", imgsz = 2048, half = True)
    model.export(format = "openvino", imgsz = (1344, 2016), half = True)


    from functions import * # load all functions
from ultralytics import YOLO
import os
import torch
import gc
import shutil
from PIL import Image

gc.collect()   # collect garbage

device = '0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


"""
------------------------------------------------------------------------------------
Set Directories
------------------------------------------------------------------------------------
"""
model_dir = ov_model_path # path to the model
image_dir = 'images' # path to the image folder
save_dir = 'output' # path to save the results

# shutil.rmtree(save_dir, ignore_errors=True)

"""
------------------------------------------------------------------------------------
Set Model Parameters (you can change these parameters to suit your needs)
------------------------------------------------------------------------------------
"""
model_params = {
    'project': save_dir, # project name
    'name': "berrybox_" + os.path.basename(proj_dir), # run name
    'save': False, # save image results
    'show_labels': True,   # hide labels
    'show_conf': True, # hide confidences
    'save_crop': False, # save cropped prediction boxes
    'line_width': 3, # bounding box line width
    'conf': 0.70, # confidence threshold
    'iou': 0.75, # NMS IoU threshold
    # 'imgsz': 2048,
    'imgsz': (1344, 2016),
    # 'imgsz': (640, 960),
    'exist_ok': False, # if True, it overwrites current 'name' saving folder
    'half': True, # use FP16 half-precision inference True/False
    'cache': False, # use cache images for faster inference
    'retina_masks': False, #use high resolution seg mask
    'device': device, # cuda device, i.e. 0 or 0,1,2,3 or cpu
    'verbose': True
}

# Load the model
model = YOLO(model_dir, task = "segment")


print('Running inference and extracting features...')

# Create an empty pandas data frame
DF = pd.DataFrame()

# List images in the image dir
image_files = [x for x in os.listdir(image_dir) if ".JPG" or ".PNG" in x.upper()]
# Target model size
newH, newW = model_params["imgsz"]

# Iterate over the image files
for i, img_name in enumerate(image_files):
    
    # Read in the image and resize
    image = Image.open(image_dir + "/" + img_name).resize((newW, newH))
    
    # Run through the model
    results = model.predict(source = image, **model_params)
    result = results[0]

    # Process the results
    # Try color correction; skip if it doesn't work
    try:
        result, patch_size = color_correction(result)
    except:
        continue
    # Was "info" found?
    if any(result.boxes.cls == get_ids(result, 'info')[0]):
        QR_info = read_QR_code(result)
    else:
        print("No 'info' detected by the model.\n")
        QR_info = img_name
    # Get features
    df1 = get_all_features_parallel(result, name= 'berry')
    df2 = get_all_features_parallel(result, name= 'rotten')
    df = pd.concat([pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), pd.concat([df1, df2], ignore_index = True)], axis = 1)    
    w,_ = df.shape
    img_name = [img_name]*w
    QR_info = [QR_info]*w
    patch_size = [np.mean(patch_size)]*w
    indeces = list(range(w))
    # If indeces is length 0; warn that no berries were found
    if len(indeces) == 0:
        print('\033[1;33mNo berries were found in the image!\033[0m')
        continue

    df_fore = pd.DataFrame({'Image_name': img_name,
                            'ID': indeces,
                            'QR_info': QR_info,
                            'Patch_size': patch_size})

    df = pd.concat([df_fore, df], axis=1)
    DF = pd.concat([DF, df], axis=0, ignore_index=True)

    img_save_folder = os.path.join(save_dir, 'predictions')
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)

    save_ROI_parallel(result, get_ids(result, 'berry'), os.path.join(img_save_folder, img_name[0]))

    print(f"\nImage {i+1} of {len(image_files)} processed." )
    
    
DF.to_csv(os.path.join(save_dir, 'features.csv'), index=False)
print('Done.')

gc.collect()    





### Code below for communicating with a raspberry pi to trigger the camera shutter

import paramiko
import time
from scp import SCPClient
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import cv2

# 1. SSH connection to Raspberry Pi (kept open for multiple operations)
def create_ssh_client(hostname, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    return ssh

def transfer_image(ssh, remote_path, local_path):
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_path, local_path)
    scp.close()

# Function to capture and process the image
def capture_and_process_image(ssh, barcode, model, csv_file='image_data.csv'):
    # 3. Trigger Nikon D7500 shutter using gphoto2 on Raspberry Pi
    remote_image_path = "/home/pi/captured_image.jpg"
    trigger_command = f"gphoto2 --capture-image-and-download --filename {remote_image_path}"
    ssh.exec_command(trigger_command)

    # Wait a bit to ensure the image is captured and transferred
    time.sleep(5)

    # 4. Transfer the image from Raspberry Pi to local machine
    local_image_path = f"./captured_image_{barcode}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    transfer_image(ssh, remote_image_path, local_image_path)

    # 5. Run YOLOv8 instance segmentation
    results = model(local_image_path)

    # Extract objects count
    objects_count = len(results[0].boxes)

    # 6. Save results (image name, date, barcode, object count) to CSV
    image_name = local_image_path.split("/")[-1]
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        'Date': [date_time],
        'Image Name': [image_name],
        'Barcode': [barcode],
        'Object Count': [objects_count]
    }

    df = pd.DataFrame(data)

    # Append to CSV if it exists
    try:
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(csv_file, index=False)

    print(f"Image {image_name} captured, processed, and saved with {objects_count} objects.")

# SSH Credentials
hostname = "raspberry_pi_ip"
username = "pi"
password = "password"
ssh = create_ssh_client(hostname, username, password)

# Load YOLOv8 model outside the loop
model = YOLO('yolov8n-seg.pt')  # Load the YOLOv8 model once

# Main loop for capturing multiple images
try:
    while True:
        # 2. Capture barcode and wait for Enter
        barcode = input("Scan the barcode (or type 'exit' to quit): ")
        if barcode.lower() == 'exit':
            break

        input("Press Enter to capture the image...")

        # Capture the image, process it, and save results
        capture_and_process_image(ssh, barcode, model)

finally:
    # Ensure the SSH connection is closed at the end
    ssh.close()
    print("SSH connection closed.")



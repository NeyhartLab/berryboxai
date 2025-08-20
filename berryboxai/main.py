## BerryBox AI
## 
## This script provides all of the functions needed to operate the BerryBox,b   
## collect images, run the imaged through an analysis pipeline
## 
## Author: Jeff Neyhart
## 
## Usage
## python berrybox.py [ADD OPTIONS HERE]
##
## 

# Import packages here
from .functions import * # load all functions
from ultralytics import YOLO
import os
import torch
import gc
import shutil
from PIL import Image # Probably remove this for opencv
import paramiko
import time
from scp import SCPClient
import pandas as pd
import cv2
import argparse
import pkg_resources
import platform
import math
import ast


# options parsers
def options():
    parser = argparse.ArgumentParser(description='BerryBoxAI v 1.0')
    parser.add_argument('-m', '--module', help='Which module to run. Can be "berry-seg" for berry segmentation and quality or "rot-det" for detecting the amount of fruit rot in a sample.', 
                        required = True, choices = ['berry-seg', 'rot-det'], default = 'berry-seg')
    parser.add_argument('--input', help='Directory containing images to process. Providing this input directory will disable the interactive mode of berrybox and instead run the inference pipeline on a batch of images. If ignored, the pipeline will run in interactive mode.', 
                        required = False, default = "NULL")
    parser.add_argument('--output', help='The directory to store the data output', required = True)
    parser.add_argument('--conf', help='Confidence level for segmenting or detecting objects from the model', required = False, default = 0)
    parser.add_argument('--iou', help='IOU level for segmenting or detecting objects from the model', required = False, default = 0)
    parser.add_argument('--imgsz', help='Image size before sending it to the model', required = False, default = (0, 0))
    parser.add_argument('--reduce-features', help='Save only the Area, Length, Width, Volume, Eccentricity, Red, Green, Blue, L*, a*, b*', default=False, action='store_true')
    parser.add_argument('--patch-size', help='The size (in cm) of the length/width of the patches in the ColorCard', required = False, default = 1.2)
    parser.add_argument('--save', help='Save the annotated images from the model output', default = False, action = 'store_true')
    parser.add_argument('--preview', help = 'Display a preview of the image with predicted features.', default = False, action = 'store_true')
    parser.add_argument('--ext', help = 'Extension of the images to find in the "input" directory.', default = '.jpg', required = False)
    parser.add_argument('--overwrite', help = 'Overwrite the existing output file, if present.', default = False, action = 'store_true')
    parser.add_argument('--model-path', help = 'The path to the .pt model weights to use. WARNING: this is a dangerous option and should only be used if you know what you are doing.', required = False, default = ".")
    parser.add_argument('--no-cc', help='Disable color correction', default = False, action = 'store_true')
    parser.add_argument('--no-qr', help='Disable the QR code and OCR reader', default = False, action = 'store_true')
    parser.add_argument('--rpiip', help='IP address of the raspberry pi', default = '169.254.111.10')
    parser.add_argument('--rpiuser', help='Username of the raspberry pi', default = 'cranpi2')
    parser.add_argument('--rpipwd', help='Password of the raspberry pi', default = 'usdacran')
    parser.add_argument('--verbose', help='Should model progress be printed to the terminal?', default = False, action = 'store_true')
    args = parser.parse_args()
    return args

# custom functions
def create_ssh_client(hostname, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    return ssh

# A function that checks the nikon camera for connection and sets configurations
def setup_nikon_camera(ssh, camera: str = "Nikon DSC D7500", sleeps = 2):
    # First check that the Emlid device is connected
    print("Checking Nikon Camera connection...")
    
    stdin, stdout, stderr = ssh.exec_command("gphoto2 --auto-detect")
    det = stdout.read().decode("utf-8")
    if not camera in str(det):
        raise RuntimeError(camera + " is not connected!")
    
    # Kill gphoto processes
    ssh.exec_command("pkill -f gphoto2")
    
    # Set configurations
    command = "gphoto2 --set-config iso=100"
    ssh.exec_command(command)
    time.sleep(sleeps) # These must be here 
    command = "gphoto2 --set-config whitebalance=7"
    ssh.exec_command(command)
    time.sleep(sleeps)
    command = "gphoto2 --set-config /main/capturesettings/f-number=7.1"
    ssh.exec_command(command)
    time.sleep(sleeps)
    command = "gphoto2 --set-config /main/capturesettings/shutterspeed=25"
    ssh.exec_command(command)

    print(camera + " is connected and configured!")


def transfer_image(ssh, remote_path, local_path):
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_path, local_path)
    scp.close()

# A function to get the date and store it in "YYYY-MM-DD" format
def get_date():
    # Get the current time
    clocaltime = time.localtime()
    ct = time.strftime("%Y-%m-%d", clocaltime)
    return str(ct)

# A function to get the time and store it in "YYYY-MM-DD_HH-MM-SS" format
def get_time():
    # Get the current time
    clocaltime = time.localtime()
    ct = time.strftime("%Y-%m-%d_%H-%M-%S", clocaltime)
    return str(ct)

# Function to display image with YOLO segmentation masks
def display_image_with_masks(image, result, class_names, show_masks = True, output_path = "", save = False):

    # Get the masks, boxes, and classes from the results
    if show_masks:
        masks = result.masks.data.numpy()  # Segmentation masks
    colors = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)]  # Green for class 0, Red for class 1
    # Generate random colors for each mask
    colors = {k: colors[i] for i, k in enumerate(class_names)}
    boxes = result.boxes.xyxy.numpy()  # Bounding boxes
    class_ids = result.boxes.cls.numpy()  # Class IDs
    confidences = result.boxes.conf.numpy()  # Confidence scores

    # Sort bounding boxes by top-to-bottom, left-to-right (row-major order)
    sorted_indices = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][1], boxes[i][0])  # Sort by y1 (top) first, then x1 (left)
    )

    # Display boxes or masks
    for rank, i in enumerate(sorted_indices, start=1):  # Start numbering from 1
        box = boxes[i]
        class_id = int(class_ids[i])
        class_name = result.names[class_id]
        color = colors[class_name]
        # Create a colored mask if called for
        if show_masks:
            mask = masks[i]
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):  # Apply color to each channel
                colored_mask[:, :, c] = mask * color[c]

            # Blend the colored mask with the original image
            image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)


        # Draw the bounding box around the object
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw the class label and confidence score
        class_name = result.names[class_id]
        label = f'Object {rank}: {class_name}, Conf: {confidences[i]:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    if save:
        cv2.imwrite(output_path, image)

    else:
        # Resize the image for a smaller preview
        og_h, og_w = image.shape[:2]
        new_h = int(og_h * 0.5)
        new_w = int(og_w * 0.5)
        image = cv2.resize(image, (new_w, new_h))

        # Display the image in a window
        window_name = "Predictions"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(window_name, image)
        # cv2.resizeWindow("Predictions", new_w, new_h)

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_ROI_boxes(image, result, class_names, output_path):
    image1 = image
    # Get the masks, boxes, and classes from the results
    boxes = result.boxes.xyxy.numpy()  # Bounding boxes
    class_ids = result.boxes.cls.numpy()  # Class IDs
    confidences = result.boxes.conf.numpy()  # Confidence scores

    # Generate random colors for each mask
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for class 0, Red for class 1
    colors = {k: colors[i] for i, k in enumerate(class_names)}

    # Display boxes or masks
    for i, box in enumerate(boxes):
        class_id = int(class_ids[i])
        class_name = result.names[class_id]
        color = colors[class_name]

        # Draw the bounding box around the object
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(image1, (x1, y1), (x2, y2), color, 2)

        # Draw the class label and confidence score
        label = f'{class_name}, Conf: {confidences[i]:.2f}'
        cv2.putText(image1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save
    cv2.imwrite(output_path, image1)


# A function to convert model weights to openvino if using a Windows PC
def convert_model_to_openvino(input_path):
    # Load the YOLO model
    model = YOLO(input_path)
    # Check which module it is and assign image size
    if "berry-seg" in input_path:
        image_size = (1856, 2784)
    elif "rot-det" in input_path:
        image_size = (1600, 2400) 
    
    # Export the model to OpenVINO format
    model.export(format='openvino', imgsz = image_size, half = True)
    print("Model " + os.path.basename(input_path) + " converted to openvino format...")

# A function to summarize results from the rot object detection model
def summarize_rot_det_results(result):
    # Find the class integers corresponding to the classes
    class_names = ["rotten", "sound"]
    results_names = {v: k for k, v in result.names.items()}

    # Get the boxes
    detected_boxes = result.boxes
    detected_classes = detected_boxes.cls.numpy()
    objects_count = len(detected_classes)
    if objects_count == 0:
        print('\033[1;33mNo berries were found in the image!\033[0m')
        return (0, 0, 0, 0, 0)

    # Count sound and rot
    class_counts = {x: (detected_classes == results_names[x]).sum() for x in class_names}
    n_rotten = class_counts["rotten"]
    n_sound = class_counts["sound"]
    n_total_berries = n_rotten + n_sound

    # Calculate rot percent
    perc_rot = round((n_rotten / n_total_berries) * 100, 3)

    # Calculate weighted percent rot based on the area of inscribed ellipse of each box
    # First calculate the areas of inscribed ellipses
    weights = []
    xyxys = detected_boxes.xyxy.numpy()
    for obj in xyxys:
        xmin, ymin, xmax, ymax = obj[:4]  # Bounding box coordinates
        # Calculate the width and height of the bounding box
        width = xmax - xmin
        height = ymax - ymin
        area = math.pi * (width / 2) * (height / 2)
        weights.append(area)

    values = [1 if x == results_names["rotten"] else 0 for x in detected_classes]
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weighted_perc_rot = round((weighted_sum / sum(weights)) * 100, 3)

    return (n_total_berries, n_rotten, n_sound, perc_rot, weighted_perc_rot)



## The main function
def main():

    # Get arguments
    args = options()
    # Determine the module
    mod = args.module
    # Determine if 'input' is provided
    input_dir = args.input
    if input_dir == "NULL":
        interactive = True
    else:
        interactive = False

    # Create a session name
    session_name = "berrybox" + "_" + mod + "_" + get_date()

    # Set up the raspberry pi and nikon camera only if in interactive mode
    if interactive:
        ## RASPBERRY PI SETUP ##
        # These settings may need to be changed in the future
        hostname = args.rpiip
        username = args.rpiuser
        password = args.rpipwd
        try:
            ssh = create_ssh_client(hostname, username, password)
        except:
            print("Could not connect to the raspberry pi. Please check the connection and power and try again.")

        # ## RGB CAMERA SETUP ##
        # Check the nikon camera connection
        setup_nikon_camera(ssh = ssh)



    ## OUTPUT DIRECTORY AND FILE SETUP ##

    # Create directories to store images and metadata
    # Create an output directory
    berrybox_dir = args.output
    if not os.path.exists(berrybox_dir):
        os.mkdir(berrybox_dir)
    save_predictions = args.save
    session_dir = berrybox_dir + "/" + session_name
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    raw_image_dir = session_dir + "/images/"
    if not os.path.exists(raw_image_dir):
        os.mkdir(raw_image_dir)
    output_dir = session_dir + "/output/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Create directory to save the inference images, if requested
    if save_predictions:
        img_save_folder = os.path.join(output_dir, 'predictions')
        if not os.path.exists(img_save_folder):
            os.mkdir(img_save_folder)

    # Create the filename to save the features
    output_feature_filename = output_dir + "/" + session_name + "_features.csv"
    if os.path.exists(output_feature_filename) & args.overwrite:
        os.remove(output_feature_filename)
    
    ## SET MODULE-SPECIFIC SETTINGS ##
    if mod == "berry-seg":
        task = "segment"
    elif mod == "rot-det":
        task = "detect"

    iou = float(args.iou)
    if iou == 0:
        if mod == "berry-seg":
            iou = 0.25
        elif mod == "rot-det":
            iou = 0.25       

    # Set image size, confidence, and iou
    image_size = args.imgsz
    if image_size == (0, 0):
        if mod == "berry-seg":
            image_size = (1856, 2784)
        elif mod == "rot-det":
            image_size = (1600, 2400)
    else:
        image_size = ast.literal_eval(image_size)

    # Set confidence
    confidence = float(args.conf)
    if confidence == 0:
        if mod == "berry-seg":
            confidence = 0.75
        elif mod == "rot-det":
            confidence = 0.50

    no_cc = args.no_cc
    no_qr = args.no_qr

    ## LOAD AND CHECK THE YOLO MODEL ##
    # This finds the model within the package structure
    package_dir = pkg_resources.resource_filename('berryboxai', 'data')

 
    # If the platform is Windows, check that the openvino model exists;
    # If it does not exist, convert it
    if platform.system() == "Windows":
        model_name = "berrybox_" + mod + "_openvino_model"
        device = "cpu"
        model_path = os.path.join(package_dir, 'weights', model_name)
        # If the path does not exist, convert the model
        if not os.path.exists(model_path):
            orig_model_name = model_name.replace("_openvino_model", ".pt")
            orig_model_path = os.path.join(package_dir, 'weights', orig_model_name)
            convert_model_to_openvino(orig_model_path)
    else:
        model_name = "berrybox_" + mod + ".pt"
        model_path = os.path.join(package_dir, 'weights', model_name)
        if platform.system() == "Darwin":
            device = "mps"
        else:
            device = "cpu"

    # If the user provided a model path, use that
    user_model_path = args.model_path
    if user_model_path != ".":
        if not os.path.isfile(user_model_path):
            raise FileNotFoundError(f"Error: The file '{user_model_path}' does not exist.")
        else:
            model_path = user_model_path

    # Load the model
    model = YOLO(model_path, task = task)

    ## SET MODEL ARGUMENTS
    model_params = {
        'save': False, # save image results
        'show_labels': True,   # hide labels
        'show_conf': True, # hide confidences
        'save_crop': False, # save cropped prediction boxes
        'line_width': 3, # bounding box line width
        'conf': confidence, # confidence threshold
        'iou': iou, # NMS IoU threshold
        'imgsz': image_size,
        'exist_ok': False, # if True, it overwrites current 'name' saving folder
        'half': True, # use FP16 half-precision inference True/False
        'cache': False, # use cache images for faster inference
        'retina_masks': False, #use high resolution seg mask
        'device': device, # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'verbose': args.verbose
    }

    # Tuple of image size
    newH, newW = model_params["imgsz"]

    # Date for printing
    current_date = get_date()

    # Patch size placeholder
    patch_size_use = 0

    if interactive:
        ## RUN THE CAPTURE PIPELINE ##
        print("\nRunning berryboxai in interactive mode using the " + mod + " module...\n")

        # Main loop for capturing multiple images
        try:
            while True:
                # 1. Capture barcode and wait for Enter
                barcode = input("Scan the barcode and hit 'Enter' (or type 'exit' and hit 'Enter' to quit): ")
                if barcode.lower() == 'exit':
                    break
                    
                # Date-time for printing
                current_datetime = get_time()

                # 2. Trigger Nikon D7500 shutter using gphoto2 on Raspberry Pi
                print("Capturing and transferring the image...")
                remote_image_path = "/home/cranpi2/berrybox/captured_image.jpg"
                trigger_command = f"gphoto2 --capture-image-and-download --filename {remote_image_path} --force-overwrite"
                ssh.exec_command(command = trigger_command)
                # Wait a bit to ensure the image is captured and transferred
                time.sleep(3)

                # 4. Transfer the image from Raspberry Pi to local machine
                local_image_path = f"{raw_image_dir}/{barcode}_{current_datetime}.jpg" 
                transfer_image(ssh, remote_image_path, local_image_path)

                print("Running the deep learning model on the image...")
                # 5. Read in the image and resize and run through the YOLO model
                image_name = local_image_path.split("/")[-1]
                image = cv2.imread(local_image_path)
                image = cv2.resize(image, (newW, newH))
                results = model.predict(source = image, **model_params)
                # Map results to cpu
                result = results[0].to("cpu")

                print("Processing and saving model results...")
                ## PROCESS RESULTS DEPENDING ON THE MODULE ##
                if (mod == "berry-seg"):
                    # 6. Process the results
                    # Try color correction; skip if it doesn't work
                    if not no_cc:
                        try:
                            result, patch_size = color_correction(result)
                            patch_size = np.min(patch_size)
                        except:
                            continue
                    else:
                        continue

                    # Update the patch size
                    if patch_size > 0:
                        if patch_size < patch_size_use or patch_size_use == 0:
                            patch_size_use = patch_size


                    # Get features
                    df1 = get_all_features_parallel(result, name= 'berry')
                    df2 = get_all_features_parallel(result, name= 'rotten')
                    df = pd.concat([pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), pd.concat([df1, df2], ignore_index = True)], axis = 1)    
                    w,_ = df.shape
                    image_name_vec = [image_name]*w
                    patch_size_vec = [patch_size]*w
                    indeces = list(range(w))
                    # If indeces is length 0; warn that no berries were found
                    if len(indeces) == 0:
                        print('\033[1;33mNo berries were found in the image!\033[0m')
                        continue

                    # 7. Save results (image name, date, barcode, object count) to CSV
                    data = {
                        'Date': current_datetime,
                        'Image Name': image_name_vec,
                        'QR_info': barcode,
                        'Object_ID': indeces,
                        'Patch_size': patch_size_vec
                    }
                    df_fore = pd.DataFrame(data)
                    # Combine with the features
                    df = pd.concat([df_fore, df], axis=1)

                    ## Calculate additional features
                    # Volume
                    df["Ellipsoid_model_volume"] = (4/3) * np.pi * (df["RP_Minor_axis_length"] / 2) * ((df["RP_Major_axis_length"] / 2) ** 2)
                    # Eccentricity
                    df["Eccentricity"] = np.sqrt(1 - ((0.5 * df["RP_Major_axis_length"]) ** 2 / (0.5 * df["RP_Minor_axis_length"]) ** 2))

                    # Assign the berry ID by sorting on bounding box coordinates
                    df = df.sort_values(by=["RP_BB_y", "RP_BB_x"])
                    df = df.reset_index(drop=True)
                    df["Object_ID"] = df.index

                    # Append to CSV if it exists
                    try:
                        existing_df = pd.read_csv(output_feature_filename)
                        df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                    except FileNotFoundError:
                        pass

                    df.to_csv(output_feature_filename, index=False)

                    print(f"Image {image_name} captured, processed, and features saved with {w} berries detected.\n\n")

                    # Save the image with predicted annotations, if requested
                    # THIS WILL NEED TO BE CHANGED FOR ROT DETECTION
                    if save_predictions:
                        display_image_with_masks(image = image, result = result, class_names = ["ColorCard", "berry", "info", "rotten"], 
                                                 output_path = os.path.join(img_save_folder, image_name_vec[0]), save = True)
                        # save_ROI_parallel(result, get_ids(result, 'berry'), os.path.join(img_save_folder, image_name_vec[0]))

                    # Show a preview of the result
                    if args.preview:
                        print("Close the preview window before proceeding to the next sample.")
                        # display_image_with_masks(image = image, results = results, class_names = ["ColorCard", "berry", "rotten"])
                        display_image_with_masks(image = image, result = result, class_names = ["ColorCard", "berry", "info", "rotten"])


                # DIFFERENT PROCESS FOR ROT DETECTION #
                elif (mod == "rot-det"):
                    # Summarize results
                    objects_count, n_rotten, n_sound, perc_rot, weighted_perc_rot = summarize_rot_det_results(result)
                    # If indeces is length 0; warn that no berries were found
                    if objects_count == 0:
                        print('\033[1;33mNo berries were found in the image!\033[0m')
                        continue

                    # 7. Save results (image name, date, barcode, object count) to CSV
                    image_name = local_image_path.split("/")[-1]

                    data = {
                        'Date': current_datetime,
                        'Image Name': image_name,
                        'QR_info': barcode,
                        'NumberSoundBerries': n_sound,
                        'NumberRottenBerries': n_rotten,
                        'FruitRotPer': perc_rot,
                        'FruitRotPerWtd': weighted_perc_rot
                    }
                    df = pd.DataFrame(data, index = [0])

                    # Append to CSV if it exists
                    try:
                        existing_df = pd.read_csv(output_feature_filename)
                        df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                    except FileNotFoundError:
                        pass

                    df.to_csv(output_feature_filename, index=False)

                    print(f"Image {image_name} captured, processed, and features saved with {n_sound} sound berries and {n_rotten} rotten berries detected.\n\n")

                    if save_predictions:
                        save_ROI_boxes(image = image, result = result, class_names = ["rotten", "sound"], output_path = os.path.join(img_save_folder, image_name))

                    # Show a preview of the result
                    if args.preview:
                        print("Close the preview window before proceeding to the next sample.")
                        display_image_with_masks(image = image, result = result, class_names = ["rotten", "sound"], show_masks = False)


        finally:
            # Modify the output data frame for berry seg
            if mod == "berry-seg":
                # Read the df back in
                df = pd.read_csv(output_feature_filename)

                # Convert features to cm and rename
                # Use the final patch size to calculate pixels to cm
                cm_per_pixel = float(args.patch_size) / patch_size_use
                df["Area"] = df["RP_Area"] * (cm_per_pixel ** 2)
                df["Length"] = df["RP_Minor_axis_length"] * cm_per_pixel
                df["Width"] = df["RP_Major_axis_length"] * cm_per_pixel
                df["Ellipsoid_model_volume"] = df["Ellipsoid_model_volume"] * (cm_per_pixel ** 3)
                df["cm_per_pixel"] = cm_per_pixel

                ## Subset features
                if args.reduce_features:
                    features_select = ["Date", "Image Name", "QR_info", "Object_ID", "Patch_size", "name", "Area", "Length", "Width", "Ellipsoid_model_volume", "Eccentricity", "Red_Color_Mean", "Red_Color_Median", "Red_Color_Std", "Green_Color_Mean", "Green_Color_Median", "Green_Color_Std", "Blue_Color_Mean", "Blue_Color_Median", "Blue_Color_Std", "L_Color_Mean", "L_Color_Median", "L_Color_Std", "a_Color_Mean", "a_Color_Median", "a_Color_Std", "b_Color_Mean", "b_Color_Median", "b_Color_Std"]
                    df = df[features_select]
                else:
                    features_drop = ["RP_Area", "RP_Major_axis_length", "RP_Minor_axis_length"]
                    df = df.drop(columns = features_drop)
                
                # Save the df
                df.to_csv(output_feature_filename, index=False)


            # Ensure the SSH connection is closed at the end
            ssh.close()
            print("SSH connection closed.")

    else:
        ## RUN THE BATCH PIPELINE ##
        print("\nRunning berryboxai in batch mode using the " + mod + " module...\n")

        # List images in the input directory
        image_extension = args.ext
        image_extension = image_extension.upper()
        image_list = os.listdir(input_dir)
        image_path_list = [os.path.join(input_dir, x) for x in image_list if image_extension in x.upper()]
        n_images = len(image_path_list)

        # Print the number of images found
        print("Using images from the directory: " + input_dir)
        print("Discovered " + str(n_images) + " images in the directory")
        print("Running the deep learning model on the images...\n")

        # Iterate over images
        for p, local_image_path in enumerate(image_path_list):

            # 5. Read in the image and resize and run through the YOLO model
            image_name = local_image_path.split("/")[-1]
            image = cv2.imread(local_image_path)
            image = cv2.resize(image, (newW, newH))
            results = model.predict(source = image, **model_params)
            # Map results to cpu
            result = results[0].to("cpu")

            ## PROCESS RESULTS DEPENDING ON THE MODULE ##
            if mod == "berry-seg":
                # 6. Process the results
                # Try color correction; skip if it doesn't work
                if not no_cc:
                    try:
                        result, patch_size = color_correction(result)
                        patch_size = np.min(patch_size)
                    except:
                        continue
                else:
                    continue

                # Update the patch size
                if patch_size > 0:
                    if patch_size < patch_size_use or patch_size_use == 0:
                        patch_size_use = patch_size

                # Try to detect the QR code if called on to do that
                if no_qr:
                    barcode = image_name
                
                else:
                    if any(result.boxes.cls == get_ids(result, 'info')[0]):
                        barcode = read_QR_code(result)
                    else:
                        barcode = image_name

                # Get features
                df1 = get_all_features_parallel(result, name= 'berry')
                df2 = get_all_features_parallel(result, name= 'rotten')
                df = pd.concat([pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), pd.concat([df1, df2], ignore_index = True)], axis = 1)    
                w,_ = df.shape
                image_name_vec = [image_name]*w
                patch_size_vec = [np.mean(patch_size)]*w
                indeces = list(range(w))
                # If indeces is length 0; warn that no berries were found
                if len(indeces) == 0:
                    print('\033[1;33mNo berries were found in the image!\033[0m')
                    continue

                # 7. Save results (image name, date, barcode, object count) to CSV
                data = {
                    'Date': current_date,
                    'Image Name': image_name_vec,
                    'QR_info': barcode,
                    'Object_ID': indeces,
                    'Patch_size': patch_size_vec
                }
                df_fore = pd.DataFrame(data)
                # Combine with the features
                df = pd.concat([df_fore, df], axis=1)

                ## Calculate additional features
                # Volume
                df["Ellipsoid_model_volume"] = (4/3) * np.pi * (df["RP_Minor_axis_length"] / 2) * ((df["RP_Major_axis_length"] / 2) ** 2)
                # Eccentricity
                df["Eccentricity"] = np.sqrt(1 - ((0.5 * df["RP_Major_axis_length"]) ** 2 / (0.5 * df["RP_Minor_axis_length"]) ** 2))

                # Assign the berry ID by sorting on bounding box coordinates
                df = df.sort_values(by=["RP_BB_y", "RP_BB_x"])
                df = df.reset_index(drop=True)
                df["Object_ID"] = df.index

                # Append to CSV if it exists
                try:
                    existing_df = pd.read_csv(output_feature_filename)
                    df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                except FileNotFoundError:
                    pass

                df.to_csv(output_feature_filename, index=False)

                print(f"Image {image_name} processed and features saved with {w} berries detected.")

                # Save the image with predicted annotations, if requested
                # THIS WILL NEED TO BE CHANGED FOR ROT DETECTION
                if save_predictions:
                    display_image_with_masks(image = image, result = result, class_names = ["ColorCard", "berry", "info", "rotten"], 
                                             output_path = os.path.join(img_save_folder, image_name_vec[0]), save = True)
                    # save_ROI_parallel(result, get_ids(result, 'berry'), os.path.join(img_save_folder, image_name_vec[0]))


            # DIFFERENT PROCESS FOR ROT DETECTION #
            elif (mod == "rot-det"):
                # Summarize results
                objects_count, n_rotten, n_sound, perc_rot, weighted_perc_rot = summarize_rot_det_results(result)
                # If indeces is length 0; warn that no berries were found
                if objects_count == 0:
                    print('\033[1;33mNo berries were found in the image!\033[0m')
                    continue

                # 7. Save results (image name, date, barcode, object count) to CSV
                image_name = local_image_path.split("/")[-1]

                data = {
                    'Image Name': image_name,
                    'NumberSoundBerries': n_sound,
                    'NumberRottenBerries': n_rotten,
                    'FruitRotPer': perc_rot,
                    'FruitRotPerWtd': weighted_perc_rot
                }
                df = pd.DataFrame(data, index = [0])

                # Append to CSV if it exists
                try:
                    existing_df = pd.read_csv(output_feature_filename)
                    df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                except FileNotFoundError:
                    pass

                df.to_csv(output_feature_filename, index=False)

                print(f"Image {image_name} processed, and features saved with {n_sound} sound berries and {n_rotten} rotten berries detected.")

                if save_predictions:
                    save_ROI_boxes(image = image, result = result, class_names = ["rotten", "sound"], output_path = os.path.join(img_save_folder, image_name))

            print(f"Image {p + 1} of {n_images} processed.\n")

        print("\nAll images processed. Results are saved in " + img_save_folder)

        # Modify the output data frame for berry seg
        if mod == "berry-seg":
            # Read the df back in
            df = pd.read_csv(output_feature_filename)

            # Convert features to cm and rename
            # Use the final patch size to calculate pixels to cm
            cm_per_pixel = float(args.patch_size) / patch_size_use
            df["Area"] = df["RP_Area"] * (cm_per_pixel ** 2)
            df["Length"] = df["RP_Minor_axis_length"] * cm_per_pixel
            df["Width"] = df["RP_Major_axis_length"] * cm_per_pixel
            df["Ellipsoid_model_volume"] = df["Ellipsoid_model_volume"] * (cm_per_pixel ** 3)
            df["cm_per_pixel"] = cm_per_pixel


            ## Subset features
            if args.reduce_features:
                features_select = ["Date", "Image Name", "QR_info", "Object_ID", "Patch_size", "name", "Area", "Length", "Width", "Ellipsoid_model_volume", "Eccentricity", "Red_Color_Mean", "Red_Color_Median", "Red_Color_Std", "Green_Color_Mean", "Green_Color_Median", "Green_Color_Std", "Blue_Color_Mean", "Blue_Color_Median", "Blue_Color_Std", "L_Color_Mean", "L_Color_Median", "L_Color_Std", "a_Color_Mean", "a_Color_Median", "a_Color_Std", "b_Color_Mean", "b_Color_Median", "b_Color_Std"]
                df = df[features_select]
            else:
                features_drop = ["RP_Area", "RP_Major_axis_length", "RP_Minor_axis_length"]
                df = df.drop(columns = features_drop)
            
            # Save the df
            df.to_csv(output_feature_filename, index=False)
   

# Runs main function
if __name__ == '__main__':
    main()



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


# options parsers
def options():
    parser = argparse.ArgumentParser(description='BerryBoxAI v 1.0')
    parser.add_argument('-m', '--module', help='Which module to run. Can be "berry-seg" for berry segmentation and quality or "rot-det" for detecting the amount of fruit rot in a sample.', 
                        required = True, choices = ['berry-seg', 'rot-det'], default = 'berry-seg')
    parser.add_argument('--input', help='Directory containing images to process. Providing this input directory will disable the interactive mode of berrybox and instead run the inference pipeline on a batch of images. If ignored, the pipeline will run in interactive mode.', 
                        required = False, default = "NULL")
    parser.add_argument('--output', help='The directory to store the data output', required = True)
    parser.add_argument('--conf', help='Confidence level for segmenting or detecting objects from the model', required = False, default = 0)
    parser.add_argument('--imgsz', help='Image size before sending it to the model', required = False, default = (0, 0))
    parser.add_argument('--save', help='Save the annotated images from the model output', default = False, action = 'store_true')
    parser.add_argument('--preview', help = 'Display a preview of the image with predicted features.', default = False, action = 'store_true')
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
def display_image_with_masks(image, results, class_names, show_masks = True):

    # Get the masks, boxes, and classes from the results
    if show_masks:
        masks = results[0].masks.data.cpu().numpy()  # Segmentation masks
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Generate random colors for each mask
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # Display boxes or masks
    for i, box in enumerate(boxes):
        color = colors[i]
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
        class_name = class_names[int(class_ids[i])]
        label = f'{class_name}, Conf: {confidences[i]:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

def save_ROI_boxes(image, results, class_names, output_path):
    image1 = image
    # Get the masks, boxes, and classes from the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Generate random colors for each mask
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # Display boxes or masks
    for i, box in enumerate(boxes):
        color = colors[i]

        # Draw the bounding box around the object
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(image1, (x1, y1), (x2, y2), color, 2)

        # Draw the class label and confidence score
        class_name = class_names[int(class_ids[i])]
        label = f'{class_name}, Conf: {confidences[i]:.2f}'
        cv2.putText(image1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save
    cv2.imwrite(output_path, image1)


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
        hostname = "169.254.111.10"
        username = "cranpi2"
        password = "usdacran"
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
    
    ## LOAD THE YOLO MODEL ##
    ['berry-seg', 'rot-det']
    if mod == "berry-seg":
        task = "segment"
    elif mod == "rot-det":
        task = "detect"
    if platform.system() == "Windows":
        model_name = "berrybox_" + mod + "_openvino_model"
        device = "cpu"
    elif platform.system() == "Darwin":
        device = "mps"
        model_name = "berrybox_" + mod + ".pt"
    else:
        model_name = "berrybox_" + mod + ".pt"

    # This finds the model within the package structure
    package_dir = pkg_resources.resource_filename('berryboxai', 'data')
    model_path = os.path.join(package_dir, 'weights', model_name)

    model = YOLO(model_path, task = task)

    # Set image size
    image_size = args.imgsz
    if image_size == (0, 0):
        if mod == "berry-seg":
            image_size = (1856, 2784)
        elif mod == "rot-det":
            image_size = (1600, 2400)

    confidence = float(args.conf)
    if confidence == 0:
        if mod == "berry-seg":
            confidence = 0.75
        elif mod == "rot-det":
            confidence = 0.40

    # Determine the device to use


    ## SET MODEL ARGUMENTS
    model_params = {
        'save': False, # save image results
        'show_labels': True,   # hide labels
        'show_conf': True, # hide confidences
        'save_crop': False, # save cropped prediction boxes
        'line_width': 3, # bounding box line width
        'conf': confidence, # confidence threshold
        'iou': 0.75, # NMS IoU threshold
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
                result = results[0]

                print("Processing and saving model results...")
                ## PROCESS RESULTS DEPENDING ON THE MODULE ##
                if (mod == "berry-seg"):
                    # 6. Process the results
                    # Try color correction; skip if it doesn't work
                    try:
                        result, patch_size = color_correction(result)
                    except:
                        continue

                    # Get features
                    df1 = get_all_features_parallel(result, name= 'berry')
                    df2 = get_all_features_parallel(result, name= 'rotten')
                    df = pd.concat([pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), pd.concat([df1, df2], ignore_index = True)], axis = 1)    
                    w,_ = df.shape
                    image_name_vec = [image_name]*w
                    patch_size = [np.mean(patch_size)]*w
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
                        'Patch_size': patch_size
                    }
                    df_fore = pd.DataFrame(data)
                    # Combine with the features
                    df = pd.concat([df_fore, df], axis=1)

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
                        save_ROI_parallel(result, get_ids(result, 'berry'), os.path.join(img_save_folder, image_name_vec[0]))

                    # Show a preview of the result
                    if args.preview:
                        print("Close the preview window before proceeding to the next sample.")
                        display_image_with_masks(image = image, results = results, class_names = ["color_card", "berry", "rotten"])

                # DIFFERENT PROCESS FOR ROT DETECTION #
                elif (mod == "rot-det"):
                    # Get the boxes
                    detected_boxes = results[0].boxes
                    detected_classes = detected_boxes.cls.cpu().numpy()
                    objects_count = len(detected_classes)
                    # If indeces is length 0; warn that no berries were found
                    if objects_count == 0:
                        print('\033[1;33mNo berries were found in the image!\033[0m')
                        continue

                    # Count rot
                    n_rotten = (detected_classes == 0).sum()
                    n_sound = (detected_classes == 1).sum()

                    # 7. Save results (image name, date, barcode, object count) to CSV
                    image_name = local_image_path.split("/")[-1]

                    data = {
                        'Date': current_datetime,
                        'Image Name': image_name,
                        'QR_info': barcode,
                        'NumberSoundBerries': n_sound,
                        'NumberRottenBerries': n_rotten,
                        'FruitRotPer': round((n_rotten / (n_rotten + n_sound) * 100), 3)
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
                        save_ROI_boxes(image = image, results = results, class_names = ["rotten", "sound"], output_path = os.path.join(img_save_folder, image_name))

                    # Show a preview of the result
                    if args.preview:
                        print("Close the preview window before proceeding to the next sample.")
                        display_image_with_masks(image = image, results = results, class_names = ["rotten", "sound"], show_masks = False)


        finally:
            # Ensure the SSH connection is closed at the end
            ssh.close()
            print("SSH connection closed.")

    else:
        ## RUN THE BATCH PIPELINE ##
        print("\nRunning berryboxai in batch mode using the " + mod + " module...\n")

        # List images in the input directory
        image_list = os.listdir(input_dir)
        image_path_list = [os.path.join(input_dir, x) for x in image_list if ".JPG" in x.upper()]

        # Print the number of images found
        print("Using images from the directory: " + input_dir)
        print("Discovered " + str(len(image_list)) + " images in the directory")
        print("Running the deep learning model on the images...")

        # Iterate over images
        for p, local_image_path in enumerate(image_path_list):

            # 5. Read in the image and resize and run through the YOLO model
            image_name = local_image_path.split("/")[-1]
            image = cv2.imread(local_image_path)
            image = cv2.resize(image, (newW, newH))
            results = model.predict(source = image, **model_params)
            result = results[0]

            print("Processing and saving model results...")
            ## PROCESS RESULTS DEPENDING ON THE MODULE ##
            if (mod == "berry-seg"):
                # 6. Process the results
                # Try color correction; skip if it doesn't work
                try:
                    result, patch_size = color_correction(result)
                except:
                    continue

                # Get features
                df1 = get_all_features_parallel(result, name= 'berry')
                df2 = get_all_features_parallel(result, name= 'rotten')
                df = pd.concat([pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), pd.concat([df1, df2], ignore_index = True)], axis = 1)    
                w,_ = df.shape
                image_name_vec = [image_name]*w
                patch_size = [np.mean(patch_size)]*w
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
                    'Patch_size': patch_size
                }
                df_fore = pd.DataFrame(data)
                # Combine with the features
                df = pd.concat([df_fore, df], axis=1)

                # Append to CSV if it exists
                try:
                    existing_df = pd.read_csv(output_feature_filename)
                    df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                except FileNotFoundError:
                    pass

                df.to_csv(output_feature_filename, index=False)

                print(f"Image {image_name} processed and features saved with {w} berries detected.\n")

                # Save the image with predicted annotations, if requested
                # THIS WILL NEED TO BE CHANGED FOR ROT DETECTION
                if save_predictions:
                    save_ROI_parallel(result, get_ids(result, 'berry'), os.path.join(img_save_folder, image_name_vec[0]))


            # DIFFERENT PROCESS FOR ROT DETECTION #
            elif (mod == "rot-det"):
                # Get the boxes
                detected_boxes = results[0].boxes
                detected_classes = detected_boxes.cls.cpu().numpy()
                objects_count = len(detected_classes)
                # If indeces is length 0; warn that no berries were found
                if objects_count == 0:
                    print('\033[1;33mNo berries were found in the image!\033[0m')
                    continue

                # Count rot
                n_rotten = (detected_classes == 0).sum()
                n_sound = (detected_classes == 1).sum()

                # 7. Save results (image name, date, barcode, object count) to CSV
                image_name = local_image_path.split("/")[-1]

                data = {
                    'Date': current_datetime,
                    'Image Name': image_name,
                    'QR_info': barcode,
                    'NumberSoundBerries': n_sound,
                    'NumberRottenBerries': n_rotten,
                    'FruitRotPer': round((n_rotten / (n_rotten + n_sound) * 100), 3)
                }
                df = pd.DataFrame(data, index = [0])

                # Append to CSV if it exists
                try:
                    existing_df = pd.read_csv(output_feature_filename)
                    df = pd.concat([existing_df, df], axis = 0, ignore_index=True)
                except FileNotFoundError:
                    pass

                df.to_csv(output_feature_filename, index=False)

                print(f"Image {image_name} processed, and features saved with {n_sound} sound berries and {n_rotten} rotten berries detected.\n\n")

                if save_predictions:
                    save_ROI_boxes(image = image, results = results, class_names = ["rotten", "sound"], output_path = os.path.join(img_save_folder, image_name))

            print(f"Image {p + 1} of {len(image_list)} processed.")

        print("\nAll images processed. Results are saved in " + img_save_folder)   

# Runs main function
if __name__ == '__main__':
    main()



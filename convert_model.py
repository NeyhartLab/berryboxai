## BerryBox AI
## 
## Convert a model to open vino format
## 
# Import packages here
from ultralytics import YOLO
import os
import argparse

def convert_yolo_to_openvino(input_path, output_dir):
    # Load the YOLO model
    model = YOLO(input_path)
    # Check which module it is and assign image size
    if "berry-seg" in input_path:
        image_size = (1856, 2784)
    elif "rot-det" in input_path:
        image_size = (1600, 2400) 
    
    # Determine output paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_model_path = os.path.join(output_dir, f'{base_name}_openvino_model')
    
    # Export the model to OpenVINO format
    model.export(format='openvino', path = output_model_path, imgsz = image_size, half = True)

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO weights to OpenVINO format')
    parser.add_argument('--input', required=True, help='Path to the input YOLO weights file')
    parser.add_argument('--output', required=True, help='Directory to save the OpenVINO weights file')
    
    args = parser.parse_args()
    
    convert_yolo_to_openvino(args.input, args.output)

if __name__ == '__main__':
    main()

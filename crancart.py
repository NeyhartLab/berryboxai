
## CranCart Controller
## 
## This script provides all of the functions needed to operate the CranCart
## and collect images (rgb or thermal) along with geographic coordinates
## 
## Author: Jeff Neyhart
## 
## Usage
## python crancart.py [ADD OPTIONS HERE]
##
## 

# Import packages here
import serial
import pynmea2
import numpy as np
import time
import requests
import os
import re
import subprocess
import argparse


## Description
## 
## 
## To-do
## 1. The output is currently saved to the internal rpi disk, but we should probably
## have the files saved to a removable hard drive
## 2. The current code for getting latitude and longitude is inefficient becasue it opens and then closes a connection;
## see lines 385 and below for code that should be used instead.
## 


# options parsers
def options():
    parser = argparse.ArgumentParser(description='CranCart hardware controller v 1.0')
    parser.add_argument('-m', '--module', help='Which module to run. Can be "thermal" for thermal images of plots, "rgb" for RGB images of plots, or "both" for both. Default is "both."', 
                        required = True, choices = ['thermal', 'rgb', 'both'], default = 'both')
    parser.add_argument('--trial', help='What trial are you in?', required = True)
    parser.add_argument('--flirip', help='IP address of the FLIR A70 camera. Default is "192.168.137.3"', default = '192.168.137.3')
    parser.add_argument('--nogps', help='Do not get lat/long coordinates for each image.', default=False, action='store_true')
    parser.add_argument('--gpsport', help='Port name for the emlid RS+ GNSS receiver. If missing, tries to find the port from USB.',
                        required = False, default = "none")
    parser.add_argument('--gpsreadings', help='Number of readings from the GNSS receiver to average to get lat/long for an image.',
                        required = False, default = 5, type = int)
    args = parser.parse_args()
    return args


# # Clean the arguments
# def clean_args(args):
#     # If mod is "thermal" or "both", the flir IP must be provided
#     mod = args.module
#     if mod == "thermal" or mod == "both":
#         if args.flir



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


# A function to check the connection to the thermal camera
def check_thermal_camera_connection(ip: str):
    # Set the full addresses for the flir rest API
    # API request for a IR image
    api_request_ir = "http://" + ip + "/api/image/current?imgformat=JPEG&tempUnit=C"
    print("Checking thermal camera connection...")
    # Try a request to the thermal camera
    try:
        req = requests.get(api_request_ir)
        if not req.status_code == 200:
            raise RuntimeError('The FLIR thermal camera is not connected! Verify connections and try again.')
        else:
            print("Thermal camera connected!")
    except:
        raise RuntimeError('The FLIR thermal camera is not connected! Verify connections and try again.')
        

# A function to check the connection to the GNSS receiver
def check_gnss_receiver_connection(port: str = "none"):
    # First check that the Emlid device is connected
    print("Checking GNSS device connection...")
    df = subprocess.check_output("lsusb")
    if not "Emlid ReachRS2+" in str(df):
        raise RuntimeError("The Emlid Reach RS2+ GNSS receiver is not connected. Verify connection and try again.")
    # Now try to find the port
    if (port == "none"):
        # The port is nearly always something like "/dev/ttyACM[#]" where[#] is an integer
        # Iterate over integers and test the connection
        i = 0
        while i < 5:
            port_i = "/dev/ttyACM" + str(i)
            if os.path.exists(port_i):
                try:
                    ser = serial.Serial(port_i, 9600, timeout = 1)
                except:
                    continue
                print("GNSS device connected!")
                return port_i
        raise RuntimeError("The Emlid Reach RS2+ GNSS receiver is not connected. Verify connection and try again.")
    # Else try the port provided
    else:
        port_i = port
        if not os.path.exists(port_i):
            raise RuntimeError("The gps port provided does not exist!")
        else:
            try:
                ser = serial.Serial(port_i, 9600, timeout = 1)
            except:
                print("The Emlid Reach RS2+ GNSS receiver is not connected. Verify connection and try again.")
            print("GNSS device connected!")
            return port_i
        
# A function that checks the nikon camera for connection and sets configurations
def check_nikon_camera(camera: str = "Nikon DSC D7500"):
    # First check that the Emlid device is connected
    print("Checking Nikon Camera connection...")
    det = subprocess.check_output(["gphoto2",  "--auto-detect"])
    if not camera in str(det):
        raise RuntimeError(camera + " is not connected!")
    
    # Kill gphoto processes
    subprocess.call(["pkill", "-f", "gphoto2"])

    # Set configurations
    subprocess.call(["gphoto2", "--set-config", "iso=100"])
    subprocess.call(["gphoto2", "--set-config", "whitebalance=7"])
    subprocess.call(["gphoto2", "--set-config", "/main/capturesettings/f-number=0"])
    subprocess.call(["gphoto2", "--set-config", "/main/capturesettings/shutterspeed=17"])

    print(camera + " is connected and configured!")
                

# A function that queries the GPS for n readings and returns the average lat and long
def get_gnss_data(port, baud_rate = 9600, n_readings = 5):
    # Define the serial connection
    # This may need to be adjusted to be a serial stream
    ser = serial.Serial(port, baud_rate, timeout = 1)
    lats = []
    longs = []
    while len(lats) < n_readings:
        line = ser.readline().decode("utf-8").strip()
        if line.startswith("$GNGGA"):
            msg = pynmea2.parse(line)
            lat = msg.latitude
            long = msg.longitude
            
            if lat == 0 or long == 0:
                continue
            else:
                lats.append(float(lat))
                longs.append(float(long))

    return np.mean(lats), np.mean(longs)

# A function to get thermal and RGB images from the thermal camera
def capture_thermal_image(irapi: str, visapi: str, file_prefix: str, file_outdir: str):
    # Adjust the image prefix
    image_prefix = file_prefix + "flira70"

    # Get the current time
    ct = get_time()
    
    # Capture an RGB image
    rgb_filename = file_outdir + "/" + image_prefix + "_rgb_" + ct + ".jpg"
    with open(rgb_filename, 'wb') as f:
        f.write(requests.get(visapi).content)
        
    # Capture a thermal image
    therm_filename = file_outdir + "/" + image_prefix + "_thermal_" + ct + ".jpg"
    with open(therm_filename, 'wb') as f:
        f.write(requests.get(irapi).content)

    # Return the rgb filename and thermal filename
    return rgb_filename, therm_filename

# A function to capture a RGB image from the DLSR camera
def capture_rgb_image(file_prefix: str, file_outdir: str, camera: str = "Nikon D7500"):
    # Adjust the image prefix
    image_prefix = file_prefix + camera.lower().replace(" ", "")

    # Get the current time
    ct = get_time()

    # Capture an RGB image
    outfile_name = file_outdir + "/" + image_prefix + "_" + ct + ".jpg"
    subprocess.call(["gphoto2", "--capture-image-and-download", "--filename", outfile_name,"--force-overwrite"],
                    stdout = subprocess.DEVNULL)

    return outfile_name


# A function to wait for Enter input
def recursive_enter(question):
    answer = input(question)
    if answer == "Q":
        print("Quitting...\n")
        return answer
    elif answer != '':
        print("I did not understand your answer\n")
        return recursive_enter("Press Enter to capture image (or 'Q' to exit).\n")
    else:
        return answer
    
# A function to wait for plot input
def recursive_plot(question):
    answer = input(question)
    if answer == "Q":
        print("Quitting...\n")
        return answer
    elif answer == '':
        print("I did not understand your answer\n")
        return recursive_plot("Enter plot number to capture image (or 'Q' to exit).\n")
    else:
        try:
            plot = int(answer)
            return plot
        except:
            print("Your answer was not an integer.\n")
            return recursive_plot("Enter plot number to capture image (or 'Q' to exit).\n")
        

        
## The main function
def main():
    # Get arguments
    args = options()
    # Determine the module
    mod = args.module
    # Trial
    trial = args.trial
    # Create a session name
    if mod == "both":
        mod1 = "rgb-thermal"
    else:
        mod1 = mod
    session_name = trial + "_" + mod1 + "_" + get_date()

    # Create a prefix for each image
    # Image prefix
    image_prefix = trial + "_"
    
    
    ## USB HARD DRIVE SETUP ##


    ## GNSS RECEIVER SETUP ##
    gpsport = args.gpsport
    run_gps = not args.nogps
    if run_gps:
        gpsport = check_gnss_receiver_connection(gpsport)
        n_readings = args.gpsreadings

    ## THERMAL CAMERA SETUP ##
    flirip = args.flirip
    # Check thermal camera connection
    if mod == "thermal" or mod == "both":
        chk = check_thermal_camera_connection(ip = flirip)
        # Set up the rest API addresses
        api_request_ir = "http://" + flirip + "/api/image/current?imgformat=JPEG&tempUnit=C"
        # API request for a visual image
        api_request_visual = "http://" + flirip + "/api/image/current?imgformat=JPEG_visual"
    
    # ## RGB CAMERA SETUP ##
    # Check the nikon camera connection
    if mod == "rgb" or mod == "both":
        check_nikon_camera()


    ## OUTPUT DIRECTORY AND FILE SETUP ##

    # Create directories to store images and metadata
    # Create an output directory
    crancart_dir = "/home/cranpi2/crancart"
    session_dir = crancart_dir + "/" + session_name
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    output_image_dir = session_dir + "/images/"
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    # Output directories for jpegs and rjpegs
    if mod == "thermal":
        output_rjpeg_dir = output_image_dir + "/raw_rjpeg_images/"
        if not os.path.exists(output_rjpeg_dir):
            os.mkdir(output_rjpeg_dir)
    elif mod == "rgb":
        output_jpeg_dir = output_image_dir + "/raw_jpeg_images/"
        if not os.path.exists(output_jpeg_dir):
            os.mkdir(output_jpeg_dir)
    else:
        output_rjpeg_dir = output_image_dir + "/raw_rjpeg_images/"
        if not os.path.exists(output_rjpeg_dir):
            os.mkdir(output_rjpeg_dir)     
        output_jpeg_dir = output_image_dir + "/raw_jpeg_images/"
        if not os.path.exists(output_jpeg_dir):
            os.mkdir(output_jpeg_dir)
          
    # Create an output metadata file
    # It will have the trial name, module, and datestamp
    metadata_filename = session_dir + "/" + trial + "_" + mod1 + "_" + get_time() + "_image_metadata.txt"
    handle = open(metadata_filename, "w")
    # First the header line with the information
    handle.write('\t'.join(["trial", "session", "image", "plot", "latitude", "longitude"]) + "\n")


    ## RUN THE CAPTURE PIPELINE ##

    # While loop
    while True:
        # Prompt the user to hit enter to capture a photo
        if run_gps:
            reply = recursive_enter("Press Enter to capture image (or 'Q' to exit).\n")
        else:
            reply = recursive_plot("Enter plot number to capture image (or 'Q' to exit).\n")
        
        if reply == "Q":
            break

        ## CAPTURE ##
        # 1. Get GPS coordinates
        if run_gps:
            img_lat, img_long = get_gnss_data(port = gpsport, n_readings = n_readings)
            plot = ""
        else:
            img_lat = ""
            img_long = ""
            plot = reply
        
        # 2. Get the thermal image and save it
        if mod == "thermal" or mod == "both":
            rgb_file, therm_file = capture_thermal_image(irapi = api_request_ir, visapi = api_request_visual,  
                                                         file_prefix = image_prefix, file_outdir = output_rjpeg_dir)
        else:
            rgb_file = None
            therm_file = None
            
        # 3. Get the RGB image and save it
        if mod == "rgb" or mod == "both":
            nikon_file = capture_rgb_image(file_prefix = image_prefix, file_outdir = output_jpeg_dir)
        else:
            nikon_file = None

        # 4. Record the name(s) of the images and lat/long to the txt file
        files_toprint = [rgb_file, therm_file, nikon_file]
        files_toprint = [x for x in files_toprint if x] # Remove Nones
        # Iterate and print
        for file_toprint in files_toprint:
            toprint = [trial, session_name, file_toprint, str(plot), str(img_lat), str(img_long)]
            handle.write('\t'.join(toprint) + '\n')
            
        # 5. Print completion statement
        print("Plot images collected and saved.\n")
            
    # Close the handle
    handle.close()

    print("Capture session complete.")
    print("Images are stored here: " + output_image_dir)
    print("Metadata is here: " + metadata_filename)


# Runs main function
if __name__ == '__main__':
    main()





## INTEGRATE THE FOLLOWING CODE ##

# import serial
# import pynmea2
# import subprocess
# import os
# import time

# def initialize_serial(serial_port):
#     # Open serial port
#     ser = serial.Serial(serial_port, baudrate=9600, timeout=1)
#     return ser

# def read_gnss_data(ser):
#     try:
#         while True:
#             # Read data from serial port
#             data = ser.readline().decode('utf-8')

#             # Check if the data is a GNSS sentence
#             if data.startswith('$GPGGA'):
#                 # Parse the NMEA sentence
#                 msg = pynmea2.parse(data)
                
#                 # Extract latitude and longitude
#                 latitude = msg.latitude
#                 longitude = msg.longitude
                
#                 print("Latitude:", latitude)
#                 print("Longitude:", longitude)

#                 return latitude, longitude

#     except KeyboardInterrupt:
#         # Close serial port on keyboard interrupt
#         ser.close()

# def capture_image():
#     # Capture image using gphoto2
#     image_name = f"image_{int(time.time())}.jpg"  # Unique name based on timestamp
#     subprocess.run(["gphoto2", "--capture-image-and-download", f"--filename={image_name}"])

#     return image_name

# def main():
#     # Replace 'COM3' with the appropriate serial port on your system
#     serial_port = '/dev/ttyUSB0'  # Example for Linux
#     ser = initialize_serial(serial_port)  # Initialize serial port

#     try:
#         while True:
#             input("Press Enter to capture image...")  # Wait for user input
#             print("Capturing image...")
#             image_name = capture_image()  # Capture image
#             print("Image captured:", image_name)

#             print("Reading GNSS data...")
#             latitude, longitude = read_gnss_data(ser)  # Read GNSS data
#             print("GNSS data read successfully.")

#             # Associate latitude and longitude with image name
#             with open("image_metadata.txt", "a") as file:
#                 file.write(f"Image: {image_name}, Latitude: {latitude}, Longitude: {longitude}\n")

#             print("Image metadata saved.")

#     except KeyboardInterrupt:
#         # Close serial port on keyboard interrupt
#         ser.close()

# if __name__ == "__main__":
#     main()



## OR THE FOLLOWING CODE ##

# import serial
# import pynmea2
# import subprocess
# import os
# import time
# import threading

# def initialize_serial(serial_port):
#     # Open serial port
#     ser = serial.Serial(serial_port, baudrate=9600, timeout=1)
#     return ser

# def read_gnss_data(ser, latitude_queue, longitude_queue):
#     try:
#         while True:
#             # Read data from serial port
#             data = ser.readline().decode('utf-8')

#             # Check if the data is a GNSS sentence
#             if data.startswith('$GPGGA'):
#                 # Parse the NMEA sentence
#                 msg = pynmea2.parse(data)
               
#                 # Extract latitude and longitude
#                 latitude = msg.latitude
#                 longitude = msg.longitude
               
#                 print("Latitude:", latitude)
#                 print("Longitude:", longitude)

#                 latitude_queue.put(latitude)
#                 longitude_queue.put(longitude)

#     except KeyboardInterrupt:
#         # Close serial port on keyboard interrupt
#         ser.close()

# def capture_continuous_images(output_dir, interval=5):
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Capture images using gphoto2 in continuous mode
#     subprocess.Popen(["gphoto2", "--capture-image", "--interval", str(interval), "--filename", os.path.join(output_dir, "image_%Y%m%d_%H%M%S.jpg")])

# def main():
#     # Replace 'COM3' with the appropriate serial port on your system
#     serial_port = '/dev/ttyUSB0'  # Example for Linux
#     output_dir = "photos"  # Directory to save photos
#     interval = 5  # Interval between photo captures in seconds

#     # Initialize serial port for GNSS receiver
#     ser = initialize_serial(serial_port)

#     # Create queues to share data between threads
#     latitude_queue = queue.Queue()
#     longitude_queue = queue.Queue()

#     # Start GNSS data reading thread
#     gnss_thread = threading.Thread(target=read_gnss_data, args=(ser, latitude_queue, longitude_queue))
#     gnss_thread.start()

#     try:
#         # Continuously capture images and associate them with GNSS coordinates
#         while True:
#             print("Capturing images...")
#             capture_continuous_images(output_dir, interval=interval)

#             # Read GNSS data (blocking until new data is available)
#             latitude = latitude_queue.get()
#             longitude = longitude_queue.get()
#             print("GNSS data read successfully.")

#             # Clear queues after each image capture
#             while not latitude_queue.empty():
#                 latitude_queue.get()
#             while not longitude_queue.empty():
#                 longitude_queue.get()

#             # Associate latitude and longitude with images
#             image_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
#             for image_file in image_files:
#                 with open(os.path.join(output_dir, image_file.replace('.jpg', '_metadata.txt')), "w") as file:
#                     file.write(f"Latitude: {latitude}, Longitude: {longitude}\n")

#             print("Image metadata saved.")

#     except KeyboardInterrupt:
#         # Close serial port and join threads on keyboard interrupt
#         ser.close()
#         gnss_thread.join()

# if __name__ == "__main__":
#     main()
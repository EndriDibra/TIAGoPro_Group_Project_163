# Author: Endri Dibra  
# Group Project: Explainable Social Navigation
# Task: Camera Object Detection 

# Importing the required libraries
import cv2
import time
from ultralytics import YOLO


# Initializing and opening the default camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera was opened successfully, exiting if not
if not camera.isOpened():

    print("Error! Camera did not open.")

    # Exiting due to no camera
    exit(0)

# Defining the set of target classes for detection
targetClasses = {

    'bicycle', 'car', 'motorbike', 'bus', 'cat', 'dog',
    'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'vase', 'person', 'cup'
}

# Loading YOLOv11n model for classical object detection
model = YOLO("yolo11n.pt")

# Retrieving the dictionary of class names from the loaded model
classNames = model.names

# Creating a list of class IDs corresponding to targetClasses for filtering
targetClassIDs = [i for i, name in classNames.items() if name.lower() in targetClasses]

# Variables for FPS calculation
prevTime = 0
fps = 0

# Starting main loop for capturing and processing camera frames
while camera.isOpened():

    # Capturing a frame from the camera
    success, frame = camera.read()

    # Checking if frame was captured successfully, breaking loop if not
    if not success:

        print("Error! Camera reading stopped.")

        # Breaking the main loop due to failure
        break

    # Running inference on the captured frame using YOLOv11n
    results = model(frame, save=False)[0]

    # Filtering detections: remove classes not in targetClassIDs
    keep = []
    
    for i, cls in enumerate(results.boxes.cls):
        
        if int(cls) in targetClassIDs:
        
            keep.append(i)

    # Keeping only selected detections
    results = results[keep]

    # Using YOLO's built-in plotting for visualization
    plottedFrame = results.plot()

    # calculating FPS based on total frame processing time
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
    prevTime = currTime

    # overlay FPS on the plottedFrame
    cv2.putText(plottedFrame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Showing the processed frame with overlays on screen
    cv2.imshow("YOLO Object Detection", plottedFrame)

    # Exiting loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Releasing camera resource after loop ends
camera.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()
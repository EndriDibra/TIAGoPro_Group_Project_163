# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Camera Object Detection and Segmentation 

# Importing the required libraries
import cv2
import time
import numpy as np
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

# Loading YOLOv11n Segmentation model for object detection
modelSeg = YOLO("yolo11n-seg.pt")

# Retrieving the dictionary of class names from the loaded model
classNames = modelSeg.names

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

    # Running inference on the captured frame using the YOLOv11n segmentation model
    results = modelSeg(frame, save=False)[0]

    # Copying the frame to prepare for overlay drawing
    overlay = frame.copy()

    # Checking if detection boxes are present in the results
    if results.boxes is not None:

        # Looping through all detected boxes
        for i in range(len(results.boxes.cls)):

            # Getting class ID of current detection
            clsID = int(results.boxes.cls[i])

            # Skipping detection if class ID is not in target list
            if clsID not in targetClassIDs:

                continue

            # Getting class name in lowercase
            className = classNames[clsID].lower()

            # Extracting bounding box coordinates as integers
            xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy

            # Drawing segmentation mask if available
            if results.masks is not None:

                # Extracting mask for current detection
                mask = results.masks.data[i].cpu().numpy()
                
                # Resizing mask to frame dimensions
                maskResized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Creating blank image for mask coloring
                coloredMask = np.zeros_like(frame, dtype=np.uint8)
                
                # Coloring green channel with mask data
                coloredMask[:, :, 1] = (maskResized * 255).astype(np.uint8)

                # Blending mask overlay onto frame copy
                overlay = cv2.addWeighted(overlay, 1.0, coloredMask, 0.4, 0)

            # Drawing bounding box and label on overlay
            conf = results.boxes.conf[i].item()

            label = f"{classNames[clsID]} {conf:.2f}"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(overlay, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # calculating FPS based on total frame processing time
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
    prevTime = currTime

    # overlay FPS on the plottedFrame
    cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Showing the processed frame with overlays on screen
    cv2.imshow("Object Detection and Segmentation", overlay)

    # Exiting loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Releasing camera resource after loop ends
camera.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()
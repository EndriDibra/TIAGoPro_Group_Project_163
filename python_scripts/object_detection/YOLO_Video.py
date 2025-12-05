# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Camera Object Detection and Segmentation  

# Importing the required libraries
import cv2
import time
from ultralytics import YOLO


# Opening the default web camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera is working
if not camera.isOpened():

    print("Error! Camera did not open.")

    exit(0)

# Loading YOLOv11n model for object detection
model = YOLO("yolo11n.pt")

# Loading YOLOv11n Segmentation model for object detection
modelSeg = YOLO("yolo11n-seg.pt")

# Variables for FPS calculation
prevTime = 0
fps = 0

# Looping through camera frames
while camera.isOpened():

    # Reading camera frames
    success, frame = camera.read()

    # Checking if camera reading is working properly
    if not success:

        print("Error! Camera reading did not work.")

    # Applying the YOLOv11n model on
    # each camera frame for object detection
    results = model(frame, save=False)
    results = results[0].plot()

    # Applying the segmentation model
    # on each camera frame to detect objects
    segResults = modelSeg(frame, save=False)
    segResults = segResults[0].plot()

    # calculating FPS based on total frame processing time
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
    prevTime = currTime

    # overlay FPS on the results
    cv2.putText(results, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # overlay FPS on the segResults
    cv2.putText(segResults, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Displaying the detected obstacles
    cv2.imshow("YOLO Object Detection", results)

    # Displaying the segmented detected obstacles
    cv2.imshow("YOLO Object Detection and Segmentation", segResults)

    # Teriminating the process by pressing key "q:quit"
    if cv2.waitKey(1) & 0XFF == ord("q"):

        break

# Closing web camera
camera.release()

# Terminating all opened windows
cv2.destroyAllWindows() 
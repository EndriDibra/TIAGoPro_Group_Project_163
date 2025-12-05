# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Image Object Detection and Segmentation 

# Importing the required libraries
import cv2
from ultralytics import YOLO


# Reading the image
image = cv2.imread("Occupancy_Grid_Map.png")

# Resizing the image to 500 x 500
image = cv2.resize(image, (500, 500))

# Loading YOLOv11n model for object detection
model = YOLO("yolo11n.pt")

# Loading YOLOv11n Segmentation model for object detection
modelSeg = YOLO("yolo11n-seg.pt")

# Applying the model on the image to detect objects
results = model(image, save=False)
results = results[0].plot()

# Applying the segmentation model on the image to detect objects
segResults = modelSeg(image, save=False)
segResults = segResults[0].plot()

# Displaying the image with the detected obstacles
cv2.imshow("YOLO Detection", results)

# Displaying the image with the segmented detected obstacles
cv2.imshow("YOLO Detection and Segmentation", segResults)

# Terminating the process by pressing key "q:quit" 
cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows() 
# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Face Counting

# Importing the required libraries
import cv2
import time
import numpy as np
import mediapipe as mp


# MediaPipe FaceMesh setup 
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=30)
mpDrawing = mp.solutions.drawing_utils

# Opening default camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera is opened successfully
if not camera.isOpened():
   
    print("Error! Camera did not open.")
   
    exit()

# Variables for FPS calculation
prevTime = 0
fps = 0

# Looping through camera frames  
while camera.isOpened():

    # Reading a frame from the camera
    success, frame = camera.read()

    # Checking if the frame was successfully captured
    if not success:
        
        print("Error! Failed to capture frame.")
        
        break

    # Converting frame to RGB for MediaPipe processing
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing the RGB frame for face mesh detection
    results = faceMesh.process(rgbFrame)

    # Initializing total face counter
    totalFaces = 0

    # Creating a blank mask for the face regions
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Copy of the original frame for drawing landmarks
    landmarkFrame = frame.copy()

    # Checking if any faces are detected
    if results.multi_face_landmarks:

        # Counting total number of detected faces
        totalFaces = len(results.multi_face_landmarks)

        # Applying a single Gaussian blur to the entire frame for performance
        blurredFrame = cv2.GaussianBlur(frame, (101, 101), 30)

        # Iterating through all detected faces
        for faceLandmarks in results.multi_face_landmarks:

            # Getting the frame dimensions
            h, w, _ = frame.shape

            # Extracting all landmark (x, y) coordinates
            points = np.array([
                
                [int(landmark.x * w), int(landmark.y * h)]
                for landmark in faceLandmarks.landmark
            ], dtype=np.int32)

            # Creating a convex hull mask around the face
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Drawing the landmarks on the landmark frame
            mpDrawing.draw_landmarks(
                
                landmarkFrame,
                faceLandmarks,
                mpFaceMesh.FACEMESH_TESSELATION,
                mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mpDrawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )

        # Blending the blurred faces onto the original frame
        frame = np.where(mask[:, :, None] == 255, blurredFrame, frame)

    # Displaying information
    label = f"Faces Detected: {totalFaces}"
    
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(landmarkFrame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # calculating FPS based on total frame processing time
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
    prevTime = currTime

    # overlay FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # overlay FPS on the landmarkFrame
    cv2.putText(landmarkFrame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Showing the processed frames
    cv2.imshow("Face Detection with Polygon Blur", frame)
    cv2.imshow("Face Detection with Landmarks", landmarkFrame)

    # Breaking the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

# Cleanup after loop ends
camera.release()
cv2.destroyAllWindows()
faceMesh.close()
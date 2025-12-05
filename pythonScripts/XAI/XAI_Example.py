# Author: Endri Dibra 
# Group Project: XAI simple example

# Importing the required libraries
import cv2
import shap
import numpy as np
import mediapipe as mp
from matplotlib import cm
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# Loading image 
image_path = "Trump.png"  
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Face detection using MediaPipe 
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    results = face_detection.process(image_rgb)

    if not results.detections:
        raise ValueError("No face detected in the image!")

    detection = results.detections[0]

    bboxC = detection.location_data.relative_bounding_box

    h, w, _ = image_rgb.shape

    x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
    x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

    face_crop = image_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

# Preparing face for model, preprocessing
face_resized = cv2.resize(face_crop, (224, 224))
face_array = np.expand_dims(face_resized, axis=0)
face_preprocessed = preprocess_input(face_array)

# Loading pretrained model 
model = MobileNetV2(weights="imagenet")

# Making prediction 
pred = model.predict(face_preprocessed)
decoded = decode_predictions(pred, top=3)[0]
print("Predictions:", decoded)

# XAI SHAP explanation 
# Using a small background set for SHAP to compute feature contributions
background = np.random.randn(5, 224, 224, 3)
background = preprocess_input(background)

# GradientExplainer with NumPy arrays (DO NOT convert to tensors)
explainer = shap.GradientExplainer(model, background)
shap_values, indexes = explainer.shap_values(face_preprocessed, ranked_outputs=1)

# Denormalize for proper visualization 
# Converting [-1,1] -> [0,1]
face_display = (face_preprocessed + 1) / 2  

# Visualizing SHAP explanation
shap.image_plot(shap_values, face_display)
plt.show()

# Taking first channel importance map
# Summing over RGB channels
shap_map = shap_values[0][0].sum(axis=-1) 

# Normalizing 0-1
shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)  

# Converting to heatmap
heatmap = cm.jet(shap_map)[:, :, :3]  # RGB
heatmap = (heatmap * 255).astype(np.uint8)

# Resizing to original face crop size
heatmap_resized = cv2.resize(heatmap, (face_crop.shape[1], face_crop.shape[0]))

# Overlaying
overlayed = cv2.addWeighted(face_crop, 0.6, heatmap_resized, 0.4, 0)
cv2.imshow("SHAP Overlay", overlayed)
cv2.waitKey(0)
cv2.destroyAllWindows()
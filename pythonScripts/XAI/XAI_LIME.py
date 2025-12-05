# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: XAI LIME Implementation

# Importing required libraries
import os
import cv2
import numpy as np
from lime import lime_image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


# Setting input image and output folder
INPUT_IMAGE = "Person.jpg"
OUTPUT_DIR = "LIME_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loading input image
image = cv2.imread(INPUT_IMAGE)

if image is None:

    raise ValueError(f"Error! Image not found: {INPUT_IMAGE}")

# Converting image to RGB
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
overlayImage = imageRGB.copy()

# Loading YOLOv11n model
yoloModel = YOLO("yolo11n.pt")


# Defining LIME prediction function using YOLO
def yoloPredict(images):
    
    preds = []
    
    for img in images:
        
        imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result = yoloModel.predict(imgBGR, verbose=False)[0]
        
        probs = np.zeros(len(yoloModel.names))
        
        for box in result.boxes:
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            probs[cls] += conf
        
        if probs.sum() > 0:
        
            probs /= probs.sum()
        
        preds.append(probs)
    
    return np.array(preds)


# Running YOLO detection on original image
results = yoloModel.predict(imageRGB, verbose=False)[0]

# Initializing combined mask for heatmap
combinedMask = np.zeros_like(imageRGB, dtype=np.float32)

# Looping through each detected object
for i, box in enumerate(results.boxes):
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    
    crop = imageRGB[y1:y2, x1:x2]

    if crop.size == 0:

        continue
    
    # Normalizing crop for LIME
    cropNorm = crop.astype(np.float32) / 255.0
    
    # Initializing LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Running LIME explanation
    explanation = explainer.explain_instance(

        cropNorm,
        classifier_fn=yoloPredict,
        top_labels=1,
        hide_color=0,
        num_samples=4000
    )
    
    # Getting LIME mask for top class
    limeImg, mask = explanation.get_image_and_mask(
       
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=80
    )
    
    # Overlaying LIME boundaries on crop
    limeOverlay = mark_boundaries(limeImg, mask)
    limeOverlay = (limeOverlay * 255).astype(np.uint8)
    
    # Blending LIME overlay with original crop
    alpha = 0.6
    overlayImage[y1:y2, x1:x2] = cv2.addWeighted(crop, 1 - alpha, limeOverlay, alpha, 0)
    
    # Drawing bounding box and class label
    cv2.rectangle(overlayImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
   
    cv2.putText(overlayImage, f"Cls {cls_id} Conf {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Adding mask to combined heatmap
    combinedMask[y1:y2, x1:x2, 0] += mask * 255

# Clipping combined mask and converting to uint8
combinedMask = np.clip(combinedMask, 0, 255).astype(np.uint8)

# Blending combined mask with original image
combinedOverlay = cv2.addWeighted(imageRGB, 0.5, combinedMask, 0.5, 0)

# Saving LIME outputs
cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay_individual.png"), overlayImage)
cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay_combined.png"), combinedOverlay)

# Displaying results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(overlayImage)
plt.title("Individual LIME Overlays")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(combinedOverlay)
plt.title("Combined LIME Heatmap")
plt.axis("off")
plt.tight_layout()
plt.show()
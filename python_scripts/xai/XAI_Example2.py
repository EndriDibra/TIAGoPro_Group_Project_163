# Author: Endri Dibra 

# Importing the required libraries
import cv2
import shap
import numpy as np
from matplotlib import cm
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# Path of image
imagePath = "TS.jpg" 

# Loading, reading image
image = cv2.imread(imagePath)

# Checking if image exists
if image is None:

    raise ValueError(f"Error! Image not found: {imagePath}")

# Preprocessing image
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imgResized = cv2.resize(imageRGB, (224, 224))
imgArray = np.expand_dims(imgResized, axis=0)
imgPreprocessed = preprocess_input(imgArray)

# Loading the pretrained TF-Keras model, MobileNetV2
model = MobileNetV2(weights="imagenet")

# Predicting the actual classes 
predictions = model.predict(imgPreprocessed)

# Decoding predictions, taking the first 5 predictions/classifications
decoded = decode_predictions(predictions, top=5)[0]
print("\nPredictions:")

# Printing the first 5 objects and their prediction score
for i, (imagenetID, label, score) in enumerate(decoded):

    print(f"{i+1}. {label}: {score*100:.2f}%")

# SHAP XAI Explanation image preprocessing
background = np.random.randn(5, 224, 224, 3)
background = preprocess_input(background.astype(np.float32))

# Appying SHAP method on image
explainer = shap.GradientExplainer(model, background)
shapValues, indexes = explainer.shap_values(imgPreprocessed, ranked_outputs=1)

# Converting image for display
imgDisplay = (imgResized.astype(np.float32) / 255.0)

# Creating a SHAP heatmap
shapMap = shapValues[0][0].sum(axis=-1)
shapMap = (shapMap - shapMap.min()) / (shapMap.max() - shapMap.min() + 1e-8)

heatMap = cm.jet(shapMap)[:, :, :3]
heatMap = (heatMap * 255).astype(np.uint8)

heatMapResized = cv2.resize(heatMap, (imageRGB.shape[1], imageRGB.shape[0]))
overlayedShap = cv2.addWeighted(imageRGB, 0.6, heatMapResized, 0.4, 0)


# LIME XAI Explanation 
def imagePredictions(images):
    
    imgsPreprocessed = preprocess_input(images.copy())
    
    preds = model.predict(imgsPreprocessed)
    
    return preds


# Appying LIME method on image
explainerLime = lime_image.LimeImageExplainer()

explanation = explainerLime.explain_instance(

    imgResized, 
    imagePredictions, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000
)

# Getting LIME mask for top class
limeImg, mask = explanation.get_image_and_mask(
    
    explanation.top_labels[0],
    positive_only=False,
    num_features=10,
    hide_rest=False
)

limeImg = mark_boundaries(limeImg / 255.0, mask)

# Resizing LIME output to original size
limeDisplay = cv2.resize((limeImg * 255).astype(np.uint8), (imageRGB.shape[1], imageRGB.shape[0]))

# Display Results 
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(imageRGB)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(overlayedShap, cv2.COLOR_BGR2RGB))
plt.title("SHAP Explanation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(limeDisplay)
plt.title("LIME Explanation")
plt.axis("off")

plt.tight_layout()
plt.show()
# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: VLM-smolVLM Object Description

# Importing the required libraries
import os
import re
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


# Setting VLM model ID
# smolVLM-Instruct for lightweight inference
SOCIAL_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Path of input image containing objects
imagePath = "Image.jpeg"

# Initializing the processor (tokenizer + feature extractor)
socialProcessor = AutoProcessor.from_pretrained(SOCIAL_MODEL_ID)

# Initializing the model and moving it to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

socialModel = AutoModelForVision2Seq.from_pretrained(SOCIAL_MODEL_ID).to(device)
socialModel.eval()

print(f"Loaded model {SOCIAL_MODEL_ID} on device: {device}...")

# Checking if the image exists
if not os.path.exists(imagePath):

    raise FileNotFoundError(f"Image not found: {imagePath}")

# Loading image
img = cv2.imread(imagePath)

if img is None:

    raise ValueError(f"Could not load image: {imagePath}")

print(f"Loaded image: {imagePath}")

# Converting OpenCV BGR image to PIL RGB
pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# Function to query smolVLM for a short description of objects in the image
def DescribeObjects(img):
    
    # Constructing a concise instruction for the model
    messages = [

        {
            "role": "user",

            "content": [

                {"type": "image"},
                {"type": "text", "text": "Describe the main objects in this image briefly. Keep it short, no extra text."}
            ]
        }
    ]
    
    # Applying chat template
    prompt = socialProcessor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Processing image and text for model inference
    inputs = socialProcessor(text=prompt, images=img, return_tensors="pt").to(device)
    
    # Generating description without gradient computation
    with torch.no_grad():
     
        outputIDs = socialModel.generate(
     
            **inputs,
            max_new_tokens=30, # small number of tokens for short description
            do_sample=True,
            temperature=0.3,
        )
    
    # Decoding the generated tokens
    description = socialProcessor.decode(outputIDs[0], skip_special_tokens=True)
    
    # Cleaning the text
    clean_desc = re.sub(r'<[^>]+>', '', description).strip()
    
    return clean_desc


# Querying the model for the short description
print("Querying VLM for object description...")
description = DescribeObjects(pilIMG)

# Printing the model output
print("\nModel Output:")
print(description)
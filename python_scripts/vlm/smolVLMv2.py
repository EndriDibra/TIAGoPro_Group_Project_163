# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: VLM-smolVLM Object Description with Chain-of-Thought (CoT)

# Importing the required libraries
import os
import re
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


# Setting model ID for smolVLM-Instruct
SOCIAL_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Path of the image to describe
imagePath = "Image.jpeg"

# Loading the processor (tokenizer + feature extractor)
processor = AutoProcessor.from_pretrained(SOCIAL_MODEL_ID)

# Loading the VLM model and moving to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForVision2Seq.from_pretrained(SOCIAL_MODEL_ID).to(device)
model.eval()

# Checking if image exists
if not os.path.exists(imagePath):

    raise FileNotFoundError(f"Image not found: {imagePath}")

# Loading the image
img = cv2.imread(imagePath)

if img is None:

    raise ValueError(f"Could not load image: {imagePath}")

# Converting to PIL format (RGB)
pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(f"Loaded image: {imagePath}")


# Function to get short object description using CoT
def DescribeObjects(img):
    
    # CoT prompt for step-by-step reasoning
    messages = [
        
        {
            "role": "user",

            "content": [

                {"type": "image"},
                {"type": "text", "text": (

                    "You are an assistant that describes objects in an image. "
                    "Think step by step: first identify people, then furniture or other objects. "
                    "Keep the final description short (2-3 sentences). "
                    "Output only text describing what you see, no extra symbols or numbers."
                )}
            ]
        }
    ]

    # Applying chat template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Preparing model inputs
    inputs = processor(text=prompt, images=pilIMG, return_tensors="pt").to(device)

    # Generating output with limited tokens for efficiency
    with torch.no_grad():
     
        outputIDs = model.generate(
     
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.3
        )

    # Decoding output text
    generatedText = processor.decode(outputIDs[0], skip_special_tokens=True)

    # Extracting assistant response
    textLower = generatedText.lower()
 
    if "assistant:" in textLower:
 
        start = textLower.find("assistant:") + len("assistant:")
 
        rawOutput = generatedText[start:].strip()
 
    else:
 
        rawOutput = generatedText.replace(prompt, "").strip()

    # Cleaning up special tokens
    cleanOutput = re.sub(r'<[^>]+>', '', rawOutput).strip()

    return cleanOutput


# Running object description
print("Querying smolVLM for object description...")
description = DescribeObjects(pilIMG)

# Displaying result
print("\nsmolVLM Object Description (CoT)")
print(description)
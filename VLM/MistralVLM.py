import os
import base64

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# Convert local image to base64
with open("src/tmp/rgb.jpg", "rb") as image_file:
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

inputs = [
    {"role": "user", "content": [{"type": "text", "text": "Analyze this image:"}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}]},
]

completion_args = {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 1
}

tools = []

response = client.beta.conversations.start(
    inputs=inputs,
    model="mistral-large-latest",
    instructions="""""",
    completion_args=completion_args,
    tools=tools,
)

print(response)

print(response.outputs[0].content)
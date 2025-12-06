import os
import base64
import json
import random
from typing import Dict, List, Optional, Tuple

from abc import ABC, abstractmethod
from pathlib import Path

# Define cache directory relative to this file
# .../src/tiago_social_vlm/tiago_social_vlm/vlm_interface.py -> .../src/cache/huggingface
CACHE_DIR = Path(__file__).parents[2] / "cache" / "huggingface"


# ============================================================
# VLM Backend Base Class
# ============================================================

class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        """Send images and prompt to VLM and get a JSON response."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass


# ============================================================
# Mock Backend (For testing without GPU/API)
# ============================================================

class MockBackend(VLMBackend):
    """Mock backend for testing without GPU or API access."""
    
    def __init__(self):
        print("Mock VLM backend initialized.")
    
    @property
    def name(self) -> str:
        return "mock"
    
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        """Return a randomized mock response for testing."""
        speed = 0.3 + (0.1 * random.random())
        return {
            "risk": random.choice(["none", "low", "medium"]),
            "behavior": "proceed_cautiously",
            "waypoint": [1.5, 0.0],
            "waypoint_frame": "base_link", 
            "target_pose": [1.0, 0.0],  # 1m forward
            "speed": speed,
            "description": "Mock VLM Response: Path clear."
        }


# ============================================================
# SmolVLM Backend (Local GPU inference)
# ============================================================

class SmolVLMBackend(VLMBackend):
    """SmolVLM2 backend for local GPU inference."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self._initialized = False
        self._init_error = None
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            print(f"Loading SmolVLM2 model from {model_path}...")
            
            self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=str(CACHE_DIR))
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                cache_dir=str(CACHE_DIR),
                _attn_implementation="flash_attention_2"
            ).to("cuda")
            
            self._initialized = True
            print("SmolVLM2 backend initialized successfully.")
            
        except ImportError as e:
            self._init_error = f"Missing dependencies for SmolVLM: {e}"
            print(f"Warning: {self._init_error}")
        except Exception as e:
            self._init_error = f"Failed to initialize SmolVLM: {e}"
            print(f"Warning: {self._init_error}")
    
    @property
    def name(self) -> str:
        return "smol"
    
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        if not self._initialized:
            print(f"SmolVLM not initialized: {self._init_error}")
            return self._get_fallback_response()
        
        try:
            import torch
            
            # Encode the RGB image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            
            # Construct the navigation prompt with JSON output instruction
            full_prompt = (
                f"{prompt}\n\n"
                "Respond with a JSON object containing these fields:\n"
                "- risk: 'high', 'medium', 'low', or 'none'\n"
                "- behavior: suggested behavior (e.g. 'proceed_cautiously', 'stop', 'slow_down')\n"
                "- target_pose: [x, y] relative waypoint in meters\n"
                "- speed: speed multiplier between 0.1 and 1.0\n"
                "- description: brief explanation\n"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_data_uri},
                        {"type": "text", "text": full_prompt},
                    ]
                },
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)
            
            generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            # Try to extract JSON from the response
            return self._parse_vlm_response(generated_text)
            
        except Exception as e:
            print(f"SmolVLM inference error: {e}")
            return self._get_fallback_response()
    
    def _parse_vlm_response(self, text: str) -> Dict:
        """Try to extract JSON from VLM text response."""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # If we can't parse JSON, return a cautious default
        print(f"Could not parse JSON from SmolVLM response: {text[:200]}...")
        return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response."""
        return {
            "risk": "medium",
            "behavior": "proceed_cautiously",
            "target_pose": [0.5, 0.0],
            "speed": 0.3,
            "description": "SmolVLM fallback: Proceeding cautiously."
        }


# ============================================================
# Mistral Backend (Cloud API)
# ============================================================

class MistralBackend(VLMBackend):
    """Mistral VLM backend using cloud API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._initialized = False
        
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            self._initialized = True
            print("Mistral VLM backend initialized successfully.")
        except ImportError:
            print("Warning: mistralai library not found.")
        except Exception as e:
            print(f"Warning: Failed to initialize Mistral client: {e}")
    
    @property
    def name(self) -> str:
        return "mistral"
    
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        if not self._initialized:
            print("Mistral not initialized, returning fallback.")
            return self._get_fallback_response()
        
        try:
            return self._call_mistral(image_path, map_img_path, prompt)
        except Exception as e:
            print(f"Mistral API call failed: {e}. Returning fallback.")
            return self._get_fallback_response()
    
    def _call_mistral(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        def encode_image(path):
            with open(path, "rb") as image_file:
                return base64.standard_b64encode(image_file.read()).decode("utf-8")

        rgb_b64 = encode_image(image_path)
        map_b64 = encode_image(map_img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{rgb_b64}"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{map_b64}"}
                ]
            }
        ]

        response = self.client.beta.conversations.start(
            inputs=messages,
            model="mistral-large-latest",
            completion_args={
                "temperature": 0.4,
                "response_format": {"type": "json_object"}
            }
        )

        content = response.outputs[0].content
        try:
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Mistral: {content}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response."""
        return {
            "risk": "medium",
            "behavior": "proceed_cautiously",
            "target_pose": [0.5, 0.0],
            "speed": 0.3,
            "description": "Mistral fallback: Proceeding cautiously."
        }


# ============================================================
# VLM Client Factory
# ============================================================

class VLMClient:
    """
    Unified VLM Client that supports multiple backends.
    
    Supported backends:
    - 'mock': Static mock responses for testing (no GPU/API needed)
    - 'smol': SmolVLM2 local GPU inference
    - 'mistral': Mistral cloud API (requires API key)
    """
    
    BACKEND_MOCK = "mock"
    BACKEND_SMOL = "smol"
    BACKEND_MISTRAL = "mistral"
    
    def __init__(self, backend: str = "mock", api_key: str = None):
        """
        Initialize the VLM Client with the specified backend.
        
        :param backend: Backend to use ('mock', 'smol', or 'mistral')
        :param api_key: API key for Mistral (required if backend='mistral')
        """
        self.backend_name = backend.lower()
        self._backend: VLMBackend = None
        
        if self.backend_name == self.BACKEND_MOCK:
            self._backend = MockBackend()
        elif self.backend_name == self.BACKEND_SMOL:
            self._backend = SmolVLMBackend()
        elif self.backend_name == self.BACKEND_MISTRAL:
            if not api_key:
                print("Warning: Mistral backend requires an API key. Falling back to mock.")
                self._backend = MockBackend()
                self.backend_name = self.BACKEND_MOCK
            else:
                self._backend = MistralBackend(api_key)
        else:
            print(f"Unknown backend '{backend}', defaulting to 'mock'")
            self._backend = MockBackend()
            self.backend_name = self.BACKEND_MOCK
        
        print(f"VLM Client initialized with backend: {self.backend_name}")
    
    @property
    def active_backend(self) -> str:
        """Return the name of the active backend."""
        return self._backend.name if self._backend else "none"
    
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        """
        Send images and prompt to the active VLM backend and get a JSON response.
        
        :param image_path: Path to the current camera image (jpg).
        :param map_img_path: Path to the current map crop (jpg).
        :param prompt: Text prompt describing the task.
        :return: JSON dict with keys: risk, behavior, target_pose, speed, description.
        """
        return self._backend.get_navigation_command(image_path, map_img_path, prompt)




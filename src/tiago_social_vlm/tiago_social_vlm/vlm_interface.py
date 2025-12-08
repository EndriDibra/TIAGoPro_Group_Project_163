import os
import base64
import json
import random
from typing import Dict, List, Optional, Tuple

from abc import ABC, abstractmethod
from pathlib import Path

import shutil
import logging

# .../src/tiago_social_vlm/tiago_social_vlm/vlm_interface.py -> .../src/cache/huggingface
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "huggingface"

debug_log_path = Path(__file__).resolve().parents[2] / "tmp" / "vlm_internal.log"
debug_log_path.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("vlm_interface")
fh = logging.FileHandler(str(debug_log_path), mode='w')  # 'w' to overwrite on each run
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)

def build_navigation_prompt(heading_deg: float = None, distance_to_goal: float = None, path_blocked: bool = False) -> str:
    """Build the navigation prompt with nav metrics and prediction-based instructions."""
    
    context_info = ""
    if distance_to_goal is not None:
        context_info += f"  - Distance to goal: {distance_to_goal:.1f} meters\n"
    if heading_deg is not None:
        context_info += f"  - Heading relative to goal: {heading_deg:.0f} degrees\n"
    
    return (
        "You are a robot navigation assistant. The robot is executing a global plan (blue line on map).\n\n"
        "IMAGE 1 (Camera View): Robot's front view.\n"
        "IMAGE 2 (Map View): Top-down view.\n"
        "  - Blue arrow: Robot position/direction\n"
        "  - Blue line: The global path the robot is following\n"
        "  - Red dots/circles: Detected humans and their personal space\n"
        "  - Green star: Final destination\n"
        f"{context_info}"
        "\n"
        "Your task:\n"
        "1. OBSERVE: Describe what you see - where are humans relative to the path?\n"
        "2. PREDICT: What will happen in the next 3-5 seconds? Will humans move into/out of the path?\n"
        "3. DECIDE: Based on your prediction, choose an action.\n\n"
        "Respond ONLY with a JSON object:\n"
        '{\n'
        '  "observation": "Brief description of current scene (humans, obstacles, path status)",\n'
        '  "prediction": "What you expect to happen next (human movement, potential conflicts)",\n'
        '  "action": "Continue" | "Slow Down" | "Yield"\n'
        '}\n\n'
        'Action Definitions:\n'
        '- "Continue": Path is clear now AND predicted to stay clear. (Speed: 100%)\n'
        '- "Slow Down": Human nearby OR may enter path soon. (Speed: 50%)\n'
        '- "Yield": Human blocking path OR predicted collision. (Speed: ~0%)'
    )


# ============================================================
# VLM Backend Base Class
# ============================================================

class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def get_navigation_command(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        """Send images and prompt to VLM and get a JSON response.
        
        Args:
            image_path: Path to camera image
            map_img_path: Path to map visualization image
            heading_deg: Heading to goal in degrees (0=forward, 90=left, -90=right)
            distance_to_goal: Distance to goal in meters
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass


    def _validate_response(self, response: Dict) -> Dict:
        """Validate and normalize the VLM response."""
        result = {
            "observation": "Unable to analyze scene.",
            "prediction": "Unknown.",
            "reasoning": "Unable to analyze scene.",  # Combined for logging
            "action": "Slow Down",  # Safe default
            "speed": 0.5,
            "speed_valid": True
        }
        
        # Handle new observation/prediction fields
        if 'observation' in response and isinstance(response['observation'], str):
            result['observation'] = response['observation']
        if 'prediction' in response and isinstance(response['prediction'], str):
            result['prediction'] = response['prediction']
        
        # Combine observation + prediction into reasoning for backward compatibility
        if 'observation' in response or 'prediction' in response:
            result['reasoning'] = f"{result['observation']} Prediction: {result['prediction']}"
        elif 'reasoning' in response and isinstance(response['reasoning'], str):
            # Fallback to old reasoning field if present
            result['reasoning'] = response['reasoning']

        # Validate Action and Map to Speed
        if 'action' in response and isinstance(response['action'], str):
            action = response['action'].lower().strip()
            result['action'] = response['action'] # Keep original case for display
            
            if "continue" in action:
                result['speed'] = 1.0
                result['action'] = "Continue"
            elif "slow" in action:
                result['speed'] = 0.5
                result['action'] = "Slow Down"
            elif "yield" in action or "stop" in action:
                result['speed'] = 0.01  # Use 0.01 instead of 0.0 - Nav2 ignores 0%
                result['action'] = "Yield"
            else:
                logger.warning(f"Unknown action: {action}, defaulting to Slow Down")
                result['speed'] = 0.5
        
        return result



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
    
    def get_navigation_command(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        """Return a randomized mock response for testing (supervisor mode compatible)."""
        # Random action choice weighted towards Continue for testing
        action_choice = random.random()
        if action_choice < 0.4:
            action = "Continue"
            speed = 1.0
        elif action_choice < 0.7:
            action = "Slow Down"
            speed = 0.5
        else:
            action = "Yield"
            speed = 0.01
            
        return {
            "reasoning": f"Mock: Path appears clear. Distance to goal: {distance_to_goal:.1f}m." if distance_to_goal else "Mock: Analyzing scene.",
            "action": action,
            "speed": speed,
            "speed_valid": True
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
            
            model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            print(f"Loading SmolVLM2-2.2B model from {model_path}...")
            
            self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=str(CACHE_DIR))
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                cache_dir=str(CACHE_DIR)
            ).to("cuda")
            
            self._initialized = True
            print(f"SmolVLM2 backend initialized successfully. Cache: {CACHE_DIR}")
            
        except ImportError as e:
            self._init_error = f"Missing dependencies for SmolVLM: {e}"
            print(f"Warning: {self._init_error}")
            logger.error(self._init_error)
        except Exception as e:
            self._init_error = f"Failed to initialize SmolVLM: {e}"
            print(f"Warning: {self._init_error}")
            logger.error(self._init_error, exc_info=True)
            
    @property
    def name(self) -> str:
        return "smol"
    
    def get_navigation_command(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        if not self._initialized:
            print(f"SmolVLM not initialized: {self._init_error}")
            logger.error(f"SmolVLM not initialized: {self._init_error}")
            return self._get_fallback_response()
        
        try:
            import torch
            
            # Encode the RGB image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            
            # Build prompt dynamically with heading and goal info
            full_prompt = build_navigation_prompt(heading_deg, distance_to_goal)
            
            
            # Encode the map image
            map_data_uri = None
            if map_img_path and os.path.exists(map_img_path):
                with open(map_img_path, "rb") as f:
                    map_bytes = f.read()
                map_base64 = base64.b64encode(map_bytes).decode("utf-8")
                map_data_uri = f"data:image/jpeg;base64,{map_base64}"
            
            content_list = [{"type": "image", "url": image_data_uri}]
            if map_data_uri:
                content_list.append({"type": "image", "url": map_data_uri})
            content_list.append({"type": "text", "text": full_prompt})

            messages = [
                {
                    "role": "user",
                    "content": content_list
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
            
            logger.info(f"Raw SmolVLM Response: {generated_text}")
            
            # Try to extract JSON from the response
            return self._parse_vlm_response(generated_text)
            
        except Exception as e:
            print(f"SmolVLM inference error: {e}")
            return self._get_fallback_response()
    
    def _parse_vlm_response(self, text: str) -> Dict:
        """Try to extract JSON from VLM text response with validation."""
        try:
            # Find the last JSON object in the response (VLM output is typically at the end)
            # Look for "Assistant:" marker and extract text after it
            if "Assistant:" in text:
                text = text.split("Assistant:")[-1]
            
            # Find JSON by matching braces properly (handles nested arrays)
            json_str = self._extract_json(text)
            if json_str:
                parsed = json.loads(json_str)
                return self._validate_response(parsed)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
        except Exception as e:
            logger.warning(f"Parse error: {e}")
        
        # If we can't parse JSON, return a cautious default
        print(f"Could not parse JSON from SmolVLM response: {text[:200]}...")
        logger.warning(f"Failed to parse VLM response: {text[:500]}")
        return self._get_fallback_response()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text using bracket matching."""
        # Find the first '{' character
        start = text.find('{')
        if start == -1:
            return None
        
        # Match braces to find the complete JSON object
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None
    

    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response (supervisor mode compatible)."""
        return {
            "reasoning": "SmolVLM fallback: Unable to analyze scene. Proceeding with caution.",
            "action": "Slow Down",
            "speed": 0.5,
            "speed_valid": True
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
    
    def get_navigation_command(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        if not self._initialized:
            print("Mistral not initialized, returning fallback.")
            return self._get_fallback_response()
        
        try:
            return self._call_mistral(image_path, map_img_path, heading_deg, distance_to_goal)
        except Exception as e:
            print(f"Mistral API call failed: {e}. Returning fallback.")
            logger.error(f"Mistral API call failed: {e}", exc_info=True)
            return self._get_fallback_response()
    
    def _call_mistral(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        
        def encode_image(path):
            with open(path, "rb") as image_file:
                return base64.standard_b64encode(image_file.read()).decode("utf-8")

        rgb_b64 = encode_image(image_path)
        map_b64 = encode_image(map_img_path)

        # Build prompt dynamically with heading and goal info
        full_prompt = build_navigation_prompt(heading_deg, distance_to_goal)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
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
        logger.info(f"Raw Mistral Response: {content}")
        try:
            clean_content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_content)
            # Validate response using base class method
            return self._validate_response(parsed)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Mistral: {content}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response (supervisor mode compatible)."""
        return {
            "reasoning": "Mistral fallback: Unable to analyze scene. Proceeding with caution.",
            "action": "Slow Down",
            "speed": 0.5,
            "speed_valid": True
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
    
    def get_navigation_command(self, image_path: str, map_img_path: str, heading_deg: float = None, distance_to_goal: float = None) -> Dict:
        """
        Send images and prompt to the active VLM backend and get a JSON response.
        
        :param image_path: Path to the current camera image (jpg).
        :param map_img_path: Path to the current map crop (jpg).
        :param heading_deg: Heading to goal in degrees (0=forward, 90=left, -90=right).
        :param distance_to_goal: Distance to goal in meters.
        :return: JSON dict with goal, speed, and validation flags.
        """
        return self._backend.get_navigation_command(image_path, map_img_path, heading_deg, distance_to_goal)

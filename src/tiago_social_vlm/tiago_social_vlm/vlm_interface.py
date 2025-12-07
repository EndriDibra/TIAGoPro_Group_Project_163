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
fh = logging.FileHandler(str(debug_log_path))
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(fh)
logger.setLevel(logging.INFO)

def build_navigation_prompt(heading_deg: float = None) -> str:
    """Build the navigation prompt, optionally including heading to goal."""
    heading_info = ""
    if heading_deg is not None:
        heading_info = f"  - Heading to goal: {heading_deg:.0f} degrees (0=forward, 90=left, -90=right)\n"
    
    return (
        "You are a robot navigation assistant helping a mobile robot navigate safely around humans.\n\n"
        "IMAGE 1 (Camera View): Shows what the robot currently sees from its front camera.\n"
        "IMAGE 2 (Map View): Shows a top-down map with:\n"
        "  - Blue dot/arrow: Robot's current position and facing direction\n"
        "  - Red dots + circles: Humans with proxemics zones (avoid entering these)\n"
        "  - Green star: Goal position\n"
        "  - Gray areas: Walls and obstacles\n"
        f"{heading_info}"
        "  - Grid: 1 meter spacing\n\n"
        "Your task: Suggest a safe intermediate waypoint to help the robot reach the goal while avoiding humans.\n\n"
        "Respond ONLY with a JSON object (no other text):\n"
        '{"scene_description": "what you see", "risk_description": "high/medium/low/none + reason", '
        '"action_description": "what to do", "goal": [x, y], "speed": 0.1-1.0}\n\n'
        "IMPORTANT: goal is [x,y] in meters relative to robot. Positive x=forward, positive y=left.\n"
        "Example: [1.5, 0.3] means 1.5m forward and 0.3m left."
    )

# Default prompt for backwards compatibility
NAVIGATION_PROMPT = build_navigation_prompt()



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

    def _save_debug_images(self, image_path: str, map_img_path: str):
        """Save debug images to tmp directory."""
        try:
            dest_dir = Path(__file__).resolve().parents[2] / "tmp"
            dest_dir.mkdir(exist_ok=True)
            
            if image_path and os.path.exists(image_path):
                shutil.copy(image_path, dest_dir / "debug_rgb.jpg")
                
            if map_img_path and os.path.exists(map_img_path):
                shutil.copy(map_img_path, dest_dir / "debug_map.jpg")
        except Exception as e:
            logger.error(f"Failed to save debug images: {e}")


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
            "scene_description": "Mock: Indoor corridor with clear path ahead.",
            "risk_description": random.choice(["none", "low: Person visible in distance", "medium: Person approaching"]),
            "action_description": "Proceed cautiously along the corridor.",
            "goal": [1.0, 0.0],
            "speed": speed,
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
    
    def get_navigation_command(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
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
            
            # Use centralized prompt
            full_prompt = NAVIGATION_PROMPT
            
            # Save debug images
            self._save_debug_images(image_path, map_img_path)
            
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
    
    def _validate_response(self, response: Dict) -> Dict:
        """Validate and normalize the VLM response."""
        result = self._get_fallback_response().copy()
        
        # Copy over valid fields
        if 'scene_description' in response and isinstance(response['scene_description'], str):
            result['scene_description'] = response['scene_description']
        if 'risk_description' in response and isinstance(response['risk_description'], str):
            result['risk_description'] = response['risk_description']
        if 'action_description' in response and isinstance(response['action_description'], str):
            result['action_description'] = response['action_description']
        
        # Validate goal - must be [x, y] with reasonable values
        if 'goal' in response:
            goal = response['goal']
            if isinstance(goal, list) and len(goal) >= 2:
                try:
                    x, y = float(goal[0]), float(goal[1])
                    # Sanity check: waypoints should be within reasonable range
                    # Note: [0, 0] is valid - it means "stop/stay in place"
                    if -5.0 <= x <= 5.0 and -5.0 <= y <= 5.0:
                        result['goal'] = [x, y]
                        result['goal_valid'] = True
                    else:
                        logger.warning(f"Goal out of range: [{x}, {y}]")
                        result['goal_valid'] = False
                except (ValueError, TypeError):
                    result['goal_valid'] = False
            else:
                result['goal_valid'] = False
        else:
            result['goal_valid'] = False
        
        # Validate speed
        if 'speed' in response:
            try:
                speed = float(response['speed'])
                if 0.1 <= speed <= 1.0:
                    result['speed'] = speed
                    result['speed_valid'] = True
                else:
                    result['speed_valid'] = False
            except (ValueError, TypeError):
                result['speed_valid'] = False
        else:
            result['speed_valid'] = False
        
        return result
    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response."""
        return {
            "scene_description": "SmolVLM fallback: Unable to analyze scene.",
            "risk_description": "medium: Proceeding with caution due to analysis failure.",
            "action_description": "Move forward slowly while sensors recover.",
            "goal": [0.5, 0.0],
            "speed": 0.3,
            "goal_valid": True,
            "speed_valid": True,
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
            logger.error(f"Mistral API call failed: {e}", exc_info=True)
            return self._get_fallback_response()
    
    def _call_mistral(self, image_path: str, map_img_path: str, prompt: str) -> Dict:
        self._save_debug_images(image_path, map_img_path)
        
        def encode_image(path):
            with open(path, "rb") as image_file:
                return base64.standard_b64encode(image_file.read()).decode("utf-8")

        rgb_b64 = encode_image(image_path)
        map_b64 = encode_image(map_img_path)

        # Use centralized prompt
        full_prompt = NAVIGATION_PROMPT

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
            return json.loads(clean_content)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Mistral: {content}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict:
        """Return a safe fallback response."""
        return {
            "scene_description": "Mistral fallback: Unable to analyze scene.",
            "risk_description": "medium: Proceeding with caution due to API failure.",
            "action_description": "Move forward slowly while connection recovers.",
            "goal": [0.5, 0.0],
            "speed": 0.3,
            "goal_valid": True,
            "speed_valid": True,
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




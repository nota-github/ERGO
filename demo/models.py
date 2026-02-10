"""
Model loading and initialization for ERGO Demo.
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .config import ERGO_MODEL_ID, ERGO_MAX_PIXELS


class ModelManager:
    """Manages loading and access to the ERGO model."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelManager._initialized:
            return
        
        self.ergo_model = None
        self.ergo_processor = None
        
        ModelManager._initialized = True
    
    def load_ergo(self, device: str = "cuda:2"):
        """Load ERGO model and processor."""
        if self.ergo_model is not None:
            return
        
        print("Loading ERGO model...")
        self.ergo_processor = AutoProcessor.from_pretrained(
            ERGO_MODEL_ID,
            max_pixels=ERGO_MAX_PIXELS
        )
        self.ergo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ERGO_MODEL_ID,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        ).to(device).eval()
        print(f"ERGO model loaded on {device}")
    
    def load_all(self, ergo_device: str = "cuda:2"):
        """Load all required models."""
        self.load_ergo(ergo_device)


# Global model manager instance
model_manager = ModelManager()


def get_ergo_model():
    """Get ERGO model (loads if not already loaded)."""
    if model_manager.ergo_model is None:
        model_manager.load_ergo()
    return model_manager.ergo_model


def get_ergo_processor():
    """Get ERGO processor (loads if not already loaded)."""
    if model_manager.ergo_processor is None:
        model_manager.load_ergo()
    return model_manager.ergo_processor

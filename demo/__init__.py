"""
ERGO Demo Package

A Gradio-based demo for ERGO - Two-Stage Reasoning driven Perception for Vision-Language Models.
"""

from .config import ERGO_MODEL_ID
from .models import model_manager
from .ui import EXAMPLES
from .app import create_demo, main

__all__ = [
    "ERGO_MODEL_ID",
    "model_manager",
    "EXAMPLES",
    "create_demo",
    "main",
]

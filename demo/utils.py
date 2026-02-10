"""
Utility functions for image processing and parsing.
"""

import math
import os
import json
import re
import random
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from .config import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
from pathlib import Path

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def round_by_factor(number: int, factor: int) -> int:
    """Round number to nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Ceiling of number to nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 256,
    max_pixels: int = 1024,
) -> Tuple[int, int]:
    """
    Resize dimensions while respecting factor alignment and pixel constraints.
    
    Args:
        height: Original height
        width: Original width
        factor: Factor to align dimensions to
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
    
    Returns:
        Tuple of (new_height, new_width)
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = round_by_factor(height / beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def map_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    width_scale: float,
    height_scale: float,
    ori_width: float,
    ori_height: float,
) -> Tuple[float, float, float, float]:
    """
    Map bounding box coordinates with scaling.
    
    Args:
        left, top, right, bottom: Original box coordinates
        width_scale, height_scale: Scaling factors
        ori_width, ori_height: Original image dimensions
    
    Returns:
        Tuple of mapped (left, top, right, bottom)
    """
    left = max(left * width_scale, 0)
    top = max(top * height_scale, 0)
    right = min(right * width_scale, ori_width)
    bottom = min(bottom * height_scale, ori_height)
    return left, top, right, bottom


def parse_zoom_bbox_from_text(content: str) -> Optional[List[int]]:
    """
    Parse zoom bounding box from model output text.
    
    Args:
        content: Model output text containing <zoom>[x1, y1, x2, y2]</zoom>
    
    Returns:
        List of [x1, y1, x2, y2] or None if not found
    """
    try:
        answer_match = re.search(r"<zoom>(.*?)</zoom>", content, re.DOTALL)
        if not answer_match:
            return None
        
        answer_content = answer_match.group(1)
        
        # Check for simple bbox format
        if re.match(r"^\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*$", answer_content):
            return [int(num.strip()) for num in answer_content.strip().strip("[]").split(",")]
        
        # Look for bbox pattern in content
        bbox_patterns = re.findall(r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]", answer_content)
        if len(bbox_patterns) == 1:
            bbox_str = bbox_patterns[0]
            return [int(num.strip()) for num in bbox_str.strip("[]").split(",")]
        
        return None
    except Exception:
        return None


def crop_with_zoom(
    image: Image.Image,
    zoom_response: str,
    width_scale: float,
    height_scale: float,
    ori_width: int,
    ori_height: int,
    factor: int = 28,
    min_pixels: int = 256,
    max_pixels: int = 1024,
) -> Tuple[Image.Image, List[int], Tuple[int, int]]:
    """
    Crop and resize image based on zoom response.
    
    Args:
        image: Original PIL Image
        zoom_response: Model response containing zoom coordinates
        width_scale, height_scale: Scaling factors
        ori_width, ori_height: Original dimensions
        factor, min_pixels, max_pixels: Resize parameters
    
    Returns:
        Tuple of (cropped_image, actual_bbox, (new_width, new_height))
    """
    w, h = image.size
    bbox = parse_zoom_bbox_from_text(zoom_response)
    
    if bbox is None or len(bbox) != 4:
        return image, [0, 0, w, h], (w, h)

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map_box(x1, y1, x2, y2, width_scale, height_scale, ori_width, ori_height)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    new_w, new_h = x2 - x1, y2 - y1
    if new_w <= 0 or new_h <= 0:
        return image, [0, 0, w, h], (w, h)
    
    new_w, new_h = smart_resize(new_w, new_h, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    cropped_image = image.crop((x1, y1, x2, y2)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return cropped_image, [x1, y1, x2, y2], (new_w, new_h)


def preprocess_image(
    img: Image.Image,
    max_pixels: int = MAX_PIXELS
) -> Tuple[Image.Image, Image.Image, int, int, float, float]:
    """
    Preprocess image for model input.
    
    Args:
        img: Input PIL Image
        max_pixels: Maximum pixels for resized image
    
    Returns:
        Tuple of (original_img, resized_img, width, height, width_scale, height_scale)
    """
    width, height = img.size
    resize_h, resize_w = smart_resize(
        int(height), int(width),
        max_pixels=max_pixels,
        min_pixels=MIN_PIXELS,
        factor=IMAGE_FACTOR
    )
    width_scale = width / resize_w
    height_scale = height / resize_h
    resized_img = img.resize((resize_w, resize_h), Image.BICUBIC)
    
    return img, resized_img, width, height, width_scale, height_scale

def load_examples(example_dir="/workspace/ERGO/Vstar_Bench") -> List[Dict[str, Any]]:
    """
    Load examples from a directory.
    
    Args:
        example_dir: Directory containing example images and questions
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    val_json = json.load(open(Path(example_dir)/"new_val.json"))
    cnt = 0
    for idx, item in enumerate(val_json):
        img_name = item["images"][0]
        # Skip any examples with an image in the banned list
        examples.append({
            "id": cnt,
            "image": str(Path(example_dir).parent / img_name),
            "question": item["problem"].split("<image>\n")[-1].split("Answer with the option's letter from the given choices directly.")[0],
            "title": f"{item['doc_id']} {img_name}",
        })
        cnt += 1
    print("Loaded", len(examples), "examples")

    return examples
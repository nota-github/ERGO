import base64
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 256,
    max_pixels: int = 1024,
) -> Tuple[int, int]:
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


def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
    left = max(left * width_scale, 0)
    top = max(top * height_scale, 0)
    right = min(right * width_scale, ori_width)
    bottom = min(bottom * height_scale, ori_height)
    return left, top, right, bottom


def parse_zoom_bbox_from_text(content: str) -> Optional[List[int]]:
    try:
        answer_match = re.search(r"<zoom>(.*?)</zoom>", content, re.DOTALL)
        if not answer_match:
            return None
        answer_content = answer_match.group(1)
        if re.match(r"^\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*$", answer_content):
            return [int(num.strip()) for num in answer_content.strip().strip("[]").split(",")]
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
    w, h = image.size
    bbox = parse_zoom_bbox_from_text(zoom_response)
    if bbox is None or len(bbox) != 4:
        return image, [0, 0, w, h], (w, h)

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map_box(x1, y1, x2, y2, width_scale, height_scale, ori_width, ori_height)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    new_w, new_h = x2 - x1, y2 - y1
    new_w, new_h = smart_resize(new_w, new_h, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    cropped_image = image.crop((x1, y1, x2, y2)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return cropped_image, [x1, y1, x2, y2], (new_w, new_h)


def extract_answer_text(model_response: str) -> Optional[str]:
    try:
        return model_response.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception:
        return None


def check_image_path(image: str) -> bool:
    if isinstance(image, str):
        try:
            if Path(image).exists():
                return True
            else:
                return False
        except Exception:
            return False
    else:
        return False


def preprocess_data(data: Dict[str, Any], dataset: str, **kwargs) -> Dict[str, Any]:
    if dataset == "vstar":
        texts = data["text"].split("\n")
        question = texts[0]
        options = texts[1:-1]
        postfix = texts[-1]
        data["question"] = question
        data["options"] = options
        data["postfix"] = postfix
    elif dataset == "hrbench":
        item = kwargs.get("item", None)
        item = item.to_dict()
        test_types = kwargs.get("test_types", None)
        id = kwargs.get("id", None)
        i = kwargs.get("i", None)
        data["image"] = item["image"]
        data["question"] = item["question"]
        data["options"] = [
            f"A:{item['A']}",
            f"B:{item['B']}",
            f"C:{item['C']}",
            f"D:{item['D']}",
        ]
        data["postfix"] = " Answer with the option's letter from the given choices directly."
        data["question_id"] = id
        data["label"] = item["answer"]
        data["category"] = test_types[i]
    elif dataset == "mmerwl":
        item = kwargs.get("item", None)
        i = kwargs.get("i", None)
        data["image"] = item["bytes"]
        data["question"] = item["question"]
        data["options"] = item["multi-choice options"]
        data["postfix"] = " Answer with the option's letter from the given choices directly."
        data["question_id"] = i
        data["label"] = item["answer"]
        data["category"] = item["category"]
    return data

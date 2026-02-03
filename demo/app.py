"""
ERGO Demo: Two-column comparison between ERGO and Qwen2.5-VL-7B-Instruct
"""

import gradio as gr
import torch
import random
import numpy as np
from PIL import Image
from threading import Thread, Lock
from queue import Queue, Empty
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer
from transformers.image_utils import load_image
from typing import List, Tuple, Optional
import re
import math
import os
import time

# ============== Constants ==============
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
ERGO_MAX_PIXELS = 1280 * 28 * 28
QWEN_MAX_PIXELS = 16384 * 28 * 28
MAX_NEW_TOKENS = 1024
DEFAULT_SEED = 42

cur_dir = os.path.dirname(os.path.abspath(__file__))

ERGO_MODEL_ID = "nota-ai/ERGO-7B"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. First, identify the coordinates "
    "of the key image area relevant to solving the problem. Append the coordinates in [x1, y1, x2, y2] "
    "format at the end of your response and stop. This will trigger cropping of the corresponding area in "
    "the original image and enlarge it for improved clarity. Once the enlarged image is available, provide the final answer."
)

QUESTION_TEMPLATE1 = "{Question} First, consider how to zoom in on the image to include only the region containing all the information necessary to answer the question. Output the thinking process in <think> </think> tags and then output the region's bounding box coordinates in the following format <zoom>[x1, y1, x2, y2]</zoom> tags."

QUESTION_TEMPLATE2 = "{Question}\nCarefully analyze both the original image and the enlarged sub-image to solve the question step by step. Present your reasoning clearly the following format strictly <think> ... </think>, and provide the final answer enclosed within <answer> ... </answer>"

QWEN_SYSTEM_PROMPT = "You are a helpful assistant."

# ============== Utility Functions ==============
def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    if new_w <= 0 or new_h <= 0:
        return image, [0, 0, w, h], (w, h)
    new_w, new_h = smart_resize(new_w, new_h, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    cropped_image = image.crop((x1, y1, x2, y2)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return cropped_image, [x1, y1, x2, y2], (new_w, new_h)


def preprocess_image(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Tuple[Image.Image, Image.Image, int, int, float, float]:
    width, height = img.size
    resize_h, resize_w = smart_resize(
        int(height), int(width), max_pixels=max_pixels, min_pixels=MIN_PIXELS, factor=IMAGE_FACTOR
    )
    width_scale = width / resize_w
    height_scale = height / resize_h
    resized_img = img.resize((resize_w, resize_h), Image.BICUBIC)
    return img, resized_img, width, height, width_scale, height_scale


# ============== Model Loading ==============
print("Loading ERGO model...")
ergo_processor = AutoProcessor.from_pretrained(ERGO_MODEL_ID, max_pixels=ERGO_MAX_PIXELS)
ergo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ERGO_MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_cache=True,
).to("cuda:2").eval()
print("ERGO model loaded on cuda:0")

print("Loading Qwen2.5-VL model...")
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, max_pixels=QWEN_MAX_PIXELS)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_cache=True,
).to("cuda:3").eval()
print("Qwen2.5-VL model loaded on cuda:1")


# ============== Streaming State with Queue ==============
class StreamingState:
    """State for streaming output with queue-based communication."""
    def __init__(self, output_queue: Queue):
        self.output_queue = output_queue  # Queue for immediate output streaming
        self.output = []  # Accumulated output
        self.done = False
        self.start_time = None
        self.end_time = None
        self.first_token_time = None
        self.token_count = 0
        self.lock = Lock()
    
    def put_update(self, update_type: str, data=None):
        """Put an update into the queue immediately."""
        self.output_queue.put({
            "type": update_type,
            "data": data,
            "time": time.time(),
            "first_token_time": self.first_token_time,
            "token_count": self.token_count,
            "done": self.done,
        })
    
    def get_ttft(self):
        """Get Time to First Token in seconds."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None
    
    def get_tpot(self):
        """Get Time Per Output Token in milliseconds."""
        if self.token_count > 1 and self.first_token_time and self.end_time:
            # Time for tokens after the first one
            generation_time = self.end_time - self.first_token_time
            return (generation_time / (self.token_count - 1)) * 1000  # Convert to ms
        return None


# ============== ERGO Inference ==============
def ergo_inference(image: Image.Image, question: str, state: StreamingState):
    """ERGO two-stage inference with streaming via queue."""
    set_random_seed(DEFAULT_SEED)
    
    state.start_time = time.time()
    state.put_update("start")
    
    # Preprocess
    orig_img, resized_img, width, height, width_scale, height_scale = preprocess_image(image, max_pixels=ERGO_MAX_PIXELS)
    all_images = [resized_img]
    
    # Stage 1
    messages_stage1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE1.format(Question=question)},
        ]},
    ]
    
    prompt = ergo_processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
    inputs = ergo_processor(text=[prompt], images=all_images, return_tensors="pt", padding=True).to(ergo_model.device)
    
    streamer = TextIteratorStreamer(ergo_processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.95, top_k=50)
    
    thread = Thread(target=ergo_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    stage1_output = ""
    zoom_detected = False
    
    for new_text in streamer:
        stage1_output += new_text
        # Track first token time
        if state.first_token_time is None:
            state.first_token_time = time.time()
            state.put_update("first_token")  # Signal first token immediately!
        
        # Count tokens
        state.token_count += len(new_text.split()) if new_text.strip() else 0
        
        if "<zoom>" in stage1_output and not zoom_detected:
            zoom_detected = True
            tmp = stage1_output.split("<zoom>")[0]
            state.output = [tmp + "\n\n**🔍 Planning Visual Operation...**\n\n"]
        elif not zoom_detected:
            state.output = [stage1_output]
        
        state.put_update("token", new_text)
    
    thread.join()
    state.output = [stage1_output + "\n\n"]
    
    # Check for zoom
    bbox = parse_zoom_bbox_from_text(stage1_output)
    
    if bbox is not None:
        state.output.append(f"**📸 Executing Visual Operation...** @crop_image(bbox={bbox})\n\n")
        state.put_update("zoom_start")
        
        cropped_img, actual_bbox, smart_wh = crop_with_zoom(
            orig_img, stage1_output, width_scale, height_scale, width, height,
            factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        
        state.output.append(f"\n**🔎 Analyzing Zoomed Region...** (size: {smart_wh[0]}x{smart_wh[1]})\n\n")
        state.put_update("zoom_image")
        
        # Stage 2
        all_images.append(cropped_img)
        messages_stage2 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION_TEMPLATE1.format(Question=question)}]},
            {"role": "assistant", "content": stage1_output},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION_TEMPLATE2.format(Question=question)}]},
        ]
        
        prompt2 = ergo_processor.apply_chat_template(messages_stage2, tokenize=False, add_generation_prompt=True)
        inputs2 = ergo_processor(text=[prompt2], images=all_images, return_tensors="pt", padding=True).to(ergo_model.device)
        
        streamer2 = TextIteratorStreamer(ergo_processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs2 = dict(**inputs2, streamer=streamer2, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.95, top_k=50)
        
        thread2 = Thread(target=ergo_model.generate, kwargs=generation_kwargs2)
        thread2.start()
        
        stage2_output = ""
        for new_text in streamer2:
            stage2_output += new_text
            state.token_count += len(new_text.split()) if new_text.strip() else 0
            state.output = state.output[:4] + [stage2_output]
            state.put_update("token", new_text)
        
        thread2.join()
        state.output = state.output[:4] + [stage2_output + "\n\n"]
    
    state.end_time = time.time()
    state.done = True
    state.put_update("done")


# ============== Qwen Inference ==============
def qwen_inference(image: Image.Image, question: str, state: StreamingState):
    """Qwen2.5-VL inference with streaming via queue."""
    set_random_seed(DEFAULT_SEED)
    
    state.start_time = time.time()
    state.put_update("start")
    
    # Preprocess image same way
    _, resized_img, _, _, _, _ = preprocess_image(image, max_pixels=QWEN_MAX_PIXELS)
    
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]
    
    prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[prompt], images=[resized_img], return_tensors="pt", padding=True).to(qwen_model.device)
    
    streamer = TextIteratorStreamer(qwen_processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, top_p=0.95, top_k=50)
    
    thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    output = ""
    for new_text in streamer:
        output += new_text
        # Track first token time
        if state.first_token_time is None:
            state.first_token_time = time.time()
            state.put_update("first_token")  # Signal first token immediately!
        
        # Count tokens
        state.token_count += len(new_text.split()) if new_text.strip() else 0
        state.output = [output]
        state.put_update("token", new_text)
    
    thread.join()
    
    state.end_time = time.time()
    state.done = True
    state.put_update("done")


# ============== Metrics Formatting ==============
def build_metrics_header(total_time: float, is_done: bool, ttft: float = None, tpot: float = None, tokens: int = 0, is_first: bool = False) -> str:
    """Build a formatted metrics header with TTFT and TPOT."""
    status = '✅' if is_done else '🔄'
    
    header = f"⏱️ **Total: {total_time:.1f}s** {status}\n\n"
    
    # First responder indicator
    if is_first and ttft is not None:
        header += "🚀 **FIRST TO RESPOND!**\n\n"
    
    # Add metrics table
    metrics_parts = []
    
    if ttft is not None:
        metrics_parts.append(f"**TTFT:** {ttft*1000:.0f}ms")
    
    if tpot is not None:
        metrics_parts.append(f"**TPOT:** {tpot:.1f}ms/token")
    
    if tokens > 0:
        metrics_parts.append(f"**Tokens:** ~{tokens}")
    
    if metrics_parts:
        header += " | ".join(metrics_parts) + "\n\n---\n\n"
    
    return header



# ============== Combined Inference ==============
def run_comparison(image, question):
    """Run both models in parallel with queue-based streaming."""
    if image is None or not question.strip():
        yield [], [], []
        return
    
    # Load and prepare image
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image
    if hasattr(img, 'convert'):
        img = img.convert("RGB")
    
    # Save input image for display in chat
    input_img_path = os.path.join(cur_dir, "tmp_input.png")
    img.save(input_img_path)
    
    # Create user message with image and question
    user_input = [
        {"role": "user", "content": {"path": input_img_path}},
        {"role": "user", "content": question},
    ]
    
    # Create separate queues for each model
    ergo_queue = Queue()
    qwen_queue = Queue()
    
    # Initialize states with queues
    ergo_state = StreamingState(ergo_queue)
    qwen_state = StreamingState(qwen_queue)
    
    # Start both threads
    ergo_thread = Thread(target=ergo_inference, args=(img.copy(), question, ergo_state))
    qwen_thread = Thread(target=qwen_inference, args=(img.copy(), question, qwen_state))
    
    ergo_thread.start()
    qwen_thread.start()
    
    # Track which model responded first
    ergo_first_token_received = False
    qwen_first_token_received = False
    first_responder = None
    
    # Non-blocking queue polling - yield as soon as any queue has data
    while not (ergo_state.done and qwen_state.done):
        updated = False
        
        # Poll ERGO queue (non-blocking)
        try:
            while True:
                msg = ergo_queue.get_nowait()
                updated = True
                if msg["type"] == "first_token" and not ergo_first_token_received:
                    ergo_first_token_received = True
                    if first_responder is None:
                        first_responder = "ERGO"
        except Empty:
            pass
        
        # Poll Qwen queue (non-blocking)
        try:
            while True:
                msg = qwen_queue.get_nowait()
                updated = True
                if msg["type"] == "first_token" and not qwen_first_token_received:
                    qwen_first_token_received = True
                    if first_responder is None:
                        first_responder = "Qwen"
        except Empty:
            pass
        
        # Build current state
        ergo_output = list(ergo_state.output)
        ergo_time = (ergo_state.end_time or time.time()) - ergo_state.start_time if ergo_state.start_time else 0
        ergo_done = ergo_state.done
        ergo_ttft = ergo_state.get_ttft()
        ergo_tpot = ergo_state.get_tpot() if ergo_done else None
        ergo_tokens = ergo_state.token_count
        
        qwen_output = list(qwen_state.output)
        qwen_time = (qwen_state.end_time or time.time()) - qwen_state.start_time if qwen_state.start_time else 0
        qwen_done = qwen_state.done
        qwen_ttft = qwen_state.get_ttft()
        qwen_tpot = qwen_state.get_tpot() if qwen_done else None
        qwen_tokens = qwen_state.token_count
        
        # Build timing headers with first responder indicator
        ergo_header = build_metrics_header(
            ergo_time, ergo_done, ergo_ttft, ergo_tpot, ergo_tokens,
            is_first=(first_responder == "ERGO")
        )
        qwen_header = build_metrics_header(
            qwen_time, qwen_done, qwen_ttft, qwen_tpot, qwen_tokens,
            is_first=(first_responder == "Qwen")
        )
        
        yield user_input, [ergo_header] + ergo_output, [qwen_header] + qwen_output
        
        # Very short sleep only if no updates (to avoid busy waiting)
        if not updated:
            time.sleep(0.01)  # 10ms polling when idle
    
    ergo_thread.join()
    qwen_thread.join()
    
    # Final output with comparison
    ergo_output = list(ergo_state.output)
    ergo_time = ergo_state.end_time - ergo_state.start_time if ergo_state.start_time else 0
    ergo_ttft = ergo_state.get_ttft()
    ergo_tpot = ergo_state.get_tpot()
    ergo_tokens = ergo_state.token_count
    
    qwen_output = list(qwen_state.output)
    qwen_time = qwen_state.end_time - qwen_state.start_time if qwen_state.start_time else 0
    qwen_ttft = qwen_state.get_ttft()
    qwen_tpot = qwen_state.get_tpot()
    qwen_tokens = qwen_state.token_count
    
    # Build final comparison
    comparison = build_final_comparison(
        ergo_ttft, ergo_tpot, ergo_time, ergo_tokens,
        qwen_ttft, qwen_tpot, qwen_time, qwen_tokens,
        first_responder
    )
    
    ergo_header = build_metrics_header(
        ergo_time, True, ergo_ttft, ergo_tpot, ergo_tokens,
        is_first=(first_responder == "ERGO")
    )
    qwen_header = build_metrics_header(
        qwen_time, True, qwen_ttft, qwen_tpot, qwen_tokens,
        is_first=(first_responder == "Qwen")
    )
    
    # Append comparison to both outputs
    ergo_final = ergo_output + [f"\n\n{comparison}"]
    qwen_final = qwen_output + [f"\n\n{comparison}"]
    
    yield user_input, [ergo_header] + ergo_final, [qwen_header] + qwen_final


# ============== Single Model Run Functions ==============
def run_ergo_only(image, question):
    """Run ERGO model only with queue-based streaming."""
    if image is None or not question.strip():
        yield []
        return
    
    # Load and prepare image
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image
    if hasattr(img, 'convert'):
        img = img.convert("RGB")
    
    # Save input image
    input_img_path = os.path.join(cur_dir, "tmp_input_ergo.png")
    img.save(input_img_path)
    
    user_input = [
        {"role": "user", "content": {"path": input_img_path}},
        {"role": "user", "content": question},
    ]
    
    # Create queue and state
    ergo_queue = Queue()
    ergo_state = StreamingState(ergo_queue)
    
    # Start inference thread
    ergo_thread = Thread(target=ergo_inference, args=(img.copy(), question, ergo_state))
    ergo_thread.start()
    
    # Stream updates
    while not ergo_state.done:
        # Drain queue (non-blocking)
        try:
            while True:
                ergo_queue.get_nowait()
        except Empty:
            pass
        
        ergo_output = list(ergo_state.output)
        ergo_time = (ergo_state.end_time or time.time()) - ergo_state.start_time if ergo_state.start_time else 0
        ergo_ttft = ergo_state.get_ttft()
        ergo_tpot = ergo_state.get_tpot() if ergo_state.done else None
        ergo_tokens = ergo_state.token_count
        
        header = build_metrics_header(ergo_time, ergo_state.done, ergo_ttft, ergo_tpot, ergo_tokens)
        yield format_for_chatbot(user_input + [header] + ergo_output)
        time.sleep(0.05)
    
    ergo_thread.join()
    
    # Final output
    ergo_output = list(ergo_state.output)
    ergo_time = ergo_state.end_time - ergo_state.start_time if ergo_state.start_time else 0
    ergo_ttft = ergo_state.get_ttft()
    ergo_tpot = ergo_state.get_tpot()
    ergo_tokens = ergo_state.token_count
    
    header = build_metrics_header(ergo_time, True, ergo_ttft, ergo_tpot, ergo_tokens)
    yield format_for_chatbot(user_input + [header] + ergo_output)


def run_qwen_only(image, question):
    """Run Qwen model only with queue-based streaming."""
    if image is None or not question.strip():
        yield []
        return
    
    # Load and prepare image
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image
    if hasattr(img, 'convert'):
        img = img.convert("RGB")
    
    # Save input image
    input_img_path = os.path.join(cur_dir, "tmp_input_qwen.png")
    img.save(input_img_path)
    
    user_input = [
        {"role": "user", "content": {"path": input_img_path}},
        {"role": "user", "content": question},
    ]
    
    # Create queue and state
    qwen_queue = Queue()
    qwen_state = StreamingState(qwen_queue)
    
    # Start inference thread
    qwen_thread = Thread(target=qwen_inference, args=(img.copy(), question, qwen_state))
    qwen_thread.start()
    
    # Stream updates
    while not qwen_state.done:
        # Drain queue (non-blocking)
        try:
            while True:
                qwen_queue.get_nowait()
        except Empty:
            pass
        
        qwen_output = list(qwen_state.output)
        qwen_time = (qwen_state.end_time or time.time()) - qwen_state.start_time if qwen_state.start_time else 0
        qwen_ttft = qwen_state.get_ttft()
        qwen_tpot = qwen_state.get_tpot() if qwen_state.done else None
        qwen_tokens = qwen_state.token_count
        
        header = build_metrics_header(qwen_time, qwen_state.done, qwen_ttft, qwen_tpot, qwen_tokens)
        yield format_for_chatbot(user_input + [header] + qwen_output)
        time.sleep(0.05)
    
    qwen_thread.join()
    
    # Final output
    qwen_output = list(qwen_state.output)
    qwen_time = qwen_state.end_time - qwen_state.start_time if qwen_state.start_time else 0
    qwen_ttft = qwen_state.get_ttft()
    qwen_tpot = qwen_state.get_tpot()
    qwen_tokens = qwen_state.token_count
    
    header = build_metrics_header(qwen_time, True, qwen_ttft, qwen_tpot, qwen_tokens)
    yield format_for_chatbot(user_input + [header] + qwen_output)


def format_for_chatbot(output_list):
    """Convert output list to Gradio chatbot messages format."""
    if not output_list:
        return []
    
    messages = []
    current_text = ""
    
    for item in output_list:
        if isinstance(item, str):
            current_text += item
        elif isinstance(item, dict) and "files" in item:
            if current_text:
                messages.append({"role": "assistant", "content": current_text})
                current_text = ""
            for f in item.get("files", []):
                messages.append({"role": "assistant", "content": {"path": f}})
        elif isinstance(item, dict) and "role" in item:
            if current_text:
                messages.append({"role": "assistant", "content": current_text})
                current_text = ""
            messages.append(item)
    
    if current_text:
        messages.append({"role": "assistant", "content": current_text})
    
    return messages


# ============== UI ==============
html_header = """
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(180deg, rgba(99, 102, 241, 0.15) 0%, transparent 100%); border-radius: 12px; margin-bottom: 1rem;">
    <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        🔭 ERGO vs Qwen2.5-VL Comparison
    </h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Compare <span style="color: #10b981;">ERGO</span> (two-stage zoom reasoning) vs <span style="color: #3b82f6;">Qwen2.5-VL-7B-Instruct</span>
    </p>
</div>
"""

with gr.Blocks(title="ERGO vs Qwen Comparison") as demo:
    
    gr.HTML(html_header)
    
    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil")
        question_input = gr.Textbox(label="Question", placeholder="Ask a question about the image...", lines=3)
    
    # Example
    gr.Examples(
        examples=[
            [os.path.join(cur_dir, "../data/demo/demo.jpg"), "What is the color of the umbrella that the man with the orange luggage is holding?"],
        ],
        inputs=[image_input, question_input],
        label="📝 Example",
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            ergo_btn = gr.Button("🟢 Run ERGO", variant="primary", size="lg")
            gr.HTML('<div style="padding: 0.5rem 1rem; background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), transparent); border-left: 3px solid #10b981; border-radius: 4px; margin: 0.5rem 0;"><span style="color: #10b981; font-weight: 600;">ERGO - Two-Stage Zoom Reasoning</span></div>')
            ergo_output = gr.Chatbot(label="ERGO Output", height=500)
        
        with gr.Column(scale=1):
            qwen_btn = gr.Button("🔵 Run Qwen", variant="primary", size="lg")
            gr.HTML('<div style="padding: 0.5rem 1rem; background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), transparent); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 0.5rem 0;"><span style="color: #3b82f6; font-weight: 600;">Qwen2.5-VL-7B-Instruct</span></div>')
            qwen_output = gr.Chatbot(label="Qwen Output", height=500)
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
    
    gr.Markdown("""
    ### 📌 How to Use
    1. Upload an image and type your question
    2. Click **🟢 Run ERGO** or **🔵 Run Qwen** to run each model independently
    3. Compare the TTFT (Time to First Token) and TPOT (Time Per Output Token) metrics
    
    ### 📖 About
    - **ERGO**: Uses two-stage zoom reasoning - first identifies relevant region, then analyzes zoomed view
    - **Qwen2.5-VL**: Standard VLM inference without zooming
    """)
    
    # ERGO button
    ergo_btn.click(
        fn=run_ergo_only,
        inputs=[image_input, question_input],
        outputs=[ergo_output],
    )
    
    # Qwen button
    qwen_btn.click(
        fn=run_qwen_only,
        inputs=[image_input, question_input],
        outputs=[qwen_output],
    )
    
    # Clear button
    clear_btn.click(
        fn=lambda: (None, "", [], []),
        outputs=[image_input, question_input, ergo_output, qwen_output],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True, share=False)

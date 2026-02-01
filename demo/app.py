"""
ERGO Demo: Two-column comparison between ERGO and Qwen2.5-VL-7B-Instruct
"""

import gradio as gr
import torch
import random
import numpy as np
from PIL import Image
from threading import Thread, Lock
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


def preprocess_image(img: Image.Image) -> Tuple[Image.Image, Image.Image, int, int, float, float]:
    width, height = img.size
    resize_h, resize_w = smart_resize(
        int(height), int(width), max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, factor=IMAGE_FACTOR
    )
    width_scale = width / resize_w
    height_scale = height / resize_h
    resized_img = img.resize((resize_w, resize_h), Image.BICUBIC)
    return img, resized_img, width, height, width_scale, height_scale


# ============== Model Loading ==============
print("Loading ERGO model...")
ergo_processor = AutoProcessor.from_pretrained(ERGO_MODEL_ID, max_pixels=MAX_PIXELS)
ergo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ERGO_MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_cache=True,
).to("cuda:2").eval()
print("ERGO model loaded on cuda:0")

print("Loading Qwen2.5-VL model...")
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, max_pixels=MAX_PIXELS)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_cache=True,
).to("cuda:3").eval()
print("Qwen2.5-VL model loaded on cuda:1")


# ============== Streaming State ==============
class StreamingState:
    def __init__(self):
        self.output = []
        self.done = False
        self.start_time = None
        self.end_time = None
        self.lock = Lock()


# ============== ERGO Inference ==============
def ergo_inference(image: Image.Image, question: str, state: StreamingState):
    """ERGO two-stage inference with streaming."""
    set_random_seed(DEFAULT_SEED)
    
    with state.lock:
        state.start_time = time.time()
    
    # Preprocess
    orig_img, resized_img, width, height, width_scale, height_scale = preprocess_image(image)
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
        with state.lock:
            if "<zoom>" in stage1_output and not zoom_detected:
                zoom_detected = True
                tmp = stage1_output.split("<zoom>")[0]
                state.output = [tmp + "\n\n**🔍 Planning Visual Operation...**\n\n"]
            elif not zoom_detected:
                state.output = [stage1_output]
    
    thread.join()
    
    with state.lock:
        state.output = [stage1_output + "\n\n"]
    
    # Check for zoom
    bbox = parse_zoom_bbox_from_text(stage1_output)
    
    if bbox is not None:
        with state.lock:
            state.output.append(f"**📸 Executing Visual Operation...** @crop_image(bbox={bbox})\n\n")
        
        cropped_img, actual_bbox, smart_wh = crop_with_zoom(
            orig_img, stage1_output, width_scale, height_scale, width, height,
            factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        
        tmp_path = os.path.join(cur_dir, "tmp_crop.png")
        cropped_img.save(tmp_path)
        
        with state.lock:
            state.output.append(f"\n**🔎 Analyzing Zoomed Region...** (size: {smart_wh[0]}x{smart_wh[1]})\n\n")
            state.output.append({"text": "", "files": [tmp_path]})
        
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
            with state.lock:
                state.output = state.output[:4] + [stage2_output]
        
        thread2.join()
        
        with state.lock:
            state.output = state.output[:4] + [stage2_output + "\n\n"]
    
    with state.lock:
        state.end_time = time.time()
        state.done = True


# ============== Qwen Inference ==============
def qwen_inference(image: Image.Image, question: str, state: StreamingState):
    """Qwen2.5-VL inference with streaming."""
    set_random_seed(DEFAULT_SEED)
    
    with state.lock:
        state.start_time = time.time()
    
    # Preprocess image same way
    _, resized_img, _, _, _, _ = preprocess_image(image)
    
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
        with state.lock:
            state.output = [output]
    
    thread.join()
    
    with state.lock:
        state.end_time = time.time()
        state.done = True


# ============== Combined Inference ==============
def run_comparison(image, question):
    """Run both models in parallel and yield updates."""
    if image is None or not question.strip():
        yield [], []
        return
    
    # Load and prepare image
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image
    if hasattr(img, 'convert'):
        img = img.convert("RGB")
    
    # Initialize states
    ergo_state = StreamingState()
    qwen_state = StreamingState()
    
    # Start both threads
    ergo_thread = Thread(target=ergo_inference, args=(img.copy(), question, ergo_state))
    qwen_thread = Thread(target=qwen_inference, args=(img.copy(), question, qwen_state))
    
    ergo_thread.start()
    qwen_thread.start()
    
    # Poll and yield updates
    while not (ergo_state.done and qwen_state.done):
        with ergo_state.lock:
            ergo_output = list(ergo_state.output)
            ergo_time = (ergo_state.end_time or time.time()) - ergo_state.start_time if ergo_state.start_time else 0
            ergo_done = ergo_state.done
        
        with qwen_state.lock:
            qwen_output = list(qwen_state.output)
            qwen_time = (qwen_state.end_time or time.time()) - qwen_state.start_time if qwen_state.start_time else 0
            qwen_done = qwen_state.done
        
        # Build timing headers
        ergo_header = f"⏱️ **{ergo_time:.1f}s** {'✅' if ergo_done else '🔄'}\n\n"
        qwen_header = f"⏱️ **{qwen_time:.1f}s** {'✅' if qwen_done else '🔄'}\n\n"
        
        yield [ergo_header] + ergo_output, [qwen_header] + qwen_output
        time.sleep(0.1)
    
    ergo_thread.join()
    qwen_thread.join()
    
    # Final output
    with ergo_state.lock:
        ergo_output = list(ergo_state.output)
        ergo_time = ergo_state.end_time - ergo_state.start_time if ergo_state.start_time else 0
    
    with qwen_state.lock:
        qwen_output = list(qwen_state.output)
        qwen_time = qwen_state.end_time - qwen_state.start_time if qwen_state.start_time else 0
    
    ergo_header = f"⏱️ **{ergo_time:.1f}s** ✅\n\n"
    qwen_header = f"⏱️ **{qwen_time:.1f}s** ✅\n\n"
    
    yield [ergo_header] + ergo_output, [qwen_header] + qwen_output


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
        with gr.Column(scale=1):
            gr.HTML('<div style="padding: 0.5rem 1rem; background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), transparent); border-left: 3px solid #10b981; border-radius: 4px; margin-bottom: 0.5rem;"><span style="color: #10b981; font-weight: 600;">🟢 ERGO - Two-Stage Zoom Reasoning</span></div>')
            ergo_output = gr.Chatbot(label="ERGO", height=500)
        
        with gr.Column(scale=1):
            gr.HTML('<div style="padding: 0.5rem 1rem; background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), transparent); border-left: 3px solid #3b82f6; border-radius: 4px; margin-bottom: 0.5rem;"><span style="color: #3b82f6; font-weight: 600;">🔵 Qwen2.5-VL-7B-Instruct</span></div>')
            qwen_output = gr.Chatbot(label="Qwen2.5-VL", height=500)
    
    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil")
        question_input = gr.Textbox(label="Question", placeholder="Ask a question about the image...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("🚀 Compare Models", variant="primary")
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
    
    gr.Markdown("""
    ### 📌 How to Use
    1. Upload an image
    2. Type your question
    3. Click "Compare Models" to see both responses side by side
    
    ### 📖 About
    - **ERGO**: Uses two-stage zoom reasoning - first identifies relevant region, then analyzes zoomed view
    - **Qwen2.5-VL**: Standard VLM inference without zooming
    """)
    
    def format_for_chatbot(output_list):
        """Convert output list to Gradio 6 chatbot messages format, preserving order."""
        if not output_list:
            return []
        
        # Process items in order, creating separate messages
        messages = []
        current_text = ""
        
        for item in output_list:
            if isinstance(item, str):
                current_text += item
            elif isinstance(item, dict) and "files" in item:
                # Flush any accumulated text first
                if current_text:
                    messages.append({"role": "assistant", "content": current_text})
                    current_text = ""
                # Add image message
                for f in item.get("files", []):
                    messages.append({"role": "assistant", "content": {"path": f}})
        
        # Flush remaining text
        if current_text:
            messages.append({"role": "assistant", "content": current_text})
        
        return messages
    
    def run_and_format(image, question):
        for ergo_out, qwen_out in run_comparison(image, question):
            yield format_for_chatbot(ergo_out), format_for_chatbot(qwen_out)
    
    submit_btn.click(
        fn=run_and_format,
        inputs=[image_input, question_input],
        outputs=[ergo_output, qwen_output],
    )
    
    clear_btn.click(
        fn=lambda: (None, "", [], []),
        outputs=[image_input, question_input, ergo_output, qwen_output],
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)

"""
Inference functions for ERGO model.
"""

import os
import time
from threading import Thread, Lock
from queue import Queue
from PIL import Image
from transformers import TextIteratorStreamer

from .config import (
    CUR_DIR,
    DEFAULT_SEED,
    ERGO_MAX_PIXELS,
    IMAGE_FACTOR,
    MIN_PIXELS,
    MAX_PIXELS,
    SYSTEM_PROMPT,
    QUESTION_TEMPLATE1,
    QUESTION_TEMPLATE2,
    GENERATION_CONFIG,
)
from .utils import (
    set_random_seed,
    preprocess_image,
    parse_zoom_bbox_from_text,
    crop_with_zoom,
)
from .models import get_ergo_model, get_ergo_processor


class StreamingState:
    """State for streaming output with queue-based communication."""
    
    def __init__(self, output_queue: Queue):
        self.output_queue = output_queue
        self.output = []
        self.done = False
        self.start_time = None
        self.end_time = None
        self.first_token_time = None
        self.token_count = 0
        self.lock = Lock()
        # Vision token stats (separate from output)
        self.vision_stats = None
    
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
            generation_time = self.end_time - self.first_token_time
            return (generation_time / (self.token_count - 1)) * 1000
        return None


def ergo_inference(image: Image.Image, question: str, state: StreamingState):
    """
    ERGO two-stage inference with streaming via queue.
    
    Stage 1: Process image and identify zoom region
    Stage 2: Process zoomed region for final answer
    """
    set_random_seed(DEFAULT_SEED)
    
    ergo_model = get_ergo_model()
    ergo_processor = get_ergo_processor()
    
    state.start_time = time.time()
    state.put_update("start")

    # Preprocess
    orig_img, resized_img, width, height, width_scale, height_scale = preprocess_image(
        image, max_pixels=ERGO_MAX_PIXELS
    )
    all_images = [resized_img]

    # Calculate vision token counts
    original_vision_token_cnt = orig_img.size[0] * orig_img.size[1] / (28 * 28)
    vision_token_cnt_resized = resized_img.size[0] * resized_img.size[1] / (28 * 28)

    # Stage 1: Identify zoom region
    messages_stage1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE1.format(Question=question.split("(A)")[0])},
        ]},
    ]
    
    prompt = ergo_processor.apply_chat_template(
        messages_stage1, tokenize=False, add_generation_prompt=True
    )
    inputs = ergo_processor(
        text=[prompt], images=all_images, return_tensors="pt", padding=True
    ).to(ergo_model.device)
    
    streamer = TextIteratorStreamer(
        ergo_processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs, streamer=streamer, **GENERATION_CONFIG
    )
    
    thread = Thread(target=ergo_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    stage1_output = ""
    zoom_detected = False
    
    for new_text in streamer:
        stage1_output += new_text
        
        if state.first_token_time is None:
            state.first_token_time = time.time()
            state.put_update("first_token")
        
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
    
    # Check for zoom coordinates
    bbox = parse_zoom_bbox_from_text(stage1_output)
    
    if bbox is not None:
        state.output.append(f"**Visual Operation...** @crop_image(bbox={bbox})\n\n")
        state.put_update("zoom_start")
        
        cropped_img, actual_bbox, smart_wh = crop_with_zoom(
            orig_img, stage1_output, width_scale, height_scale, width, height,
            factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        
        # Save cropped image for display
        tmp_path = os.path.join(CUR_DIR, "tmp_crop.png")
        cropped_img.save(tmp_path)
        
        state.output.append(f"\n**Analyzing Zoomed Region...** (size: {smart_wh[0]}x{smart_wh[1]})\n\n")
        vision_token_cnt_zoomed = smart_wh[0] * smart_wh[1] / (28 * 28)
        state.output.append({"text": "", "files": [tmp_path]})
        state.put_update("zoom_image", tmp_path)
        
        # Stage 2: Analyze zoomed region
        all_images.append(cropped_img)
        messages_stage2 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE1.format(Question=question.split("(A)")[0])}
            ]},
            {"role": "assistant", "content": stage1_output},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE2.format(Question=question)}
            ]},
        ]
        
        prompt2 = ergo_processor.apply_chat_template(
            messages_stage2, tokenize=False, add_generation_prompt=True
        )
        inputs2 = ergo_processor(
            text=[prompt2], images=all_images, return_tensors="pt", padding=True
        ).to(ergo_model.device)
        
        streamer2 = TextIteratorStreamer(
            ergo_processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs2 = dict(
            **inputs2, streamer=streamer2, **GENERATION_CONFIG
        )
        
        thread2 = Thread(target=ergo_model.generate, kwargs=generation_kwargs2)
        thread2.start()
        
        stage2_output = ""
        for new_text in streamer2:
            stage2_output += new_text
            state.token_count += len(new_text.split()) if new_text.strip() else 0
            state.output = state.output[:5] + [stage2_output]
            state.put_update("token", new_text)
        
        thread2.join()
        
        # Store vision token stats separately
        total_ergo_tokens = vision_token_cnt_resized + vision_token_cnt_zoomed
        pct_of_original = (
            100.0 * total_ergo_tokens / original_vision_token_cnt
            if original_vision_token_cnt else 0
        )
        
        state.vision_stats = {
            "original": original_vision_token_cnt,
            "stage1": vision_token_cnt_resized,
            "stage2": vision_token_cnt_zoomed,
            "total": total_ergo_tokens,
            "reduction_pct": pct_of_original,
        }
        
        state.output = state.output[:5] + [stage2_output]
    
    state.end_time = time.time()
    state.done = True
    state.put_update("done")



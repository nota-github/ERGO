import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ergo.const import QUESTION_TEMPLATE1, QUESTION_TEMPLATE2, SYSTEM_PROMPT
from ergo.utils import crop_with_zoom, smart_resize

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_NEW_TOKENS = 512
DEFAULT_SEED = 42


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_processor(device: torch.device) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "nota-ai/ERGO-7B",
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("nota-ai/ERGO-7B")
    model.eval()
    return model, processor


def preprocess_image(input_path: str) -> Tuple[Image.Image, Image.Image, int, int, float, float]:
    img = Image.open(input_path)
    width, height = img.size
    resize_h, resize_w = smart_resize(
        int(height), int(width), max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, factor=IMAGE_FACTOR
    )
    width_scale = width / resize_w
    height_scale = height / resize_h
    resized_img = img.resize((resize_w, resize_h), Image.BICUBIC)
    return img, resized_img, width, height, width_scale, height_scale


def build_initial_messages(question: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE1.format(Question=question)},
            ],
        },
    ]


def generate_once(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: List[dict],
    images: List[Image.Image],
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/demo/demo.jpg")
    parser.add_argument(
        "--question",
        type=str,
        default="Is the orange luggage on the left or right side of the purple umbrella?",
    )
    parser.add_argument("--save_output", action="store_true")
    args = parser.parse_args()

    set_random_seed(DEFAULT_SEED)
    device = resolve_device()
    model, processor = load_model_and_processor(device)

    img, resized_img, width, height, width_scale, height_scale = preprocess_image(args.input_path)

    messages = build_initial_messages(args.question)
    intermediate_output_text = generate_once(
        model=model,
        processor=processor,
        messages=messages,
        images=[resized_img],
        device=device,
    )
    print("\nIntermediate_output_text", intermediate_output_text)

    cropped_img, _bbox, _smart_wh = crop_with_zoom(
        img,
        intermediate_output_text,
        width_scale,
        height_scale,
        width,
        height,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    if args.save_output:
        os.makedirs("./output", exist_ok=True)
        cropped_img.save("./output/demo_cropped.jpg")

    messages.extend(
        [
            {
                "role": "assistant",
                "content": intermediate_output_text,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE2.format(Question=args.question)},
                ],
            },
        ]
    )

    final_output_text = generate_once(
        model=model,
        processor=processor,
        messages=messages,
        images=[resized_img, cropped_img],
        device=device,
    )
    print("\nFinal_output_text", final_output_text)


if __name__ == "__main__":
    main()

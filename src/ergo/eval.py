import argparse
import asyncio
import base64
import json
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from PIL import Image
from tqdm import tqdm

from ergo.const import QUESTION_TEMPLATE1, QUESTION_TEMPLATE2, SYSTEM_PROMPT
from ergo.utils import (
    check_image_path,
    crop_with_zoom,
    encode_pil_image_to_base64,
    extract_answer_text,
    preprocess_data,
    smart_resize,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["vstar", "mmerwl", "hrbench"])
parser.add_argument("--data_root", type=str, default="./data")
parser.add_argument("--api_url", type=str, default="http://localhost:8008/v1")
parser.add_argument("--max_vision_token_num", type=int, default=1280)
parser.add_argument("--max_concurrency", type=int, default=16)
args = parser.parse_args()


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = int(args.max_vision_token_num) * 28 * 28


async def process(
    async_client: AsyncOpenAI,
    item: Dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Optional[Dict[str, Any]]:
    try:
        async with sem:
            image = item["image"]
            question = item["question"]
            options = item["options"]
            gt = item["label"]
            question_id = item["question_id"]
            category = item["category"]
            is_path = check_image_path(image)
            if is_path:
                pil_img = Image.open(image)
            else:
                pil_img = Image.open(BytesIO(base64.b64decode(image)))

            width, height = pil_img.size
            resize_h, resize_w = smart_resize(
                int(height), int(width), max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, factor=IMAGE_FACTOR
            )
            width_scale = width / resize_w
            height_scale = height / resize_h
            pil_img_resize = pil_img.resize((resize_w, resize_h), Image.BICUBIC)
            base64_image = encode_pil_image_to_base64(pil_img_resize)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE1.format(Question=question),
                        },
                    ],
                },
            ]
            params = {
                "model": "ergo",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024,
                "stop": ["<|im_end|>\n".strip()],
                "seed": 42,
            }

            response = None
            for attempt_index in range(max_retries):
                try:
                    response = await async_client.chat.completions.create(**params)
                    break
                except (
                    APITimeoutError,
                    APIConnectionError,
                    RateLimitError,
                    APIError,
                    asyncio.TimeoutError,
                ):
                    if attempt_index == max_retries - 1:
                        raise
                    await asyncio.sleep(base_delay * (2**attempt_index))

            response_message = response.choices[0].message.content

            cropped_img, _bbox, _smart_wh = crop_with_zoom(
                pil_img,
                response_message,
                width_scale,
                height_scale,
                width,
                height,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            cropped_b64 = encode_pil_image_to_base64(cropped_img)

            full_question = question + "\n" + "\n".join(options) + "\n" + item["postfix"]
            messages.extend(
                [
                    {"role": "assistant", "content": response_message},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{cropped_b64}"},
                            },
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE2.format(Question=full_question),
                            },
                        ],
                    },
                ]
            )

            final_response_obj = None
            for attempt_index in range(max_retries):
                try:
                    final_response_obj = await async_client.chat.completions.create(**params)
                    break
                except (
                    APITimeoutError,
                    APIConnectionError,
                    RateLimitError,
                    APIError,
                    asyncio.TimeoutError,
                ):
                    if attempt_index == max_retries - 1:
                        raise
                    await asyncio.sleep(base_delay * (2**attempt_index))

            final_response = final_response_obj.choices[0].message.content
            model_answer_raw = extract_answer_text(final_response) or final_response
            pred = "".join(re.findall(r"[a-zA-Z]", model_answer_raw))
            return {
                "gt": gt,
                "pred": pred,
                "question_id": question_id,
                "category": category,
                "question": question,
                "options": options,
                "response_message": response_message,
            }
    except Exception:
        return None


async def main():
    async_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.api_url,
    )

    sem = asyncio.Semaphore(args.max_concurrency)

    datas = []
    test_types = None
    if args.dataset == "vstar":
        test_types = ["direct_attributes", "relative_position"]
        ann_path = Path(args.data_root) / "vstar_bench" / "test_questions.jsonl"
        with open(ann_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data["image"] = str(Path(args.data_root) / "vstar_bench" / data["image"])
                datas.append(preprocess_data(data, "vstar"))

    elif args.dataset == "hrbench":
        test_types = ["4k", "8k"]
        ann_4k_path = Path(args.data_root) / "HR-Bench" / "hr_bench_4k.parquet"
        ann_8k_path = Path(args.data_root) / "HR-Bench" / "hr_bench_8k.parquet"
        ann_datas_pandas = [pd.read_parquet(ann_4k_path), pd.read_parquet(ann_8k_path)]
        for i, ann_data_pandas in enumerate(ann_datas_pandas):
            for id, item in ann_data_pandas.iterrows():
                data = dict()
                datas.append(preprocess_data(data, "hrbench", item=item, test_types=test_types, id=id, i=i))

    elif args.dataset == "mmerwl":
        test_types = []
        for test_type in [
            "Autonomous_Driving",
            "Diagram and Table",
            "Monitoring",
            "OCR with Complex Context",
            "Remote Sensing",
        ]:
            for task in ["Perception", "Reasoning"]:
                test_types.append(f"{task}/{test_type}")

        for i in range(4):
            ann_path = str(
                Path(args.data_root) / "MME-RealWorld-lite-lmms-eval" / "data" / f"train-0000{i}-of-00004.parquet"
            )
            df = pd.read_parquet(ann_path)
            for i, item in df.iterrows():
                data = dict()
                datas.append(preprocess_data(data, "mmerwl", item=item, i=i))

    tasks = []
    for data in datas:
        tasks.append(asyncio.create_task(process(async_client, data, sem)))

    results = []
    with tqdm(total=len(datas), desc="Processing ") as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result is not None:
                results.append(result)
            pbar.update(1)
    accuracy_results = {}
    total_correct = 0
    total_count = 0
    for test_type in test_types:
        correct = 0
        count = 0
        for result in results:
            if test_type == result["category"]:
                count += 1
                if result["gt"].lower() == result["pred"].lower():
                    correct += 1
        total_correct += correct
        total_count += count
        accuracy = correct / count if count > 0 else 0
        accuracy_results[test_type] = {
            "category": test_type,
            "accuracy": accuracy,
            "correct": correct,
            "count": count,
        }

    total_accuracy = total_correct / total_count if total_count > 0 else 0
    accuracy_results["total"] = {
        "accuracy": total_accuracy,
        "correct": total_correct,
        "count": total_count,
    }

    print("\n===== Accuracy Report =====")
    for test_type in test_types:
        v = accuracy_results.get(test_type, None)
        if v is None:
            continue
        acc_pct = v["accuracy"] * 100
        print(f"{test_type:>20}: {acc_pct:.2f}% ({v['correct']}/{v['count']})")

    total_v = accuracy_results.get("total")
    if total_v is not None:
        print("-" * 34)
        print(f"{'TOTAL':>20}: {total_v['accuracy'] * 100:.2f}% ({total_v['correct']}/{total_v['count']})")


if __name__ == "__main__":
    asyncio.run(main())

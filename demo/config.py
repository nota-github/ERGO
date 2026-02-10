"""
Configuration constants and prompts for ERGO Demo.
"""

import os

# ============== Path Constants ==============
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== Image Processing Constants ==============
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
ERGO_MAX_PIXELS = 1280 * 28 * 28

# ============== Generation Constants ==============
MAX_NEW_TOKENS = 1024
DEFAULT_SEED = 42

# ============== Model IDs ==============
ERGO_MODEL_ID = "nota-ai/ERGO-7B"

# ============== System Prompts ==============
SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. First, identify the coordinates "
    "of the key image area relevant to solving the problem. Append the coordinates in [x1, y1, x2, y2] "
    "format at the end of your response and stop. This will trigger cropping of the corresponding area in "
    "the original image and enlarge it for improved clarity. Once the enlarged image is available, provide the final answer."
)

# ============== Question Templates ==============
QUESTION_TEMPLATE1 = (
    "{Question} First, consider how to zoom in on the image to include only the region "
    "containing all the information necessary to answer the question. Output the thinking "
    "process in <think> </think> tags and then output the region's bounding box coordinates "
    "in the following format <zoom>[x1, y1, x2, y2]</zoom> tags."
)

QUESTION_TEMPLATE2 = (
    "{Question}\nCarefully analyze both the original image and the enlarged sub-image to "
    "solve the question step by step. Present your reasoning clearly the following format "
    "strictly <think> ... </think>, and provide the final answer enclosed within "
    "<answer> ... </answer>"
)

# ============== Generation Parameters ==============
GENERATION_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 50,
}
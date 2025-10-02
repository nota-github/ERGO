SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. First, identify the coordinates "
    "of the key image area relevant to solving the problem. Append the coordinates in [x1, y1, x2, y2] "
    "format at the end of your response and stop. This will trigger cropping of the corresponding area in "
    "the original image and enlarge it for improved clarity. Once the enlarged image is available, provide the final answer."
)

QUESTION_TEMPLATE1 = "{Question} First, consider how to zoom in on the image to include only the region containing all the information necessary to answer the question. Output the thinking process in <think> </think> tags and then output the region's bounding box coordinates in the following format <zoom>[x1, y1, x2, y2]</zoom> tags."

QUESTION_TEMPLATE2 = "{Question}\nCarefully analyze both the original image and the enlarged sub-image to solve the question step by step. Present your reasoning clearly the following format strictly <think> ... <think>, and provide the final answer enclosed within <answer> ... </answer>"

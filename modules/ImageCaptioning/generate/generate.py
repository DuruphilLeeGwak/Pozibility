import time
import sys
import os
from PIL import Image

from transformers import pipeline

# ImageCaptioning 폴더를 sys.path에 추가 (NBGenerator에서 호출될 때와 ImageCaptioning에서 직접 호출될 때 모두 대응)
current_dir = os.path.dirname(os.path.abspath(__file__))
image_captioning_dir = os.path.dirname(current_dir)
if image_captioning_dir not in sys.path:
    sys.path.insert(0, image_captioning_dir)

from utils.utils import resize_by_input, im_show, extract_json

pipe = pipeline("image-text-to-text", model="OpenGVLab/InternVL3_5-4B-HF", trust_remote_code=True)

def image_captioning(img_path, prompt):
    img_size = im_show(img_path)
    img_obj = Image.open(img_path)
    img_resized = resize_by_input(img_obj, img_size[0], img_size[1], 512)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_resized},
                {"type": "text", "text": prompt}
            ]
        },
    ]

    t0 = time.time()
    output_txt = pipe(text=messages)
    clean_txt = extract_json(output_txt[0]["generated_text"][-1]["content"])
    # print(f"Elapsed time: {time.time() - t0 :.2f}")
    return clean_txt
import re
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def extract_json(text):
    # ```json ... ``` 제거
    text = text.replace("```json", "").replace("```", "").strip()

    # { } 블록만 추출
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("❌ JSON 블록을 찾지 못했습니다!")

    json_str = match.group(0)
    return json.loads(json_str)

def resize_by_input(img, w, h, fix_size = 512) -> Image.Image:
    """
    이미지의 width 또는 height 중 하나라도 1024 이상이면
    1024 x 1024 정사각형으로 리사이즈한다.
    그렇지 않으면 원본 이미지를 그대로 반환한다.
    """
    # 둘 중 하나라도 512 이상이면 리사이징
    if w >= fix_size or h >= fix_size:
        return img.resize((fix_size, fix_size), Image.LANCZOS)
    else:
        return img

def im_show(img_path):
    img = Image.open(img_path)
    img_np = np.array(img) ## 행렬로 변환된 이미지
    plt.imshow(img_np) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
    plt.axis('off')
    plt.show() ## 이미지 인터프린터에 출력
    # print("📏 Image size:", img.size)        # (width, height)
    return img.size
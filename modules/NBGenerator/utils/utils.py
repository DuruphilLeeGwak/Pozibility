import numpy as np
import matplotlib.pyplot as plt
import os
import io

from PIL import Image, ImageOps

__all__ = ["im_show", "save_binary_file", "img_to_bytes"]


def im_show(img_path):
    img = Image.open(img_path)
    img_np = np.array(img) ## 행렬로 변환된 이미지
    plt.imshow(img_np) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
    plt.axis('off')
    plt.show() ## 이미지 인터프린터에 출력
    return img.size

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")

def img_to_bytes(img):
    """PIL.Image 객체 또는 이미지 파일 경로를 PNG 포맷의 바이트 데이터로 변환합니다."""
    # 파일 경로 문자열인 경우 PIL Image로 열기
    if isinstance(img, str):
        img = Image.open(img)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
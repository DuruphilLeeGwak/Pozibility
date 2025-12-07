import json
from utils import *
from prompt.prompt import *
from generate.generate import image_captioning

# from transformers import pipe
import argparse

def main():
    parser = argparse.ArgumentParser(description='ImageCaptioning Module')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()
    
    img_path = args.img_path
    type = args.type

    if type == "img_org":
        prompt = prompt_img_org
    elif type == "nano_org":
        prompt = prompt_nano_org
    elif type == "nano_rendered":
        prompt = prompt_nano_rendered
    else:
        raise ValueError("Invalid type")

    result = image_captioning(img_path, prompt)
    with open("prompt.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    main()
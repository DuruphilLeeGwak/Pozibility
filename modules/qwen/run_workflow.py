import os
import io
import time
import json
import shutil
import requests
import argparse

from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = 'workflow/qwen_edit_workflow.json'
OUTPUT_DIR = 'output'


def load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def override_workflow(workflow, img_path, sktn_path, positive_prompt):
    for node_id, node in workflow.items():
        cls = node.get("class_type")
        meta = node.get("_meta", {})
        title = meta.get("title", "")

        if cls == "LoadImage":
            # Image 1 (model)
            if node_id == '78':
                node["inputs"]["image"] = img_path

            # Image 2 (Pose Reference)
            elif node_id == '179':
                node["inputs"]["image"] = sktn_path

        if cls == "TextEncodeQwenImageEditPlus" and "Positive" in title:
            if positive_prompt is not None:
                node["inputs"]["prompt"] = positive_prompt

    return workflow

def send_prompt(workflow):
    payload = {
        "prompt": workflow
    }
    resp = requests.post(f"{COMFY_URL}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["prompt_id"]

def qwen_inference(img_path, sktn_path, positive_prompt):
    workflow = load_workflow()
    workflow = override_workflow(workflow, img_path, sktn_path, positive_prompt)

    prompt_id = send_prompt(workflow)

    timeout = 360
    start = time.time()
    while True:
        hist_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        hist_resp.raise_for_status()
        hist = hist_resp.json()

        if prompt_id in hist and "outputs" in hist[prompt_id]:
            outputs = hist[prompt_id]["outputs"]

            for node_id, out in outputs.items():
                if "images" in out and out["images"]:
                    return hist[prompt_id]

        if time.time() - start > timeout:
            raise TimeoutError("ComfyUI inference timeout")

        time.sleep(0.5)

def save_output(result):
    outputs = result["outputs"]

    images = None

    if "94" in outputs and "images" in outputs["94"]:
        images = outputs["94"]["images"]
    else:
        for node_id, out in outputs.items():
            if "images" in out and out["images"]:
                images = out["images"]
                break

    if not images:
        raise RuntimeError("No image outputs found in history")

    img_info = images[0]
    filename = img_info["filename"]
    subfolder = img_info.get("subfolder", "")
    img_type = img_info.get("type", "output")

    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": img_type,
    }
    resp = requests.get(f"{COMFY_URL}/view", params=params)
    resp.raise_for_status()

    # PIL Image
    pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "qwen_output.jpg")
    pil_img.save(save_path)
    

def run_workflow(img_path, sktn_path, positive_prompt):
    result = qwen_inference(img_path, sktn_path, positive_prompt)
    save_output(result)
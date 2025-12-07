import subprocess
import time
import requests
import sys
import argparse
import json

COMFYUI_PATH = "ComfyUI"  # ComfyUI 디렉토리
PYTHON_EXE = "python"             # 또는 python

def start_comfyui():
    # ComfyUI 서버를 백그라운드로 실행
    process = subprocess.Popen(
        [PYTHON_EXE, "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("comfy main.py on")
    return process


def wait_server_ready(timeout=1200):
    url = "http://127.0.0.1:8188/system_stats"
    start = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("ComfyUI READY")
                return True
        except:
            pass

        if time.time() - start > timeout:
            raise TimeoutError("ComfyUI server start timeout")
        time.sleep(1)


def stop_comfyui(process):
    # graceful 종료
    process.terminate()  
    try:
        process.wait(timeout=100)
    except subprocess.TimeoutExpired:
        print("Force kill ComfyUI")
        process.kill()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen-Edit Module')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--skeleton_path', type=str, required=True)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()
    
    img_path = f'{args.img_path}'
    sktn_path = f'{args.skeleton_path}'
    prompt = args.prompt

    if prompt is None:
        prompt = """
        preserve facial identity
        """
    else:
        with open(prompt, "r", encoding="utf-8") as f:
            tmp = json.load(f)
        prompt = f'{tmp}'

    # 1) 서버 실행
    server = start_comfyui()

    try:
        # 2) 서버 부팅 대기
        wait_server_ready()

        # 3) 여기에서 inference 실행
        from run_workflow import run_workflow

        run_workflow(
            img_path=img_path,
            sktn_path=sktn_path,
            positive_prompt=prompt
        )
        print("Inference done!")

    except Exception as e:
        print("ERROR:", e)

    finally:
        # 4) 서버 종료
        stop_comfyui(server)
        print("ComfyUI server stopped.")

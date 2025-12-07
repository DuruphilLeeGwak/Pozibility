import sys
import os
import importlib.util

# ImageCaptioning 디렉토리 경로 계산 (NBGenerator와 ImageCaptioning은 형제 디렉토리)
nbgenerator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/ubuntu/NBGenerator
parent_dir = os.path.dirname(nbgenerator_dir)  # /home/ubuntu
image_captioning_dir = os.path.join(parent_dir, 'ImageCaptioning')  # /home/ubuntu/ImageCaptioning

# ImageCaptioning의 generate 모듈을 명시적으로 로드
generate_spec = importlib.util.spec_from_file_location(
    "image_captioning_generate",
    os.path.join(image_captioning_dir, "generate", "generate.py")
)
image_captioning_generate = importlib.util.module_from_spec(generate_spec)
generate_spec.loader.exec_module(image_captioning_generate)
image_captioning = image_captioning_generate.image_captioning

# ImageCaptioning의 prompt 모듈을 명시적으로 로드
prompt_spec = importlib.util.spec_from_file_location(
    "image_captioning_prompt",
    os.path.join(image_captioning_dir, "prompt", "prompt.py")
)
image_captioning_prompt = importlib.util.module_from_spec(prompt_spec)
prompt_spec.loader.exec_module(image_captioning_prompt)
prompt_img_org = image_captioning_prompt.prompt_img_org
prompt_nano_org = image_captioning_prompt.prompt_nano_org
prompt_nano_rendered = image_captioning_prompt.prompt_nano_rendered


__all__ = ["generate_prompt"]

def _format_caption_as_text(caption_dict):
    """JSON 캡션 딕셔너리를 읽기 쉬운 텍스트로 변환"""
    if isinstance(caption_dict, dict):
        parts = []
        for key, value in caption_dict.items():
            if value and value != "nothing":
                parts.append(f"{key}: {value}")
        return "(" + ", ".join(parts) + ")" if parts else ""
    return str(caption_dict)

def generate_prompt(hallucination_type, original_img_path, ref_img_path):
    print(f'original_img_path: {original_img_path}')
    print(f'ref_img_path: {ref_img_path}')
    # 선택지 매핑 테이블
    options = {
        1: 'simply returned original image',
        2: 'pose',
        3: "facial identity",
        4: "outfit",
        5: "proportion",
        6: "background",
        7: "person",
        8: "perspective or depth",
        9: "object", 
    }

    # 입력 문자열 → 정수 리스트 변환
    try:
        indices = [int(x.strip()) for x in hallucination_type.split(",")]
    except ValueError:
        raise ValueError("❌ 숫자만 입력해야 합니다. 예: 1,2")

    # --------------------------------------------
    # ✔ SPECIAL CASE: 입력이 1 또는 2만 포함된 경우
    # --------------------------------------------
    if set(indices).issubset({1, 2}):
        # (Image 1)에는 prompt_img_org 적용
        # (Image 2)에는 prompt_nano_rendered 적용
        image1_caption_dict = image_captioning(original_img_path, prompt_img_org)
        image2_caption_dict = image_captioning(ref_img_path, prompt_nano_rendered)
        
        image1_caption = _format_caption_as_text(image1_caption_dict)
        image2_caption = _format_caption_as_text(image2_caption_dict)
        
        prompt = (
            f"Pose transfer: Apply only the pose of the skeleton keypoint rendered image to the image of a {image1_caption_dict['sex']} wearing {image1_caption_dict['outfit']} on {image1_caption_dict['background']}. "
            f"Strictly maintain the facial identity and the perspective of the original image."
        )
        return prompt

    # --------------------------------------------
    # ✔ 나머지는 기존 로직 수행 (3~9번)
    # --------------------------------------------
    
    # object 제거 여부 (9번 선택 시)
    remove_unwanted_object = 9 in indices
    
    # 9번 케이스에서는 image1에 prompt_nano_org 적용, 그 외에는 prompt_img_org 적용
    # (Image 2)에는 항상 prompt_img_org 적용
    if remove_unwanted_object:
        # image1_caption_dict = image_captioning(original_img_path, prompt_nano_org)
        image1_caption_dict = image_captioning(original_img_path, prompt_img_org)
    else:
        image1_caption_dict = image_captioning(original_img_path, prompt_img_org)
    image2_caption_dict = image_captioning(ref_img_path, prompt_img_org)
    # image2_caption_dict = image_captioning(ref_img_path, prompt_nano_org)
    
    image1_caption = _format_caption_as_text(image1_caption_dict)
    print(f"Image 1 caption: {image1_caption}")
    image2_caption = _format_caption_as_text(image2_caption_dict)
    print(f"Image 2 caption: {image2_caption}")
    
    # 3~8번은 일반 hallucination 항목으로 구성
    base_indices = [i for i in indices if i in range(3, 9)]  
    hallucination_list = [options[i] for i in base_indices]

    # prompt-friendly string 변환
    if len(hallucination_list) == 0:
        hallucination_str = ""
    elif len(hallucination_list) == 1:
        hallucination_str = hallucination_list[0]
    else:
        hallucination_str = ", ".join(hallucination_list[:-1]) + " and " + hallucination_list[-1]

    # 기본 prompt 구성
    if hallucination_str:
        # prompt = f"Pose Transfer: stirctly keep the pose of the {image1_caption_dict['sex']} wearing {image1_caption_dict['outfit']} on {image1_caption_dict['background']}, but substitute {hallucination_str} with that in the other image."
        prompt = f"Pose Transfer: stirctly keep the pose ({image1_caption_dict['pose']}) of Image 1 (the one with too much blank spaces), but substitute {hallucination_str} with that of Image 2 (a compact image)."
    else:
        # hallucination_str이 비어있는 경우는 거의 없지만, pose 키가 있는지 확인
        pose_text = image1_caption_dict.get('pose', '')
        if pose_text:
            prompt = f"Pose transfer: strictly apply only the pose ({image1_caption_dict['pose']}) to Image 1 ({image1_caption_dict['sex']} wearing {image1_caption_dict.get('outfit', 'clothing')} on {image1_caption_dict.get('background', 'background')})."
        else:
            prompt = f"Pose transfer: strictly apply only the pose of Image 1 ({image1_caption_dict['sex']} wearing {image1_caption_dict.get('outfit', 'clothing')} on {image1_caption_dict.get('background', 'background')} )."

    # unwanted objects 제거 문구 추가
    if remove_unwanted_object:
        object_text = image1_caption_dict.get('object', 'unwanted objects')
        if object_text and object_text != "nothing":
            prompt += (
                f"Only remove {object_text} from that image (Image 1), which does not exist in the other image (Image 2)."
            )
        else:
            prompt += (
                "Only remove unwanted objects from that image (Image 1), which does not exist in the other image (Image 2)."
            )
    # print(f"NB input prompt: {prompt}")
    return prompt
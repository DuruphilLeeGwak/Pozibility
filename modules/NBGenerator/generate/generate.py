import os
import re
import sys
import mimetypes
import importlib.util
from dotenv import load_dotenv
from google import genai
from google.genai import types

# NBGenerator의 utils 모듈을 명시적으로 로드 (ImageCaptioning과의 충돌 방지)
nbgenerator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(nbgenerator_dir, "utils", "utils.py")
utils_spec = importlib.util.spec_from_file_location("nbgenerator_utils", utils_path)
nbgenerator_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(nbgenerator_utils)

# 필요한 함수들을 명시적으로 가져오기
img_to_bytes = nbgenerator_utils.img_to_bytes
save_binary_file = nbgenerator_utils.save_binary_file

__all__ = ["generate_nanobanana"]

def generate_nanobanana(qwen_image, ref_image, prefix, prompt):
    print(f"qwen_path: {qwen_image}")
    print(f"ref_path: {ref_image}")
    
    # --------------------------------------
    # qwen_image 경로에서 마지막 _ 뒤의 숫자 추출
    # --------------------------------------
    qwen_basename = os.path.basename(qwen_image)  # 파일명만 추출
    qwen_name_without_ext = os.path.splitext(qwen_basename)[0]  # 확장자 제거
    
    # 마지막 _ 뒤의 숫자 추출
    match = re.search(r'_(\d+)$', qwen_name_without_ext)
    if match:
        qwen_number = match.group(1)
    else:
        # _ 뒤 숫자가 없으면 기본값 사용
        qwen_number = "0"
    
    # --------------------------------------
    # output 폴더 생성
    # --------------------------------------
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # --------------------------------------
    # .env 파일에서 환경 변수를 로드합니다.
    # --------------------------------------
    load_dotenv()
    
    # --------------------------------------
    # 환경 변수에서 API 키를 가져옵니다.
    # --------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. "
            "환경 변수를 설정하거나 export GEMINI_API_KEY='your-api-key'를 실행하세요."
        )
    
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise ValueError(
            f"❌ Gemini API 클라이언트 초기화 실패: {str(e)}. "
            "API 키가 유효한지 확인하세요."
        ) from e

    model = "gemini-3-pro-image-preview"

    contents = [
        types.Content(
            role="user",
            parts=[
                # --------------------------------------
                # 1) QWEN 할루시네이션
                # --------------------------------------
                types.Part(
                    inline_data=types.Blob(
                        data=img_to_bytes(qwen_image),
                        mime_type="image/png"
                    )
                ),

                # --------------------------------------
                # 2) 원본 이미지 또는 원본에 스켈레톤 렌더링된 이미지
                # --------------------------------------
                types.Part(
                    inline_data=types.Blob(
                        data=img_to_bytes(ref_image),
                        mime_type="image/png"
                    )
                ),


                # --------------------------------------
                # TEXT PROMPT (사용자 의도 명시)
                # --------------------------------------
                types.Part(
                    text= prompt
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(image_size="1K"),
    )

    try:
        file_name = None
        file_extension = None
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                not chunk.candidates
                or not chunk.candidates[0].content
                or not chunk.candidates[0].content.parts
            ):
                continue

            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                file_name = f"result_{qwen_number}"
                # file_index += 1
                data_buffer = part.inline_data.data
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type)
                output_path = os.path.join(output_dir, f"{file_name}{file_extension}")
                save_binary_file(output_path, data_buffer)

            else:
                print(chunk.text)
        
        if file_name is None or file_extension is None:
            raise RuntimeError("❌ 이미지 생성에 실패했습니다. API 응답에 이미지 데이터가 포함되지 않았습니다.")
        
        return os.path.join(output_dir, f"{file_name}{file_extension}")
    
    except types.ClientError as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "authentication" in error_msg.lower():
            raise ValueError(
                f"❌ API 키 인증 실패: {error_msg}. "
                "GEMINI_API_KEY 환경 변수가 올바른지 확인하세요."
            ) from e
        elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            raise RuntimeError(
                f"❌ API 할당량 초과 또는 요청 제한: {error_msg}. "
                "잠시 후 다시 시도하세요."
            ) from e
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            raise PermissionError(
                f"❌ API 접근 권한 없음: {error_msg}. "
                "API 키에 필요한 권한이 있는지 확인하세요."
            ) from e
        else:
            raise RuntimeError(f"❌ Gemini API 클라이언트 오류: {error_msg}") from e
    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        if "network" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            raise ConnectionError(
                f"❌ 네트워크 오류 ({error_type}): {error_msg}. "
                "인터넷 연결을 확인하고 다시 시도하세요."
            ) from e
        elif "invalid" in error_msg.lower() or "bad request" in error_msg.lower():
            raise ValueError(
                f"❌ 잘못된 요청 파라미터 ({error_type}): {error_msg}. "
                "입력 이미지나 프롬프트를 확인하세요."
            ) from e
        else:
            raise RuntimeError(
                f"❌ 예상치 못한 오류 발생 ({error_type}): {error_msg}"
            ) from e

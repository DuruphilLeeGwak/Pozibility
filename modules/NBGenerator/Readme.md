# NBGenerator

NBGenerator는 Qwen-Image-Edit 모듈에서 생성된 1차 결과 이미지의 할루시네이션(Hallucination) 문제를 수정하는 모듈입니다. Gemini API를 활용하여 포즈 전송(Pose Transfer) 및 이미지 편집을 수행합니다.

## 📁 프로젝트 구조

```
NBGenerator/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 패키지 의존성
├── .env                    # 환경 변수 (GEMINI_API_KEY)
├── output/                 # 생성된 결과 이미지 저장 폴더
├── test_data/              # 테스트용 데이터
│
├── prompt/                 # 프롬프트 생성 모듈
│   └── prompt_generator.py # 할루시네이션 타입에 따른 프롬프트 생성
│
├── generate/               # 이미지 생성 모듈
│   └── generate.py         # Gemini API를 통한 이미지 생성
│
└── utils/                  # 유틸리티 함수
    └── utils.py            # 이미지 처리 및 파일 저장 유틸리티
```

## 📦 각 폴더 설명

### `prompt/`
- **목적**: 할루시네이션 타입에 따라 적절한 프롬프트를 생성
- **주요 함수**: `generate_prompt(hallucination_type)`
  - 할루시네이션 타입 번호를 입력받아 Gemini API에 전달할 프롬프트 문자열 생성
  - 9가지 할루시네이션 타입 지원 (1~9번)
  - 특수 케이스: 타입 1 또는 2만 선택된 경우 포즈 전송 전용 프롬프트 생성

### `generate/`
- **목적**: Gemini API를 통해 최종 이미지 생성
- **주요 함수**: `generate_nanobanana(qwen_image, ref_image, prefix, prompt)`
  - Qwen 결과 이미지와 참조 이미지를 입력받아 Gemini API로 이미지 생성
  - `.env` 파일에서 `GEMINI_API_KEY` 로드
  - 생성된 이미지를 `output/` 폴더에 저장
  - 다양한 API 오류에 대한 예외 처리 포함

### `utils/`
- **목적**: 공통 유틸리티 함수 제공
- **주요 함수**:
  - `img_to_bytes(img)`: PIL Image 객체 또는 파일 경로를 PNG 바이트 데이터로 변환
  - `save_binary_file(file_name, data)`: 바이너리 데이터를 파일로 저장
  - `im_show(img_path)`: 이미지 파일을 시각화

## 🚀 설치 방법

1. **저장소 클론**
```bash
git clone <repository-url>
cd NBGenerator
```

2. **패키지 설치**
```bash
pip install -r requirements.txt
```

3. **환경 변수 설정**
프로젝트 루트에 `.env` 파일을 생성하고 Gemini API 키를 설정합니다:
```
GEMINI_API_KEY=your-gemini-api-key-here
```

## 💻 사용 방법

### 기본 사용법

```bash
python main.py \
  --qwen_result <qwen_result_image_path> \
  --img_original <original_image_path> \
  --img_rendered <rendered_keypoint_image_path> \
  --hallucination_type <hallucination_type_numbers>
```

### 인자 설명

- `--qwen_result` (필수): Qwen-Image-Edit 모듈에서 생성된 1차 결과 이미지 경로
- `--img_original` (필수): 원본 이미지 경로
- `--img_rendered` (필수): 렌더링된 키포인트 이미지 경로 (스켈레톤 오버레이)
- `--hallucination_type` (필수): 수정할 할루시네이션 타입 번호 (쉼표로 구분, 예: "1,2" 또는 "2,4")

### 할루시네이션 타입

| 번호 | 설명 |
|------|------|
| 1 | 단순히 입력한 이미지를 그대로 반환 |
| 2 | 포즈가 원하는 대로 반영되지 않음 |
| 3 | 얼굴이 원하는 대로 반영되지 않음 |
| 4 | 의상이 다름 |
| 5 | 인물 비율이 원본과 다름 |
| 6 | 뒷배경이 변형됨 |
| 7 | 전혀 다른 인물을 생성함 |
| 8 | 원근과 구도가 변형됨 |
| 9 | 전혀 상관없는 물건 또는 기재도구가 포함됨 |

### 사용 예시

```bash
# 예시 1: 비율 문제 수정
python main.py \
  --qwen_result test_data/qwen_001.png \
  --img_original test_data/ref_full_001.jpg \
  --img_rendered test_data/ref_full_001_overlay.png \
  --hallucination_type "5"

# 예시 2: 포즈 문제 수정 (rendered_image 사용)
python main.py \
  --qwen_result test_data/qwen_002.png \
  --img_original test_data/ref_full_002.png \
  --img_rendered test_data/ref_full_002_overlay.png \
  --hallucination_type "1"

# 예시 3: 여러 문제 동시 수정
python main.py \
  --qwen_result test_data/qwen_003.png \
  --img_original test_data/ref_full_003.jpg \
  --img_rendered test_data/ref_full_003_overlay.png \
  --hallucination_type "3,4,9"
```

## 🔄 모듈 동작 방법

### 1. 입력 처리 (`main.py`)
1. 명령줄 인자 파싱 (argparse)
2. `hallucination_type` 파싱 및 검증
3. 할루시네이션 타입에 따른 참조 이미지 선택:
   - 타입 1 또는 2 포함 → `img_rendered` 사용
   - 그 외 → `img_original` 사용

### 2. 프롬프트 생성 (`prompt/prompt_generator.py`)
1. `hallucination_type` 문자열을 정수 리스트로 변환
2. 특수 케이스 처리:
   - 타입 1 또는 2만 포함된 경우: 포즈 전송 전용 프롬프트 반환
3. 일반 케이스 처리:
   - 선택된 할루시네이션 타입을 영어 문자열로 변환
   - 포즈 전송 프롬프트 생성
   - 타입 9 선택 시 불필요한 객체 제거 문구 추가

### 3. 이미지 생성 (`generate/generate.py`)
1. `.env` 파일에서 `GEMINI_API_KEY` 로드
2. `qwen_image` 경로에서 파일명의 마지막 `_` 뒤 숫자 추출 (예: `qwen_001.png` → `001`)
3. Gemini API 클라이언트 초기화
4. API 요청 구성:
   - Image 1: Qwen 결과 이미지 (할루시네이션이 포함된 이미지)
   - Image 2: 참조 이미지 (원본 또는 렌더링된 키포인트 이미지)
   - Text: 생성된 프롬프트
5. 스트리밍 방식으로 응답 수신
6. 생성된 이미지를 `output/result_{숫자}.{확장자}` 형식으로 저장

### 4. 출력
- 생성된 이미지는 `output/` 폴더에 저장됩니다
- 파일명 형식: `result_{qwen_image의 마지막 _ 뒤 숫자}.jpg`
  - 예: `qwen_001.png` → `output/result_001.jpg`

## ⚠️ 주의사항

1. **API 키**: `.env` 파일에 유효한 `GEMINI_API_KEY`가 설정되어 있어야 합니다
2. **이미지 경로**: 모든 이미지 경로는 유효한 파일을 가리켜야 합니다
3. **할루시네이션 타입**: 1~9 사이의 숫자만 입력 가능하며, 쉼표로 구분합니다
4. **출력 폴더**: `output/` 폴더는 자동으로 생성되지만, 기존 파일은 덮어씌워질 수 있습니다

## 🐛 에러 처리

모듈은 다음과 같은 에러를 처리합니다:
- API 키 누락 또는 잘못된 키
- 네트워크 연결 오류
- API 할당량 초과
- 잘못된 요청 파라미터
- 이미지 생성 실패



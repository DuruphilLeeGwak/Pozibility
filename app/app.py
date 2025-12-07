import streamlit as st
import subprocess
import time
import os
from PIL import Image

st.set_page_config(layout="wide")

hallucination_type = None

# -----------------------
# 초기 페이지 상태 설정
# -----------------------
if "page" not in st.session_state:
    st.session_state["page"] = "upload"


# -----------------------
# 페이지 이동 함수
# -----------------------
def go_to(page_name):
    st.session_state["page"] = page_name
    st.rerun()


# ============================================================
# 1. 업로드 페이지
# ============================================================
if st.session_state["page"] == "upload":
    st.title("이미지 업로드")

    col_src, col_pose = st.columns(2, gap="large")

    with col_src:
        src_img = st.file_uploader("당신의 사진 업로드", type=["png","jpg","jpeg"], key="src_img")
        if src_img is not None:
            st.image(src_img, caption="Src_img 미리보기", width=400)

    with col_pose:
        pose_img = st.file_uploader("포즈 사진 업로드", type=["png","jpg","jpeg"], key="pose_ref")
        if pose_img is not None:
            st.image(pose_img, caption="Pose_ref 미리보기", width=400)

    space, run_col = st.columns([6,1])
    run = run_col.button("RUN", use_container_width=True)

    if run:
        if not src_img or not pose_img:
            st.warning("두 이미지를 모두 업로드해주세요.")
        else:
            # 이미지 저장
            save_dir = "../data/input"
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "pose_ref.png"), "wb") as f:
                f.write(pose_img.getvalue())

            with open(os.path.join(save_dir, "src_img.png"), "wb") as f:
                f.write(src_img.getvalue())

            # bash 실행 페이지로 이동
            go_to("running")


# ============================================================
# 2. 실행 중 페이지 (로딩 + bash 실행)
# ============================================================
elif st.session_state["page"] == "running":

    st.title("처리 중…")
    st.write("모델이 실행되는 동안 잠시만 기다려주세요.")

    # 로딩 애니메이션
    with st.spinner("포즈 변환 중입니다...\n롤 한 판 하고 오세요.\n알림은 안 드립니다."):
        # 실제 bash 실행
        process = subprocess.run(
            ["bash", "pose_to_qwen.sh"],
            capture_output=True,
            text=True
        )

        # 실행 완료 → 결과 페이지로 이동
        go_to("result")


# ============================================================
# 3. 결과 페이지
# ============================================================
elif st.session_state["page"] == "result":

    OUTPUT_DIR = "../data/qwen_outputs"

    st.title("Qwen 결과 이미지")

    # 2-컬럼 프레임 구성
    col_img, col_opts = st.columns([3, 1])  # 왼쪽: 넓게, 오른쪽: 옵션

    # -------------------------
    # 왼쪽: 이미지 표시
    # -------------------------
    with col_img:
        if not os.path.exists(OUTPUT_DIR):
            st.error(f"폴더가 없습니다: {os.path.abspath(OUTPUT_DIR)}")
        else:
            exts = (".png", ".jpg", ".jpeg", ".webp")
            files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(exts)]
            files.sort()

            if len(files) == 0:
                st.warning("이미지가 없습니다.")
            else:
                # 최신 이미지 1장만 표시 (원하면 전체 그리드로 변경 가능)
                latest_img_path = os.path.join(OUTPUT_DIR, files[-1])
                img = Image.open(latest_img_path)

                st.image(img, caption=files[-1], width=700)


    # -------------------------
    # 오른쪽: 할루시네이션 옵션 UI
    # -------------------------
    with col_opts:
        st.subheader("할루시네이션 유형 선택")

        hallucination_types = {
                1: "simply returned original image",
                2: "pose",
                3: "facial identity",
                4: "outfit",
                5: "proportion",
                6: "background",
                7: "person",
                8: "perspective or depth",
                9: "object",
        }

        choice_label = st.radio(
            "할루시네이션 유형",
            list(hallucination_types.values()),
            horizontal=False
        )
        for k, v in hallucination_types.items():
            if v == choice_label:
                st.session_state.selected_hallu = k
                break
    space, run_col = st.columns([6,1])
    run = run_col.button("FIX", use_container_width=True)

    if run:
        go_to("running2")

elif st.session_state["page"] == "running2":

    st.title("처리 중…")
    st.write("모델이 실행되는 동안 잠시만 기다려주세요.")

    selected = st.session_state.get("selected_hallu", None)

    # 로딩 애니메이션
    with st.spinner("수정 중입니다...\n죄송합니다.\n롤 한 판만 더 하고 오시지요."):
        # 실제 bash 실행
        process = subprocess.run(
            ["bash", "nanobanana.sh", str(selected)],
            capture_output=True,
            text=True
        )

        # 실행 완료 → 결과 페이지로 이동
        go_to("result2")

elif st.session_state["page"] == "result2":

    OUTPUT_DIR = "../data/outputs"

    st.title("Nano Banana 결과 이미지")

    # 2-컬럼 프레임 구성
    col_img, col_opts = st.columns([3, 1])  # 왼쪽: 넓게, 오른쪽: 옵션

    # -------------------------
    # 왼쪽: 이미지 표시
    # -------------------------
    with col_img:
        if not os.path.exists(OUTPUT_DIR):
            st.error(f"폴더가 없습니다: {os.path.abspath(OUTPUT_DIR)}")
        else:
            exts = (".png", ".jpg", ".jpeg", ".webp")
            files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(exts)]
            files.sort()

            if len(files) == 0:
                st.warning("이미지가 없습니다.")
            else:
                # 최신 이미지 1장만 표시 (원하면 전체 그리드로 변경 가능)
                latest_img_path = os.path.join(OUTPUT_DIR, files[-1])
                img = Image.open(latest_img_path)

                st.image(img, caption=files[-1], width=700)
import streamlit as st
import subprocess
import time
import os
import io
from PIL import Image

st.set_page_config(layout="wide")

hallucination_type = None

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #7C98B3;
        color: white;
        border-radius: 8px;
        height: 48px;
        width: 100%;
        border: none;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #7C98B3;
        border: 2px solid #81F495;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state["page"] = "upload"

def go_to(page_name):
    st.session_state["page"] = page_name
    st.rerun()

# Upload
if st.session_state["page"] == "upload":
    _, col_main, _ = st.columns([1, 6, 1])
    with col_main:
        st.title("POZibilty")

        space, run_col = st.columns([6,1])
        st.markdown("---")
        run = run_col.button("RUN", use_container_width=True)

        col_src, _, col_pose = st.columns([10, 2, 10])

        with col_src:
            st.markdown("## Your Picture")
            src_img = st.file_uploader("", type=["png","jpg","jpeg"], key="src_img")
            _, sub_col, _ = st.columns([1, 2, 1])
            with sub_col:
                if src_img is not None:
                    st.image(src_img, caption="", width="content")

        with col_pose:
            st.markdown("## Your Pose")
            pose_img = st.file_uploader("", type=["png","jpg","jpeg"], key="pose_ref")
            _, sub_col, _ = st.columns([1, 2, 1])
            with sub_col:
                if pose_img is not None:
                    st.image(pose_img, caption="", width="content")

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

    with st.spinner("포즈 변환 중입니다...\n롤 한 판 하고 오세요."):
        process = subprocess.run(
            ["bash", "pose_to_qwen.sh"],
            capture_output=True,
            text=True
        )
        #time.sleep(12)

        go_to("result")


# ============================================================
# 3. 결과 페이지
# ============================================================
elif st.session_state["page"] == "result":

    OUTPUT_DIR = "../data/qwen_outputs"

    _, col_main, _ = st.columns([1, 6, 1])
    with col_main:
        st.markdown("# Result")
        st.markdown("---")

        # 2-컬럼 프레임 구성
        col_img, col_opts = st.columns([3, 1.2])  # 왼쪽: 넓게, 오른쪽: 옵션

        # -------------------------
        # 왼쪽: 이미지 표시
        # -------------------------
        with col_img:
            _, sub_col, _ = st.columns([1,2,1])
            with sub_col:
                if not os.path.exists(OUTPUT_DIR):
                    st.error(f"폴더가 없습니다: {os.path.abspath(OUTPUT_DIR)}")
                else:
                    exts = (".png", ".jpg", ".jpeg", ".webp")
                    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(exts)]
                    files.sort()

                    if len(files) == 0:
                        st.warning("이미지가 없습니다.")
                    else:
                        latest_img_path = os.path.join(OUTPUT_DIR, files[-1])
                        img = Image.open(latest_img_path)
                        st.session_state.qwen_output = img

                        st.image(img, caption="", width="content")


    # -------------------------
    # 오른쪽: 할루시네이션 옵션 UI
    # -------------------------
    with col_opts:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_img = buf.getvalue()
        download=st.download_button(
            label="Download",
            data=byte_img,
            file_name="qwen_output.png",
            mime="image/png"
        )
        expander = st.expander(
            "1차로 생성한 이미지에서\n\n어떠한 점이 마음에 안 드시나요?"
        )

        hallucination_types = {
                1: "원본 이미지를 그대로 리턴했어요",
                2: "포즈가 반영이 안됐어요",
                3: "얼굴이 변형됐어요",
                4: "의상이 바뀌었어요",
                5: "비율이 변형됐어요",
                6: "배경이 바뀌었어요",
                7: "완전히 다른 사람이 생성됐어요",
                8: "원근 또는 구도가 변형됐어요",
                9: "전혀 상관없는 물건이 생성됐어요",
        }

        choice_label = expander.radio(
            "할루시네이션 유형",
            list(hallucination_types.values()),
            horizontal=False
        )
        for k, v in hallucination_types.items():
            if v == choice_label:
                st.session_state.selected_hallu = k
                break
        space, run_col = expander.columns([2,1])
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
        #time.sleep(8)

        # 실행 완료 → 결과 페이지로 이동
        go_to("result2")

elif st.session_state["page"] == "result2":

    OUTPUT_DIR = "../data/outputs"
    _, col_main, _ = st.columns([1, 6, 1])
    with col_main:
        # 2-컬럼 프레임 구성
        st.markdown("## Nano Banana Result")
        st.markdown("---")
        col_nb, col_bt = st.columns([8, 1])
        with col_nb:
            _, col_img, _ = st.columns([1, 2, 1])
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
                        nb = Image.open(latest_img_path)

                        st.image(nb, caption="", width="content")
        with col_bt:
            buf = io.BytesIO()
            nb.save(buf, format="PNG")
            byte_img = buf.getvalue()
            download=st.download_button(
                label="Download",
                data=byte_img,
                file_name="output.png",
                mime="image/png"
            )
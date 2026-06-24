import base64
import streamlit as st
from langchain_openai import ChatOpenAI


def init_page():
    st.set_page_config(page_title="Image Recognizer", page_icon="🤗")
    st.header("Image Recognizer 🤗")
    st.sidebar.title("Options")


def main():
    init_page()

    llm = ChatOpenAI(
            model="unsloth/gemma-4-E2B-it-GGUF",
            base_url="http://172.30.192.1:12345/v1",  # LM Studio v1 엔드포인트 유지
            api_key="lm-studio",
            temperature=0.7
        )


    uploaded_file = st.file_uploader(
        label="이미지를 업로드해 주세요😇",
        # LLM이 처리 가능한 이미지 파일만 허용
        type=["png", "jpg", "webp", "gif"],
    )

    if uploaded_file:
        if user_input := st.chat_input("궁금한 내용을 입력해 주세요!"):
            # 읽어온 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "auto"},
                        },
                    ],
                )
            ]
            st.markdown("### 질문")
            st.write(user_input)  # 사용자의 질문
            st.image(uploaded_file)  # 업로드한 이미지 표시
            st.markdown("### 답변")
            st.write_stream(llm.stream(query))

    else:
        st.write("먼저 이미지를 업로드해 주세요😇")


if __name__ == "__main__":
    main()
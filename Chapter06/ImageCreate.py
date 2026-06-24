import base64
import requests  # OpenAI 대신 표준 REST API 통신을 위해 사용
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# LM Studio 서버 설정 정보 기본 정의
LM_STUDIO_BASE_URL = "http://172.30.192.1:12345/v1"
MODEL_NAME = "unsloth/gemma-4-E2B-it-GGUF"

IMAGE_PROMPT_TEMPLATE = """
먼저, 아래의 사용자의 요청과 업로드된 이미지를 주의 깊게 읽어주세요.
다음으로, 업로드된 이미지를 기반으로 이미지를 생성해 달라는 사용자의 요청에 따라
이미지 생성용 프롬프트를 작성해 주세요.
프롬프트는 반드시 영어로 작성해야 합니다.

주의: 이미지 속 사람이나 특정 장소, 랜드마크, 상표 등을 식별하지 말아 주세요.
묘사는 사진 속 시각적 요소를 중립적으로 설명하는 방식으로 해주세요.

사용자 입력: {user_input}

프롬프트에서는 사용자가 업로드한 사진에 무엇이 담겨 있는지,
어떻게 구성되어 있는지를 설명해 주세요.
사진의 구도와 줌 정도도 설명해 주세요.
사진의 내용을 재현하는 것이 중요합니다.

이미지 생성용 프롬프트를 영어로 출력해 주세요:
"""

def generate_image(prompt: str) -> str:
    """
    로컬 이미지 생성 모델로 이미지를 생성하고 base64 문자열을 반환.
    ※ 주의: LM Studio 자체는 이미지 생성(DALL-E 기능)을 지원하지 않으므로, 
    만약 로컬 Stable Diffusion API 등을 연동한다면 이 부분을 해당 API 주소로 수정해야 합니다.
    현재는 에러 방지를 위해 표준 OpenAI 호환 포맷 규격으로 REST API 요청을 보내도록 처리했습니다.
    """
    headers = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
    
    try:
        # LM Studio 주소의 이미지 생성 엔드포인트 호출 시도
        response = requests.post(f"{LM_STUDIO_BASE_URL}/images/generations", json=payload, headers=headers)
        response_data = response.json()
        return response_data["data"][0]["b64_json"]
    except Exception as e:
        st.error(f"이미지 생성 중 오류 발생 (LM Studio가 이미지 생성 API를 지원하는지 확인 필요): {e}")
        return ""

# RunnableLambda로 감싸서 LCEL 체인에서 사용 가능하게 만든다
image_generator = RunnableLambda(generate_image)


def init_page():
    st.set_page_config(page_title="Image Converter", page_icon="🎨")
    st.header("Image Converter 🎨")


def main():
    init_page()

    # ChatOpenAI 라이브러리를 사용하지만, 내부 주소는 완벽하게 LM Studio를 바라보도록 설정
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        temperature=0.7
    )

    generated_image_base64 = None

    uploaded_file = st.file_uploader(
        label="이미지를 업로드해 주세요 📷",
        type=["png", "jpg", "webp", "gif"],
    )

    # 세션 상태에 user_input 유지하기 위해 초기화
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    if uploaded_file:
        if user_input := st.chat_input("이미지를 어떻게 가공할지 알려주세요!"):
            st.session_state.current_input = user_input
            
            # 읽은 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": IMAGE_PROMPT_TEMPLATE.format(user_input=user_input),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "auto"},
                        },
                    ],
                )
            ]

            # LLM에게 이미지 생성용 프롬프트를 작성하게 함
            st.markdown("### Image Prompt")
            image_prompt = st.write_stream(llm.stream(query))

            # 이미지 생성 수행
            with st.spinner("로컬 이미지 모델이 그림을 그리는 중입니다..."):
                generated_image_base64 = image_generator.invoke(image_prompt)
    else:
        st.write("먼저 이미지를 업로드해 주세요 📷")

    # 생성된 이미지 표시
    if generated_image_base64:
        st.markdown("### Question")
        st.write(st.session_state.current_input)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("### Generated Image")
        
        # base64를 디코딩하여 이미지로 표시
        image_bytes = base64.b64decode(generated_image_base64)
        st.image(image_bytes, use_container_width=True)


if __name__ == "__main__":
    main()
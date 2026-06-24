import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 누락되었던 파서 추가

def main():
    st.set_page_config(page_title="My Great Chat")
    st.header("My Great Chat")
    # 1. 채팅 이력 초기화
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 2. LLM 모델 설정
    # temperature: 0~1까지 큰값일수록 무작위 성이 높아짐.
    llm = ChatOpenAI(
        model="unsloth/gemma-4-E2B-it-GGUF",
        base_url="http://172.30.192.1:12345/v1",  # LM Studio v1 엔드포인트 유지
        api_key="lm-studio",
        temperature=0.7
    )

    # 3️. LLM에 전달할 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트 입니다."),
            ("placeholder", "{history}"),
            ("user", "{user_input}"),
        ]
    )

    # LLM 응답을 텍스트로 변환해주는 파서
    output_parser = StrOutputParser()

    # 4. 사용자 질문을 ChatGPT로 전달해 응답을 받는 체인을 생성
    chain = prompt | llm | output_parser

    # 5️. 사용자 입력 처리
    if user_input := st.chat_input("궁금한 것을 입력해 주세요"):
        with st.spinner("Chat이 답변중..."):
            response = chain.invoke({
                "history": st.session_state.message_history,
                "user_input": user_input,
            })

        st.session_state.message_history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.message_history.append(
            {"role": "assistant", "content": response}
        )
        
        # 6. 대화 이력 출력
        for msg in st.session_state.get("message_history", []): 
            st.chat_message(msg["role"]).markdown(msg["content"])

if __name__ == "__main__":
    main()
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 누락되었던 파서 추가

from dotenv import load_dotenv
load_dotenv()
def main():
    
    st.set_page_config(page_title="My Great Chat")
    st.header("My Great Chat")
    # 1. 채팅 이력 초기화
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 2. LLM 모델 설정
    llm = ChatOpenAI()

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
    
    user_input="안녕 GPT"
    # 5️. 사용자 입력 처리
    response = chain.invoke({
                "user_input": user_input
            })
    print(response)

if __name__ == "__main__":
    main()
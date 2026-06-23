import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.output_parsers import StrOutputParser

def main():
    st.set_page_config(page_title="My Greate Chat")
    st.header("My Great Chat")

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    llm = ChatOpenAI(
        model="unsloth/gemma-4-E2B-it-GGUF",
        base_url="http://172.30.192.1:12345",
        api_key="lm-studio",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트 입니다."),
            MessagePlaceholder(variable_name="history"),
            ("user", "{user_input}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    if user_input := st.chat_input("궁금한 것을 입력해 주세요"):
        with st.spinner("Chat이 답변중..."):
            response = chain.invoke({
                "history": st.session_state.message_history,
                "user_input": user_input,
            })

        st.session_state.message_history.append(
            {"role":"user", "content": user_input}
        )

        st.session_state.message_history.append({"role":"assistant", "content":response})

        for msg in st.session_state.get("message_history", []): st.chat_message(msg["role"]).markdown(msg["content"])

if __name__ == "__main__":
    main()

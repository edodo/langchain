import fitz
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def init_page():
    st.set_page_config(page_title="Upload PDF(s)", page_icon="📄")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("DB 초기화", key="clear")
    if clear_button and "vectorstore" in st.session_state:
        del st.session_state.vectorstore


def get_pdf_text():
    # file_uploader로 PDF를 업로드한다
    pdf_file = st.file_uploader(
        label="PDF를 업로드하세요 😇", type="pdf"  # PDF 파일만 업로드 가능
    )
    if pdf_file:
        pdf_text = ""
        with st.spinner("PDF 로딩 중 ..."):
            # PyMuPDF로 PDF를 읽어들인다
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # 500자 단위로 분할
            chunk_overlap=50,     # 문맥이 끊기는 것을 방지하기 위해 50자 중첩 (0보다 권장)
            length_function=len,
        )
        return text_splitter.split_text(pdf_text)
    else:
        return None


def build_vector_store(pdf_text):
    with st.spinner("벡터 스토어 저장 중 ..."):
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            # 벡터 DB 초기화와 문서 추가를 동시에 수행
            embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
            st.session_state.vectorstore = FAISS.from_texts(pdf_text, embeddings)


def page_pdf_upload_and_build_vector_db():
    st.title("PDF 업로드 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)


def main():
    init_page()
    init_messages()
    page_pdf_upload_and_build_vector_db()


if __name__ == "__main__":
    main()
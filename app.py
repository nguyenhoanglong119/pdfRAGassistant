import streamlit as st
from models import load_llm
from utils import process_pdf
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="wide")
st.title("🤖 PDF RAG Assistant")

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

    if uploaded_file:
        with st.spinner("🔄 Đang xử lý PDF..."):
            rag_chain = process_pdf(uploaded_file, st.session_state.embeddings)
            st.session_state.rag_chain = rag_chain
        st.success("✅ Xử lý xong!")

st.markdown("### 💬 Chat với tài liệu PDF")
if st.session_state.rag_chain:
    user_question = st.text_input("Nhập câu hỏi:")
    if user_question:
        with st.spinner("🤔 Đang tìm câu trả lời..."):
            response = st.session_state.rag_chain.invoke(user_question)
            answer = response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()
            st.success("✅ Trả lời:")
            st.write(answer)
else:
    st.info("⏳ Vui lòng upload PDF để bắt đầu.")

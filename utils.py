import tempfile, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

def process_pdf(file, embedding_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = SemanticChunker(
        embeddings=embedding_model,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = splitter.split_documents(documents)
    os.remove(tmp_path)

    vector_db = Chroma.from_documents(docs, embedding=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)[:1200]

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    return rag_chain

# 🤖 PDF RAG Assistant

Ứng dụng giúp bạn **chat với nội dung của file PDF** bằng tiếng Việt, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp mô hình ngôn ngữ lớn (LLM).  
Phù hợp để học tập, tra cứu tài liệu, hoặc hỗ trợ đọc hiểu tự động.

---

## 🧠 Tính năng

- Tải và xử lý file PDF
- Semantic Chunking văn bản với HuggingFace Embedding
- Tạo vector database với Chroma
- Truy xuất thông tin và sinh câu trả lời bằng mô hình LLM (`Flan-T5-Base`)
- Giao diện đơn giản bằng Streamlit

---

## 🏗️ Kiến trúc chính

```bash
User
 │
 ▼
[Streamlit UI] ──> [RAG Pipeline]
                     ├─ Vector DB (Chroma)
                     ├─ Semantic Chunking (LangChain)
                     └─ LLM (HuggingFace Pipeline)


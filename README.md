# 🤖 RAGChatBot – Chat with Your PDFs Using GenAI

RAGChatBot is an intelligent document-based chatbot built using the Retrieval-Augmented Generation (RAG) architecture. It allows users to upload any PDF and ask questions, with answers generated using state-of-the-art large language models and contextual retrieval from the uploaded documents.

## 🔍 Features

- 📄 Upload and parse PDF documents
- 🧠 Uses Google's Gemini or Groq for natural language generation
- 🗂️ Text chunking and embedding via LangChain
- ⚡ Real-time Q&A with context-aware responses
- 🧱 Clean UI using Streamlit
- 🧾 Uses `unstructured.io` to handle complex PDFs robustly

---

## 🛠️ Tech Stack

| Layer             | Tool/Library                      |
|------------------|------------------------------------|
| Interface         | Streamlit                         |
| PDF Parsing       | Unstructured                      |
| Text Embedding    | LangChain + Google Embeddings     |
| Vector Storage    | In-Memory / FAISS (customizable)  |
| LLM Integration   | Google Gemini / Groq              |
| Environment Setup | Python + dotenv                   |

---

## 🚀 Getting Started

### 1. Clone the Repository

bash
git clone https://github.com/310511/RAGCHATBOT.git
cd RAGCHATBOT

2. Set up a Virtual Environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Requirements
   
pip install -r requirements.txt

4. Configure Environment Variables
   
Create a .env file and add your API keys:

GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

5. Run the Application
   
streamlit run app.py



🧠 How It Works

User uploads a PDF.

The file is parsed using unstructured.partition.pdf.

Text is chunked using LangChain's RecursiveCharacterTextSplitter.

Each chunk is embedded and stored in a vector store.

User inputs a query → relevant chunks retrieved → sent to LLM.

LLM generates an accurate and context-aware response.



🧩 Future Improvements

Support for multiple document types (Word, HTML)

Chat history and session memory

Option to switch between LLM providers (Gemini, OpenAI, etc.)

Integration with FAISS or ChromaDB for persistent vector storage



🙌 Acknowledgements

Unstructured.io

LangChain

Streamlit

Google Generative AI


📬 Contact
Developed during an internship by Utsav Gautam.
Feel free to raise issues or contribute to enhance the chatbot further!

# 📚 AI Study Assistant
### AI-Powered PDF Learning Assistant using RAG, LangChain & Google Gemini

An intelligent study assistant that transforms PDF documents into interactive learning material. Users can upload lecture notes, textbooks, or research papers and instantly generate summaries, ask context-aware questions, create quizzes, and build flashcards using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🔐 User accounts (sign up / log in) with bcrypt-hashed passwords
- 📄 Upload and process PDF documents
- 🤖 Context-aware Question Answering using Retrieval-Augmented Generation (RAG)
- 🎙️ Voice-based interaction — ask questions by microphone, hear answers read aloud
- 💾 Persistent per-user chat history (saved to SQLite, survives logout/restart)
- 📝 Automatic document summarization
- ❓ AI-generated Multiple Choice Quizzes
- 🎴 AI-generated Flashcards for revision
- 🔍 Semantic document search with vector embeddings
- 📑 Page-level source citations for generated answers
- 📤 Export summaries, quizzes, and flashcards in TXT or JSON format
- ⚡ Interactive Streamlit web interface

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Framework | LangChain |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Auth & History | SQLite + bcrypt |
| Voice I/O | streamlit-mic-recorder (speech-to-text) + gTTS (text-to-speech) |
| Language | Python |

---

## 🏗️ Architecture

```
               PDF Upload
                    │
                    ▼
         PDF Text Extraction (PyPDF)
                    │
                    ▼
      Text Chunking (LangChain Splitter)
                    │
                    ▼
      HuggingFace Embeddings Generation
                    │
                    ▼
          ChromaDB Vector Storage
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
  RAG Chat      Summarizer     Quiz & Flashcards
      │             │             │
      └─────────────┼─────────────┘
                    ▼
             Streamlit Interface
```

---

## 📂 Project Structure

```
ai_study_assistant/
│
├── app.py
├── requirements.txt
├── .env.example
├── app_data.db          # SQLite: users + chat history (auto-created, gitignored)
│
├── src/
│   ├── config.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   ├── summarizer.py
│   ├── quiz_generator.py
│   ├── flashcard_generator.py
│   ├── db.py              # SQLite connection + schema
│   ├── auth.py            # registration, login, password hashing
│   ├── chat_history.py    # persistent per-user chat history
│   ├── voice.py           # text-to-speech (spoken answers)
│   ├── utils.py
│   └── __init__.py
│
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/ai-study-assistant.git

cd ai-study-assistant
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure API Key

Create a `.env` file.

```env
GOOGLE_API_KEY=your_api_key_here
```

Get a free API key from

https://aistudio.google.com/app/apikey

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser at

```
http://localhost:8501
```

> **Note on voice input:** browsers only allow microphone access on `localhost` or
> over HTTPS. Running locally (as above) works out of the box; if you deploy this
> behind a custom domain, make sure it's served over HTTPS or the mic button won't
> get permission from the browser.

---

## 👤 Accounts & chat history

On first launch, create an account from the **Sign up** tab (username, email,
password — min. 8 characters). Credentials are hashed with bcrypt and stored in a
local SQLite file (`app_data.db`, auto-created, already gitignored).

Every question you ask — typed or spoken — is saved against your account and the
active document, so it's still there next time you log in and re-process the same
PDF. Use the **"Clear chat history for this document"** button in the sidebar to
wipe it for a given file.

---

## 📸 Workflow

1. Sign up / log in
2. Upload a PDF
3. Process the document
4. Ask questions by typing or by voice — answers can be read aloud
5. Revisit past chat history any time (saved to your account)
6. Generate AI summary
7. Create quizzes
8. Generate flashcards
9. Export learning material

---

## 💡 Key Concepts Used

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Large Language Models (LLMs)
- Prompt Engineering
- Document Chunking
- Dense Retrieval
- Similarity Search

---

## 📈 Resume Highlights

- Built an end-to-end AI-powered study assistant using Retrieval-Augmented Generation (RAG).
- Implemented semantic document search using HuggingFace embeddings and ChromaDB.
- Integrated Google Gemini with LangChain for context-aware question answering.
- Developed automated summarization, quiz generation, and flashcard creation pipelines.
- Added user authentication (bcrypt-hashed credentials) and persistent per-user chat history backed by SQLite.
- Implemented voice-based interaction: browser microphone input transcribed via speech recognition, with spoken text-to-speech answers.
- Designed an interactive Streamlit application for real-time PDF-based learning.

---

## 🔮 Future Enhancements

- Multi-document support
- OCR support for scanned PDFs
- Cloud deployment (AWS/GCP/Azure)
- Learning progress dashboard
- Password reset / email verification
- Swap SQLite for Postgres for multi-instance deployments

---




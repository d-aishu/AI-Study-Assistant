"""
AI Study Assistant — RAG-powered PDF Q&A, summarization, quiz and
flashcard generation, with user accounts, persistent chat history, and
voice-based interaction.

Stack: Streamlit (UI) + LangChain (orchestration) + Gemini (LLM) +
ChromaDB (vector store) + Hugging Face sentence-transformers (embeddings) +
SQLite (users/history) + gTTS / streamlit-mic-recorder (voice).

Run with:  streamlit run app.py
"""
import streamlit as st
from streamlit_mic_recorder import speech_to_text

from src.config import Config
from src.pdf_processor import process_pdf, full_text_from_documents
from src.vector_store import build_vectorstore
from src.rag_chain import ask_question
from src.summarizer import summarize_documents
from src.quiz_generator import generate_quiz
from src.flashcard_generator import generate_flashcards
from src.utils import (
    make_collection_name,
    flashcards_to_text,
    quiz_to_text,
    flashcards_to_json,
    quiz_to_json,
    friendly_error_message,
)
from src.db import init_db
from src import auth
from src import chat_history as chat_db
from src.voice import text_to_speech

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")
init_db()

# ================================================================== AUTH
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


def render_login_signup():
    st.title("📚 AI Study Assistant")
    st.caption("Log in or create a free account to start studying.")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            if not username or not password:
                st.error("Enter both a username and password.")
            else:
                user = auth.authenticate_user(username, password)
                if user:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            error = auth.validate_registration(new_username, new_email, new_password, confirm_password)
            if error:
                st.error(error)
            else:
                try:
                    auth.register_user(new_username, new_email, new_password)
                    st.success("Account created! Switch to the 'Log in' tab to sign in.")
                except ValueError as e:
                    st.error(str(e))


if not st.session_state.auth_user:
    render_login_signup()
    st.stop()

current_user = st.session_state.auth_user  # {"id", "username", "email"}

# ---------------------------------------------------------------- session state
defaults = {
    "vectordb": None,
    "doc_chunks": None,
    "doc_name": None,
    "chat_history": [],
    "summary": None,
    "quiz": None,
    "flashcards": None,
    "tts_cache": {},  # index -> generated mp3 bytes, so replaying doesn't regenerate audio
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 AI Study Assistant")
    st.caption(f"👤 Logged in as **{current_user['username']}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.vectordb = None
        st.session_state.doc_chunks = None
        st.session_state.doc_name = None
        st.session_state.chat_history = []
        st.session_state.summary = None
        st.session_state.quiz = None
        st.session_state.flashcards = None
        st.rerun()

    st.divider()

    st.subheader("1. Gemini API Key")
    api_key_input = st.text_input(
        "Google Gemini API key",
        value=Config.GOOGLE_API_KEY,
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        Config.GOOGLE_API_KEY = api_key_input

    st.subheader("2. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a study PDF", type=["pdf"])

    process_clicked = st.button("Process document", type="primary", use_container_width=True)

    if process_clicked:
        if not Config.is_ready():
            st.error("Please enter your Gemini API key first.")
        elif not uploaded_file:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    chunks = process_pdf(file_bytes, uploaded_file.name)
                    collection_name = make_collection_name(uploaded_file.name, file_bytes)
                    vectordb = build_vectorstore(chunks, collection_name)

                    st.session_state.vectordb = vectordb
                    st.session_state.doc_chunks = chunks
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.summary = None
                    st.session_state.quiz = None
                    st.session_state.flashcards = None
                    st.session_state.tts_cache = {}

                    # Load this user's past chat history for this specific document
                    st.session_state.chat_history = chat_db.load_chat_history(
                        current_user["id"], document_name=uploaded_file.name
                    )

                    st.success(f"Processed '{uploaded_file.name}' into {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

    if st.session_state.doc_name:
        st.info(f"📄 Active document:\n**{st.session_state.doc_name}**")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history for this document", use_container_width=True):
                chat_db.clear_chat_history(current_user["id"], st.session_state.doc_name)
                st.session_state.chat_history = []
                st.session_state.tts_cache = {}
                st.rerun()

    st.divider()
    st.caption(
        "Embeddings: Hugging Face (local) · LLM: Gemini · "
        "Vector store: ChromaDB · Orchestration: LangChain"
    )

# ---------------------------------------------------------------- main area
st.title("AI Study Assistant")

if not st.session_state.vectordb:
    st.markdown(
        """
        👋 **Welcome!** To get started:
        1. Enter your free Gemini API key in the sidebar.
        2. Upload a PDF (lecture notes, textbook chapter, research paper).
        3. Click **Process document**.

        Then you'll be able to ask questions (by typing OR by voice), get a
        summary, and auto-generate a quiz and flashcards from the material.
        Your chat history is saved to your account automatically.
        """
    )
    st.stop()

tab_chat, tab_summary, tab_quiz, tab_flashcards = st.tabs(
    ["💬 Ask Questions", "📝 Summary", "🧠 Quiz", "🗂️ Flashcards"]
)


def render_chat_turn(idx, turn):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**Page {s['page']}** — {s['snippet']}")

        # Voice output: generate (once, cached) and play the answer aloud
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("🔊 Listen", key=f"listen_{idx}"):
                if idx not in st.session_state.tts_cache:
                    with st.spinner("Generating audio..."):
                        try:
                            st.session_state.tts_cache[idx] = text_to_speech(turn["answer"])
                        except Exception as e:
                            st.error(f"Couldn't generate audio: {e}")
        if idx in st.session_state.tts_cache:
            st.audio(st.session_state.tts_cache[idx], format="audio/mp3")


def handle_new_question(question: str):
    """Shared by both typed and voice-transcribed questions."""
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and thinking..."):
            try:
                result = ask_question(st.session_state.vectordb, question)
                st.write(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"**Page {s['page']}** — {s['snippet']}")

                turn = {"question": question, "answer": result["answer"], "sources": result["sources"]}
                st.session_state.chat_history.append(turn)

                # Persist to this user's permanent chat history
                chat_db.save_chat_turn(
                    current_user["id"],
                    st.session_state.doc_name,
                    question,
                    result["answer"],
                    result["sources"],
                )
            except Exception as e:
                st.error(f"Something went wrong: {friendly_error_message(e)}")


# ---------------------------------------------------------------- Chat / Q&A tab
with tab_chat:
    st.subheader("Ask questions about your document")
    st.caption("Type a question, or use the mic to ask out loud.")

    for i, turn in enumerate(st.session_state.chat_history):
        render_chat_turn(i, turn)

    col_mic, col_hint = st.columns([1, 4])
    with col_mic:
        voice_question = speech_to_text(
            language="en",
            start_prompt="🎤 Ask by voice",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="voice_stt",
        )
    with col_hint:
        st.caption("Voice questions use your browser's microphone and Google speech recognition.")

    if voice_question:
        handle_new_question(voice_question)

    question = st.chat_input("Ask something about the document...")
    if question:
        handle_new_question(question)

# ---------------------------------------------------------------- Summary tab
with tab_summary:
    st.subheader("Document summary")
    col1, col2 = st.columns([1, 3])
    with col1:
        style = st.radio("Summary style", ["detailed", "brief"], index=0)
        if st.button("Generate summary", use_container_width=True):
            with st.spinner("Summarizing document..."):
                try:
                    st.session_state.summary = summarize_documents(
                        st.session_state.doc_chunks, style=style
                    )
                except Exception as e:
                    st.error(f"Failed to summarize: {friendly_error_message(e)}")
    with col2:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            st.download_button(
                "Download summary (.txt)",
                st.session_state.summary,
                file_name="summary.txt",
            )
        else:
            st.caption("Click 'Generate summary' to summarize the whole document.")

# ---------------------------------------------------------------- Quiz tab
with tab_quiz:
    st.subheader("Auto-generated quiz")
    num_q = st.slider("Number of questions", 3, 15, 5)
    if st.button("Generate quiz"):
        with st.spinner("Writing quiz questions..."):
            try:
                st.session_state.quiz = generate_quiz(st.session_state.doc_chunks, num_questions=num_q)
                st.session_state["quiz_answers"] = {}
            except Exception as e:
                st.error(f"Failed to generate quiz: {friendly_error_message(e)}")

    if st.session_state.quiz:
        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. ({q.get('difficulty','medium')}) {q['question']}**")
                options = q["options"]
                labels = [f"{k}. {v}" for k, v in options.items()]
                choice = st.radio(
                    f"q_{i}", labels, key=f"quiz_radio_{i}", label_visibility="collapsed"
                )
                user_answers[i] = choice.split(".")[0].strip()
                st.write("")
            submitted = st.form_submit_button("Submit quiz")

        if submitted:
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                correct = q["correct_answer"]
                given = user_answers[i]
                is_correct = given == correct
                score += int(is_correct)
                icon = "✅" if is_correct else "❌"
                st.markdown(
                    f"{icon} **Q{i+1}:** your answer **{given}**, correct answer **{correct}**  \n"
                    f"_{q.get('explanation','')}_"
                )
            st.success(f"Score: {score} / {len(st.session_state.quiz)}")

        st.download_button(
            "Download quiz (.txt)",
            quiz_to_text(st.session_state.quiz),
            file_name="quiz.txt",
        )
        st.download_button(
            "Download quiz (.json)",
            quiz_to_json(st.session_state.quiz),
            file_name="quiz.json",
        )

# ---------------------------------------------------------------- Flashcards tab
with tab_flashcards:
    st.subheader("Flashcards")
    num_cards = st.slider("Number of flashcards", 5, 30, 10)
    if st.button("Generate flashcards"):
        with st.spinner("Creating flashcards..."):
            try:
                st.session_state.flashcards = generate_flashcards(
                    st.session_state.doc_chunks, num_cards=num_cards
                )
            except Exception as e:
                st.error(f"Failed to generate flashcards: {friendly_error_message(e)}")

    if st.session_state.flashcards:
        cols = st.columns(2)
        for i, card in enumerate(st.session_state.flashcards):
            with cols[i % 2]:
                with st.expander(f"🗂️ [{card.get('topic','General')}]  {card['front']}"):
                    st.write(card["back"])

        st.download_button(
            "Download flashcards (.txt)",
            flashcards_to_text(st.session_state.flashcards),
            file_name="flashcards.txt",
        )
        st.download_button(
            "Download flashcards (.json)",
            flashcards_to_json(st.session_state.flashcards),
            file_name="flashcards.json",
        )

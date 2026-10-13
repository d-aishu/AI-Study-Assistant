"""
Centralized configuration for the AI Study Assistant.
Reads settings from environment variables (loaded from a .env file if present)
so that API keys and tunable parameters never need to be hard-coded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- API keys ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # --- Models ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- Storage ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    # --- Chunking / retrieval ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 4))

    @classmethod
    def is_ready(cls) -> bool:
        """True once a Gemini API key has been supplied (via .env or the UI)."""
        return bool(cls.GOOGLE_API_KEY)

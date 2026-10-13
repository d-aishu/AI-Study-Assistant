"""
Wraps ChromaDB + Hugging Face sentence-transformer embeddings so the
rest of the app never has to touch embedding/vector-store internals.
"""
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from src.config import Config

_embeddings_cache = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the Hugging Face embedding model once and cache it for the
    process lifetime — loading it repeatedly is slow.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def build_vectorstore(
    documents: List[Document],
    collection_name: str,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Embed `documents` with Hugging Face embeddings and index them in a
    fresh Chroma collection. Each Streamlit session/document gets its
    own collection name so multiple uploads don't mix results.
    """
    persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vectordb


def load_vectorstore(
    collection_name: str,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Reload a previously persisted Chroma collection from disk."""
    persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def get_retriever(vectordb: Chroma, k: Optional[int] = None):
    """Return a similarity-search retriever with configurable top-k."""
    k = k or Config.RETRIEVER_K
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

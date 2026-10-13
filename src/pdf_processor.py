"""
Handles turning an uploaded PDF into clean text and then into
LangChain Document chunks ready for embedding.
"""
from typing import List
from io import BytesIO

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.config import Config


def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
    """
    Extract text page-by-page from a PDF and wrap each page in a
    LangChain Document, keeping page number + source filename as metadata.
    This per-page metadata later lets the assistant cite sources.
    """
    reader = PdfReader(BytesIO(file_bytes))
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_num},
            )
        )

    if not documents:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "It may be a scanned/image-only PDF that needs OCR."
        )

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level documents into smaller overlapping chunks so that
    retrieval returns focused, relevant context instead of whole pages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_pdf(file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
    """Convenience wrapper: extract -> chunk in one call."""
    pages = extract_text_from_pdf(file_bytes, filename)
    return chunk_documents(pages)


def full_text_from_documents(documents: List[Document]) -> str:
    """Reconstruct plain text from a list of documents (used for summarization)."""
    return "\n\n".join(doc.page_content for doc in documents)

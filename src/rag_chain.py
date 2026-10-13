"""
Retrieval-Augmented Generation: retrieves relevant chunks from ChromaDB
and asks Gemini to answer strictly from that context, with citations.
"""
import re
import time
from typing import Dict, List, Callable, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.config import Config

QA_PROMPT_TEMPLATE = """You are a helpful, precise study assistant.
Answer the question using ONLY the context below. If the answer is not
contained in the context, say "I couldn't find that in the document"
rather than guessing.

Context:
{context}

Question: {question}

Answer clearly and concisely, using bullet points if it helps readability:"""

# Shared across every LLM instance the app creates so that summarization,
# quiz, flashcard, and Q&A calls all draw from the SAME budget. Google's
# actual free-tier quota for this project is 5 requests/minute, so we
# throttle client-side well under that (~1 request every 20s) to leave
# headroom for retries and other tabs being used in the same minute.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=1 / 20,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Instantiate the Gemini chat model with the current API key.
    Requests are rate-limited client-side to stay under the free-tier
    quota. Built-in retries are disabled (max_retries=0) in favor of
    invoke_with_retry(), which reads Google's own suggested wait time
    out of the 429 error instead of guessing a backoff schedule.
    """
    return ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
        rate_limiter=_rate_limiter,
        max_retries=0,
    )


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "ResourceExhausted" in type(e).__name__ or "quota" in msg.lower()


def _parse_retry_delay_seconds(e: Exception, default: float = 20.0) -> float:
    """
    Google's 429 error body includes a retry_delay hint, e.g.
    'retry_delay { seconds: 56 }' or 'retry in 56.4s'. Extract it so we
    wait exactly as long as the server actually asked for, instead of
    guessing with a generic exponential backoff.
    """
    msg = str(e)
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if match:
        return float(match.group(1)) + 3  # small safety buffer
    match = re.search(r"retry in ([\d.]+)\s*s", msg)
    if match:
        return float(match.group(1)) + 3
    return default


def invoke_with_retry(fn: Callable[[], Any], max_attempts: int = 3) -> Any:
    """
    Call fn() (a zero-arg callable wrapping some_chain.invoke(...)) and,
    if it fails with a rate-limit error, wait exactly as long as Google's
    API told us to before retrying — up to max_attempts total tries.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == max_attempts - 1:
                raise
            last_error = e
            time.sleep(_parse_retry_delay_seconds(e))
    raise last_error


def build_qa_chain(vectordb, k: int = None) -> RetrievalQA:
    """Build a RetrievalQA chain wired to the given Chroma vector store."""
    retriever = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": k or Config.RETRIEVER_K}
    )
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def ask_question(vectordb, question: str, k: int = None) -> Dict:
    """
    Run the RAG pipeline for a single question.
    Returns {"answer": str, "sources": List[dict]}.
    """
    chain = build_qa_chain(vectordb, k=k)
    result = invoke_with_retry(lambda: chain.invoke({"query": question}))

    sources = []
    for doc in result.get("source_documents", []):
        sources.append(
            {
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "document"),
                "snippet": doc.page_content[:200].strip() + "...",
            }
        )

    return {"answer": result["result"], "sources": sources}

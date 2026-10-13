"""Small shared helpers used across the Streamlit app."""
import hashlib
import json
from typing import List, Dict

from langchain.docstore.document import Document


def friendly_error_message(e: Exception) -> str:
    """
    Translate common Gemini API errors into a message that tells the
    person what to actually do, instead of a raw stack trace.
    """
    msg = str(e)
    if "429" in msg or "quota" in msg.lower():
        return (
            "Gemini's free-tier rate limit (5 requests/minute) was hit. "
            "The app already retries automatically with backoff — if this "
            "keeps happening, wait about a minute before trying again, or "
            "upgrade to a paid Gemini API plan for higher limits: "
            "https://ai.google.dev/gemini-api/docs/rate-limits"
        )
    return msg


def make_collection_name(filename: str, file_bytes: bytes) -> str:
    """
    Derive a stable, unique Chroma collection name from the file content,
    so re-uploading the same PDF reuses its existing index.
    """
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    safe_name = "".join(c if c.isalnum() else "_" for c in filename)[:30]
    return f"doc_{safe_name}_{digest}"


def select_representative_text(documents: List[Document], max_chars: int = 60000) -> str:
    """
    Build a text blob for prompts like quiz/flashcard generation that
    represents the WHOLE document rather than just its first N characters.

    If the full document fits under max_chars, return it as-is. Otherwise,
    evenly sample chunks spread across the entire chunk list (beginning,
    middle, and end) so later sections of the PDF aren't silently dropped.
    """
    total_chars = sum(len(d.page_content) for d in documents)
    if total_chars <= max_chars:
        return "\n\n".join(d.page_content for d in documents)

    n = len(documents)
    avg_len = max(1, total_chars // n)
    target_count = max(1, min(n, max_chars // avg_len))

    if target_count == 1:
        indices = [0]
    else:
        # Evenly spaced indices spanning from the first chunk (0) to the
        # last chunk (n-1) inclusive, so the end of the document is
        # always represented alongside the beginning and middle.
        indices = [
            round(i * (n - 1) / (target_count - 1)) for i in range(target_count)
        ]

    selected = [documents[i] for i in sorted(set(indices))]
    return "\n\n".join(d.page_content for d in selected)


def flashcards_to_text(flashcards: List[Dict]) -> str:
    """Render flashcards as plain text, e.g. for download."""
    lines = []
    for i, card in enumerate(flashcards, start=1):
        lines.append(f"Card {i} [{card.get('topic', 'General')}]")
        lines.append(f"Q: {card['front']}")
        lines.append(f"A: {card['back']}")
        lines.append("")
    return "\n".join(lines)


def quiz_to_text(questions: List[Dict]) -> str:
    """Render quiz questions as plain text, e.g. for download."""
    lines = []
    for i, q in enumerate(questions, start=1):
        lines.append(f"Q{i}. ({q.get('difficulty', 'medium')}) {q['question']}")
        for key, val in q["options"].items():
            lines.append(f"   {key}. {val}")
        lines.append(f"   Correct answer: {q['correct_answer']}")
        lines.append(f"   Explanation: {q.get('explanation', '')}")
        lines.append("")
    return "\n".join(lines)


def flashcards_to_json(flashcards: List[Dict]) -> str:
    return json.dumps(flashcards, indent=2)


def quiz_to_json(questions: List[Dict]) -> str:
    return json.dumps(questions, indent=2)

"""
Generates study flashcards (term/question -> answer pairs) from
document text, again using strict JSON output from Gemini.
"""
import json
import re
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from src.rag_chain import get_llm, invoke_with_retry
from src.utils import select_representative_text

FLASHCARD_PROMPT = PromptTemplate(
    template="""You are a study-aid generator. Based ONLY on the material below,
create {num_cards} flashcards for spaced-repetition study.

Rules:
- Mix definition-style cards ("What is X?") and concept-check cards
  ("Why does X happen?")
- Keep answers concise (1-3 sentences)
- Do not invent facts not present in the material
- Return ONLY valid JSON, no markdown fences, no commentary, matching this schema:

{{
  "flashcards": [
    {{"front": "question or term", "back": "concise answer", "topic": "short topic label"}}
  ]
}}

Study material:
{text}

JSON output:""",
    input_variables=["text", "num_cards"],
)


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def generate_flashcards(documents: List[Document], num_cards: int = 10) -> List[Dict]:
    """
    Generate flashcards from a list of Document chunks.
    Returns a list of {"front", "back", "topic"} dicts.
    """
    llm = get_llm(temperature=0.5)
    text = select_representative_text(documents, max_chars=60000)

    chain = FLASHCARD_PROMPT | llm
    result = invoke_with_retry(lambda: chain.invoke({"text": text, "num_cards": num_cards}))

    try:
        parsed = _extract_json(result.content)
        # Gemini sometimes returns a bare JSON array instead of the
        # requested {"flashcards": [...]} wrapper — handle both shapes.
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("flashcards", parsed.get("cards", []))
        raise ValueError(f"Unexpected JSON shape: {type(parsed)}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(
            f"Could not parse flashcard JSON from the model's response: {e}\nRaw output: {result.content[:500]}"
        )

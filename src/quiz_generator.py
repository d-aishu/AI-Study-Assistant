"""
Generates multiple-choice quiz questions from document text.
Asks Gemini to return strict JSON so the Streamlit UI can render
interactive quiz widgets instead of raw text.
"""
import json
import re
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from src.rag_chain import get_llm, invoke_with_retry
from src.utils import select_representative_text

QUIZ_PROMPT = PromptTemplate(
    template="""You are an expert exam writer. Based ONLY on the study material
below, create {num_questions} multiple-choice questions to test understanding.

Rules:
- Each question must have exactly 4 options (A-D)
- Only one option is correct
- Vary difficulty (easy, medium, hard)
- Do not invent facts that aren't in the material
- Return ONLY valid JSON, no markdown fences, no commentary, matching this schema:

{{
  "questions": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_answer": "A",
      "explanation": "why this answer is correct, 1 sentence",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Study material:
{text}

JSON output:""",
    input_variables=["text", "num_questions"],
)


def _extract_json(raw: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences despite instructions; strip them."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def generate_quiz(documents: List[Document], num_questions: int = 5) -> List[Dict]:
    """
    Generate a quiz from a list of Document chunks.
    Returns a list of question dicts, or raises ValueError if parsing fails.
    """
    llm = get_llm(temperature=0.5)
    text = select_representative_text(documents, max_chars=60000)

    chain = QUIZ_PROMPT | llm
    result = invoke_with_retry(lambda: chain.invoke({"text": text, "num_questions": num_questions}))

    try:
        parsed = _extract_json(result.content)
        # Gemini sometimes returns a bare JSON array instead of the
        # requested {"questions": [...]} wrapper — handle both shapes.
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("questions", [])
        raise ValueError(f"Unexpected JSON shape: {type(parsed)}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(
            f"Could not parse quiz JSON from the model's response: {e}\nRaw output: {result.content[:500]}"
        )

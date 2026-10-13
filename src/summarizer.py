"""
Summarizes a full document.

Gemini 2.5 Flash has a 1M-token context window, so most study PDFs fit
in a SINGLE API call — that's the fast path, and it matters a lot on
free-tier accounts limited to 5 requests/minute. Map-reduce (summarize
each chunk, then combine) is only used as a fallback for documents too
large to fit in one call.
"""
from typing import List

from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from src.rag_chain import get_llm, invoke_with_retry
from src.utils import select_representative_text

# Documents under this size are summarized in a single call using their
# full text (no sampling, no map-reduce). ~300k chars is roughly 75k
# tokens — comfortably inside Gemini's 1M-token context window while
# leaving lots of room for the response and prompt overhead.
FULL_DOCUMENT_CHAR_BUDGET = 300_000

SINGLE_CALL_PROMPT = PromptTemplate(
    template="""You are a study assistant. Read the following study material
in full and produce a single well-structured summary with:
- A 2-3 sentence overview
- 5-10 key takeaway bullet points, covering the material from start to finish
- Any important terms/definitions mentioned

Study material:
{text}

Summary:""",
    input_variables=["text"],
)

BRIEF_PROMPT = PromptTemplate(
    template="""Write a concise summary (1 short paragraph + 5 bullet
points of key ideas) of the following study material:

{text}

Summary:""",
    input_variables=["text"],
)

MAP_PROMPT = PromptTemplate(
    template="""Summarize the key points of the following text in 3-5 bullet points.
Be factual and concise.

Text:
{text}

Bullet-point summary:""",
    input_variables=["text"],
)

COMBINE_PROMPT = PromptTemplate(
    template="""You are given several bullet-point summaries of sections of one document.
Combine them into a single, well-structured summary with:
- A 2-3 sentence overview
- 5-8 key takeaway bullet points
- Any important terms/definitions mentioned

Section summaries:
{text}

Final combined summary:""",
    input_variables=["text"],
)


def _batch_documents(documents: List[Document], target_batches: int) -> List[Document]:
    """
    Merge many small chunks into a handful of larger 'batch' documents
    before running map-reduce. This keeps the number of Gemini API
    calls as low as possible for very large documents.
    """
    n = len(documents)
    if n <= target_batches:
        return documents

    batch_size = -(-n // target_batches)  # ceil division
    batched = []
    for i in range(0, n, batch_size):
        group = documents[i : i + batch_size]
        combined = "\n\n".join(d.page_content for d in group)
        batched.append(Document(page_content=combined, metadata={"batch": i // batch_size}))
    return batched


def summarize_documents(documents: List[Document], style: str = "detailed") -> str:
    """
    Summarize a list of Document chunks.
    style: "detailed" (full-document, single call when it fits — falls
           back to map-reduce only for very large docs) or "brief"
           (single-pass on a representative sample, fastest option).
    """
    llm = get_llm(temperature=0.3)
    total_chars = sum(len(d.page_content) for d in documents)

    if style == "brief":
        combined_text = select_representative_text(documents, max_chars=40000)
        chain = BRIEF_PROMPT | llm
        result = invoke_with_retry(lambda: chain.invoke({"text": combined_text}))
        return result.content

    if total_chars <= FULL_DOCUMENT_CHAR_BUDGET:
        # Fast path: the whole document fits in one request — 1 API call
        # total, which is the friendliest option for free-tier quotas.
        full_text = "\n\n".join(d.page_content for d in documents)
        chain = SINGLE_CALL_PROMPT | llm
        result = invoke_with_retry(lambda: chain.invoke({"text": full_text}))
        return result.content

    # Fallback for documents too large for one call: batch into just a
    # few large groups (not one call per chunk) and map-reduce.
    target_batches = min(4, max(2, -(-total_chars // FULL_DOCUMENT_CHAR_BUDGET)))
    batched_docs = _batch_documents(documents, target_batches=target_batches)
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=False,
    )
    result = invoke_with_retry(lambda: chain.invoke({"input_documents": batched_docs}))
    return result["output_text"]

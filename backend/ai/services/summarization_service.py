import asyncio
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.llm_service import extract_text_from_content, llm_service
from app.models.document import Document
from app.models.document_chunks import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Intent Detection Patterns
# ---------------------------------------------------------------------

SUMMARIZE_TRIGGER_PATTERN = re.compile(
    r"\b("
    r"summarize|summarise|"
    r"summary(?:\s+of)?|"
    r"overview(?:\s+of)?|"
    r"tl;?dr(?:\s+of)?|"
    r"give\s+(?:me\s+)?the\s+gist(?:\s+of)?|"
    r"gist(?:\s+of)?"
    r")\b",
    re.IGNORECASE,
)

WHOLE_DOCUMENT_REF_PATTERN = re.compile(
    r"\b("
    r"(?:this|the|my|our)\s+(?:document|doc|file|pdf|upload|paper|notes?)|"
    r"(?:the\s+)?(?:document|doc|file|pdf|upload)\s+(?:i\s+)?(?:just\s+)?uploaded|"
    r"(?:what\s+i|document\s+i|file\s+i)\s+(?:just\s+)?uploaded|"
    r"the\s+entire\s+(?:document|doc|file|pdf|upload)|"
    r"the\s+whole\s+(?:document|doc|file|pdf|upload)|"
    r"everything\s+in\s+(?:this|the|my)\s+(?:document|doc|file|pdf)"
    r")\b",
    re.IGNORECASE,
)

# Topic-focused indicators that mean the user is asking about a specific section or topic
# rather than the whole document. These must stay on the standard RAG retrieval path.
TOPIC_FOCUSED_INDICATORS = re.compile(
    r"\b("
    r"(?:section|chapter|part|paragraph|page|topic|point|aspect|clause|article|table|figure)\s+(?:on|about|regarding|of|covering)|"
    r"what\s+(?:it|this\s+document|the\s+document|they)\s+say(?:s)?\s+(?:on|about|regarding)|"
    r"what\s+is\s+said\s+about|"
    r"information\s+(?:on|about|regarding)|"
    r"details\s+(?:on|about|regarding)|"
    r"findings?\s+(?:on|about|regarding)|"
    r"policy\s+(?:on|about|regarding)|"
    r"rule\s+(?:on|about|regarding)|"
    r"how\s+to\s+|"
    r"why\s+(?:does|is|are|did|was|were)\s+|"
    r"when\s+(?:does|is|are|did|was|were)\s+|"
    r"who\s+(?:is|are|was|were)\s+"
    r")\b",
    re.IGNORECASE,
)

DOCUMENT_TOPIC_LINK_PATTERN = re.compile(
    r"\b(?:this|the)\s+(?:document|doc|file|pdf|upload)\s+(?:on|about|regarding)\s+[a-zA-Z0-9_-]+",
    re.IGNORECASE,
)

STANDALONE_SUMMARIZE_COMMAND = re.compile(
    r"^(?:can\s+you\s+)?(?:please\s+)?(?:"
    r"summarize|summarise|"
    r"give\s+(?:me\s+)?(?:a\s+)?summary|"
    r"give\s+(?:me\s+)?(?:a\s+)?(?:quick\s+)?overview|"
    r"overview|"
    r"tl;?dr|"
    r"give\s+(?:me\s+)?the\s+gist|"
    r"gist"
    r")(?:\s+it|\s+this)?(?:\s+please)?[\.\?!]*$",
    re.IGNORECASE,
)


def is_summarize_document_query(query: str) -> bool:
    """
    Deterministically detects whether a user query is asking for a whole-document summary.
    Returns True for whole-document requests ("summarize this document", "tl;dr of handbook.pdf").
    Returns False for topic-focused queries ("summarize the section on refund policy")
    which must remain on standard RAG retrieval.
    """
    if not query:
        return False

    q_stripped = query.strip()

    # 1. Standalone command like "summarize", "tl;dr", "give me a summary", "can you summarize it?"
    if STANDALONE_SUMMARIZE_COMMAND.match(q_stripped):
        return True

    # 2. Must contain a summary trigger word/phrase
    if not SUMMARIZE_TRIGGER_PATTERN.search(q_stripped):
        return False

    # 3. Exclude topic-focused queries
    if TOPIC_FOCUSED_INDICATORS.search(q_stripped):
        return False
    if DOCUMENT_TOPIC_LINK_PATTERN.search(q_stripped):
        return False

    # 4. Check for whole-document references or explicit filenames
    if WHOLE_DOCUMENT_REF_PATTERN.search(q_stripped):
        return True

    # Check for document filenames with extensions (e.g. handbook.pdf, notes.txt)
    if re.search(r"\b[\w\.-]+\.(?:pdf|docx|txt|pptx|xlsx|md)\b", q_stripped, re.IGNORECASE):
        return True

    return False


# ---------------------------------------------------------------------
# Rate Limit & Execution Constants
# ---------------------------------------------------------------------

class RateLimitExceededError(Exception):
    """Raised when LLM returns 429 / RESOURCE_EXHAUSTED and bounded retries are exhausted."""
    pass


# Token / character sizing:
# Gemini Flash has generous context limits, but to protect free-tier rate limits (~5 RPM)
# and avoid timeouts, we size single-call budgets carefully and cap map-reduce iterations.
SINGLE_CALL_CHAR_BUDGET = 32000    # ~8,000 tokens: single call fits most documents
BATCH_CHAR_TARGET = 24000          # ~6,000 tokens per map batch
MAX_MAP_CALLS = 4                  # Maximum consecutive batch map calls
MAX_TOTAL_LLM_CALLS = 5            # 4 map + 1 reduce = 5 calls maximum (safeguards 5 RPM quota)


async def call_llm_with_backoff(prompt: str, max_retries: int = 2) -> str:
    """
    Invokes the LLM with bounded exponential backoff on HTTP 429 / RESOURCE_EXHAUSTED errors.
    Does not loop indefinitely; raises RateLimitExceededError if retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            llm = llm_service.get_chat_model(temperature=0.2, streaming=False)
            response = await llm.ainvoke(prompt)
            return extract_text_from_content(response.content)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "429" in err_str
                or "resource_exhausted" in err_str
                or "quota" in err_str
                or "rate limit" in err_str
            )
            if is_rate_limit:
                if attempt < max_retries:
                    wait_sec = 2.0 * (attempt + 1)
                    logger.warning(
                        "LLM rate limit encountered during summarization. Backing off for %.1fs (attempt %d/%d)...",
                        wait_sec,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                logger.error("LLM rate limit retries exhausted during summarization: %s", e)
                raise RateLimitExceededError("LLM service is rate-limited.") from e
            raise


# ---------------------------------------------------------------------
# Target Document Resolution
# ---------------------------------------------------------------------

async def resolve_target_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    knowledge_base_id: Optional[uuid.UUID],
    query: str,
) -> Tuple[Optional[Document], Optional[str], List[Document]]:
    """
    Resolves the target Document for whole-document summarization using scoped DB queries.
    Reuses the exact same scoping rules as retrieval: user ownership, session isolation,
    and knowledge base association.

    Returns:
        (target_document, resolution_status, candidate_documents)
        Where resolution_status can be:
        - None: successfully resolved target_document
        - "no_documents": no documents exist in scope
        - "clarification_needed": multiple documents in scope with no clear target
    """
    conditions = [Document.user_id == user_id]

    if knowledge_base_id is not None and session_id is not None:
        conditions.append(
            (Document.knowledge_base_id == knowledge_base_id) | (Document.session_id == session_id)
        )
    elif knowledge_base_id is not None:
        conditions.append(Document.knowledge_base_id == knowledge_base_id)
    elif session_id is not None:
        conditions.append(Document.session_id == session_id)
    else:
        return None, "no_documents", []

    stmt = select(Document).where(*conditions).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    documents = list(result.scalars().all())

    if not documents:
        return None, "no_documents", []

    q_lower = query.lower()

    # 1. Did the user name a document by filename or title?
    matched_by_name = []
    for doc in documents:
        fname = doc.filename.lower()
        # Full filename match
        if fname in q_lower:
            matched_by_name.append(doc)
            continue
        # Base name without extension if length >= 3
        if "." in fname:
            base_name = fname.rsplit(".", 1)[0]
            if len(base_name) >= 3 and base_name in q_lower:
                matched_by_name.append(doc)

    if matched_by_name:
        # Return the best/longest matching document
        matched_by_name.sort(key=lambda d: len(d.filename), reverse=True)
        return matched_by_name[0], None, documents

    # 2. Check if user explicitly refers to "just uploaded" / "recently uploaded"
    is_recent_ref = bool(
        re.search(
            r"\b(?:just|recently|latest)\s+uploaded\b|\buploaded\s+(?:just\s+now|recently)\b",
            q_lower,
        )
    )

    # Filter session uploads (documents attached directly to this session, not KB-level)
    session_uploads = [
        d for d in documents if d.session_id == session_id and d.knowledge_base_id is None
    ]

    if is_recent_ref:
        if session_uploads:
            return session_uploads[0], None, documents
        return documents[0], None, documents

    # 3. Exactly one session upload in scope: use it!
    if len(session_uploads) == 1:
        return session_uploads[0], None, documents

    # 4. Exactly one document total in scope: use it!
    if len(documents) == 1:
        return documents[0], None, documents

    # 5. Otherwise, multiple candidates and no clear target -> clarify!
    return None, "clarification_needed", documents


# ---------------------------------------------------------------------
# Whole-Document Summarization Execution
# ---------------------------------------------------------------------

async def generate_document_summary(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    knowledge_base_id: Optional[uuid.UUID],
    query: str,
) -> Dict[str, Any]:
    """
    Executes the whole-document summarization pipeline:
    1. Resolves target document under fail-closed scoping.
    2. Handles document status (processing, failed, ready).
    3. Bypasses relevance threshold and top_k, loading all chunks in order.
    4. Executes single-call or map-reduce summarization with rate-limit protection.
    5. Returns formatted context and final response.
    """
    target_doc, resolution_status, candidates = await resolve_target_document(
        db=db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
        query=query,
    )

    if resolution_status == "no_documents":
        return {
            "final_response": "There are no documents available to summarize in this conversation. Please upload a document first.",
            "retrieved_chunks": [],
            "formatted_context": "",
            "target_document": None,
        }

    if resolution_status == "clarification_needed":
        doc_list = "\n".join(f"- {d.filename}" for d in candidates)
        return {
            "final_response": f"Which document would you like me to summarize? Please specify one of the following:\n{doc_list}",
            "retrieved_chunks": [],
            "formatted_context": "",
            "target_document": None,
        }

    # Target document resolved: check status
    if target_doc.status == "processing":
        return {
            "final_response": f"'{target_doc.filename}' is still being processed. Please try again shortly once processing is complete.",
            "retrieved_chunks": [],
            "formatted_context": "",
            "target_document": target_doc.filename,
        }

    if target_doc.status == "failed":
        err_detail = f": {target_doc.error_message}" if target_doc.error_message else ""
        return {
            "final_response": f"Processing failed for '{target_doc.filename}'{err_detail}. Please try re-uploading the document.",
            "retrieved_chunks": [],
            "formatted_context": "",
            "target_document": target_doc.filename,
        }

    # Status is 'ready': fetch all chunks ordered by chunk_index
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == target_doc.id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    result = await db.execute(stmt)
    chunks = list(result.scalars().all())

    if not chunks:
        return {
            "final_response": f"'{target_doc.filename}' does not contain any extractable text to summarize.",
            "retrieved_chunks": [],
            "formatted_context": "",
            "target_document": target_doc.filename,
        }

    # Format chunks for response citation / UI
    formatted_chunks = [
        {
            "chunk_id": str(c.id),
            "document_id": str(target_doc.id),
            "filename": target_doc.filename,
            "page_number": c.page_number or 1,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "score": 1.0,
            "source_type": "document_summary",
        }
        for c in chunks
    ]

    total_chars = sum(len(c.content) for c in chunks)
    logger.info(
        "📄 [Summarization] Generating summary for '%s' (%d chunks, %d total chars)...",
        target_doc.filename,
        len(chunks),
        total_chars,
    )

    try:
        # Strategy A: Single-Call Summarization (fits within budget)
        if total_chars <= SINGLE_CALL_CHAR_BUDGET:
            full_text = "\n\n".join(
                f"[Page {c.page_number or 1}]\n{c.content}" for c in chunks
            )
            prompt = (
                f"You are an expert document summarizer.\n"
                f"Provide a comprehensive, clear, and well-structured summary of the document '{target_doc.filename}'.\n"
                f"Include:\n"
                f"- **Overview / Main Purpose**: High-level summary of what the document is.\n"
                f"- **Key Takeaways & Findings**: Bullet points of essential facts, numbers, decisions, or rules.\n"
                f"- **Important Details & Structure**: Summary of main sections or topics.\n"
                f"- **Conclusion**: Final takeaway or summary outcome.\n\n"
                f"Document Content:\n{full_text}"
            )
            summary_text = await call_llm_with_backoff(prompt)

        # Strategy B: Map-Reduce Batches for Large Documents
        else:
            logger.info("📄 [Summarization] Document exceeds single-call budget. Using map-reduce...")
            batches: List[List[DocumentChunk]] = []
            current_batch: List[DocumentChunk] = []
            current_batch_chars = 0

            for chunk in chunks:
                chunk_len = len(chunk.content)
                if current_batch and (current_batch_chars + chunk_len > BATCH_CHAR_TARGET):
                    batches.append(current_batch)
                    current_batch = [chunk]
                    current_batch_chars = chunk_len
                else:
                    current_batch.append(chunk)
                    current_batch_chars += chunk_len
            if current_batch:
                batches.append(current_batch)

            # Cap the number of map batches to protect LLM rate limits
            is_truncated = len(batches) > MAX_MAP_CALLS
            covered_batches = batches[:MAX_MAP_CALLS]
            covered_chunks = [c for b in covered_batches for c in b]

            covered_start_page = covered_chunks[0].page_number or 1
            covered_end_page = covered_chunks[-1].page_number or 1

            batch_summaries: List[str] = []
            for idx, batch in enumerate(covered_batches, start=1):
                b_start = batch[0].page_number or 1
                b_end = batch[-1].page_number or 1
                batch_text = "\n\n".join(f"[Page {c.page_number or 1}]\n{c.content}" for c in batch)
                map_prompt = (
                    f"You are an expert document summarizer.\n"
                    f"Summarize the following section (Pages {b_start}-{b_end}, Part {idx}/{len(covered_batches)}) "
                    f"of '{target_doc.filename}'. Extract all key facts, numbers, headings, and takeaways concisely:\n\n"
                    f"{batch_text}"
                )
                b_summary = await call_llm_with_backoff(map_prompt)
                batch_summaries.append(f"### Section {idx} (Pages {b_start}-{b_end})\n{b_summary}")

            # Reduce step: synthesize all section summaries into one cohesive summary
            combined_sections = "\n\n".join(batch_summaries)
            reduce_prompt = (
                f"You are an expert document summarizer.\n"
                f"Below are summaries of consecutive sections from '{target_doc.filename}'.\n"
                f"Synthesize them into a unified, cohesive, and comprehensive whole-document summary.\n"
                f"Structure with an Overview, Key Takeaways, Section Summaries, and Conclusion:\n\n"
                f"{combined_sections}"
            )
            summary_text = await call_llm_with_backoff(reduce_prompt)

            if is_truncated:
                total_pages = chunks[-1].page_number or len(chunks)
                truncation_note = (
                    f"\n\n*(Note: Due to document length constraints, this summary covers pages {covered_start_page} to {covered_end_page} "
                    f"({len(covered_chunks)} of {len(chunks)} chunks). The remainder of the document was omitted to protect processing limits.)*"
                )
                summary_text = f"{summary_text.strip()}{truncation_note}"

    except RateLimitExceededError:
        summary_text = (
            "The AI service is currently busy due to rate limits. Please try requesting the summary again in a few moments."
        )

    formatted_context = f"[Document Summary: {target_doc.filename}]\n{summary_text}"

    return {
        "final_response": summary_text,
        "retrieved_chunks": formatted_chunks,
        "formatted_context": formatted_context,
        "target_document": target_doc.filename,
    }

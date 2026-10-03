import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> List[Dict[str, Any]]:
    """
    Recursively splits text on semantic boundaries (\n\n, \n, . , space)
    to preserve document structure, context, and table readability.
    """
    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d). Clamped to %d.",
            chunk_overlap, chunk_size, chunk_size // 2
        )
        chunk_overlap = chunk_size // 2

    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_recursively(content: str) -> List[str]:
        if len(content) <= chunk_size:
            return [content]

        sep = ""
        for s in separators:
            if s == "":
                sep = ""
                break
            if s in content:
                sep = s
                break

        if sep == "":
            return list(content)

        parts = content.split(sep)
        result = []
        for i, part in enumerate(parts):
            if len(part) > chunk_size:
                result.extend(_split_recursively(part))
            else:
                result.append(part)

            if i < len(parts) - 1:
                if len(sep) > chunk_size:
                    result.extend(list(sep))
                else:
                    result.append(sep)

        return result

    units = _split_recursively(text)

    final_chunks = []
    current_chunk = ""

    for unit in units:
        if current_chunk and len(current_chunk) + len(unit) > chunk_size:
            doc_text = current_chunk.strip()
            if doc_text:
                final_chunks.append(doc_text)

            if chunk_overlap > 0 and doc_text:
                overlap_str = (
                    doc_text[-chunk_overlap:]
                    if len(doc_text) > chunk_overlap
                    else doc_text
                )
                if len(doc_text) > chunk_overlap:
                    match = re.search(r'([.?!]\s+|\n+)', overlap_str)
                    if match:
                        overlap_str = overlap_str[match.end():]
                    else:
                        match = re.search(r'\s+', overlap_str)
                        if match:
                            overlap_str = overlap_str[match.end():]
                        else:
                            overlap_str = ""
            else:
                overlap_str = ""

            while overlap_str and len(overlap_str) + len(unit) > chunk_size:
                match = re.search(r'([.?!]\s+|\n+)', overlap_str)
                if match:
                    overlap_str = overlap_str[match.end():]
                else:
                    match = re.search(r'\s+', overlap_str)
                    if match:
                        overlap_str = overlap_str[match.end():]
                    else:
                        overlap_str = ""

            current_chunk = overlap_str + unit
        else:
            current_chunk += unit

    if current_chunk:
        doc_text = current_chunk.strip()
        if doc_text:
            final_chunks.append(doc_text)

    return [{"content": c, "length": len(c)} for c in final_chunks]
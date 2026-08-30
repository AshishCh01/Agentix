import re
from typing import Any, Dict, List


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> List[Dict[str, Any]]:
    """
    Recursively splits text on semantic boundaries (\n\n, \n, . , space)
    to preserve document structure, context, and table readability.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(content: str, max_len: int) -> List[str]:
        if len(content) <= max_len:
            return [content]

        # Select the highest priority separator present in content
        separator = ""
        for s in separators:
            if s == "":
                separator = ""
                break
            if s in content:
                separator = s
                break

        if separator != "":
            splits = content.split(separator)
        else:
            splits = list(content)

        final_chunks = []
        current_doc = []
        total_len = 0

        for s in splits:
            if len(s) > max_len:
                # A single split unit alone still exceeds the cap; flush what's
                # accumulated so far, then recursively re-split it with the next
                # separator (falling through to per-character as the last resort)
                # so no oversized chunk ever survives.
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        final_chunks.append(doc_text)
                    current_doc = []
                    total_len = 0
                final_chunks.extend(_split_text(s, max_len))
                continue

            s_len = len(s) + (len(separator) if current_doc else 0)
            if total_len + s_len > max_len:
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        final_chunks.append(doc_text)

                    # Retain overlap from ending of previous chunk
                    while total_len > chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + len(separator)

                    # current_doc may now be trimmed or emptied; recompute
                    # whether s needs a leading separator before it's appended.
                    s_len = len(s) + (len(separator) if current_doc else 0)

            current_doc.append(s)
            total_len += s_len

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text:
                final_chunks.append(doc_text)

        return final_chunks

    raw_chunks = _split_text(text, chunk_size)

    return [{"content": c, "length": len(c)} for c in raw_chunks if c.strip()]
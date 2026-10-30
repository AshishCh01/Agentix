import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _pack_chunks(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> List[str]:
    """
    Recursively splits text on semantic boundaries (\n\n, \n, . , space)
    and packs units into chunks respecting chunk_size and chunk_overlap.
    """
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

            # Ensure newline separator between lines if overlap comes from line-based text
            sep = "\n" if overlap_str and "\n" in doc_text and not overlap_str.endswith("\n") and not unit.startswith("\n") else ""

            while overlap_str and len(overlap_str) + len(sep) + len(unit) > chunk_size:
                match = re.search(r'([.?!]\s+|\n+)', overlap_str)
                if match:
                    overlap_str = overlap_str[match.end():]
                else:
                    match = re.search(r'\s+', overlap_str)
                    if match:
                        overlap_str = overlap_str[match.end():]
                    else:
                        overlap_str = ""
                sep = "\n" if overlap_str and "\n" in doc_text and not overlap_str.endswith("\n") and not unit.startswith("\n") else ""

            current_chunk = overlap_str + sep + unit
        else:
            current_chunk += unit

    if current_chunk:
        doc_text = current_chunk.strip()
        if doc_text:
            final_chunks.append(doc_text)

    return final_chunks


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    header_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively splits text on semantic boundaries (\n\n, \n, . , space)
    to preserve document structure, context, and table readability.
    Repeats contextual headers (such as Excel sheet names) across chunk boundaries.
    """
    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d). Clamped to %d.",
            chunk_overlap, chunk_size, chunk_size // 2
        )
        chunk_overlap = chunk_size // 2

    if not text or not text.strip():
        return []

    prefix = ""
    body_text = text
    if header_prefix:
        prefix = header_prefix.strip()
        body_text = text
    elif text.startswith("Sheet: "):
        lines = text.split("\n", 1)
        prefix = lines[0].strip()
        body_text = lines[1] if len(lines) > 1 else ""

    # If the text already fits in a single chunk, return it directly
    if len(text.strip()) <= chunk_size:
        clean_text = text.strip()
        return [{"content": clean_text, "length": len(clean_text)}]

    # If a contextual header prefix is present, repeat it on every generated chunk
    if prefix:
        if not body_text.strip():
            clean_text = prefix
            return [{"content": clean_text, "length": len(clean_text)}]

        prefix_line = prefix + "\n"
        effective_chunk_size = max(50, chunk_size - len(prefix_line))
        effective_overlap = min(chunk_overlap, effective_chunk_size // 2)

        raw_chunks = _pack_chunks(
            body_text,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
        )

        final_chunks = []
        for c in raw_chunks:
            full_content = f"{prefix_line}{c}".strip()
            final_chunks.append({
                "content": full_content,
                "length": len(full_content),
            })
        return final_chunks

    # Standard chunking without header prefix
    raw_chunks = _pack_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [{"content": c, "length": len(c)} for c in raw_chunks]

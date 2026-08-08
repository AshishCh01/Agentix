from typing import List, Dict, Any


def chunk_text(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits raw text into overlapping chunks using sliding window tokenization/character splits.
    Returns a list of dicts with chunk content and token sequence numbers.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "content": chunk_content,
                    "char_count": len(chunk_content),
                }
            )
            chunk_index += 1

        if end == text_length:
            break

        start += chunk_size - overlap

    return chunks
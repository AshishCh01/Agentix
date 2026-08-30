from ai.services.chunking_service import chunk_text


def test_consecutive_chunks_share_overlap():
    """
    Chunks split from a long input should carry the trailing overlap of the
    previous chunk into the start of the next one, not lose it entirely.
    """
    sentences = [
        f"This is sentence number {i} with some extra padding words to fill space."
        for i in range(60)
    ]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1

    for i in range(len(chunks) - 1):
        tail = chunks[i]["content"][-50:]
        head = chunks[i + 1]["content"][:100]
        # Some meaningful suffix of the previous chunk's tail should appear
        # at the start of the next chunk.
        assert any(tail[-k:] in head for k in range(10, 51)), (
            f"No overlap found between chunk {i} tail {tail!r} "
            f"and chunk {i + 1} head {head!r}"
        )


def test_no_chunk_exceeds_chunk_size_for_oversized_unit():
    """
    A single split unit with no nearby separator (e.g. a very long unbroken
    string) must still be recursively re-split so no output chunk exceeds
    chunk_size.
    """
    long_word = "x" * 1500
    text = f"Intro sentence before the blob. {long_word} Outro sentence after the blob."

    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c["length"] <= 300 for c in chunks)
    # The oversized blob should have actually been split into multiple pieces.
    assert any(c["content"] == "x" * 300 for c in chunks)


def test_no_chunk_exceeds_chunk_size_for_long_unbroken_line():
    """
    A long line with no whitespace/punctuation at all (last-resort
    per-character splitting) must still respect chunk_size.
    """
    text = "y" * 5000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=80)
    assert all(c["length"] <= 400 for c in chunks)
    # Overlapping chunks legitimately double-count shared characters, so the
    # total length only needs to be at least as long as the source text.
    assert sum(c["length"] for c in chunks) >= len(text)


def test_short_text_returns_single_chunk():
    text = "Hello world. This is a short document that fits in one chunk easily."
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text


def test_empty_and_whitespace_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []

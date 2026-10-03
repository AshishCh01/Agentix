from ai.services.chunking_service import chunk_text
import random
import string
import logging


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


def test_chunking_synthetic_paragraphs():
    # Regression test for 300-450 char paragraphs with 600/120 settings
    random.seed(42)
    paras = []
    for _ in range(30):
        length = random.randint(300, 450)
        words = []
        for _ in range(length // 5):
            word = "".join(random.choices(string.ascii_lowercase, k=random.randint(2, 7)))
            words.append(word)
        para = " ".join(words).replace(" ", ". ", 5)
        paras.append(para)
    text = "\n\n".join(paras)

    chunks = chunk_text(text, chunk_size=600, chunk_overlap=120)
    
    assert len(chunks) > 0
    for i, c in enumerate(chunks):
        assert c["length"] <= 600
        # check overlap
        if i > 0:
            prev = chunks[i-1]["content"]
            curr = c["content"]
            
            overlap_found = False
            for k in range(min(len(prev), len(curr), 120), 0, -1):
                if prev[-k:] == curr[:k]:
                    overlap_found = True
                    break
            assert overlap_found, f"No overlap found between chunk {i-1} and chunk {i}"


def test_chunking_deterministic():
    text = "A simple deterministic test string. " * 100
    chunks1 = chunk_text(text, 200, 50)
    chunks2 = chunk_text(text, 200, 50)
    assert chunks1 == chunks2


def test_chunking_clamp_warning(caplog):
    text = "A simple test string."
    with caplog.at_level(logging.WARNING):
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=150)
    assert any("clamped" in record.message.lower() for record in caplog.records)
    assert len(chunks) > 0


def test_chunking_hindi_emoji_and_mix():
    text = "नमस्ते दुनिया 🌍 " * 100 + "\n\n" + "A short paragraph." + "\n\n" + "A very long single word: " + "a"*1000 + "\n\n" + "ID,Name\n1,Test\n2,Test2\n" * 20
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 0
    for c in chunks:
        assert c["length"] <= 300


def test_chunking_preserves_all_words():
    text = "One two three four five six seven eight nine ten."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    for word in ["One", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten."]:
        assert any(word in c["content"] for c in chunks)


def test_chunking_never_starts_mid_word():
    text = "This is a sentence. And another sentence. With words."
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=10)
    for i in range(1, len(chunks)):
        assert not chunks[i]["content"].startswith(" ")

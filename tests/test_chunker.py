import pytest

from src.rag.chunker import chunk_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text():
    assert chunk_text("   ") == []


def test_short_text_returned_unchanged():
    text = "hello   world\nagain"
    assert chunk_text(text) == [text]


def test_long_text_chunks_and_overlap():
    chunks = chunk_text(words(500), chunk_words=250, overlap_words=30)
    # windows start at 0, 220, 440; the third reaches the end -> stop
    assert len(chunks) == 3
    first, second, third = (c.split() for c in chunks)
    assert len(first) == 250 and first[0] == "w0"
    assert second[0] == "w220"
    assert first[-30:] == second[:30]           # 30-word overlap
    assert third[-1] == "w499" and len(third) == 60


def test_every_word_is_covered():
    text = words(1000)
    covered = set(w for c in chunk_text(text, 100, 10) for w in c.split())
    assert covered == set(text.split())


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(words(10), chunk_words=5, overlap_words=5)

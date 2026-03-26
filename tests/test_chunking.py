import pytest

from app.services.chunking import chunk_text


def test_chunk_text_returns_empty_list_for_empty_input() -> None:
    assert chunk_text("") == []


def test_chunk_text_returns_single_chunk_when_text_is_short() -> None:
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

    assert chunks == ["This is a short text."]


def test_chunk_text_splits_text_into_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert chunks == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
    ]


def test_chunk_text_raises_error_when_chunk_size_is_invalid() -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than 0."):
        chunk_text("hello world", chunk_size=0, chunk_overlap=0)


def test_chunk_text_raises_error_when_chunk_overlap_is_negative() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be greater than or equal to 0.",
    ):
        chunk_text("hello world", chunk_size=10, chunk_overlap=-1)


def test_chunk_text_raises_error_when_chunk_overlap_is_too_large() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size.",
    ):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
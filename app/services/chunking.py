def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += chunk_size - chunk_overlap

    return chunks
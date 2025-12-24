from typing import List

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.
    Useful for RAG pipelines.
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    sample_text = "This is a sample regulatory document text " * 100
    chunks = chunk_text(sample_text)

    print(f"Total chunks created: {len(chunks)}")

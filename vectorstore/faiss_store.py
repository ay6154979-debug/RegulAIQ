import faiss
import os
import pickle
from typing import List
import numpy as np


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]):
        """
        Add embeddings and corresponding text chunks to FAISS index.
        """
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5):
        """
        Search similar chunks for a given query embedding.
        """
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    def save(self, path: str):
        """
        Persist FAISS index and chunks to disk.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path: str):
        """
        Load FAISS index and chunks from disk.
        """
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.text_chunks = pickle.load(f)


if __name__ == "__main__":
    print("FAISS Vector Store ready")

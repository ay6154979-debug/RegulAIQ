from typing import List
from embeddings.embedding_generator import EmbeddingGenerator
from vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    """
    Retriever to fetch relevant document chunks using semantic search.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: FAISSVectorStore,
        top_k: int = 5
    ):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve top-k relevant chunks for a given query.
        """
        query_embedding = self.embedding_generator.generate_embedding(query)
        results = self.vector_store.search(query_embedding, self.top_k)
        return results


if __name__ == "__main__":
    print("Retriever module ready")

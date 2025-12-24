from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate vector embeddings for text chunks.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "This regulation applies to pharmaceutical manufacturing.",
        "The applicant must submit stability data."
    ]

    generator = EmbeddingGenerator()
    vectors = generator.generate_embeddings(sample_chunks)

    print("Embedding shape:", vectors.shape)

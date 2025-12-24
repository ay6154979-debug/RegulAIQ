# File: qa/qa_engine.py
from typing import List
from retrieval.retriever import Retriever
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class QARAGEngine:
    """
    Explainable QA-RAG system for regulatory documents.
    
    Steps:
    1. Retrieve top-k relevant chunks using semantic retriever.
    2. Generate answer using OpenAI LLM.
    3. Return explainable answer based on retrieved chunks.
    """

    def __init__(self, retriever: Retriever, llm_api_key: str, model: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        self.model = model
        self.client = OpenAI(api_key=llm_api_key)
        logging.info("QA-RAG Engine initialized with model: %s", self.model)

    def answer_query(self, query: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        """
        Retrieve relevant chunks and generate answer using LLM.
        
        Args:
            query (str): User question.
            max_tokens (int): Maximum tokens for LLM response.
            temperature (float): Creativity of LLM output.
        
        Returns:
            str: Explainable answer based on retrieved chunks.
        """
        try:
            logging.info("Processing query: %s", query)

            # Step 1: Retrieve relevant chunks
            chunks = self.retriever.retrieve(query)
            if not chunks:
                logging.warning("No relevant chunks found for query.")
                return "No relevant information found in the documents."

            # Step 2: Prepare prompt
            prompt = self._build_prompt(query, chunks)

            # Step 3: Call LLM
            response = self.client.Completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].text.strip()
            logging.info("Answer generated successfully.")
            return answer

        except Exception as e:
            logging.error("Error in QA-RAG Engine: %s", str(e))
            return "An error occurred while processing the query."

    @staticmethod
    def _build_prompt(query: str, chunks: List[str]) -> str:
        """
        Construct the LLM prompt with retrieved chunks.
        """
        prompt = "Answer the question based on these regulatory documents:\n\n"
        for i, chunk in enumerate(chunks, 1):
            prompt += f"Document {i}:\n{chunk}\n\n"
        prompt += f"Question: {query}\nAnswer:"
        return prompt


if __name__ == "__main__":
    logging.info("QA-RAG Engine module is ready for testing.")

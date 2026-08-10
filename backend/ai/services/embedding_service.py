from typing import Any, List, Optional, Dict


class EmbeddingService:
    def __init__(self):
        self._model: Optional[Any] = None
        self._reranker: Optional[Any] = None

    @property
    def model(self) -> Any:
        """
        Lazy-loads PyTorch, sentence-transformers, and model weights on FIRST actual upload/embedding call.
        """
        if self._model is None:
            print("First embedding request detected. Loading PyTorch and BAAI/bge-base-en-v1.5 into RAM...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model

    @property
    def reranker(self) -> Any:
        """
        Lazy-loads a cross-encoder model for hybrid search reranking.
        """
        if self._reranker is None:
            print("Loading CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)...")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a 768-dimensional vector embedding for a single text string.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text strings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks using a CrossEncoder and updates their scores.
        Sorts the chunks in descending order by the new score.
        """
        if not chunks:
            return []
        
        # Prepare inputs: list of [query, chunk_content] pairs
        inputs = [[query, chunk["content"]] for chunk in chunks]
        scores = self.reranker.predict(inputs)
        
        for i, chunk in enumerate(chunks):
            chunk["score"] = float(scores[i])
            chunk["source_type"] = "reranked"
            
        # Sort chunks in descending order of score
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks


embedding_service = EmbeddingService()

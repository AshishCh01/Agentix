from typing import Any, List, Optional


class EmbeddingService:
    def __init__(self):
        self._model: Optional[Any] = None

    @property
    def model(self) -> Any:
        """
        Lazy-loads PyTorch, sentence-transformers, and model weights on FIRST actual upload/embedding call.
        """
        if self._model is None:
            print("⚡ First embedding request detected. Loading PyTorch and BAAI/bge-base-en-v1.5 into RAM...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model

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


embedding_service = EmbeddingService()



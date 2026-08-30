import asyncio
from enum import Enum
from typing import Any, List, Optional, Dict

from google import genai

from app.config.settings import settings


class EmbeddingTask(str, Enum):
    """
    gemini-embedding-2 has no task_type request field. Task intent is instead
    conveyed by prefixing the text sent to the API, so every embedding call
    must declare which prefix applies.
    """
    QUERY = "query"
    DOCUMENT = "document"


class EmbeddingService:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self._reranker: Optional[Any] = None

    @property
    def client(self) -> genai.Client:
        """
        Lazy-inits the Gemini GenAI client on first use.
        """
        if self._client is None:
            api_key = str(settings.GEMINI_API_KEY).strip().strip("'\"")
            self._client = genai.Client(api_key=api_key)
        return self._client

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

    @staticmethod
    def _build_prefixed_text(text: str, mode: EmbeddingTask, title: Optional[str] = None) -> str:
        """
        gemini-embedding-2 takes task intent as a text prefix rather than a
        task_type request field.
        """
        if mode == EmbeddingTask.QUERY:
            return f"task: search result | query: {text}"
        return f"title: {title or 'none'} | text: {text}"

    async def generate_embedding(
        self,
        text: str,
        mode: EmbeddingTask = EmbeddingTask.QUERY,
        title: Optional[str] = None,
    ) -> List[float]:
        """
        Generates a gemini-embedding-2 vector, truncated to
        settings.EMBEDDING_DIMENSIONS (pgvector's HNSW index cannot index
        the model's native 3072-dim output, which exceeds its 2000-dim cap).
        `mode` selects the prefix used to signal task intent to the model.
        """
        prefixed_text = self._build_prefixed_text(text, mode, title)
        response = await self.client.aio.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=[prefixed_text],
            config={"output_dimensionality": settings.EMBEDDING_DIMENSIONS},
        )
        return response.embeddings[0].values

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        mode: EmbeddingTask = EmbeddingTask.DOCUMENT,
        title: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generates gemini-embedding-2 vectors for a list of text strings,
        all sharing the same task `mode` (and, for document mode, the same
        source `title`).
        """
        if not texts:
            return []

        prefixed_texts = [self._build_prefixed_text(t, mode, title) for t in texts]
        response = await self.client.aio.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=prefixed_texts,
            config={"output_dimensionality": settings.EMBEDDING_DIMENSIONS},
        )
        return [e.values for e in response.embeddings]

    async def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks using a CrossEncoder and updates their scores.
        Sorts the chunks in descending order by the new score.
        """
        if not chunks:
            return []

        # Prepare inputs: list of [query, chunk_content] pairs
        inputs = [[query, chunk["content"]] for chunk in chunks]

        def _predict() -> Any:
            return self.reranker.predict(inputs)

        scores = await asyncio.to_thread(_predict)

        for i, chunk in enumerate(chunks):
            chunk["score"] = float(scores[i])
            chunk["source_type"] = "reranked"

        # Sort chunks in descending order of score
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks


embedding_service = EmbeddingService()

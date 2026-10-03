import asyncio
import logging
import threading
from enum import Enum
from typing import Any, List, Optional, Dict

import torch
from google import genai

from app.config.settings import settings

logger = logging.getLogger(__name__)


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
        self._reranker_lock = threading.Lock()
        self._reranker_unavailable = False

    @property
    def client(self) -> genai.Client:
        """
        Lazy-inits the Gemini GenAI client on first use.
        """
        if self._client is None:
            api_key = str(settings.GEMINI_API_KEY).strip().strip("'\"")
            self._client = genai.Client(api_key=api_key)
        return self._client
    
    def load_reranker(self) -> None:
        """
        Sync, idempotent, thread-safe method to load the reranker model.
        """
        if settings.RERANK_PROVIDER != "local":
            self._reranker_unavailable = True
            return

        if self._reranker is not None or self._reranker_unavailable:
            return

        with self._reranker_lock:
            # Double checked locking
            if self._reranker is not None or self._reranker_unavailable:
                return

            try:
                if settings.RERANK_NUM_THREADS > 0:
                    torch.set_num_threads(settings.RERANK_NUM_THREADS)
                
                logger.info(f"Loading CrossEncoder reranker ({settings.RERANK_MODEL_NAME})...")
                from sentence_transformers import CrossEncoder
                
                model = CrossEncoder(
                    settings.RERANK_MODEL_NAME, 
                    device="cpu", 
                    max_length=settings.RERANK_MAX_LENGTH
                )
                
                if settings.RERANK_QUANTIZE:
                    try:
                        logger.info("Quantizing the reranker model...")
                        model.model = torch.quantization.quantize_dynamic(
                            model.model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    except Exception as e:
                        logger.warning(f"Reranker quantization failed, continuing with unquantized model: {e}")
                
                self._reranker = model
            except Exception as e:
                logger.error(f"❌ Failed to load local reranker model: {e}")
                self._reranker_unavailable = True

    def warmup_reranker(self) -> None:
        """
        Loads the reranker and runs a dummy predict to warm it up.
        """
        self.load_reranker()
        if self._reranker is not None:
            logger.info("Warming up reranker with dummy predict...")
            with torch.inference_mode():
                self._reranker.predict([["query", "passage"]], batch_size=1, show_progress_bar=False)
            logger.info("Reranker warmup complete.")

    @property
    def reranker(self) -> Any:
        """
        Lazy-loads a cross-encoder model for hybrid search reranking.
        """
        if self._reranker is None and not self._reranker_unavailable:
            self.load_reranker()
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
        If the reranker is unavailable, fails open and returns the chunks in original order.
        """
        if not chunks:
            return []

        # Fail open
        if self.reranker is None:
            return chunks

        inputs = [[query, chunk["content"]] for chunk in chunks]

        def _predict() -> Any:
            with torch.inference_mode():
                return self.reranker.predict(
                    inputs, 
                    batch_size=settings.RERANK_BATCH_SIZE, 
                    show_progress_bar=False
                )

        try:
            raw_logits = await asyncio.to_thread(_predict)
            
            # Use torch.sigmoid for consistent conversion, or mathematically equivalent
            scores = torch.sigmoid(torch.tensor(raw_logits)).tolist()
            logits = raw_logits.tolist() if hasattr(raw_logits, 'tolist') else list(raw_logits)

            for i, chunk in enumerate(chunks):
                chunk["rerank_logit"] = float(logits[i])
                chunk["rerank_score"] = float(scores[i])
                # DO NOT overwrite 'score' or 'source_type'

            # Sort chunks in descending order of score, stable for ties
            chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return chunks
        except Exception as e:
            logger.warning(f"Reranker failed during inference: {e}. Failing open.")
            return chunks


embedding_service = EmbeddingService()

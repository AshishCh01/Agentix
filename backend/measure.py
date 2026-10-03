import asyncio
import time
import psutil
from app.config.settings import settings
from ai.services.embedding_service import embedding_service
import torch

def measure():
    process = psutil.Process()
    print("--- WITHOUT QUANTIZE ---")
    settings.RERANK_QUANTIZE = False
    
    t0 = time.time()
    embedding_service.load_reranker()
    t1 = time.time()
    mem1 = process.memory_info().rss / 1024 / 1024
    
    print(f"Load time: {t1 - t0:.4f}s")
    print(f"Memory after load: {mem1:.2f} MB")
    
    dummy_chunks_8 = [{"content": f"Dummy passage {i}"} for i in range(8)]
    dummy_chunks_20 = [{"content": f"Dummy passage {i}"} for i in range(20)]
    
    t0 = time.time()
    asyncio.run(embedding_service.rerank_chunks("query", dummy_chunks_8))
    t1 = time.time()
    print(f"Rerank 8 candidates latency: {t1 - t0:.4f}s")
    
    t0 = time.time()
    asyncio.run(embedding_service.rerank_chunks("query", dummy_chunks_20))
    t1 = time.time()
    print(f"Rerank 20 candidates latency: {t1 - t0:.4f}s")
    
    print("\n--- WITH QUANTIZE ---")
    settings.RERANK_QUANTIZE = True
    embedding_service._reranker = None
    
    t0 = time.time()
    embedding_service.load_reranker()
    t1 = time.time()
    mem2 = process.memory_info().rss / 1024 / 1024
    
    print(f"Load time: {t1 - t0:.4f}s")
    print(f"Memory after quantize load: {mem2:.2f} MB")
    
    t0 = time.time()
    asyncio.run(embedding_service.rerank_chunks("query", dummy_chunks_8))
    t1 = time.time()
    print(f"Rerank 8 candidates latency: {t1 - t0:.4f}s")
    
    t0 = time.time()
    asyncio.run(embedding_service.rerank_chunks("query", dummy_chunks_20))
    t1 = time.time()
    print(f"Rerank 20 candidates latency: {t1 - t0:.4f}s")

if __name__ == "__main__":
    measure()

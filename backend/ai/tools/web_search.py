import logging
import asyncio
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _run_ddgs_sync(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Synchronous DDGS execution to be run in a thread pool."""
    results = []
    
    # Prefer official modern ddgs package, fallback to legacy duckduckgo_search
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=max_results))
        for item in raw_results:
            raw_score = (
                item.get("score")
                if item.get("score") is not None
                else item.get("similarity_score")
                if item.get("similarity_score") is not None
                else item.get("relevance_score")
                if item.get("relevance_score") is not None
                else item.get("rating")
            )
            score = float(raw_score) if raw_score is not None else None
            results.append({
                "title": item.get("title", "Untitled Source"),
                "url": item.get("href", "") or item.get("url", ""),
                "content": item.get("body", "") or item.get("description", ""),
                "score": score,
            })
    return results


async def web_search_tool(query: str, max_results: int = 4) -> Dict[str, Any]:
    """
    Executes a web search query asynchronously to prevent blocking the 
    FastAPI event loop. Outputs formatted sources for citation UI.
    """
    try:
        # Offload blocking network call to a separate thread
        results = await asyncio.to_thread(_run_ddgs_sync, query, max_results)

        if not results:
            return {
                "query": query,
                "results": [],
                "sources": [],
                "formatted_context": "No relevant web search results found.",
            }

        context_blocks = []
        sources = []
        
        for idx, res in enumerate(results):
            context_blocks.append(
                f"Source: {res['title']} ({res['url']})\nContent: {res['content']}"
            )
            raw_score = (
                res.get("score")
                if res.get("score") is not None
                else res.get("similarity_score")
                if res.get("similarity_score") is not None
                else res.get("relevance_score")
                if res.get("relevance_score") is not None
                else res.get("rating")
            )
            score = float(raw_score) if raw_score is not None else None

            # Map explicitly to the ChatSource Pydantic schema
            sources.append({
                "filename": res["title"],
                "url": res["url"],
                "chunk_index": idx,
                "score": score,
                "page_number": None,
                "source_type": res.get("source_type", "web"),
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)

        return {
            "query": query,
            "results": results,
            "sources": sources,
            "formatted_context": formatted_context,
        }

    except Exception as e:
        logger.error(f"❌ DuckDuckGo web search error: {str(e)}")
        return {
            "query": query,
            "results": [],
            "sources": [],
            "formatted_context": (
                "Web search service is currently unavailable. "
                "Proceeding with assistant general knowledge."
            ),
        }
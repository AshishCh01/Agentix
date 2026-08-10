import logging
import asyncio
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _run_ddgs_sync(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Synchronous DDGS execution to be run in a thread pool."""
    results = []
    
    # Handle the dependency alias safely
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        from ddgs import DDGS

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=max_results))
        for item in raw_results:
            results.append({
                "title": item.get("title", "Untitled Source"),
                "url": item.get("href", ""),
                "content": item.get("body", ""),
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
            # Map explicitly to the ChatSource Pydantic schema
            sources.append({
                "filename": res["title"],
                "url": res["url"],
                "chunk_index": idx,
                "similarity_score": 1.0, 
                "page_number": None
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
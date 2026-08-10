import logging
from typing import Any, Dict
from ddgs import DDGS

logger = logging.getLogger(__name__)


async def web_search_tool(query: str, max_results: int = 4) -> Dict[str, Any]:
    """
    Executes a web search query using DuckDuckGo without requiring an API key.
    Safely catches exceptions to ensure LangGraph state transitions do not crash.
    """
    try:
        results = []
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
            for item in raw_results:
                results.append({
                    "title": item.get("title", "Untitled Source"),
                    "url": item.get("href", ""),
                    "content": item.get("body", ""),
                })

        context_blocks = [
            f"Source: {res['title']} ({res['url']})\nContent: {res['content']}"
            for res in results
        ]
        formatted_context = "\n\n---\n\n".join(context_blocks)

        return {
            "query": query,
            "results": results,
            "formatted_context": formatted_context or "No relevant web search results found.",
        }

    except Exception as e:
        logger.error(f"❌ DuckDuckGo web search error: {str(e)}")
        return {
            "query": query,
            "results": [],
            "formatted_context": (
                "Web search service is currently unavailable. "
                "Proceeding with assistant general knowledge."
            ),
        }
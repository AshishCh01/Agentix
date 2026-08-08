import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


async def web_search_tool(query: str, max_results: int = 4) -> Dict[str, Any]:
    """
    Executes a web search via DuckDuckGo and returns structured snippets.
    """
    results: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            try:
                raw_results = list(ddgs.text(query, max_results=max_results))
            except Exception:
                raw_results = list(ddgs.text(keywords=query, max_results=max_results))

            for item in raw_results:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href") or item.get("link", ""),
                        "snippet": item.get("body") or item.get("snippet", ""),
                    }
                )

        formatted_snippets = "\n\n".join(
            [
                f"Source [{i+1}] ({res['url']}):\n{res['snippet']}"
                for i, res in enumerate(results)
                if res["snippet"]
            ]
        )

        logger.info(f"Web search fetched {len(results)} snippets for query: '{query}'")

        return {
            "query": query,
            "results": results,
            "formatted_context": formatted_snippets,
        }
    except Exception as e:
        logger.error(f"Web search tool error: {str(e)}")
        return {
            "query": query,
            "results": [],
            "formatted_context": "",
            "error": str(e),
        }
from langchain.tools import tool
from typing import Optional
import logging
import requests
from config.settings import settings

logger = logging.getLogger(__name__)


@tool
def image_search_tool(
    query: str,
    limit: Optional[int] = 5,
) -> str:
    """
    Search for images related to an emergency event using Tavily.

    Args:
        query: The image search query
        limit: Number of image results to return (default 5)

    Returns:
        Formatted string of image URLs with descriptions
    """
    if not settings.TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not configured."

    try:
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "include_images": True,
            "include_image_descriptions": True,
            "max_results": limit,
        }
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=15,
        )

        if not response.ok:
            return f"Error: HTTP {response.status_code} from Tavily"

        data = response.json()
        images = data.get("images", [])

        if not images:
            return f"No images found for: {query}"

        lines = [f"Images found for: '{query}'\n"]
        for i, img in enumerate(images[:limit], 1):
            if isinstance(img, dict):
                lines.append(
                    f"{i}. URL: {img.get('url', img)}\n"
                    f"   Description: {img.get('description', 'No description')}\n"
                )
            else:
                lines.append(f"{i}. URL: {img}\n")

        logger.info(f"[image_search_tool] {len(images)} images for: {query}")
        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Error: Image search request timed out"
    except Exception as e:
        logger.error(f"[image_search_tool] Error: {e}", exc_info=True)
        return f"Error: {str(e)}"
